//! `photon`: render a scene to a PNG on every CPU core.
//!
//! ```text
//! cargo run --release -p photon-cli -- cornell --spp 512 --out cornell.png
//! ```
//!
//! **Rust concepts:** scoped threads (`std::thread::scope`) borrowing data
//! from the parent stack frame, an `AtomicUsize` as a lock-free work queue,
//! `mpsc` channels to send results back, and error handling with
//! `Result<_, String>` and `?`.

mod png;

use photon::scenes::{self, SCENES};
use photon::{Camera, Color, Frame, render_rows, tonemap};
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::Instant;

struct Options {
    scene: String,
    width: usize,
    height: Option<usize>,
    spp: u32,
    depth: u32,
    threads: usize,
    exposure: f32,
    out: Option<String>,
}

const USAGE: &str = "\
Usage: photon [SCENE] [OPTIONS]

Options:
  --width N      Image width in pixels           (default 960)
  --height N     Image height (default: scene aspect ratio)
  --spp N        Samples per pixel               (default 128)
  --depth N      Maximum bounces per path        (default 12)
  --threads N    Worker threads                  (default: all cores)
  --exposure F   Brightness multiplier           (default 1.0)
  --out FILE     Output PNG                      (default SCENE.png)
  --list         Show available scenes
  --help         Show this message";

fn parse_args() -> Result<Options, String> {
    let mut opts = Options {
        scene: "cornell".into(),
        width: 960,
        height: None,
        spp: 128,
        depth: 12,
        threads: std::thread::available_parallelism().map_or(4, |n| n.get()),
        exposure: 1.0,
        out: None,
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        // A closure that grabs the flag's value from the same iterator.
        let mut value = |name: &str| args.next().ok_or(format!("{name} needs a value"));
        match arg.as_str() {
            "--width" => opts.width = num(value("--width")?, "--width")?,
            "--height" => opts.height = Some(num(value("--height")?, "--height")?),
            "--spp" => opts.spp = num(value("--spp")?, "--spp")?,
            "--depth" => opts.depth = num(value("--depth")?, "--depth")?,
            "--threads" => opts.threads = num(value("--threads")?, "--threads")?,
            "--exposure" => opts.exposure = num(value("--exposure")?, "--exposure")?,
            "--out" => opts.out = Some(value("--out")?),
            "--list" => {
                for s in SCENES {
                    println!("{:<10} {}", s.name, s.description);
                }
                std::process::exit(0);
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            flag if flag.starts_with('-') => return Err(format!("unknown option {flag}\n\n{USAGE}")),
            name => opts.scene = name.to_string(),
        }
    }
    if opts.width == 0 || opts.spp == 0 || opts.threads == 0 {
        return Err("--width, --spp and --threads must be positive".into());
    }
    Ok(opts)
}

/// Generic over any `T: FromStr`, so one function parses usize, u32 and f32.
fn num<T: std::str::FromStr>(s: String, name: &str) -> Result<T, String> {
    s.parse().map_err(|_| format!("invalid value for {name}: {s}"))
}

fn main() {
    if let Err(e) = run() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let opts = parse_args()?;
    let scene =
        scenes::by_name(&opts.scene).ok_or_else(|| format!("unknown scene '{}' (try --list)", opts.scene))?;
    let default_aspect = if scene.name == "cornell" { 1.0 } else { 16.0 / 9.0 };
    let height = opts.height.unwrap_or((opts.width as f32 / default_aspect).round() as usize);
    let frame = Frame { width: opts.width, height, max_depth: opts.depth, seed: 0x5EED };
    let camera = Camera::new(&scene.camera, frame.aspect());

    eprintln!(
        "Rendering '{}': {}x{} @ {} spp, {} objects, {} BVH nodes, {} threads",
        scene.name,
        frame.width,
        frame.height,
        opts.spp,
        scene.objects().len(),
        scene.bvh().node_count(),
        opts.threads,
    );

    // Work is handed out in bands of rows. Each thread repeatedly grabs the
    // next band index with an atomic `fetch_add`: no mutex, no contention.
    const BAND: usize = 4;
    let bands = frame.height.div_ceil(BAND);
    let next_band = AtomicUsize::new(0);
    let (tx, rx) = mpsc::channel::<(usize, Vec<Color>)>();
    let mut hdr = vec![Color::ZERO; frame.width * frame.height];
    let start = Instant::now();

    // `scope` guarantees every spawned thread finishes before it returns,
    // so the threads may borrow `scene`, `camera` and `next_band` directly.
    // No `Arc`, no `'static` bound, no cloning the scene per thread.
    std::thread::scope(|s| {
        for _ in 0..opts.threads {
            let tx = tx.clone();
            let (scene, camera, next_band) = (&scene, &camera, &next_band);
            s.spawn(move || {
                loop {
                    let band = next_band.fetch_add(1, Ordering::Relaxed);
                    if band >= bands {
                        break;
                    }
                    let rows = band * BAND..((band + 1) * BAND).min(frame.height);
                    let mut buf = vec![Color::ZERO; rows.len() * frame.width];
                    for sample in 0..opts.spp {
                        render_rows(scene, camera, &frame, rows.clone(), sample, &mut buf);
                    }
                    if tx.send((band, buf)).is_err() {
                        break;
                    }
                }
            });
        }
        // Drop the original sender so `rx` ends once every worker is done.
        drop(tx);

        for (done, (band, buf)) in rx.iter().enumerate() {
            let offset = band * BAND * frame.width;
            hdr[offset..offset + buf.len()].copy_from_slice(&buf);
            progress(done + 1, bands, start);
        }
    });

    let secs = start.elapsed().as_secs_f64();
    let paths = (frame.width * frame.height) as f64 * opts.spp as f64;
    eprintln!("\nDone in {secs:.2}s ({:.2} M paths/s)", paths / secs / 1e6);

    let inv = 1.0 / opts.spp as f32;
    let rgb: Vec<u8> = hdr.iter().flat_map(|&c| tonemap(c * inv, opts.exposure)).collect();

    let path = opts.out.unwrap_or_else(|| format!("{}.png", scene.name));
    let file = std::fs::File::create(&path).map_err(|e| format!("can't create {path}: {e}"))?;
    let mut w = BufWriter::new(file);
    png::write_png(&mut w, frame.width, frame.height, &rgb)
        .and_then(|_| w.flush())
        .map_err(|e| format!("can't write {path}: {e}"))?;
    eprintln!("Wrote {path}");
    Ok(())
}

fn progress(done: usize, total: usize, start: Instant) {
    const WIDTH: usize = 40;
    let frac = done as f64 / total as f64;
    let filled = (frac * WIDTH as f64) as usize;
    let elapsed = start.elapsed().as_secs_f64();
    let eta = if done > 0 { elapsed / frac - elapsed } else { 0.0 };
    eprint!(
        "\r[{}{}] {:5.1}%  eta {:>5.1}s",
        "█".repeat(filled),
        "·".repeat(WIDTH - filled),
        frac * 100.0,
        eta
    );
}
