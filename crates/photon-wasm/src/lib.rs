//! Photon for the browser, with a hand-written WebAssembly interface.
//!
//! Most projects use `wasm-bindgen` to glue Rust and JavaScript together.
//! This crate deliberately doesn't, to show what's underneath: WebAssembly
//! functions can only exchange numbers (`i32`, `i64`, `f32`, `f64`). To
//! share anything bigger, Rust returns a *pointer*, which is just an offset
//! into the module's linear memory, and JavaScript reads the bytes there
//! through a typed-array view:
//!
//! ```js
//! const ptr = wasm.exports.tonemap(1.0);
//! const pixels = new Uint8ClampedArray(wasm.exports.memory.buffer, ptr, w * h * 4);
//! ```
//!
//! **Rust concepts:** `extern "C"` and `#[unsafe(no_mangle)]` to export
//! functions with stable names, `thread_local!` + `RefCell` for global
//! mutable state without `static mut`, and `#[repr(C)]` (see `Vec3`) to make
//! a struct's memory layout something other languages can rely on.
//!
//! The same module is loaded twice in the page: in Web Workers that call
//! `render_band` (the heavy lifting, one worker per CPU core), and on the
//! main thread, which merges results with `accum_ptr` and displays them via
//! `tonemap`. Because every sample is seeded by its pixel and index, it
//! doesn't matter which worker renders what.

use photon::scenes::SCENES;
use photon::{Camera, CameraConfig, Color, Frame, Point3, Scene, render_rows, tonemap as tonemap_px};
use std::cell::RefCell;

struct Engine {
    scene: Scene,
    scene_index: usize,
    frame: Frame,
    camera_cfg: CameraConfig,
    camera: Camera,
    /// Output of the latest `render_band` call.
    band: Vec<Color>,
    /// Running sum of all samples, per pixel (main thread only).
    accum: Vec<Color>,
    /// Samples accumulated so far, per row. Rows progress independently, so
    /// each is averaged by its own count.
    counts: Vec<u32>,
    rgba: Vec<u8>,
    camera_out: [f32; 9],
}

impl Engine {
    fn new(scene_index: usize) -> Self {
        let scene = (SCENES[scene_index].build)();
        let camera_cfg = scene.camera;
        let frame = Frame { width: 1, height: 1, max_depth: 8, seed: 0x5EED };
        Engine {
            camera: Camera::new(&camera_cfg, 1.0),
            scene,
            scene_index,
            frame,
            camera_cfg,
            band: Vec::new(),
            accum: Vec::new(),
            counts: Vec::new(),
            rgba: Vec::new(),
            camera_out: [0.0; 9],
        }
    }

    fn set_camera(&mut self, cfg: CameraConfig) {
        self.camera_cfg = cfg;
        self.camera = Camera::new(&cfg, self.frame.aspect());
    }
}

thread_local! {
    // WebAssembly (without the threads proposal) is single-threaded, but
    // Rust still won't let us touch a plain `static mut` without `unsafe`.
    // `thread_local!` + `RefCell` gives safe, checked mutable access.
    static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
    static TEXT: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn with_engine<R>(f: impl FnOnce(&mut Engine) -> R) -> R {
    ENGINE.with_borrow_mut(|e| f(e.as_mut().expect("configure() must be called first")))
}

/// Number of built-in scenes.
#[unsafe(no_mangle)]
pub extern "C" fn scene_count() -> u32 {
    SCENES.len() as u32
}

/// Writes `"name\ndescription"` as UTF-8 and returns a pointer to it; the
/// length is available from [`text_len`].
#[unsafe(no_mangle)]
pub extern "C" fn scene_info(index: u32) -> *const u8 {
    let info = &SCENES[index as usize % SCENES.len()];
    TEXT.with_borrow_mut(|t| {
        t.clear();
        t.extend_from_slice(info.name.as_bytes());
        t.push(b'\n');
        t.extend_from_slice(info.description.as_bytes());
        t.as_ptr()
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn text_len() -> u32 {
    TEXT.with_borrow(|t| t.len() as u32)
}

/// Select a scene and image size. Switching scenes also resets the camera
/// to that scene's default. Clears accumulated samples.
#[unsafe(no_mangle)]
pub extern "C" fn configure(scene: u32, width: u32, height: u32, max_depth: u32) {
    let scene = scene as usize % SCENES.len();
    ENGINE.with_borrow_mut(|slot| {
        if slot.as_ref().is_none_or(|e| e.scene_index != scene) {
            *slot = Some(Engine::new(scene));
        }
        let e = slot.as_mut().expect("just initialized");
        let (w, h) = (width.max(1) as usize, height.max(1) as usize);
        e.frame = Frame { width: w, height: h, max_depth: max_depth.clamp(1, 64), ..e.frame };
        e.set_camera(e.camera_cfg);
        e.accum.clear();
        e.accum.resize(w * h, Color::ZERO);
        e.counts.clear();
        e.counts.resize(h, 0);
        e.rgba.resize(w * h * 4, 255);
    });
}

#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn set_camera(
    fx: f32,
    fy: f32,
    fz: f32,
    ax: f32,
    ay: f32,
    az: f32,
    vfov: f32,
    aperture: f32,
    focus: f32,
) {
    with_engine(|e| {
        e.set_camera(CameraConfig {
            look_from: Point3::new(fx, fy, fz),
            look_at: Point3::new(ax, ay, az),
            vfov,
            aperture,
            focus_dist: focus,
            ..e.camera_cfg
        })
    });
}

/// Pointer to 9 floats: look_from (3), look_at (3), vfov, aperture, focus.
#[unsafe(no_mangle)]
pub extern "C" fn camera_ptr() -> *const f32 {
    with_engine(|e| {
        let c = e.camera_cfg;
        e.camera_out = [
            c.look_from.x,
            c.look_from.y,
            c.look_from.z,
            c.look_at.x,
            c.look_at.y,
            c.look_at.z,
            c.vfov,
            c.aperture,
            c.focus_dist,
        ];
        e.camera_out.as_ptr()
    })
}

/// Render one sample for every pixel in rows `[row_start, row_end)`.
/// Returns a pointer to `(row_end - row_start) * width * 3` floats.
#[unsafe(no_mangle)]
pub extern "C" fn render_band(row_start: u32, row_end: u32, sample: u32) -> *const f32 {
    with_engine(|e| {
        let rows = row_start as usize..(row_end as usize).min(e.frame.height);
        e.band.clear();
        e.band.resize(rows.len() * e.frame.width, Color::ZERO);
        render_rows(&e.scene, &e.camera, &e.frame, rows, sample, &mut e.band);
        // Sound because `Vec3` is `#[repr(C)]`: three consecutive f32s.
        e.band.as_ptr().cast()
    })
}

/// The `width * height * 3` float accumulation buffer, for JS to add into.
#[unsafe(no_mangle)]
pub extern "C" fn accum_ptr() -> *mut f32 {
    with_engine(|e| e.accum.as_mut_ptr().cast())
}

/// Per-row sample counts (`height` u32s), maintained by JS.
#[unsafe(no_mangle)]
pub extern "C" fn counts_ptr() -> *mut u32 {
    with_engine(|e| e.counts.as_mut_ptr())
}

/// Average and tone map the accumulation buffer into RGBA bytes.
#[unsafe(no_mangle)]
pub extern "C" fn tonemap(exposure: f32) -> *const u8 {
    with_engine(|e| {
        let w = e.frame.width;
        for (y, &n) in e.counts.iter().enumerate() {
            let inv = if n > 0 { 1.0 / n as f32 } else { 0.0 };
            let src = &e.accum[y * w..(y + 1) * w];
            let dst = &mut e.rgba[y * w * 4..(y + 1) * w * 4];
            for (c, px) in src.iter().zip(dst.chunks_exact_mut(4)) {
                let [r, g, b] = tonemap_px(*c * inv, exposure);
                px.copy_from_slice(&[r, g, b, 255]);
            }
        }
        e.rgba.as_ptr()
    })
}

/// Distance from the camera, measured along its view direction, to the
/// surface under normalized screen position `(sx, sy)`. Used for
/// click-to-focus. Returns -1 if nothing is there.
#[unsafe(no_mangle)]
pub extern "C" fn pick(sx: f32, sy: f32) -> f32 {
    with_engine(|e| {
        let pinhole = Camera::new(&CameraConfig { aperture: 0.0, ..e.camera_cfg }, e.frame.aspect());
        let ray = pinhole.ray(sx, sy, &mut photon::rng::Pcg32::new(0, 0));
        let forward = (e.camera_cfg.look_at - e.camera_cfg.look_from).normalized();
        match e.scene.hit(&ray) {
            Some((_, hit)) => (hit.point - ray.origin).dot(forward),
            None => -1.0,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The FFI is plain Rust functions, so it's testable natively too.
    #[test]
    fn render_accumulate_and_tonemap() {
        configure(0, 8, 4, 4);
        let band = render_band(0, 4, 0);
        // Simulate what the JS does: add the band into the accumulator.
        unsafe {
            let src = std::slice::from_raw_parts(band, 8 * 4 * 3);
            let dst = std::slice::from_raw_parts_mut(accum_ptr(), 8 * 4 * 3);
            for (d, s) in dst.iter_mut().zip(src) {
                *d += s;
            }
            std::slice::from_raw_parts_mut(counts_ptr(), 4).fill(1);
        }
        let rgba = unsafe { std::slice::from_raw_parts(tonemap(1.0), 8 * 4 * 4) };
        assert!(rgba.chunks(4).all(|px| px[3] == 255));
        assert!(rgba.chunks(4).any(|px| px[0] > 0), "image shouldn't be all black");
    }

    #[test]
    fn pick_hits_the_back_wall_of_the_cornell_box() {
        configure(0, 16, 16, 4);
        let d = pick(0.5, 0.5);
        assert!(d > 800.0 && d < 1400.0, "distance was {d}");
    }

    #[test]
    fn scene_info_round_trips() {
        let p = scene_info(0);
        let text = unsafe { std::slice::from_raw_parts(p, text_len() as usize) };
        assert!(std::str::from_utf8(text).unwrap().starts_with("cornell\n"));
    }
}
