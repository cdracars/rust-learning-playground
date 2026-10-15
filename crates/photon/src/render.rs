//! Turning rays into pixels: sampling, accumulation and tone mapping.
//!
//! **Rust concepts:** slices (`&mut [Color]`) as views into a larger
//! buffer, `std::ops::Range`, and `#[inline]`.

use crate::camera::Camera;
use crate::rng::Pcg32;
use crate::scene::Scene;
use crate::vec3::Color;
use std::ops::Range;

#[derive(Debug, Clone, Copy)]
pub struct Frame {
    pub width: usize,
    pub height: usize,
    pub max_depth: u32,
    /// Mixed into every pixel's RNG seed. Different seeds give different
    /// (but equally valid) noise patterns.
    pub seed: u64,
}

impl Frame {
    pub fn aspect(&self) -> f32 {
        self.width as f32 / self.height as f32
    }
}

/// The RNG for one pixel and one sample index. Because it only depends on
/// *which* sample is being computed, not on which thread computes it or in
/// what order, renders are bit-for-bit reproducible across thread counts.
fn pixel_rng(frame: &Frame, x: usize, y: usize, sample: u32) -> Pcg32 {
    let pixel = (y * frame.width + x) as u64;
    Pcg32::new(frame.seed ^ ((sample as u64) << 32), pixel)
}

/// Very bright, rare paths ("fireflies") take thousands of samples to
/// average out. Clamping them adds a little bias but removes a lot of noise.
const FIREFLY_CLAMP: f32 = 10.0;

#[inline]
pub fn sample_pixel(scene: &Scene, cam: &Camera, frame: &Frame, x: usize, y: usize, sample: u32) -> Color {
    let mut rng = pixel_rng(frame, x, y, sample);
    // Jitter within the pixel: free anti-aliasing.
    let s = (x as f32 + rng.next_f32()) / frame.width as f32;
    let t = (y as f32 + rng.next_f32()) / frame.height as f32;
    let c = scene.trace(cam.ray(s, t, &mut rng), &mut rng, frame.max_depth);
    if !(c.x.is_finite() && c.y.is_finite() && c.z.is_finite()) {
        return Color::ZERO;
    }
    let peak = c.max_component();
    if peak > FIREFLY_CLAMP { c * (FIREFLY_CLAMP / peak) } else { c }
}

/// Add one sample per pixel for every row in `rows` into `out`, which holds
/// exactly those rows.
pub fn render_rows(
    scene: &Scene,
    cam: &Camera,
    frame: &Frame,
    rows: Range<usize>,
    sample: u32,
    out: &mut [Color],
) {
    debug_assert_eq!(out.len(), rows.len() * frame.width);
    // `chunks_exact_mut` hands out one non-overlapping `&mut` row at a time.
    for (y, row) in rows.zip(out.chunks_exact_mut(frame.width)) {
        for (x, px) in row.iter_mut().enumerate() {
            *px += sample_pixel(scene, cam, frame, x, y, sample);
        }
    }
}

/// ACES filmic tone curve (Narkowicz fit): maps unbounded HDR radiance into
/// `[0, 1]` with a pleasant shoulder instead of harsh clipping.
fn aces(x: f32) -> f32 {
    ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
}

/// Linear light to the sRGB curve that displays expect.
fn linear_to_srgb(x: f32) -> f32 {
    if x <= 0.003_130_8 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 }
}

/// Convert an averaged HDR color into an 8-bit display pixel.
pub fn tonemap(c: Color, exposure: f32) -> [u8; 3] {
    let f = |v: f32| (linear_to_srgb(aces(v * exposure)) * 255.0 + 0.5) as u8;
    [f(c.x), f(c.y), f(c.z)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tonemap_black_and_bright() {
        assert_eq!(tonemap(Color::ZERO, 1.0), [0, 0, 0]);
        assert_eq!(tonemap(Color::splat(1000.0), 1.0), [255, 255, 255]);
    }

    #[test]
    fn tonemap_is_monotonic() {
        let mut prev = 0;
        for i in 0..200 {
            let v = tonemap(Color::splat(i as f32 * 0.05), 1.0)[0];
            assert!(v >= prev);
            prev = v;
        }
    }
}
