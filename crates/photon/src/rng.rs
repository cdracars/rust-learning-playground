//! A small, fast, deterministic random number generator (PCG32).
//!
//! **Rust concepts:** wrapping integer arithmetic (overflow is a panic in
//! debug builds unless you ask for wrapping), `&mut self` methods, and why
//! a renderer wants *its own* RNG instead of a global one.
//!
//! Every thread (or web worker) owns a `Pcg32`, so there's no locking, and
//! seeding with the pixel position and sample index makes renders
//! reproducible.

use crate::vec3::Vec3;
use std::f32::consts::TAU;

/// PCG-XSH-RR, see <https://www.pcg-random.org>.
#[derive(Debug, Clone)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    const MULTIPLIER: u64 = 6_364_136_223_846_793_005;

    /// `stream` selects one of 2^63 independent sequences.
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut rng = Self { state: 0, inc: (stream << 1) | 1 };
        rng.next_u32();
        rng.state = rng.state.wrapping_add(seed);
        rng.next_u32();
        rng
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(Self::MULTIPLIER).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in `[0, 1)`. Uses the top 24 bits: exactly what fits in an
    /// `f32` mantissa.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// Uniformly distributed direction on the unit sphere.
    pub fn unit_vector(&mut self) -> Vec3 {
        let z = self.range(-1.0, 1.0);
        let phi = TAU * self.next_f32();
        let r = (1.0 - z * z).max(0.0).sqrt();
        Vec3::new(r * phi.cos(), r * phi.sin(), z)
    }

    /// Uniform point inside the unit disk in the XY plane (for depth of
    /// field: a point on the camera's lens).
    pub fn in_unit_disk(&mut self) -> Vec3 {
        let r = self.next_f32().sqrt();
        let theta = TAU * self.next_f32();
        Vec3::new(r * theta.cos(), r * theta.sin(), 0.0)
    }

    /// Uniform point inside the unit ball.
    pub fn in_unit_ball(&mut self) -> Vec3 {
        self.unit_vector() * self.next_f32().cbrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_sequence() {
        let mut a = Pcg32::new(42, 7);
        let mut b = Pcg32::new(42, 7);
        for _ in 0..100 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn different_streams_differ() {
        let mut a = Pcg32::new(42, 1);
        let mut b = Pcg32::new(42, 2);
        let same = (0..100).filter(|_| a.next_u32() == b.next_u32()).count();
        assert!(same < 3);
    }

    #[test]
    fn floats_are_in_unit_interval_and_roughly_uniform() {
        let mut rng = Pcg32::new(1, 1);
        let n = 100_000;
        let mut sum = 0.0f64;
        for _ in 0..n {
            let x = rng.next_f32();
            assert!((0.0..1.0).contains(&x));
            sum += x as f64;
        }
        let mean = sum / n as f64;
        assert!((mean - 0.5).abs() < 0.01, "mean was {mean}");
    }

    #[test]
    fn unit_vectors_are_unit_length() {
        let mut rng = Pcg32::new(3, 3);
        for _ in 0..1000 {
            assert!((rng.unit_vector().length() - 1.0).abs() < 1e-4);
            assert!(rng.in_unit_disk().length() <= 1.0 + 1e-6);
        }
    }
}
