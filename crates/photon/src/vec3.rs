//! A tiny 3D vector type.
//!
//! **Rust concepts:** `Copy` types, operator overloading through the
//! `std::ops` traits, `macro_rules!` to stamp out repetitive impls,
//! associated constants, and `const fn`.

use std::ops::{Add, AddAssign, Div, Index, Mul, MulAssign, Neg, Sub};

/// Three `f32`s. Used for points, directions and RGB colors alike.
///
/// `Copy` means a `Vec3` is duplicated bit-for-bit on assignment instead of
/// being *moved*, so you can keep using a value after passing it to a
/// function. That's only allowed because every field is itself `Copy`.
///
/// `#[repr(C)]` pins the memory layout to "x, y, z, in order, no padding",
/// so a `&[Vec3]` can be handed to JavaScript as a flat array of floats.
/// Without it, Rust is free to reorder fields.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(C)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Type aliases don't create new types; they just document intent.
pub type Point3 = Vec3;
pub type Color = Vec3;

impl Vec3 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 1.0, 1.0);

    /// `const fn` can run at compile time, which is what lets us use it to
    /// define the associated constants above.
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    pub const fn splat(v: f32) -> Self {
        Self::new(v, v, v)
    }

    pub fn dot(self, o: Self) -> f32 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Self) -> Self {
        Self::new(self.y * o.z - self.z * o.y, self.z * o.x - self.x * o.z, self.x * o.y - self.y * o.x)
    }

    pub fn length_squared(self) -> f32 {
        self.dot(self)
    }

    pub fn length(self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn normalized(self) -> Self {
        self / self.length()
    }

    pub fn min(self, o: Self) -> Self {
        Self::new(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }

    pub fn max(self, o: Self) -> Self {
        Self::new(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }

    pub fn max_component(self) -> f32 {
        self.x.max(self.y).max(self.z)
    }

    /// Perceived brightness (Rec. 709 weights).
    pub fn luminance(self) -> f32 {
        0.2126 * self.x + 0.7152 * self.y + 0.0722 * self.z
    }

    pub fn near_zero(self) -> bool {
        const EPS: f32 = 1e-6;
        self.x.abs() < EPS && self.y.abs() < EPS && self.z.abs() < EPS
    }

    /// Mirror reflection of `self` about the surface normal `n`.
    pub fn reflect(self, n: Self) -> Self {
        self - n * (2.0 * self.dot(n))
    }

    /// Snell's law. `self` and `n` must be unit vectors; `eta` is the ratio
    /// of refractive indices (outside / inside).
    pub fn refract(self, n: Self, eta: f32) -> Self {
        let cos_theta = (-self).dot(n).min(1.0);
        let perp = (self + n * cos_theta) * eta;
        let parallel = n * -(1.0 - perp.length_squared()).abs().sqrt();
        perp + parallel
    }

    /// Linear interpolation: `t = 0` gives `self`, `t = 1` gives `o`.
    pub fn lerp(self, o: Self, t: f32) -> Self {
        self * (1.0 - t) + o * t
    }

    /// Rotate around the Y axis by `angle` radians.
    pub fn rotate_y(self, angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self::new(c * self.x + s * self.z, self.y, -s * self.x + c * self.z)
    }
}

/// Indexing by axis (`v[0]` is x). Handy for the BVH, which splits along
/// whichever axis is longest.
impl Index<usize> for Vec3 {
    type Output = f32;

    fn index(&self, axis: usize) -> &f32 {
        match axis {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("Vec3 axis out of range: {axis}"),
        }
    }
}

impl Neg for Vec3 {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.x, -self.y, -self.z)
    }
}

/// Writing `impl Add for Vec3`, `impl Sub for Vec3`, ... by hand is pure
/// boilerplate. A declarative macro generates them all. `$tr:ident` matches
/// an identifier such as `Add`; `$op:tt` matches a single token such as `+`.
macro_rules! componentwise_ops {
    ($($tr:ident, $method:ident, $op:tt);* $(;)?) => {$(
        // Vec3 (op) Vec3, component by component.
        impl $tr for Vec3 {
            type Output = Self;
            fn $method(self, o: Self) -> Self {
                Self::new(self.x $op o.x, self.y $op o.y, self.z $op o.z)
            }
        }
        // Vec3 (op) f32, applied to every component.
        impl $tr<f32> for Vec3 {
            type Output = Self;
            fn $method(self, s: f32) -> Self {
                Self::new(self.x $op s, self.y $op s, self.z $op s)
            }
        }
    )*};
}

componentwise_ops! {
    Add, add, +;
    Sub, sub, -;
    Mul, mul, *;
    Div, div, /;
}

/// `f32 * Vec3` needs its own impl because the left-hand side decides which
/// `Mul` implementation is used.
impl Mul<Vec3> for f32 {
    type Output = Vec3;

    fn mul(self, v: Vec3) -> Vec3 {
        v * self
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

impl MulAssign for Vec3 {
    fn mul_assign(&mut self, o: Self) {
        *self = *self * o;
    }
}

impl MulAssign<f32> for Vec3 {
    fn mul_assign(&mut self, s: f32) {
        *self = *self * s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: Vec3, b: Vec3) -> bool {
        (a - b).length() < 1e-5
    }

    #[test]
    fn arithmetic_is_componentwise() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(4.0, 5.0, 6.0);
        assert_eq!(a + b, Vec3::new(5.0, 7.0, 9.0));
        assert_eq!(b - a, Vec3::splat(3.0));
        assert_eq!(a * b, Vec3::new(4.0, 10.0, 18.0));
        assert_eq!(2.0 * a, a * 2.0);
        assert_eq!(-a, Vec3::new(-1.0, -2.0, -3.0));
    }

    #[test]
    fn dot_and_cross() {
        let x = Vec3::new(1.0, 0.0, 0.0);
        let y = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(x.dot(y), 0.0);
        assert_eq!(x.cross(y), Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn reflect_flips_normal_component() {
        let v = Vec3::new(1.0, -1.0, 0.0);
        let n = Vec3::new(0.0, 1.0, 0.0);
        assert_eq!(v.reflect(n), Vec3::new(1.0, 1.0, 0.0));
    }

    #[test]
    fn refract_with_eta_one_is_identity() {
        let v = Vec3::new(1.0, -1.0, 0.0).normalized();
        let n = Vec3::new(0.0, 1.0, 0.0);
        assert!(approx(v.refract(n, 1.0), v));
    }

    #[test]
    fn rotate_y_quarter_turn() {
        let v = Vec3::new(1.0, 0.0, 0.0).rotate_y(std::f32::consts::FRAC_PI_2);
        assert!(approx(v, Vec3::new(0.0, 0.0, -1.0)));
    }
}
