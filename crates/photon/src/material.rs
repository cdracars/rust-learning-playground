//! Materials, textures and skies: how light interacts with what it hits.
//!
//! **Rust concepts:** enums as closed sets of behaviours, struct-like enum
//! variants, and `match` with guards and bindings.

use crate::vec3::{Color, Point3, Vec3};

#[derive(Debug, Clone, Copy)]
pub enum Texture {
    Solid(Color),
    /// A 3D checkerboard: each unit cube (after scaling) alternates color.
    Checker {
        even: Color,
        odd: Color,
        scale: f32,
    },
}

impl Texture {
    pub fn value(&self, p: Point3) -> Color {
        match *self {
            Texture::Solid(c) => c,
            Texture::Checker { even, odd, scale } => {
                let cell = (p.x * scale).floor() + (p.y * scale).floor() + (p.z * scale).floor();
                if cell.rem_euclid(2.0) < 1.0 { even } else { odd }
            }
        }
    }
}

impl From<Color> for Texture {
    /// Lets builders accept a plain color wherever a texture is expected.
    fn from(c: Color) -> Self {
        Texture::Solid(c)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Material {
    /// Matte surface that scatters light evenly in all directions.
    Diffuse(Texture),
    /// Mirror. `roughness` 0 is perfect, 1 is very blurry.
    Metal { albedo: Color, roughness: f32 },
    /// Glass, water, diamond... `ior` is the index of refraction.
    Glass { ior: f32, tint: Color },
    /// Emits light; doesn't reflect any.
    Light(Color),
}

impl Material {
    pub fn diffuse(c: impl Into<Texture>) -> Self {
        Material::Diffuse(c.into())
    }

    pub fn metal(albedo: Color, roughness: f32) -> Self {
        Material::Metal { albedo, roughness: roughness.clamp(0.0, 1.0) }
    }

    pub fn glass(ior: f32) -> Self {
        Material::Glass { ior, tint: Color::ONE }
    }

    pub fn light(color: Color, intensity: f32) -> Self {
        Material::Light(color * intensity)
    }
}

/// Schlick's approximation: glass reflects more at grazing angles.
pub fn schlick_reflectance(cosine: f32, eta: f32) -> f32 {
    let r0 = ((1.0 - eta) / (1.0 + eta)).powi(2);
    r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
}

/// What a ray sees when it escapes the scene.
#[derive(Debug, Clone, Copy)]
pub enum Sky {
    Solid(Color),
    Gradient {
        horizon: Color,
        zenith: Color,
    },
    /// A procedural sunset with a glowing sun disk.
    Sunset {
        sun_dir: Vec3,
    },
}

impl Sky {
    pub fn color(&self, dir: Vec3) -> Color {
        match *self {
            Sky::Solid(c) => c,
            Sky::Gradient { horizon, zenith } => {
                let t = 0.5 * (dir.y + 1.0);
                horizon.lerp(zenith, t)
            }
            Sky::Sunset { sun_dir } => {
                let up = dir.y.max(0.0);
                let zenith = Color::new(0.08, 0.12, 0.35);
                let mid = Color::new(0.85, 0.35, 0.25);
                let horizon = Color::new(1.6, 0.75, 0.3);
                let sky = if up < 0.25 {
                    horizon.lerp(mid, up / 0.25)
                } else {
                    mid.lerp(zenith, ((up - 0.25) / 0.75).powf(0.6))
                };
                let sky = if dir.y < 0.0 {
                    horizon.lerp(Color::new(0.15, 0.08, 0.1), (-dir.y * 4.0).min(1.0))
                } else {
                    sky
                };
                let cos = dir.dot(sun_dir).max(0.0);
                let disk = if cos > 0.9995 { 60.0 } else { 0.0 };
                let glow = cos.powi(64) * 3.0 + cos.powi(8) * 0.6;
                sky + Color::new(1.0, 0.7, 0.4) * (disk + glow)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_alternates() {
        let t = Texture::Checker { even: Color::ONE, odd: Color::ZERO, scale: 1.0 };
        assert_eq!(t.value(Point3::new(0.5, 0.5, 0.5)), Color::ONE);
        assert_eq!(t.value(Point3::new(1.5, 0.5, 0.5)), Color::ZERO);
        assert_eq!(t.value(Point3::new(-0.5, 0.5, 0.5)), Color::ZERO);
    }

    #[test]
    fn schlick_is_total_at_grazing_angle() {
        assert!((schlick_reflectance(0.0, 1.5) - 1.0).abs() < 1e-6);
        assert!(schlick_reflectance(1.0, 1.5) < 0.05);
    }
}
