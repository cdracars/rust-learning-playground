//! Rays and a thin-lens camera with depth of field.
//!
//! **Rust concepts:** plain-data config structs, `Default` for sensible
//! starting values, and struct update syntax (`..Default::default()`).

use crate::rng::Pcg32;
use crate::vec3::{Point3, Vec3};

#[derive(Debug, Clone, Copy)]
pub struct Ray {
    pub origin: Point3,
    pub dir: Vec3,
}

impl Ray {
    pub fn new(origin: Point3, dir: Vec3) -> Self {
        Self { origin, dir }
    }

    pub fn at(&self, t: f32) -> Point3 {
        self.origin + self.dir * t
    }
}

/// Everything needed to describe a camera. Kept separate from [`Camera`] so
/// it's cheap to tweak and rebuild (the browser does this on every drag).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraConfig {
    pub look_from: Point3,
    pub look_at: Point3,
    pub up: Vec3,
    /// Vertical field of view in degrees.
    pub vfov: f32,
    /// Lens diameter. Zero gives a pinhole camera where everything is sharp.
    pub aperture: f32,
    /// Distance to the plane that is in perfect focus.
    pub focus_dist: f32,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            look_from: Point3::new(0.0, 0.0, 5.0),
            look_at: Point3::ZERO,
            up: Vec3::new(0.0, 1.0, 0.0),
            vfov: 40.0,
            aperture: 0.0,
            focus_dist: 5.0,
        }
    }
}

/// A camera ready to generate rays for a given image aspect ratio.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    origin: Point3,
    upper_left: Point3,
    horizontal: Vec3,
    vertical: Vec3,
    u: Vec3,
    v: Vec3,
    lens_radius: f32,
}

impl Camera {
    pub fn new(cfg: &CameraConfig, aspect: f32) -> Self {
        let h = (cfg.vfov.to_radians() / 2.0).tan();
        let viewport_h = 2.0 * h;
        let viewport_w = aspect * viewport_h;

        // Orthonormal basis: w points backwards, u right, v up.
        let w = (cfg.look_from - cfg.look_at).normalized();
        let u = cfg.up.cross(w).normalized();
        let v = w.cross(u);

        let horizontal = u * (viewport_w * cfg.focus_dist);
        let vertical = v * (viewport_h * cfg.focus_dist);
        let upper_left = cfg.look_from - horizontal / 2.0 + vertical / 2.0 - w * cfg.focus_dist;

        Self {
            origin: cfg.look_from,
            upper_left,
            horizontal,
            vertical,
            u,
            v,
            lens_radius: cfg.aperture / 2.0,
        }
    }

    /// A ray through the image at normalized coordinates `(s, t)`, where
    /// `(0, 0)` is the top-left corner and `(1, 1)` the bottom-right.
    pub fn ray(&self, s: f32, t: f32, rng: &mut Pcg32) -> Ray {
        let offset = if self.lens_radius > 0.0 {
            let rd = rng.in_unit_disk() * self.lens_radius;
            self.u * rd.x + self.v * rd.y
        } else {
            Vec3::ZERO
        };
        let origin = self.origin + offset;
        let target = self.upper_left + self.horizontal * s - self.vertical * t;
        Ray::new(origin, (target - origin).normalized())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn center_ray_points_at_target() {
        let cfg = CameraConfig { look_from: Point3::new(0.0, 0.0, 10.0), ..Default::default() };
        let cam = Camera::new(&cfg, 16.0 / 9.0);
        let mut rng = Pcg32::new(0, 0);
        let ray = cam.ray(0.5, 0.5, &mut rng);
        assert!((ray.dir - Vec3::new(0.0, 0.0, -1.0)).length() < 1e-5);
    }

    #[test]
    fn top_left_ray_goes_up_and_left() {
        let cam = Camera::new(&CameraConfig::default(), 1.0);
        let ray = cam.ray(0.0, 0.0, &mut Pcg32::new(0, 0));
        assert!(ray.dir.x < 0.0 && ray.dir.y > 0.0);
    }
}
