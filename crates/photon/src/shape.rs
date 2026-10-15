//! Geometry: spheres and quads (parallelograms), plus bounding boxes.
//!
//! **Rust concepts:** enums with data ("sum types") and exhaustive `match`,
//! returning `Option` instead of sentinel values, and the `?` operator on
//! `Option`.
//!
//! Why an `enum Shape` rather than a `trait Shape` with `Box<dyn Shape>`?
//! Both work. Trait objects are *open* (anyone can add a shape) but every
//! call goes through a vtable and every object is a separate heap
//! allocation. An enum is *closed* (all variants live here) but values sit
//! contiguously in a `Vec` and calls are a plain `match` the compiler can
//! inline. For a hot inner loop with a fixed set of shapes, the enum wins.

use crate::camera::Ray;
use crate::rng::Pcg32;
use crate::vec3::{Point3, Vec3};

/// The geometric part of a ray hit (no material yet).
#[derive(Debug, Clone, Copy)]
pub struct HitGeom {
    pub t: f32,
    pub point: Point3,
    /// Always points *against* the incoming ray.
    pub normal: Vec3,
    /// True when the ray hit the outside of the surface.
    pub front_face: bool,
}

impl HitGeom {
    fn new(ray: &Ray, t: f32, outward_normal: Vec3) -> Self {
        let front_face = ray.dir.dot(outward_normal) < 0.0;
        let normal = if front_face { outward_normal } else { -outward_normal };
        Self { t, point: ray.at(t), normal, front_face }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Shape {
    Sphere {
        center: Point3,
        radius: f32,
    },
    /// The parallelogram `corner + a*u + b*v` for `a, b` in `[0, 1]`.
    /// The other fields are precomputed once to make intersection cheap.
    Quad {
        corner: Point3,
        u: Vec3,
        v: Vec3,
        normal: Vec3,
        d: f32,
        w: Vec3,
    },
}

impl Shape {
    pub fn sphere(center: Point3, radius: f32) -> Self {
        Shape::Sphere { center, radius }
    }

    pub fn quad(corner: Point3, u: Vec3, v: Vec3) -> Self {
        let n = u.cross(v);
        let normal = n.normalized();
        Shape::Quad { corner, u, v, normal, d: normal.dot(corner), w: n / n.dot(n) }
    }

    /// Find the nearest intersection with `t` in `(t_min, t_max)`.
    pub fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitGeom> {
        match *self {
            Shape::Sphere { center, radius } => {
                // Solve |origin + t*dir - center|^2 = r^2 for t. `dir` is a
                // unit vector so the quadratic's `a` term is 1.
                let oc = ray.origin - center;
                let half_b = oc.dot(ray.dir);
                let c = oc.length_squared() - radius * radius;
                let discriminant = half_b * half_b - c;
                if discriminant < 0.0 {
                    return None;
                }
                let sqrtd = discriminant.sqrt();
                // Try the nearer root first, then the farther one.
                let t =
                    [-half_b - sqrtd, -half_b + sqrtd].into_iter().find(|t| (t_min..t_max).contains(t))?;
                let outward = (ray.at(t) - center) / radius;
                Some(HitGeom::new(ray, t, outward))
            }
            Shape::Quad { corner, u, v, normal, d, w } => {
                let denom = normal.dot(ray.dir);
                if denom.abs() < 1e-8 {
                    return None; // Parallel to the plane.
                }
                let t = (d - normal.dot(ray.origin)) / denom;
                if !(t_min..t_max).contains(&t) {
                    return None;
                }
                // Express the hit point in the quad's (u, v) coordinates.
                let planar = ray.at(t) - corner;
                let alpha = w.dot(planar.cross(v));
                let beta = w.dot(u.cross(planar));
                let inside = |x: f32| (0.0..=1.0).contains(&x);
                (inside(alpha) && inside(beta)).then(|| HitGeom::new(ray, t, normal))
            }
        }
    }

    pub fn bounds(&self) -> Aabb {
        match *self {
            Shape::Sphere { center, radius } => {
                let r = Vec3::splat(radius.abs());
                Aabb::new(center - r, center + r)
            }
            Shape::Quad { corner, u, v, .. } => {
                let mut b = Aabb::EMPTY;
                for p in [corner, corner + u, corner + v, corner + u + v] {
                    b = b.grow(p);
                }
                b.padded(1e-4)
            }
        }
    }

    /// A uniformly distributed point on the surface and the normal there.
    /// Used to aim shadow rays at lights.
    pub fn sample_point(&self, rng: &mut Pcg32) -> (Point3, Vec3) {
        match *self {
            Shape::Sphere { center, radius } => {
                let n = rng.unit_vector();
                (center + n * radius, n)
            }
            Shape::Quad { corner, u, v, normal, .. } => {
                (corner + u * rng.next_f32() + v * rng.next_f32(), normal)
            }
        }
    }

    /// Surface area: needed to sample a point on a light uniformly.
    pub fn area(&self) -> f32 {
        match *self {
            Shape::Sphere { radius, .. } => 4.0 * std::f32::consts::PI * radius * radius,
            Shape::Quad { u, v, .. } => u.cross(v).length(),
        }
    }
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Point3,
    pub max: Point3,
}

impl Aabb {
    /// Inverted infinite box: growing it by any point yields that point.
    pub const EMPTY: Self = Self { min: Vec3::splat(f32::INFINITY), max: Vec3::splat(f32::NEG_INFINITY) };

    pub fn new(min: Point3, max: Point3) -> Self {
        Self { min, max }
    }

    pub fn grow(self, p: Point3) -> Self {
        Self::new(self.min.min(p), self.max.max(p))
    }

    pub fn union(self, o: Self) -> Self {
        Self::new(self.min.min(o.min), self.max.max(o.max))
    }

    pub fn padded(self, eps: f32) -> Self {
        Self::new(self.min - Vec3::splat(eps), self.max + Vec3::splat(eps))
    }

    pub fn centroid(&self) -> Point3 {
        (self.min + self.max) * 0.5
    }

    pub fn extent(&self) -> Vec3 {
        self.max - self.min
    }

    pub fn surface_area(&self) -> f32 {
        let e = self.extent();
        if e.x < 0.0 {
            return 0.0; // EMPTY
        }
        2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
    }

    pub fn longest_axis(&self) -> usize {
        let e = self.extent();
        if e.x >= e.y && e.x >= e.z {
            0
        } else if e.y >= e.z {
            1
        } else {
            2
        }
    }

    /// The "slab test". Returns the entry distance if the ray hits the box
    /// within `(t_min, t_max)`. `inv_dir` is `1 / ray.dir`, precomputed once
    /// per ray because division is slow and this runs millions of times.
    #[inline]
    pub fn hit(&self, origin: Point3, inv_dir: Vec3, t_min: f32, t_max: f32) -> Option<f32> {
        let t0 = (self.min - origin) * inv_dir;
        let t1 = (self.max - origin) * inv_dir;
        let near = t0.min(t1).max_component().max(t_min);
        let far = t0.max(t1);
        let far = far.x.min(far.y).min(far.z).min(t_max);
        (near <= far).then_some(near)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ray_hits_sphere_front() {
        let s = Shape::sphere(Point3::new(0.0, 0.0, -5.0), 1.0);
        let r = Ray::new(Point3::ZERO, Vec3::new(0.0, 0.0, -1.0));
        let hit = s.hit(&r, 1e-3, f32::INFINITY).expect("should hit");
        assert!((hit.t - 4.0).abs() < 1e-5);
        assert!(hit.front_face);
        assert_eq!(hit.normal, Vec3::new(0.0, 0.0, 1.0));
    }

    #[test]
    fn ray_inside_sphere_hits_back_face() {
        let s = Shape::sphere(Point3::ZERO, 2.0);
        let r = Ray::new(Point3::ZERO, Vec3::new(1.0, 0.0, 0.0));
        let hit = s.hit(&r, 1e-3, f32::INFINITY).unwrap();
        assert!((hit.t - 2.0).abs() < 1e-5);
        assert!(!hit.front_face);
        assert_eq!(hit.normal, Vec3::new(-1.0, 0.0, 0.0));
    }

    #[test]
    fn ray_misses_sphere() {
        let s = Shape::sphere(Point3::new(0.0, 5.0, -5.0), 1.0);
        let r = Ray::new(Point3::ZERO, Vec3::new(0.0, 0.0, -1.0));
        assert!(s.hit(&r, 1e-3, f32::INFINITY).is_none());
    }

    #[test]
    fn quad_hit_inside_and_miss_outside() {
        let q =
            Shape::quad(Point3::new(-1.0, -1.0, -3.0), Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.0, 2.0, 0.0));
        let inside = Ray::new(Point3::ZERO, Vec3::new(0.0, 0.0, -1.0));
        assert!((q.hit(&inside, 1e-3, 100.0).unwrap().t - 3.0).abs() < 1e-5);
        let outside = Ray::new(Point3::new(5.0, 0.0, 0.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(q.hit(&outside, 1e-3, 100.0).is_none());
        assert!((q.area() - 4.0).abs() < 1e-5);
    }

    #[test]
    fn aabb_slab_test() {
        let b = Aabb::new(Vec3::splat(-1.0), Vec3::splat(1.0));
        let dir = Vec3::new(0.0, 0.0, -1.0);
        let inv = Vec3::ONE / dir; // Infinite components are fine here.
        assert_eq!(b.hit(Point3::new(0.0, 0.0, 5.0), inv, 0.0, 100.0), Some(4.0));
        assert_eq!(b.hit(Point3::new(3.0, 0.0, 5.0), inv, 0.0, 100.0), None);
    }
}
