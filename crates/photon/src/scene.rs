//! The scene and the light transport algorithm (the "integrator").
//!
//! **Rust concepts:** the newtype pattern (`MatId`), the builder pattern
//! with `&mut self` chaining, `let ... else`, and borrowing: `trace` takes
//! `&self`, so many threads can render one `Scene` at the same time without
//! any locks. The compiler proves that's safe because `Scene` is `Sync`.
//!
//! ## How path tracing works, in one paragraph
//!
//! For each pixel we shoot a ray into the scene and follow it as it
//! bounces: off a mirror, through glass, scattering off a matte wall.
//! `throughput` tracks how much light can still make it back to the camera
//! along this path (each bounce absorbs some). Whenever the path touches a
//! light, or escapes to the sky, we add that light times the throughput.
//! One path per pixel is very noisy; averaging hundreds converges to a
//! physically accurate image with soft shadows, color bleeding, caustics
//! and depth of field for free.
//!
//! On matte surfaces we also use *next event estimation*: pick a random
//! point on a random light and check directly whether it's visible. This
//! makes small lights converge far faster than waiting for a random bounce
//! to stumble into them.

use crate::bvh::Bvh;
use crate::camera::{CameraConfig, Ray};
use crate::material::{Material, Sky, schlick_reflectance};
use crate::rng::Pcg32;
use crate::shape::{HitGeom, Shape};
use crate::vec3::{Color, Vec3};
use std::f32::consts::PI;

/// Offset to avoid a ray re-hitting the surface it just left ("shadow acne").
const T_MIN: f32 = 1e-3;

/// A handle to a material in a [`SceneBuilder`].
///
/// Wrapping the `u32` in its own type means you can't accidentally pass an
/// object index where a material index is expected. It costs nothing at
/// runtime: the wrapper is erased by the compiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatId(u32);

#[derive(Debug, Clone, Copy)]
pub struct Object {
    pub shape: Shape,
    pub material: MatId,
}

#[derive(Debug, Clone)]
pub struct Scene {
    pub name: &'static str,
    pub camera: CameraConfig,
    pub sky: Sky,
    objects: Vec<Object>,
    materials: Vec<Material>,
    /// Indices of emissive objects, for next event estimation.
    lights: Vec<u32>,
    bvh: Bvh,
}

impl Scene {
    pub fn objects(&self) -> &[Object] {
        &self.objects
    }

    pub fn bvh(&self) -> &Bvh {
        &self.bvh
    }

    /// Nearest hit along the ray, if any.
    pub fn hit(&self, ray: &Ray) -> Option<(&Object, HitGeom)> {
        let mut best = None;
        self.bvh.closest_hit(ray, T_MIN, f32::INFINITY, |i, t_max| {
            let obj = &self.objects[i as usize];
            let hit = obj.shape.hit(ray, T_MIN, t_max)?;
            best = Some((obj, hit));
            Some(hit.t)
        });
        best
    }

    fn occluded(&self, ray: &Ray, t_max: f32) -> bool {
        self.bvh
            .any_hit(ray, T_MIN, t_max, |i| self.objects[i as usize].shape.hit(ray, T_MIN, t_max).is_some())
    }

    /// Follow one light path and return the radiance it carries back.
    pub fn trace(&self, mut ray: Ray, rng: &mut Pcg32, max_depth: u32) -> Color {
        let mut radiance = Color::ZERO;
        let mut throughput = Color::ONE;
        // After a matte bounce, light was already counted by next event
        // estimation; counting it again on a direct hit would double it.
        let mut count_emission = true;

        for depth in 0..max_depth {
            let Some((obj, hit)) = self.hit(&ray) else {
                radiance += throughput * self.sky.color(ray.dir);
                break;
            };

            match self.materials[obj.material.0 as usize] {
                Material::Light(emitted) => {
                    if count_emission {
                        radiance += throughput * emitted;
                    }
                    break;
                }
                Material::Diffuse(texture) => {
                    let albedo = texture.value(hit.point);
                    radiance += throughput * albedo * self.sample_direct(&hit, rng);
                    // Cosine-weighted bounce: normal + random unit vector.
                    let dir = hit.normal + rng.unit_vector();
                    let dir = if dir.near_zero() { hit.normal } else { dir.normalized() };
                    ray = Ray::new(hit.point, dir);
                    throughput *= albedo;
                    count_emission = false;
                }
                Material::Metal { albedo, roughness } => {
                    let reflected = ray.dir.reflect(hit.normal);
                    let dir = (reflected + rng.in_unit_ball() * roughness).normalized();
                    if dir.dot(hit.normal) <= 0.0 {
                        break; // Scattered below the surface: absorbed.
                    }
                    ray = Ray::new(hit.point, dir);
                    throughput *= albedo;
                    count_emission = true;
                }
                Material::Glass { ior, tint } => {
                    let eta = if hit.front_face { 1.0 / ior } else { ior };
                    let cos_theta = (-ray.dir).dot(hit.normal).min(1.0);
                    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
                    let must_reflect = eta * sin_theta > 1.0; // Total internal reflection.
                    let dir = if must_reflect || schlick_reflectance(cos_theta, eta) > rng.next_f32() {
                        ray.dir.reflect(hit.normal)
                    } else {
                        ray.dir.refract(hit.normal, eta)
                    };
                    ray = Ray::new(hit.point, dir.normalized());
                    throughput *= tint;
                    count_emission = true;
                }
            }

            // Russian roulette: randomly end dim paths, boosting survivors
            // so the average stays correct (unbiased) while saving work.
            if depth >= 3 {
                let survive = throughput.max_component().clamp(0.05, 0.95);
                if rng.next_f32() > survive {
                    break;
                }
                throughput = throughput / survive;
            }
        }
        radiance
    }

    /// Next event estimation: light arriving directly from one random point
    /// on one random light, divided by π (the Lambertian BRDF without albedo).
    fn sample_direct(&self, hit: &HitGeom, rng: &mut Pcg32) -> Color {
        if self.lights.is_empty() {
            return Color::ZERO;
        }
        let pick = (rng.next_u32() as usize) % self.lights.len();
        let light = &self.objects[self.lights[pick] as usize];
        let Material::Light(emitted) = self.materials[light.material.0 as usize] else {
            unreachable!("only emissive objects are registered as lights");
        };

        let (point, light_normal) = light.shape.sample_point(rng);
        let to_light = point - hit.point;
        let dist2 = to_light.length_squared();
        let dist = dist2.sqrt();
        let l = to_light / dist;
        let cos_surface = hit.normal.dot(l);
        let cos_light = light_normal.dot(l).abs();
        if cos_surface <= 0.0 || cos_light < 1e-6 {
            return Color::ZERO;
        }
        if self.occluded(&Ray::new(hit.point, l), dist * (1.0 - 1e-3)) {
            return Color::ZERO;
        }
        // Convert the area sample to solid angle: pdf = dist² / (cos_l · area),
        // and we picked this light with probability 1 / lights.len().
        let weight = cos_surface * cos_light * light.shape.area() * self.lights.len() as f32 / (PI * dist2);
        emitted * weight
    }
}

/// Accumulates objects and materials, then builds the BVH once at the end.
#[derive(Debug, Default)]
pub struct SceneBuilder {
    objects: Vec<Object>,
    materials: Vec<Material>,
}

impl SceneBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn material(&mut self, m: Material) -> MatId {
        self.materials.push(m);
        MatId(self.materials.len() as u32 - 1)
    }

    pub fn add(&mut self, shape: Shape, material: MatId) -> &mut Self {
        self.objects.push(Object { shape, material });
        self
    }

    /// An axis-aligned box of `size`, rotated around Y, with its min corner
    /// at `position` before rotation. Made from six quads.
    pub fn add_box(&mut self, position: Vec3, size: Vec3, rotate_y_deg: f32, m: MatId) -> &mut Self {
        let angle = rotate_y_deg.to_radians();
        let rot = |v: Vec3| v.rotate_y(angle);
        let (dx, dy, dz) =
            (Vec3::new(size.x, 0.0, 0.0), Vec3::new(0.0, size.y, 0.0), Vec3::new(0.0, 0.0, size.z));
        let faces = [
            (Vec3::ZERO, dx, dy),
            (dz, dx, dy),
            (Vec3::ZERO, dz, dy),
            (dx, dz, dy),
            (Vec3::ZERO, dx, dz),
            (dy, dx, dz),
        ];
        for (corner, u, v) in faces {
            self.add(Shape::quad(position + rot(corner), rot(u), rot(v)), m);
        }
        self
    }

    pub fn build(self, name: &'static str, camera: CameraConfig, sky: Sky) -> Scene {
        let bounds: Vec<_> = self.objects.iter().map(|o| o.shape.bounds()).collect();
        let bvh = Bvh::build(&bounds);
        let lights = self
            .objects
            .iter()
            .enumerate()
            .filter(|(_, o)| matches!(self.materials[o.material.0 as usize], Material::Light(_)))
            .map(|(i, _)| i as u32)
            .collect();
        Scene { name, camera, sky, objects: self.objects, materials: self.materials, lights, bvh }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vec3::Point3;

    /// A "furnace test": a white diffuse sphere inside a uniformly white sky
    /// must look exactly as bright as the sky, since energy is conserved and
    /// nothing absorbs. Any bias in the integrator shows up here.
    #[test]
    fn white_furnace_conserves_energy() {
        let mut b = SceneBuilder::new();
        let white = b.material(Material::diffuse(Color::ONE));
        b.add(Shape::sphere(Point3::ZERO, 1.0), white);
        let scene = b.build("furnace", CameraConfig::default(), Sky::Solid(Color::ONE));

        let mut rng = Pcg32::new(9, 9);
        let n = 20_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let ray = Ray::new(Point3::new(0.0, 0.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
            sum += scene.trace(ray, &mut rng, 64).luminance();
        }
        let mean = sum / n as f32;
        assert!((mean - 1.0).abs() < 0.03, "furnace mean {mean}");
    }

    /// Next event estimation and plain bouncing must agree: a matte floor
    /// lit by a quad light should get the same brightness either way.
    #[test]
    fn direct_light_estimate_is_unbiased() {
        let mut b = SceneBuilder::new();
        let floor = b.material(Material::diffuse(Color::splat(0.5)));
        let lamp = b.material(Material::light(Color::ONE, 4.0));
        b.add(
            Shape::quad(
                Point3::new(-50.0, 0.0, -50.0),
                Vec3::new(100.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 100.0),
            ),
            floor,
        );
        b.add(
            Shape::quad(Point3::new(-1.0, 2.0, -1.0), Vec3::new(2.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 2.0)),
            lamp,
        );
        let scene = b.build("nee", CameraConfig::default(), Sky::Solid(Color::ZERO));

        // Analytic-ish reference by brute force: no NEE, cosine sampling only.
        let p = Point3::new(0.3, 0.0, 0.2);
        let hit = HitGeom { t: 1.0, point: p, normal: Vec3::new(0.0, 1.0, 0.0), front_face: true };
        let mut rng = Pcg32::new(5, 5);
        let n = 200_000;
        let (mut nee, mut brute) = (0.0, 0.0);
        for _ in 0..n {
            nee += scene.sample_direct(&hit, &mut rng).luminance();
            let dir = (hit.normal + rng.unit_vector()).normalized();
            // A "let chain": two pattern matches joined with `&&`.
            if let Some((o, _)) = scene.hit(&Ray::new(p, dir))
                && let Material::Light(e) = scene.materials[o.material.0 as usize]
            {
                brute += e.luminance();
            }
        }
        let (nee, brute) = (nee / n as f32, brute / n as f32);
        assert!((nee - brute).abs() / brute < 0.03, "nee {nee} vs brute {brute}");
    }
}
