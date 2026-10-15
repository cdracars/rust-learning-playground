//! A gallery of built-in scenes.
//!
//! **Rust concepts:** function pointers in a `const` table, iterator
//! adaptors (`flat_map`, `filter`), and `&'static str`.

use crate::camera::CameraConfig;
use crate::material::{Material, Sky, Texture};
use crate::rng::Pcg32;
use crate::scene::{Scene, SceneBuilder};
use crate::shape::Shape;
use crate::vec3::{Color, Point3, Vec3};

pub struct SceneInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub build: fn() -> Scene,
}

/// Every built-in scene. The CLI and the browser both read this table, so
/// adding a scene here makes it show up everywhere.
pub const SCENES: &[SceneInfo] = &[
    SceneInfo {
        name: "cornell",
        description: "The classic Cornell box with a glass sphere and a brushed-gold block",
        build: cornell,
    },
    SceneInfo {
        name: "sunset",
        description: "Glass, gold and copper spheres under a procedural sunset",
        build: sunset,
    },
    SceneInfo {
        name: "neon",
        description: "A synthwave grid of glowing lines reflected in glossy black",
        build: neon,
    },
    SceneInfo {
        name: "weekend",
        description: "The cover image of 'Ray Tracing in One Weekend': ~480 random spheres",
        build: weekend,
    },
];

pub fn by_name(name: &str) -> Option<Scene> {
    SCENES.iter().find(|s| s.name == name).map(|s| (s.build)())
}

fn rgb(hex: u32) -> Color {
    let c = |shift: u32| ((hex >> shift) & 0xff) as f32 / 255.0;
    // Convert sRGB to linear so colors match what you'd pick in a color picker.
    let lin = |v: f32| v.powf(2.2);
    Color::new(lin(c(16)), lin(c(8)), lin(c(0)))
}

pub fn cornell() -> Scene {
    let mut b = SceneBuilder::new();
    let red = b.material(Material::diffuse(Color::new(0.65, 0.05, 0.05)));
    let white = b.material(Material::diffuse(Color::splat(0.73)));
    let green = b.material(Material::diffuse(Color::new(0.12, 0.45, 0.15)));
    let lamp = b.material(Material::light(Color::new(1.0, 0.85, 0.6), 15.0));
    let gold = b.material(Material::metal(Color::new(0.95, 0.7, 0.35), 0.25));
    let glass = b.material(Material::glass(1.5));

    let s = 555.0;
    let x = Vec3::new(s, 0.0, 0.0);
    let y = Vec3::new(0.0, s, 0.0);
    let z = Vec3::new(0.0, 0.0, s);
    b.add(Shape::quad(x, y, z), green)
        .add(Shape::quad(Point3::ZERO, y, z), red)
        .add(Shape::quad(Point3::ZERO, x, z), white)
        .add(Shape::quad(y, x, z), white)
        .add(Shape::quad(z, x, y), white)
        .add(
            Shape::quad(
                Point3::new(213.0, 554.0, 227.0),
                Vec3::new(130.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 105.0),
            ),
            lamp,
        )
        .add_box(Point3::new(265.0, 0.0, 295.0), Vec3::new(165.0, 330.0, 165.0), 15.0, white)
        // Balanced on top of the block (its rotated top-face center).
        .add(Shape::sphere(Point3::new(366.0, 390.0, 353.0), 60.0), gold)
        .add(Shape::sphere(Point3::new(170.0, 90.0, 170.0), 90.0), glass);

    let camera = CameraConfig {
        look_from: Point3::new(278.0, 278.0, -800.0),
        look_at: Point3::new(278.0, 278.0, 0.0),
        vfov: 40.0,
        focus_dist: 900.0,
        ..Default::default()
    };
    b.build("cornell", camera, Sky::Solid(Color::ZERO))
}

pub fn sunset() -> Scene {
    let mut b = SceneBuilder::new();
    let floor = b.material(Material::diffuse(Texture::Checker {
        even: rgb(0xe8e0d0),
        odd: rgb(0x403530),
        scale: 1.0,
    }));
    let glass = b.material(Material::glass(1.5));
    let gold = b.material(Material::metal(rgb(0xffcc66), 0.05));
    let copper = b.material(Material::metal(rgb(0xd9825b), 0.35));
    let matte = b.material(Material::diffuse(rgb(0x5b8fd9)));
    let teal = b.material(Material::Glass { ior: 1.45, tint: rgb(0x9fe8e0) });

    b.add(
        Shape::quad(Point3::new(-60.0, 0.0, -60.0), Vec3::new(120.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 120.0)),
        floor,
    )
    .add(Shape::sphere(Point3::new(0.0, 1.0, 0.0), 1.0), glass)
    // A negative radius flips the normals: a hollow bubble inside the glass.
    .add(Shape::sphere(Point3::new(0.0, 1.0, 0.0), -0.9), glass)
    .add(Shape::sphere(Point3::new(-2.2, 1.0, -0.6), 1.0), gold)
    .add(Shape::sphere(Point3::new(2.2, 1.0, -0.6), 1.0), copper)
    .add(Shape::sphere(Point3::new(-1.0, 0.45, 1.6), 0.45), matte)
    .add(Shape::sphere(Point3::new(1.2, 0.5, 1.5), 0.5), teal);

    let camera = CameraConfig {
        look_from: Point3::new(0.0, 1.6, 7.5),
        look_at: Point3::new(0.0, 0.8, 0.0),
        vfov: 36.0,
        aperture: 0.08,
        focus_dist: 7.4,
        ..Default::default()
    };
    let sun_dir = Vec3::new(-0.3, 0.12, -1.0).normalized();
    b.build("sunset", camera, Sky::Sunset { sun_dir })
}

pub fn neon() -> Scene {
    let mut b = SceneBuilder::new();
    let floor = b.material(Material::metal(Color::splat(0.35), 0.12));
    let pink = b.material(Material::light(rgb(0xff2a9d), 6.0));
    let cyan = b.material(Material::light(rgb(0x22d3ee), 6.0));
    let sun = b.material(Material::light(rgb(0xff7a3d), 3.0));
    let chrome = b.material(Material::metal(Color::splat(0.95), 0.0));
    let glass = b.material(Material::glass(1.6));
    let violet = b.material(Material::metal(rgb(0xb48cff), 0.3));

    b.add(
        Shape::quad(Point3::new(-40.0, 0.0, -60.0), Vec3::new(80.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 80.0)),
        floor,
    );

    // Thin glowing lines laid on the floor: the grid.
    let w = 0.04;
    for i in -10..=10 {
        let x = i as f32 * 2.0;
        b.add(
            Shape::quad(
                Point3::new(x - w, 0.01, -60.0),
                Vec3::new(2.0 * w, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 80.0),
            ),
            pink,
        );
    }
    for i in 0..20 {
        let z = 18.0 - i as f32 * 3.0;
        b.add(
            Shape::quad(
                Point3::new(-20.0, 0.01, z - w),
                Vec3::new(40.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, 2.0 * w),
            ),
            cyan,
        );
    }

    b.add(Shape::sphere(Point3::new(0.0, 6.0, -55.0), 14.0), sun)
        .add(Shape::sphere(Point3::new(0.0, 1.5, 0.0), 1.5), chrome)
        .add(Shape::sphere(Point3::new(-3.4, 1.0, 1.0), 1.0), glass)
        .add(Shape::sphere(Point3::new(3.4, 1.0, 1.0), 1.0), violet);

    let camera = CameraConfig {
        look_from: Point3::new(0.0, 2.2, 10.0),
        look_at: Point3::new(0.0, 1.6, 0.0),
        vfov: 42.0,
        aperture: 0.05,
        focus_dist: 10.0,
        ..Default::default()
    };
    b.build("neon", camera, Sky::Gradient { horizon: rgb(0x2b0b3f), zenith: rgb(0x05030f) })
}

pub fn weekend() -> Scene {
    let mut b = SceneBuilder::new();
    let ground = b.material(Material::diffuse(Texture::Checker {
        even: Color::new(0.2, 0.3, 0.1),
        odd: Color::splat(0.9),
        scale: 1.0,
    }));
    b.add(
        Shape::quad(Point3::new(-100.0, 0.0, -100.0), Vec3::new(200.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 200.0)),
        ground,
    );

    let mut rng = Pcg32::new(2024, 1);
    let big = [Point3::new(0.0, 1.0, 0.0), Point3::new(-4.0, 1.0, 0.0), Point3::new(4.0, 1.0, 0.0)];
    let glass = b.material(Material::glass(1.5));

    // `flat_map` turns the nested loops into one iterator of (a, b) pairs.
    let grid = (-11..11).flat_map(|a| (-11..11).map(move |c| (a, c)));
    for (a, c) in grid {
        let center = Point3::new(a as f32 + 0.9 * rng.next_f32(), 0.2, c as f32 + 0.9 * rng.next_f32());
        if big.iter().any(|&p| (center - p).length() < 1.2) {
            continue;
        }
        let choice = rng.next_f32();
        let m = if choice < 0.75 {
            let albedo = Color::new(rng.next_f32(), rng.next_f32(), rng.next_f32())
                * Color::new(rng.next_f32(), rng.next_f32(), rng.next_f32());
            b.material(Material::diffuse(albedo))
        } else if choice < 0.92 {
            let albedo = Color::new(rng.range(0.5, 1.0), rng.range(0.5, 1.0), rng.range(0.5, 1.0));
            b.material(Material::metal(albedo, rng.range(0.0, 0.5)))
        } else {
            glass
        };
        b.add(Shape::sphere(center, 0.2), m);
    }

    let brown = b.material(Material::diffuse(Color::new(0.4, 0.2, 0.1)));
    let steel = b.material(Material::metal(Color::new(0.7, 0.6, 0.5), 0.0));
    b.add(Shape::sphere(big[0], 1.0), glass)
        .add(Shape::sphere(big[1], 1.0), brown)
        .add(Shape::sphere(big[2], 1.0), steel);

    let camera = CameraConfig {
        look_from: Point3::new(13.0, 2.0, 3.0),
        look_at: Point3::ZERO,
        vfov: 20.0,
        aperture: 0.1,
        focus_dist: 10.0,
        ..Default::default()
    };
    let sky = Sky::Gradient { horizon: Color::ONE, zenith: Color::new(0.5, 0.7, 1.0) };
    b.build("weekend", camera, sky)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_scene_builds_and_is_findable() {
        for info in SCENES {
            let scene = by_name(info.name).expect("scene should exist");
            assert_eq!(scene.name, info.name);
            assert!(!scene.objects().is_empty());
        }
        assert!(by_name("nope").is_none());
    }
}
