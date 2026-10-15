//! # Photon
//!
//! A small, dependency-free, physically based path tracer. The same code
//! runs natively (see the `photon-cli` crate, which renders on every CPU
//! core) and in the browser (see `photon-wasm`, compiled to WebAssembly).
//!
//! Read the modules roughly in this order, smallest ideas first:
//!
//! 1. [`vec3`]: the math type everything is built on
//! 2. [`rng`]: random numbers, the "Monte Carlo" in Monte Carlo rendering
//! 3. [`camera`]: rays, and where they come from
//! 4. [`shape`]: does this ray hit this object?
//! 5. [`bvh`]: ...and how to avoid asking that question millions of times
//! 6. [`material`]: what happens to light at a surface
//! 7. [`scene`]: putting it together; the path tracing loop lives here
//! 8. [`render`]: pixels, sampling and tone mapping
//! 9. [`scenes`]: the built-in scene gallery

pub mod bvh;
pub mod camera;
pub mod material;
pub mod render;
pub mod rng;
pub mod scene;
pub mod scenes;
pub mod shape;
pub mod vec3;

pub use camera::{Camera, CameraConfig, Ray};
pub use render::{Frame, render_rows, tonemap};
pub use scene::{Scene, SceneBuilder};
pub use vec3::{Color, Point3, Vec3};
