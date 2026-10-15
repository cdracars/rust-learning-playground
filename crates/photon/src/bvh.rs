//! Bounding Volume Hierarchy: the data structure that makes ray tracing fast.
//!
//! Testing a ray against every object is O(n). A BVH wraps objects in a
//! tree of boxes; if a ray misses a box it skips everything inside, so a
//! typical query is closer to O(log n). The "weekend" scene has ~480
//! spheres; the BVH makes it render roughly 20x faster.
//!
//! **Rust concepts:** a tree stored in a flat `Vec` with `u32` indices
//! instead of `Box<Node>` pointers (cache-friendly, trivially `Send + Sync`,
//! no lifetimes to fight), slices and `split_at_mut`-style partitioning with
//! `select_nth_unstable_by`, and generic closures (`impl FnMut`) that the
//! compiler monomorphizes into a specialized traversal loop.

use crate::camera::Ray;
use crate::shape::Aabb;
use crate::vec3::Vec3;

#[derive(Debug, Clone, Copy)]
struct Node {
    bounds: Aabb,
    /// Leaf: index of the first primitive in `Bvh::indices`.
    /// Interior: index of the left child (the right child is `left + 1`).
    start: u32,
    /// Number of primitives. Zero means this is an interior node.
    count: u32,
}

#[derive(Debug, Clone, Default)]
pub struct Bvh {
    nodes: Vec<Node>,
    /// Primitive indices, reordered so every leaf owns a contiguous range.
    indices: Vec<u32>,
}

const MAX_LEAF_SIZE: usize = 2;
const SAH_BINS: usize = 12;

impl Bvh {
    /// Build from one bounding box per primitive.
    pub fn build(bounds: &[Aabb]) -> Self {
        let mut bvh =
            Bvh { nodes: Vec::with_capacity(bounds.len() * 2), indices: (0..bounds.len() as u32).collect() };
        if !bounds.is_empty() {
            bvh.nodes.push(Node { bounds: Aabb::EMPTY, start: 0, count: 0 });
            bvh.subdivide(0, 0, bounds.len(), bounds);
        }
        bvh
    }

    fn subdivide(&mut self, node: usize, start: usize, end: usize, bounds: &[Aabb]) {
        let items = &mut self.indices[start..end];
        let node_bounds = items.iter().fold(Aabb::EMPTY, |acc, &i| acc.union(bounds[i as usize]));
        self.nodes[node].bounds = node_bounds;

        let make_leaf = |nodes: &mut Vec<Node>| {
            nodes[node].start = start as u32;
            nodes[node].count = (end - start) as u32;
        };
        if items.len() <= MAX_LEAF_SIZE {
            return make_leaf(&mut self.nodes);
        }

        let centroid_bounds =
            items.iter().fold(Aabb::EMPTY, |acc, &i| acc.grow(bounds[i as usize].centroid()));
        let Some((axis, split_pos)) = best_split(items, bounds, centroid_bounds, node_bounds) else {
            return make_leaf(&mut self.nodes);
        };

        // Partition so everything left of `mid` has its centroid below the
        // split. If the SAH split is degenerate fall back to the median.
        let key = |i: &u32| bounds[*i as usize].centroid()[axis];
        let mut mid = partition(items, |i| key(i) < split_pos);
        if mid == 0 || mid == items.len() {
            mid = items.len() / 2;
            items.select_nth_unstable_by(mid, |a, b| key(a).total_cmp(&key(b)));
        }

        let left = self.nodes.len();
        self.nodes.push(Node { bounds: Aabb::EMPTY, start: 0, count: 0 });
        self.nodes.push(Node { bounds: Aabb::EMPTY, start: 0, count: 0 });
        self.nodes[node].start = left as u32;
        self.nodes[node].count = 0;
        self.subdivide(left, start, start + mid, bounds);
        self.subdivide(left + 1, start + mid, end, bounds);
    }

    /// Visit primitives the ray might hit, nearest boxes first.
    ///
    /// `test(primitive, t_max)` returns `Some(t)` on a hit closer than
    /// `t_max`; the search then shrinks to that distance so farther boxes
    /// get culled. Returns the closest primitive and its `t`.
    pub fn closest_hit(
        &self,
        ray: &Ray,
        t_min: f32,
        mut t_max: f32,
        mut test: impl FnMut(u32, f32) -> Option<f32>,
    ) -> Option<(u32, f32)> {
        if self.nodes.is_empty() {
            return None;
        }
        let inv_dir = Vec3::ONE / ray.dir;
        let mut best = None;

        // An explicit stack instead of recursion: no function-call overhead
        // and no risk of overflowing the (small) wasm stack.
        let mut stack = [0u32; 64];
        let mut sp = 1; // stack[0] is the root.

        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if node.bounds.hit(ray.origin, inv_dir, t_min, t_max).is_none() {
                continue;
            }
            if node.count > 0 {
                let range = node.start as usize..(node.start + node.count) as usize;
                for &prim in &self.indices[range] {
                    if let Some(t) = test(prim, t_max) {
                        t_max = t;
                        best = Some((prim, t));
                    }
                }
            } else {
                // Push the farther child first so the nearer one pops first.
                let (l, r) = (node.start, node.start + 1);
                let dl = self.nodes[l as usize].bounds.hit(ray.origin, inv_dir, t_min, t_max);
                let dr = self.nodes[r as usize].bounds.hit(ray.origin, inv_dir, t_min, t_max);
                let (near, far) = match (dl, dr) {
                    (Some(a), Some(b)) if b < a => (Some(r), Some(l)),
                    (Some(_), Some(_)) => (Some(l), Some(r)),
                    (Some(_), None) => (Some(l), None),
                    (None, Some(_)) => (Some(r), None),
                    (None, None) => (None, None),
                };
                for child in [far, near].into_iter().flatten() {
                    stack[sp] = child;
                    sp += 1;
                }
            }
        }
        best
    }

    /// Does *anything* block the ray within `(t_min, t_max)`? Shadow rays
    /// only need a yes/no answer, so we can stop at the first hit.
    pub fn any_hit(&self, ray: &Ray, t_min: f32, t_max: f32, mut test: impl FnMut(u32) -> bool) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let inv_dir = Vec3::ONE / ray.dir;
        let mut stack = [0u32; 64];
        let mut sp = 1;
        while sp > 0 {
            sp -= 1;
            let node = &self.nodes[stack[sp] as usize];
            if node.bounds.hit(ray.origin, inv_dir, t_min, t_max).is_none() {
                continue;
            }
            if node.count > 0 {
                let range = node.start as usize..(node.start + node.count) as usize;
                if self.indices[range].iter().any(|&prim| test(prim)) {
                    return true;
                }
            } else {
                stack[sp] = node.start;
                stack[sp + 1] = node.start + 1;
                sp += 2;
            }
        }
        false
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Choose a split plane with the Surface Area Heuristic: the probability a
/// random ray hits a box is proportional to its surface area, so we
/// minimize `area(left) * count(left) + area(right) * count(right)`.
/// Candidates are evaluated cheaply by dropping centroids into bins.
fn best_split(
    items: &[u32],
    bounds: &[Aabb],
    centroid_bounds: Aabb,
    node_bounds: Aabb,
) -> Option<(usize, f32)> {
    let leaf_cost = items.len() as f32;
    let mut best: Option<(usize, f32, f32)> = None; // (axis, position, cost)

    for axis in 0..3 {
        let lo = centroid_bounds.min[axis];
        let hi = centroid_bounds.max[axis];
        if hi - lo < 1e-6 {
            continue; // All centroids coincide on this axis.
        }
        let scale = SAH_BINS as f32 / (hi - lo);
        let mut bins = [(Aabb::EMPTY, 0u32); SAH_BINS];
        for &i in items {
            let b = bounds[i as usize];
            let bin = (((b.centroid()[axis] - lo) * scale) as usize).min(SAH_BINS - 1);
            bins[bin].0 = bins[bin].0.union(b);
            bins[bin].1 += 1;
        }
        // Sweep from the right to get suffix areas, then from the left.
        let mut right_area = [0.0f32; SAH_BINS];
        let mut right_count = [0u32; SAH_BINS];
        let (mut acc, mut n) = (Aabb::EMPTY, 0);
        for b in (1..SAH_BINS).rev() {
            acc = acc.union(bins[b].0);
            n += bins[b].1;
            right_area[b] = acc.surface_area();
            right_count[b] = n;
        }
        let (mut acc, mut n) = (Aabb::EMPTY, 0);
        for b in 0..SAH_BINS - 1 {
            acc = acc.union(bins[b].0);
            n += bins[b].1;
            let cost = (acc.surface_area() * n as f32 + right_area[b + 1] * right_count[b + 1] as f32)
                / node_bounds.surface_area().max(1e-12);
            if best.is_none_or(|(_, _, c)| cost < c) {
                best = Some((axis, lo + (b + 1) as f32 / scale, cost));
            }
        }
    }

    // `0.5` approximates the relative cost of a box test vs a primitive test.
    best.filter(|&(_, _, cost)| 0.5 + cost < leaf_cost).map(|(axis, pos, _)| (axis, pos))
}

/// In-place partition: moves elements matching `pred` to the front and
/// returns how many there were. (The standard library's `partition` method
/// on slices is still unstable, so here's the classic two-pointer version.)
fn partition<T>(items: &mut [T], pred: impl Fn(&T) -> bool) -> usize {
    let mut first = 0;
    for i in 0..items.len() {
        if pred(&items[i]) {
            items.swap(first, i);
            first += 1;
        }
    }
    first
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Pcg32;
    use crate::shape::Shape;

    /// The BVH is an optimization, so it must give *exactly* the same answer
    /// as testing every shape. Throw thousands of random rays at random
    /// spheres and compare.
    #[test]
    fn bvh_matches_brute_force() {
        let mut rng = Pcg32::new(123, 0);
        let shapes: Vec<Shape> = (0..300)
            .map(|_| {
                let c = Vec3::new(rng.range(-20.0, 20.0), rng.range(-20.0, 20.0), rng.range(-20.0, 20.0));
                Shape::sphere(c, rng.range(0.1, 2.0))
            })
            .collect();
        let bounds: Vec<Aabb> = shapes.iter().map(Shape::bounds).collect();
        let bvh = Bvh::build(&bounds);

        for _ in 0..5000 {
            let origin = rng.in_unit_ball() * 30.0;
            let ray = Ray::new(origin, rng.unit_vector());

            let brute = shapes
                .iter()
                .enumerate()
                .filter_map(|(i, s)| s.hit(&ray, 1e-3, f32::INFINITY).map(|h| (i as u32, h.t)))
                .min_by(|a, b| a.1.total_cmp(&b.1));
            let fast = bvh.closest_hit(&ray, 1e-3, f32::INFINITY, |i, t_max| {
                shapes[i as usize].hit(&ray, 1e-3, t_max).map(|h| h.t)
            });
            assert_eq!(brute, fast);
        }
    }

    #[test]
    fn empty_bvh_hits_nothing() {
        let bvh = Bvh::build(&[]);
        let ray = Ray::new(Vec3::ZERO, Vec3::new(0.0, 0.0, 1.0));
        assert!(bvh.closest_hit(&ray, 0.0, 1.0, |_, _| Some(0.5)).is_none());
    }

    #[test]
    fn partition_moves_matches_to_front() {
        let mut v = [5, 1, 8, 2, 9, 3];
        let n = partition(&mut v, |&x| x < 5);
        assert_eq!(n, 3);
        assert!(v[..n].iter().all(|&x| x < 5));
        assert!(v[n..].iter().all(|&x| x >= 5));
    }
}
