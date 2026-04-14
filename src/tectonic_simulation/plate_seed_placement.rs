use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

use glam::DVec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use super::fibonnaci_spiral::SphericalFibonacci;
use super::spherical_delaunay_triangulation::SphericalDelaunay;

/// Point-level adjacency graph extracted from a Delaunay triangulation.
pub struct Adjacency {
    /// Flat neighbor storage — `neighbors[offsets[i]..offsets[i+1]]` are i's neighbors.
    neighbors: Vec<u32>,
    offsets: Vec<u32>,
}

impl Adjacency {
    pub fn from_delaunay(point_count: usize, delaunay: &SphericalDelaunay) -> Self {
        let mut counts = vec![0u32; point_count];
        for &v in &delaunay.triangles {
            counts[v as usize] += 2; // upper bound: each triangle adds ≤2 edges per vertex
        }
        let mut adj: Vec<Vec<u32>> = (0..point_count).map(|i| Vec::with_capacity(counts[i] as usize / 2)).collect();

        for tri in 0..delaunay.triangle_count() {
            let base = tri * 3;
            let v = [delaunay.triangles[base], delaunay.triangles[base + 1], delaunay.triangles[base + 2]];
            for edge in 0..3 {
                let a = v[edge];
                let b = v[(edge + 1) % 3];
                if !adj[a as usize].contains(&b) {
                    adj[a as usize].push(b);
                }
                if !adj[b as usize].contains(&a) {
                    adj[b as usize].push(a);
                }
            }
        }

        let mut offsets = Vec::with_capacity(point_count + 1);
        let mut neighbors = Vec::new();
        offsets.push(0);
        for list in &adj {
            neighbors.extend_from_slice(list);
            offsets.push(neighbors.len() as u32);
        }

        Self { neighbors, offsets }
    }

    pub fn neighbors_of(&self, point: u32) -> &[u32] {
        let start = self.offsets[point as usize] as usize;
        let end = self.offsets[point as usize + 1] as usize;
        &self.neighbors[start..end]
    }
}

/// Controls how noise warps the flood-fill distances to produce irregular plate boundaries.
pub struct WarpParams {
    /// How strongly noise distorts the boundaries. 0 = uniform Voronoi, 1 = heavily warped.
    pub amplitude: f64,
    /// Base frequency of the noise field. Higher values produce smaller-scale irregularity.
    pub frequency: f64,
    /// Number of FBM octaves. More octaves add finer detail to boundary shapes.
    pub octaves: usize,
    /// Noise seed — independent of the plate seed placement seed.
    pub noise_seed: u32,
    /// Per-plate growth speed range [min, max]. Each plate gets a random multiplier in this
    /// range — low values grow fast (large plates), high values grow slow (small plates).
    /// Use [1.0, 1.0] for uniform growth.
    pub speed_range: [f64; 2],
}

impl Default for WarpParams {
    fn default() -> Self {
        Self {
            amplitude: 0.7,
            frequency: 2.0,
            octaves: 4,
            noise_seed: 0,
            speed_range: [0.3, 1.7],
        }
    }
}

const UNASSIGNED: u32 = u32::MAX;

/// Picks `plate_count` seed indices spread across the sphere via farthest-point sampling,
/// then Dijkstra flood-fills with noise-warped edge weights to assign every point to a plate.
///
/// The noise warp distorts effective distances so plate boundaries become irregular,
/// producing organic continent-like shapes instead of uniform Voronoi cells.
pub fn assign_plates(
    points: &[DVec3],
    fibonacci: &SphericalFibonacci,
    delaunay: &SphericalDelaunay,
    plate_count: u32,
    seed: u64,
    warp: &WarpParams,
) -> PlateAssignment {
    assert!(plate_count >= 2, "need at least 2 plates");
    assert!((plate_count as usize) <= points.len(), "more plates than points");

    let adjacency = Adjacency::from_delaunay(points.len(), delaunay);
    let seeds = pick_seeds(points, fibonacci, plate_count, seed);
    let plate_ids = flood_fill(points, &adjacency, &seeds, warp);

    PlateAssignment { plate_ids, seeds }
}

/// Result of plate partitioning.
pub struct PlateAssignment {
    /// `plate_ids[i]` is the plate index (0..plate_count) that point `i` belongs to.
    pub plate_ids: Vec<u32>,
    /// The point index chosen as seed for each plate.
    pub seeds: Vec<u32>,
}

impl PlateAssignment {
    pub fn plate_count(&self) -> u32 {
        self.seeds.len() as u32
    }
}

/// Farthest-point sampling: pick the first seed pseudo-randomly, then greedily pick
/// the point farthest from all existing seeds. Produces well-separated plate centers.
fn pick_seeds(points: &[DVec3], fibonacci: &SphericalFibonacci, count: u32, seed: u64) -> Vec<u32> {
    let n = fibonacci.point_count();

    // First seed: deterministic from user seed via simple hash.
    let first = (splitmix64(seed) % n as u64) as u32;
    let mut seeds = Vec::with_capacity(count as usize);
    seeds.push(first);

    // min_dot[i] tracks the maximum dot product (minimum angular distance) to any seed.
    let mut min_dot = vec![f64::NEG_INFINITY; n as usize];
    update_distances(points, first, &mut min_dot);

    for _ in 1..count {
        // Pick the point whose closest seed is farthest away (smallest max dot = largest angle).
        let farthest = (0..n)
            .filter(|i| !seeds.contains(i))
            .min_by(|&a, &b| min_dot[a as usize].partial_cmp(&min_dot[b as usize]).unwrap())
            .unwrap();
        seeds.push(farthest);
        update_distances(points, farthest, &mut min_dot);
    }

    seeds
}

fn update_distances(points: &[DVec3], new_seed: u32, min_dot: &mut [f64]) {
    let seed_point = points[new_seed as usize];
    for (i, dot) in min_dot.iter_mut().enumerate() {
        let d = points[i].dot(seed_point);
        if d > *dot {
            *dot = d;
        }
    }
}

struct Entry {
    cost: f64,
    point: u32,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for Entry {}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse for min-heap (BinaryHeap is max-heap by default).
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Dijkstra flood-fill from all seeds with noise-warped edge weights.
/// Edge cost = `arc_distance(a, b) * speed[plate] * (1 + amplitude * noise(midpoint))`.
/// Each plate's speed is drawn from `warp.speed_range` — low speed = fast growth = large plate.
fn flood_fill(points: &[DVec3], adjacency: &Adjacency, seeds: &[u32], warp: &WarpParams) -> Vec<u32> {
    let fbm: Fbm<Perlin> = Fbm::new(warp.noise_seed)
        .set_octaves(warp.octaves)
        .set_frequency(warp.frequency);

    let plate_speeds = generate_plate_speeds(seeds.len(), warp);

    let mut plate_ids = vec![UNASSIGNED; points.len()];
    let mut costs = vec![f64::INFINITY; points.len()];
    let mut heap = BinaryHeap::with_capacity(points.len());

    for (plate, &seed) in seeds.iter().enumerate() {
        plate_ids[seed as usize] = plate as u32;
        costs[seed as usize] = 0.0;
        heap.push(Entry { cost: 0.0, point: seed });
    }

    while let Some(Entry { cost, point }) = heap.pop() {
        if cost > costs[point as usize] {
            continue;
        }

        let plate = plate_ids[point as usize];
        let speed = plate_speeds[plate as usize];
        let p = points[point as usize];

        for &neighbor in adjacency.neighbors_of(point) {
            let q = points[neighbor as usize];
            let edge_cost = speed * warped_edge_cost(p, q, warp.amplitude, &fbm);
            let new_cost = cost + edge_cost;

            if new_cost < costs[neighbor as usize] {
                costs[neighbor as usize] = new_cost;
                plate_ids[neighbor as usize] = plate;
                heap.push(Entry { cost: new_cost, point: neighbor });
            }
        }
    }

    plate_ids
}

/// Simple Dijkstra flood-fill with uniform arc-distance edge costs.
/// Used during resampling to reassign plate ownership from centroids.
pub fn flood_fill_from_seeds(points: &[DVec3], adjacency: &Adjacency, seeds: &[u32]) -> Vec<u32> {
    let mut plate_ids = vec![UNASSIGNED; points.len()];
    let mut costs = vec![f64::INFINITY; points.len()];
    let mut heap = BinaryHeap::with_capacity(points.len());

    for (plate, &seed) in seeds.iter().enumerate() {
        plate_ids[seed as usize] = plate as u32;
        costs[seed as usize] = 0.0;
        heap.push(Entry { cost: 0.0, point: seed });
    }

    while let Some(Entry { cost, point }) = heap.pop() {
        if cost > costs[point as usize] { continue; }

        let plate = plate_ids[point as usize];
        let p = points[point as usize];

        for &neighbor in adjacency.neighbors_of(point) {
            let q = points[neighbor as usize];
            let edge_cost = p.dot(q).clamp(-1.0, 1.0).acos();
            let new_cost = cost + edge_cost;

            if new_cost < costs[neighbor as usize] {
                costs[neighbor as usize] = new_cost;
                plate_ids[neighbor as usize] = plate;
                heap.push(Entry { cost: new_cost, point: neighbor });
            }
        }
    }

    plate_ids
}

fn generate_plate_speeds(plate_count: usize, warp: &WarpParams) -> Vec<f64> {
    let [lo, hi] = warp.speed_range;
    let mut rng_state = splitmix64(warp.noise_seed as u64 ^ 0xDEAD_BEEF);
    (0..plate_count)
        .map(|_| {
            rng_state = splitmix64(rng_state);
            let t = (rng_state as f64) / (u64::MAX as f64);
            lo + t * (hi - lo)
        })
        .collect()
}

fn warped_edge_cost(a: DVec3, b: DVec3, amplitude: f64, fbm: &Fbm<Perlin>) -> f64 {
    let arc = a.dot(b).clamp(-1.0, 1.0).acos();
    let mid = (a + b).normalize();
    let warp = 1.0 + amplitude * fbm.get([mid.x, mid.y, mid.z]);
    arc * warp.max(0.1)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_assignment(point_count: u32, plate_count: u32) -> (Vec<DVec3>, PlateAssignment) {
        let fib = SphericalFibonacci::new(point_count);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, plate_count, 42, &WarpParams::default());
        (points, assignment)
    }

    #[test]
    fn every_point_is_assigned() {
        let (_, assignment) = make_test_assignment(500, 8);
        for (i, &plate) in assignment.plate_ids.iter().enumerate() {
            assert_ne!(plate, UNASSIGNED, "point {i} unassigned");
        }
    }

    #[test]
    fn plate_ids_in_range() {
        let (_, assignment) = make_test_assignment(500, 8);
        for (i, &plate) in assignment.plate_ids.iter().enumerate() {
            assert!(plate < 8, "point {i} has plate id {plate}, expected < 8");
        }
    }

    #[test]
    fn all_plates_have_points() {
        let (_, assignment) = make_test_assignment(500, 8);
        for plate in 0..8 {
            let count = assignment.plate_ids.iter().filter(|&&p| p == plate).count();
            assert!(count > 0, "plate {plate} has no points");
        }
    }

    #[test]
    fn seeds_own_their_plate() {
        let (_, assignment) = make_test_assignment(500, 12);
        for (plate, &seed) in assignment.seeds.iter().enumerate() {
            assert_eq!(
                assignment.plate_ids[seed as usize], plate as u32,
                "seed {seed} not assigned to its own plate {plate}"
            );
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let a = make_test_assignment(200, 6).1;
        let b = make_test_assignment(200, 6).1;
        assert_eq!(a.plate_ids, b.plate_ids);
        assert_eq!(a.seeds, b.seeds);
    }

    #[test]
    fn different_seed_gives_different_result() {
        let fib = SphericalFibonacci::new(200);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let warp = WarpParams::default();
        let a = assign_plates(&points, &fib, &del, 6, 42, &warp);
        let b = assign_plates(&points, &fib, &del, 6, 99, &warp);
        assert_ne!(a.seeds, b.seeds);
    }

    #[test]
    fn seeds_are_well_separated() {
        let (points, assignment) = make_test_assignment(1000, 10);
        let mut min_dot = f64::INFINITY;
        for (i, &a) in assignment.seeds.iter().enumerate() {
            for &b in &assignment.seeds[i + 1..] {
                let d = points[a as usize].dot(points[b as usize]);
                if d < min_dot {
                    min_dot = d;
                }
            }
        }
        assert!(min_dot < 0.5, "seeds too clustered: min dot product {min_dot}");
    }

    #[test]
    fn plates_are_contiguous() {
        let (points, assignment) = make_test_assignment(500, 8);
        let del = SphericalDelaunay::from_points(&points);
        let adjacency = Adjacency::from_delaunay(points.len(), &del);

        for plate in 0..8u32 {
            let seed = assignment.seeds[plate as usize];
            let mut visited = vec![false; points.len()];
            let mut queue = VecDeque::new();
            visited[seed as usize] = true;
            queue.push_back(seed);
            let mut count = 1usize;
            while let Some(current) = queue.pop_front() {
                for &neighbor in adjacency.neighbors_of(current) {
                    if !visited[neighbor as usize] && assignment.plate_ids[neighbor as usize] == plate {
                        visited[neighbor as usize] = true;
                        queue.push_back(neighbor);
                        count += 1;
                    }
                }
            }
            let expected = assignment.plate_ids.iter().filter(|&&p| p == plate).count();
            assert_eq!(count, expected, "plate {plate} is not contiguous");
        }
    }

    #[test]
    fn forty_plates_on_large_sphere() {
        let (_, assignment) = make_test_assignment(10_000, 40);
        for (i, &plate) in assignment.plate_ids.iter().enumerate() {
            assert!(plate < 40, "point {i} has plate id {plate}");
        }
        for plate in 0..40 {
            let count = assignment.plate_ids.iter().filter(|&&p| p == plate).count();
            assert!(count > 0, "plate {plate} empty");
        }
    }

    #[test]
    fn different_noise_seed_different_shapes() {
        let fib = SphericalFibonacci::new(1000);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let a = assign_plates(&points, &fib, &del, 10, 42, &WarpParams { noise_seed: 1, ..Default::default() });
        let b = assign_plates(&points, &fib, &del, 10, 42, &WarpParams { noise_seed: 2, ..Default::default() });
        assert_ne!(a.plate_ids, b.plate_ids);
    }
}
