use glam::DVec3;
use noise::{Fbm, NoiseFn, Perlin};

use super::fibonnaci_spiral::SphericalFibonacci;
use super::spherical_delaunay_triangulation::SphericalDelaunay;
use super::util::splitmix64;

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

const UNASSIGNED: u32 = u32::MAX;

/// Floor for noise-warped edge cost (used by plate rifting).
pub(super) const WARP_COST_FLOOR: f64 = 0.1;

// ── Directional growth constants ────────────────────────────────────────────

/// Minimum per-plate growth rate (steps per round).
const RATE_MIN: f64 = 0.7;
/// Growth rate range added to RATE_MIN (rng² distribution for right-skew).
const RATE_RANGE: f64 = 2.3;
/// Base directional strength — how much alignment with growth dir matters.
const DIR_BASE: f64 = 0.15;
/// Extra directional strength inversely scaled by growth rate.
const DIR_SCALE: f64 = 0.25;
/// Maximum directional strength.
const DIR_STRENGTH_CAP: f64 = 0.85;
/// Compactness weight — penalizes frontier cells far from the seed.
const COMPACT_WEIGHT: f64 = 0.3;
/// How many expected-radii before the compactness penalty kicks in.
const COMPACT_THRESHOLD_MULT: f64 = 1.8;
/// Multiplier on the compactness penalty.
const COMPACT_PENALTY_MULT: f64 = 4.0;
/// Area governor: plates exceeding this many times the expected area grow slower.
const AREA_GOVERNOR_MULT: f64 = 2.0;

// ── Public API ──────────────────────────────────────────────────────────────

/// Picks `plate_count` seed indices spread across the sphere via farthest-point sampling,
/// then grows plates via round-robin directional expansion.
///
/// Each plate gets a random preferred growth direction and rate, producing
/// elongated, peninsula-like shapes instead of uniform Voronoi cells.
pub fn assign_plates(
    points: &[DVec3],
    fibonacci: &SphericalFibonacci,
    delaunay: &SphericalDelaunay,
    plate_count: u32,
    seed: u64,
) -> PlateAssignment {
    assert!(plate_count >= 2, "need at least 2 plates");
    assert!((plate_count as usize) <= points.len(), "more plates than points");

    let adjacency = Adjacency::from_delaunay(points.len(), delaunay);
    let seeds = pick_seeds(points, fibonacci, plate_count, seed);
    let plate_ids = directional_growth(points, &adjacency, &seeds, seed);

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

// ── Directional growth ─────────────────────────────────────────────────────

/// Simple seeded RNG: returns a float in [0, 1) and advances the state.
fn rng_f64(state: &mut u64) -> f64 {
    *state = splitmix64(*state);
    (*state as f64) / (u64::MAX as f64)
}

/// Returns a random integer in [0, n).
fn rng_usize(state: &mut u64, n: usize) -> usize {
    *state = splitmix64(*state);
    (*state as usize) % n
}

/// Round-robin directional growth. Each plate has a preferred growth direction,
/// growth rate, and compactness constraint. Produces elongated plate shapes.
fn directional_growth(
    points: &[DVec3],
    adjacency: &Adjacency,
    seeds: &[u32],
    seed: u64,
) -> Vec<u32> {
    let n = points.len();
    let num_plates = seeds.len();
    let mut rng = splitmix64(seed ^ 0xCAFE_BABE);

    // Per-plate properties.
    let mut growth_rate = Vec::with_capacity(num_plates);
    let mut growth_dir = Vec::with_capacity(num_plates);
    let mut dir_strength = Vec::with_capacity(num_plates);

    for &seed_idx in seeds {
        let r1 = rng_f64(&mut rng);
        let r2 = rng_f64(&mut rng);
        let rate = RATE_MIN + r1 * r2 * RATE_RANGE;
        growth_rate.push(rate);

        // Random tangent direction at the seed point.
        let normal = points[seed_idx as usize].normalize();
        let rx = rng_f64(&mut rng) - 0.5;
        let ry = rng_f64(&mut rng) - 0.5;
        let rz = rng_f64(&mut rng) - 0.5;
        let rand = DVec3::new(rx, ry, rz);
        let tangent = (rand - normal * rand.dot(normal)).normalize_or_zero();
        growth_dir.push(if tangent.length_squared() > 0.01 {
            tangent
        } else {
            // Degenerate — pick any tangent.
            let up = if normal.y.abs() < 0.9 { DVec3::Y } else { DVec3::X };
            normal.cross(up).normalize()
        });

        let ds = (rng_f64(&mut rng) * (DIR_BASE + DIR_SCALE / rate)).min(DIR_STRENGTH_CAP);
        dir_strength.push(ds);
    }

    // Initialize plate assignment and per-plate frontiers.
    let mut plate_ids = vec![UNASSIGNED; n];
    let mut frontiers: Vec<Vec<u32>> = Vec::with_capacity(num_plates);
    let mut plate_area = vec![0usize; num_plates];

    for (plate, &seed_idx) in seeds.iter().enumerate() {
        plate_ids[seed_idx as usize] = plate as u32;
        frontiers.push(vec![seed_idx]);
        plate_area[plate] = 1;
    }

    let expected_area = (n - num_plates).max(1) / num_plates.max(1);
    let inv_n = 1.0 / n as f64;
    let mut remaining = n - num_plates;

    while remaining > 0 {
        let mut any_progress = false;

        for plate in 0..num_plates {
            let frontier = &mut frontiers[plate];
            if frontier.is_empty() {
                continue;
            }

            let rate = growth_rate[plate];
            let dir = growth_dir[plate];
            let ds = dir_strength[plate];
            let ds_half = ds * 0.5;

            let mut steps = (rate * (0.5 + rng_f64(&mut rng))).ceil() as usize;
            steps = steps.max(1);

            // Governor: halve steps for oversized plates.
            if plate_area[plate] > expected_area * AREA_GOVERNOR_MULT as usize {
                steps = (steps / 2).max(1);
            }

            // Compactness threshold: expected chord distance for a circular plate.
            let compact_threshold =
                (plate_area[plate].max(1) as f64 * inv_n / std::f64::consts::PI).sqrt()
                    * 2.0
                    * COMPACT_THRESHOLD_MULT;

            let seed_pos = points[seeds[plate] as usize];

            for _ in 0..steps {
                if frontier.is_empty() {
                    break;
                }

                // Sample a few frontier cells and pick the best-scoring one.
                let samples = frontier.len().min(3 + (ds * 5.0) as usize);
                let mut best_idx = 0;
                let mut best_score = f64::NEG_INFINITY;

                for _ in 0..samples {
                    let idx = rng_usize(&mut rng, frontier.len());
                    let cell = frontier[idx];
                    let p = points[cell as usize];
                    let diff = p - seed_pos;
                    let dist_sq = diff.length_squared();
                    let dist = dist_sq.sqrt().max(1e-12);
                    let alignment = diff.dot(dir) / dist;

                    let excess = (dist_sq * 0.5 - compact_threshold).max(0.0);
                    let compact_penalty = excess * COMPACT_WEIGHT * COMPACT_PENALTY_MULT;

                    let score =
                        alignment * ds + rng_f64(&mut rng) * (1.0 - ds_half) - compact_penalty;
                    if score > best_score {
                        best_score = score;
                        best_idx = idx;
                    }
                }

                // Pop the chosen frontier cell (swap-remove).
                let current = frontier[best_idx];
                let last = frontier.len() - 1;
                frontier.swap(best_idx, last);
                frontier.pop();

                // Expand to unclaimed neighbors.
                for &nb in adjacency.neighbors_of(current) {
                    if plate_ids[nb as usize] == UNASSIGNED {
                        plate_ids[nb as usize] = plate as u32;
                        frontier.push(nb);
                        plate_area[plate] += 1;
                        remaining -= 1;
                        any_progress = true;
                    }
                }
            }
        }

        if !any_progress {
            break;
        }
    }

    // Assign orphaned regions to the nearest claimed neighbor.
    let mut orphans = true;
    while orphans {
        orphans = false;
        for r in 0..n {
            if plate_ids[r] != UNASSIGNED {
                continue;
            }
            for &nb in adjacency.neighbors_of(r as u32) {
                if plate_ids[nb as usize] != UNASSIGNED {
                    plate_ids[r] = plate_ids[nb as usize];
                    orphans = true;
                    break;
                }
            }
        }
    }

    plate_ids
}

// ── Noise-warped edge cost (used by plate rifting) ─────────────────────────

pub(super) fn warped_edge_cost(a: DVec3, b: DVec3, amplitude: f64, fbm: &Fbm<Perlin>) -> f64 {
    let arc = a.dot(b).clamp(-1.0, 1.0).acos();
    let mid = (a + b).normalize();
    let warp = 1.0 + amplitude * fbm.get([mid.x, mid.y, mid.z]);
    arc * warp.max(WARP_COST_FLOOR)
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    fn make_test_assignment(point_count: u32, plate_count: u32) -> (Vec<DVec3>, PlateAssignment) {
        let fib = SphericalFibonacci::new(point_count);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, plate_count, 42);
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
        let a = assign_plates(&points, &fib, &del, 6, 42);
        let b = assign_plates(&points, &fib, &del, 6, 99);
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
    fn plates_have_size_variety() {
        let (_, assignment) = make_test_assignment(5000, 12);
        let mut sizes: Vec<usize> = (0..12)
            .map(|p| assignment.plate_ids.iter().filter(|&&id| id == p).count())
            .collect();
        sizes.sort();
        let ratio = *sizes.last().unwrap() as f64 / *sizes.first().unwrap() as f64;
        assert!(ratio > 1.5, "expected size variety, got max/min ratio {ratio:.1}");
    }
}
