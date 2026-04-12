use std::collections::VecDeque;

use glam::DVec3;

use super::fibonnaci_spiral::SphericalFibonacci;
use super::spherical_delaunay_triangulation::SphericalDelaunay;

/// Point-level adjacency graph extracted from a Delaunay triangulation.
struct Adjacency {
    /// Flat neighbor storage — `neighbors[offsets[i]..offsets[i+1]]` are i's neighbors.
    neighbors: Vec<u32>,
    offsets: Vec<u32>,
}

impl Adjacency {
    fn from_delaunay(point_count: usize, delaunay: &SphericalDelaunay) -> Self {
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

    fn neighbors_of(&self, point: u32) -> &[u32] {
        let start = self.offsets[point as usize] as usize;
        let end = self.offsets[point as usize + 1] as usize;
        &self.neighbors[start..end]
    }
}

const UNASSIGNED: u32 = u32::MAX;

/// Picks `plate_count` seed indices spread across the sphere via farthest-point sampling,
/// then flood-fills the adjacency graph to assign every point to its nearest seed's plate.
pub fn assign_plates(points: &[DVec3], fibonacci: &SphericalFibonacci, delaunay: &SphericalDelaunay, plate_count: u32, seed: u64) -> PlateAssignment {
    assert!(plate_count >= 2, "need at least 2 plates");
    assert!((plate_count as usize) <= points.len(), "more plates than points");

    let adjacency = Adjacency::from_delaunay(points.len(), delaunay);
    let seeds = pick_seeds(points, fibonacci, plate_count, seed);
    let plate_ids = flood_fill(points.len(), &adjacency, &seeds);

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

/// Simultaneous BFS from all seeds — each point is claimed by the first seed that reaches it.
fn flood_fill(point_count: usize, adjacency: &Adjacency, seeds: &[u32]) -> Vec<u32> {
    let mut plate_ids = vec![UNASSIGNED; point_count];
    let mut queue = VecDeque::with_capacity(point_count);

    for (plate, &seed) in seeds.iter().enumerate() {
        plate_ids[seed as usize] = plate as u32;
        queue.push_back(seed);
    }

    while let Some(current) = queue.pop_front() {
        let plate = plate_ids[current as usize];
        for &neighbor in adjacency.neighbors_of(current) {
            if plate_ids[neighbor as usize] == UNASSIGNED {
                plate_ids[neighbor as usize] = plate;
                queue.push_back(neighbor);
            }
        }
    }

    plate_ids
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
        // With farthest-point sampling, no two seeds should be very close.
        let mut min_dot = f64::INFINITY;
        for (i, &a) in assignment.seeds.iter().enumerate() {
            for &b in &assignment.seeds[i + 1..] {
                let d = points[a as usize].dot(points[b as usize]);
                if d < min_dot {
                    min_dot = d;
                }
            }
        }
        // min_dot < ~0.5 means seeds are at least ~60° apart for 10 plates on a sphere.
        assert!(min_dot < 0.5, "seeds too clustered: min dot product {min_dot}");
    }

    #[test]
    fn plates_are_contiguous() {
        let fib = SphericalFibonacci::new(500);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, 8, 42);
        let adjacency = Adjacency::from_delaunay(points.len(), &del);

        // BFS from each seed should reach every point in its plate without crossing plates.
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
}
