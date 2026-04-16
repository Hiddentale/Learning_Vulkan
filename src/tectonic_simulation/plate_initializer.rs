use std::collections::{HashSet, VecDeque};

use glam::DVec3;

use super::plate_seed_placement::PlateAssignment;
use super::plates::{CrustData, CrustType, Plate};
use super::spherical_delaunay_triangulation::SphericalDelaunay;
use super::util::{arbitrary_tangent, splitmix64};

const OCEANIC_THICKNESS: f64 = 7.0;
const OCEANIC_ELEVATION: f64 = -4.0;
const CONTINENTAL_THICKNESS: f64 = 35.0;
const CONTINENTAL_ELEVATION: f64 = 0.3;
const CONTINENTAL_FRACTION: f64 = 0.3;
const MIN_ANGULAR_SPEED: f64 = 0.005;
const MAX_ANGULAR_SPEED: f64 = 0.02;

pub struct InitParams {
    pub seed: u64,
    /// Fraction of surface area that starts as continental crust.
    pub continental_fraction: f64,
}

impl Default for InitParams {
    fn default() -> Self {
        Self { seed: 0, continental_fraction: CONTINENTAL_FRACTION }
    }
}

/// Build initial plates from a partitioned sphere.
///
/// Picks a cluster of adjacent plates as continental (super-continent),
/// assigns default crust parameters, and generates random rotation for each plate.
pub fn initialize_plates(
    points: &[DVec3],
    delaunay: &SphericalDelaunay,
    assignment: &PlateAssignment,
    params: &InitParams,
) -> Vec<Plate> {
    let plate_count = assignment.seeds.len();
    let plate_adj = build_plate_adjacency(delaunay, &assignment.plate_ids, plate_count);
    let continental = pick_continental_plates(&plate_adj, assignment, params);
    let mut rng = params.seed;

    (0..plate_count)
        .map(|plate_id| {
            let point_indices = collect_plate_points(&assignment.plate_ids, plate_id as u32);
            let is_continental = continental.contains(&(plate_id as u32));
            let crust = build_crust(points, &point_indices, is_continental);
            let (axis, speed) = random_rotation(&mut rng);
            Plate { point_indices, crust, rotation_axis: axis, angular_speed: speed }
        })
        .collect()
}

fn collect_plate_points(plate_ids: &[u32], plate: u32) -> Vec<u32> {
    plate_ids
        .iter()
        .enumerate()
        .filter(|(_, &pid)| pid == plate)
        .map(|(i, _)| i as u32)
        .collect()
}

fn build_crust(points: &[DVec3], point_indices: &[u32], continental: bool) -> Vec<CrustData> {
    point_indices
        .iter()
        .map(|&pi| {
            let tangent = arbitrary_tangent(points[pi as usize]);
            if continental {
                CrustData {
                    crust_type: CrustType::Continental,
                    thickness: CONTINENTAL_THICKNESS,
                    elevation: CONTINENTAL_ELEVATION,
                    age: 0.0,
                    local_direction: tangent,
                    orogeny_type: None,
                }
            } else {
                CrustData::oceanic(OCEANIC_THICKNESS, OCEANIC_ELEVATION, 0.0, tangent)
            }
        })
        .collect()
}

/// Plate adjacency graph: two plates are neighbors if any of their points share a Delaunay edge.
fn build_plate_adjacency(
    delaunay: &SphericalDelaunay,
    plate_ids: &[u32],
    plate_count: usize,
) -> Vec<HashSet<u32>> {
    let mut adj: Vec<HashSet<u32>> = vec![HashSet::new(); plate_count];

    for tri in 0..delaunay.triangle_count() {
        let base = tri * 3;
        let v = [
            delaunay.triangles[base],
            delaunay.triangles[base + 1],
            delaunay.triangles[base + 2],
        ];
        for edge in 0..3 {
            let pa = plate_ids[v[edge] as usize];
            let pb = plate_ids[v[(edge + 1) % 3] as usize];
            if pa != pb {
                adj[pa as usize].insert(pb);
                adj[pb as usize].insert(pa);
            }
        }
    }

    adj
}

/// BFS from a random seed plate, claiming adjacent plates as continental
/// until the target surface fraction is reached. Produces a clustered super-continent.
fn pick_continental_plates(
    plate_adj: &[HashSet<u32>],
    assignment: &PlateAssignment,
    params: &InitParams,
) -> HashSet<u32> {
    let total_points = assignment.plate_ids.len();
    let target = (total_points as f64 * params.continental_fraction) as usize;

    let mut plate_sizes = vec![0usize; plate_adj.len()];
    for &pid in &assignment.plate_ids {
        plate_sizes[pid as usize] += 1;
    }

    let first = (splitmix64(params.seed) % plate_adj.len() as u64) as u32;
    let mut continental = HashSet::new();
    let mut queue = VecDeque::new();
    let mut covered = 0usize;

    continental.insert(first);
    covered += plate_sizes[first as usize];
    queue.push_back(first);

    while covered < target {
        let Some(current) = queue.pop_front() else { break };
        for &neighbor in &plate_adj[current as usize] {
            if continental.contains(&neighbor) {
                continue;
            }
            continental.insert(neighbor);
            covered += plate_sizes[neighbor as usize];
            queue.push_back(neighbor);
            if covered >= target {
                break;
            }
        }
    }

    continental
}

fn random_rotation(rng: &mut u64) -> (DVec3, f64) {
    *rng = splitmix64(*rng);
    let z = (*rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
    *rng = splitmix64(*rng);
    let theta = (*rng as f64 / u64::MAX as f64) * std::f64::consts::TAU;
    let r = (1.0 - z * z).sqrt();
    let axis = DVec3::new(r * theta.cos(), r * theta.sin(), z).normalize();

    *rng = splitmix64(*rng);
    let t = *rng as f64 / u64::MAX as f64;
    let speed = MIN_ANGULAR_SPEED + t * (MAX_ANGULAR_SPEED - MIN_ANGULAR_SPEED);

    (axis, speed)
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::tectonic_simulation::fibonnaci_spiral::SphericalFibonacci;
    use crate::tectonic_simulation::plate_seed_placement::{assign_plates, WarpParams};
    use crate::tectonic_simulation::plates::CrustType;

    fn setup(point_count: u32, plate_count: u32) -> (Vec<DVec3>, SphericalDelaunay, PlateAssignment) {
        let fib = SphericalFibonacci::new(point_count);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, plate_count, 42, &WarpParams::default());
        (points, del, assignment)
    }

    #[test]
    fn all_points_covered() {
        let (points, del, assignment) = setup(500, 8);
        let plates = initialize_plates(&points, &del, &assignment, &InitParams::default());
        let total: usize = plates.iter().map(|p| p.point_count()).sum();
        assert_eq!(total, 500);
    }

    #[test]
    fn crust_parallel_to_indices() {
        let (points, del, assignment) = setup(500, 8);
        let plates = initialize_plates(&points, &del, &assignment, &InitParams::default());
        for plate in &plates {
            assert_eq!(plate.crust.len(), plate.point_indices.len());
        }
    }

    #[test]
    fn continental_fraction_approximately_correct() {
        let (points, del, assignment) = setup(1000, 12);
        let plates = initialize_plates(&points, &del, &assignment, &InitParams::default());
        let continental_count: usize = plates
            .iter()
            .flat_map(|p| &p.crust)
            .filter(|c| c.crust_type == CrustType::Continental)
            .count();
        let fraction = continental_count as f64 / 1000.0;
        assert!(fraction > 0.15 && fraction < 0.50, "continental fraction {fraction} out of range");
    }

    #[test]
    fn rotation_axes_are_normalized() {
        let (points, del, assignment) = setup(500, 8);
        let plates = initialize_plates(&points, &del, &assignment, &InitParams::default());
        for plate in &plates {
            let len = plate.rotation_axis.length();
            assert!((len - 1.0).abs() < 1e-10, "axis not normalized: {len}");
        }
    }

    #[test]
    fn plates_have_distinct_rotations() {
        let (points, del, assignment) = setup(500, 8);
        let plates = initialize_plates(&points, &del, &assignment, &InitParams::default());
        for i in 0..plates.len() {
            for j in (i + 1)..plates.len() {
                let same_axis = (plates[i].rotation_axis - plates[j].rotation_axis).length() < 1e-10;
                let same_speed = (plates[i].angular_speed - plates[j].angular_speed).abs() < 1e-10;
                assert!(!(same_axis && same_speed), "plates {i} and {j} have identical rotation");
            }
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let (points, del, assignment) = setup(500, 8);
        let params = InitParams::default();
        let a = initialize_plates(&points, &del, &assignment, &params);
        let b = initialize_plates(&points, &del, &assignment, &params);
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.point_indices, pb.point_indices);
            assert_eq!(pa.angular_speed, pb.angular_speed);
        }
    }

    #[test]
    fn surface_velocity_perpendicular_to_point() {
        let (points, del, assignment) = setup(500, 8);
        let plates = initialize_plates(&points, &del, &assignment, &InitParams::default());
        let plate = &plates[0];
        let p = points[plate.point_indices[0] as usize];
        let v = plate.surface_velocity(p);
        assert!(v.dot(p).abs() < 1e-10, "velocity not tangent to sphere");
    }
}
