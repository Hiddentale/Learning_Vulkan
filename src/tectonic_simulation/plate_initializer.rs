use std::collections::{HashMap, HashSet, VecDeque};

use glam::{DQuat, DVec3};

use super::plate_seed_placement::PlateAssignment;
use super::plates::{BoundingCap, CrustData, CrustType, Plate, SampleCache, TriangleNeighbors};
use super::simulate::NO_PLATE;
use super::spherical_delaunay_triangulation::SphericalDelaunay;
use super::util::{arbitrary_tangent, splitmix64};

const OCEANIC_THICKNESS: f64 = 7.0;
const OCEANIC_ELEVATION: f64 = -4.0;
const CONTINENTAL_THICKNESS: f64 = 35.0;
const CONTINENTAL_ELEVATION: f64 = 2.0;
const CONTINENTAL_FRACTION: f64 = 0.3;
const MIN_ANGULAR_SPEED: f64 = 0.005;
const MAX_ANGULAR_SPEED: f64 = 0.02;

pub struct InitParams {
    pub seed: u64,
    pub continental_fraction: f64,
}

impl Default for InitParams {
    fn default() -> Self {
        Self {
            seed: 0,
            continental_fraction: CONTINENTAL_FRACTION,
        }
    }
}

/// Build initial plates from a partitioned sphere.
///
/// Each plate gets a reference sub-mesh (portion of the global Delaunay),
/// per-vertex crust data, and a random Euler-pole rotation. At t=0 the
/// accumulated rotation is identity (reference frame = world frame).
///
/// Returns (plates, sample_cache).
pub fn initialize_plates(
    points: &[DVec3],
    delaunay: &SphericalDelaunay,
    assignment: &PlateAssignment,
    params: &InitParams,
) -> (Vec<Plate>, Vec<SampleCache>) {
    let plate_count = assignment.seeds.len();
    let plate_adj = build_plate_adjacency(delaunay, &assignment.plate_ids, plate_count);
    let continental = pick_continental_plates(&plate_adj, assignment, params);
    let mut rng = params.seed;

    // Partition triangles by plate (majority vote on vertex assignments).
    let n_tri = delaunay.triangles.len() / 3;
    let mut plate_tris: Vec<Vec<usize>> = vec![Vec::new(); plate_count];
    for t in 0..n_tri {
        let owner = triangle_owner(t, delaunay, &assignment.plate_ids);
        plate_tris[owner as usize].push(t);
    }

    let mut plates = Vec::with_capacity(plate_count);

    // Per-plate: collect vertices, build sub-mesh, assign crust.
    let mut all_global_indices: Vec<Vec<u32>> = Vec::with_capacity(plate_count);

    for plate_id in 0..plate_count {
        let (local_verts, global_to_local) = collect_plate_vertices(&plate_tris[plate_id], delaunay);
        let is_continental = continental.contains(&(plate_id as u32));

        // At t=0, reference frame = world frame, so reference_points = world points.
        let reference_points: Vec<DVec3> =
            local_verts.iter().map(|&gi| points[gi as usize]).collect();

        let crust: Vec<CrustData> = reference_points
            .iter()
            .map(|&p| {
                let tangent = arbitrary_tangent(p);
                if is_continental {
                    CrustData {
                        crust_type: CrustType::Continental,
                        thickness: CONTINENTAL_THICKNESS,
                        elevation: CONTINENTAL_ELEVATION,
                        age: 0.0,
                        local_direction: tangent,
                        orogeny_type: None,
                        subducted_distance: 0.0,
                    }
                } else {
                    CrustData::oceanic(OCEANIC_THICKNESS, OCEANIC_ELEVATION, 0.0, tangent)
                }
            })
            .collect();

        let triangles: Vec<[u32; 3]> = plate_tris[plate_id]
            .iter()
            .map(|&t| {
                let base = t * 3;
                [
                    global_to_local[&delaunay.triangles[base]],
                    global_to_local[&delaunay.triangles[base + 1]],
                    global_to_local[&delaunay.triangles[base + 2]],
                ]
            })
            .collect();

        let adjacency = build_triangle_adjacency(&plate_tris[plate_id], delaunay);

        let (axis, speed) = random_rotation(&mut rng);

        let mut plate = Plate {
            reference_points,
            crust,
            triangles,
            adjacency,
            bounding_cap: BoundingCap {
                center: DVec3::Y,
                cos_radius: -1.0,
            },
            rotation: DQuat::IDENTITY,
            rotation_axis: axis,
            angular_speed: speed,
        };
        plate.recompute_bounding_cap();

        plates.push(plate);
        all_global_indices.push(local_verts);
    }

    let cache = build_initial_cache(&plates, &all_global_indices, points.len());
    (plates, cache)
}

// ── Helpers ───────────────────────────────────────────────────────────

/// Majority-vote triangle owner from vertex plate assignments.
fn triangle_owner(tri: usize, del: &SphericalDelaunay, plate_ids: &[u32]) -> u32 {
    let base = tri * 3;
    let p = [
        plate_ids[del.triangles[base] as usize],
        plate_ids[del.triangles[base + 1] as usize],
        plate_ids[del.triangles[base + 2] as usize],
    ];
    if p[0] == p[1] || p[0] == p[2] {
        p[0]
    } else {
        p[1] // p[1]==p[2] or true triple junction — pick p[1] deterministically
    }
}

fn collect_plate_vertices(
    plate_tris: &[usize],
    del: &SphericalDelaunay,
) -> (Vec<u32>, HashMap<u32, u32>) {
    let mut global_to_local: HashMap<u32, u32> = HashMap::new();
    let mut local_verts: Vec<u32> = Vec::new();

    for &t in plate_tris {
        let base = t * 3;
        for i in 0..3 {
            let gi = del.triangles[base + i];
            global_to_local.entry(gi).or_insert_with(|| {
                let idx = local_verts.len() as u32;
                local_verts.push(gi);
                idx
            });
        }
    }

    (local_verts, global_to_local)
}

fn build_triangle_adjacency(
    global_tri_indices: &[usize],
    del: &SphericalDelaunay,
) -> Vec<TriangleNeighbors> {
    let mut global_to_local_tri: HashMap<usize, u32> =
        HashMap::with_capacity(global_tri_indices.len());
    for (local_t, &global_t) in global_tri_indices.iter().enumerate() {
        global_to_local_tri.insert(global_t, local_t as u32);
    }

    global_tri_indices
        .iter()
        .map(|&global_t| {
            let mut neighbors: TriangleNeighbors = [None; 3];
            for edge in 0..3 {
                let he = del.halfedges[global_t * 3 + edge];
                if he != u32::MAX {
                    let neighbor_global = (he / 3) as usize;
                    if let Some(&neighbor_local) = global_to_local_tri.get(&neighbor_global) {
                        neighbors[edge] = Some(neighbor_local);
                    }
                }
            }
            neighbors
        })
        .collect()
}

fn build_initial_cache(
    plates: &[Plate],
    global_indices: &[Vec<u32>],
    sample_count: usize,
) -> Vec<SampleCache> {
    let mut cache = vec![
        SampleCache {
            plate: NO_PLATE,
            triangle: u32::MAX,
            bary: [0.0; 3],
        };
        sample_count
    ];

    for (k, globals) in global_indices.iter().enumerate() {
        let mut incident = vec![u32::MAX; globals.len()];
        for (t, tri) in plates[k].triangles.iter().enumerate() {
            for &v in tri {
                if incident[v as usize] == u32::MAX {
                    incident[v as usize] = t as u32;
                }
            }
        }

        for (local, &global) in globals.iter().enumerate() {
            let tri = incident[local];
            if tri == u32::MAX {
                continue;
            }
            let tri_verts = plates[k].triangles[tri as usize];
            let local_u32 = local as u32;
            let bary = if tri_verts[0] == local_u32 {
                [1.0, 0.0, 0.0]
            } else if tri_verts[1] == local_u32 {
                [0.0, 1.0, 0.0]
            } else {
                [0.0, 0.0, 1.0]
            };
            cache[global as usize] = SampleCache {
                plate: k as u32,
                triangle: tri,
                bary,
            };
        }
    }

    cache
}

/// Plate adjacency graph from the Delaunay triangulation.
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

/// BFS from a random seed plate, claiming neighbors as continental
/// until the target surface fraction is reached.
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
        let Some(current) = queue.pop_front() else {
            break;
        };
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
    use crate::tectonic_simulation::plate_seed_placement::assign_plates;

    fn setup(
        point_count: u32,
        plate_count: u32,
    ) -> (Vec<DVec3>, SphericalDelaunay, PlateAssignment) {
        let fib = SphericalFibonacci::new(point_count);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, plate_count, 42);
        (points, del, assignment)
    }

    #[test]
    fn all_points_covered() {
        let (points, del, assignment) = setup(500, 8);
        let (plates, _cache) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        let total: usize = plates.iter().map(|p| p.point_count()).sum();
        // Plate vertices can overlap at boundaries (shared by adjacent plates).
        assert!(total >= 500, "total plate vertices {total} < 500");
    }

    #[test]
    fn crust_parallel_to_reference_points() {
        let (points, del, assignment) = setup(500, 8);
        let (plates, _) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        for plate in &plates {
            assert_eq!(plate.crust.len(), plate.reference_points.len());
        }
    }

    #[test]
    fn continental_fraction_approximately_correct() {
        let (points, del, assignment) = setup(1000, 12);
        let (plates, _) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        let continental_count: usize = plates
            .iter()
            .flat_map(|p| &p.crust)
            .filter(|c| c.crust_type == CrustType::Continental)
            .count();
        let total: usize = plates.iter().map(|p| p.point_count()).sum();
        let fraction = continental_count as f64 / total as f64;
        assert!(
            fraction > 0.15 && fraction < 0.50,
            "continental fraction {fraction} out of range"
        );
    }

    #[test]
    fn rotation_axes_are_normalized() {
        let (points, del, assignment) = setup(500, 8);
        let (plates, _) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        for plate in &plates {
            let len = plate.rotation_axis.length();
            assert!((len - 1.0).abs() < 1e-10, "axis not normalized: {len}");
        }
    }

    #[test]
    fn initial_rotation_is_identity() {
        let (points, del, assignment) = setup(500, 8);
        let (plates, _) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        for plate in &plates {
            let diff = (plate.rotation - DQuat::IDENTITY).length();
            assert!(diff < 1e-10, "initial rotation not identity");
        }
    }

    #[test]
    fn every_plate_has_triangles() {
        let (points, del, assignment) = setup(500, 8);
        let (plates, _) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        for (i, plate) in plates.iter().enumerate() {
            assert!(!plate.triangles.is_empty(), "plate {i} has no triangles");
        }
    }

    #[test]
    fn adjacency_parallel_to_triangles() {
        let (points, del, assignment) = setup(500, 8);
        let (plates, _) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        for plate in &plates {
            assert_eq!(plate.adjacency.len(), plate.triangles.len());
        }
    }

    #[test]
    fn sample_cache_covers_all_points() {
        let (points, del, assignment) = setup(500, 8);
        let (_, cache) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        assert_eq!(cache.len(), 500);
        let assigned = cache.iter().filter(|c| c.plate != NO_PLATE).count();
        // Most points should be assigned; a few at triple junctions might not.
        assert!(assigned >= 490, "only {assigned}/500 samples assigned");
    }

    #[test]
    fn deterministic_with_same_seed() {
        let (points, del, assignment) = setup(500, 8);
        let params = InitParams::default();
        let (a, _) = initialize_plates(&points, &del, &assignment, &params);
        let (b, _) = initialize_plates(&points, &del, &assignment, &params);
        for (pa, pb) in a.iter().zip(b.iter()) {
            assert_eq!(pa.point_count(), pb.point_count());
            assert_eq!(pa.angular_speed, pb.angular_speed);
        }
    }
}
