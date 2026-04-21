use std::collections::{BinaryHeap, HashSet};

use glam::DVec3;
use noise::{Fbm, MultiFractal, Perlin};

use super::plates::{BoundingCap, CrustData, CrustType, Plate, TriangleNeighbors};
use super::util::{arbitrary_tangent, splitmix64, MinHeapEntry};

/// Base Poisson rate λ_0 for rifting probability per check.
const BASE_RIFT_RATE: f64 = 0.15;
/// Minimum simulation steps between rifting checks.
pub(super) const RIFT_CHECK_INTERVAL: usize = 20;
/// Minimum points a plate must have to be eligible for rifting.
const MIN_PLATE_POINTS: usize = 50;
/// Plates with continental fraction below this don't rift.
const MIN_CONTINENTAL_FRACTION: f64 = 0.3;
/// Angular perturbation applied to sub-plate rotation axes (radians).
const AXIS_PERTURBATION: f64 = 0.5;
/// Dot product threshold for choosing a stable cross-product axis.
const TANGENT_AXIS_THRESHOLD: f64 = 0.9;
/// Speed factor range for sub-plate speed variation.
const SPEED_FACTOR_BASE: f64 = 0.8;
const SPEED_FACTOR_RANGE: f64 = 0.4;
/// Minimum points for multi-split.
const MIN_POINTS_FOR_MULTI_SPLIT: usize = 100;
/// Minimum points per sub-plate.
const POINTS_PER_SUB_PLATE: usize = 25;
/// Noise warp amplitude for fracture line irregularity.
const RIFT_WARP_AMPLITUDE: f64 = 0.7;
/// Noise frequency for fracture warp.
const RIFT_WARP_FREQUENCY: f64 = 3.0;
/// FBM octaves for fracture warp.
const RIFT_WARP_OCTAVES: usize = 3;
/// Freshly rifted oceanic crust thickness (km).
const RIFT_OCEAN_THICKNESS: f64 = 7.0;
/// Freshly rifted oceanic crust elevation (km).
const RIFT_OCEAN_ELEVATION: f64 = -1.0;

/// Check whether a plate should rift this timestep.
pub(super) fn should_rift(
    plate: &Plate,
    plate_index: usize,
    total_points: usize,
    plate_count: usize,
    time: f64,
    seed: u64,
) -> bool {
    if plate.point_count() < MIN_PLATE_POINTS {
        return false;
    }

    let continental_fraction = continental_fraction(plate);
    if continental_fraction < MIN_CONTINENTAL_FRACTION {
        return false;
    }

    let avg_plate_size = total_points as f64 / plate_count.max(1) as f64;
    let area_ratio = plate.point_count() as f64 / avg_plate_size;

    let lambda = BASE_RIFT_RATE * continental_fraction * area_ratio;
    let probability = lambda * (-lambda).exp();

    let time_bits = time.to_bits();
    let hash = splitmix64(seed ^ plate_index as u64 ^ time_bits);
    let roll = (hash as f64) / (u64::MAX as f64);

    roll < probability
}

/// Split a plate into 2–4 sub-plates via noise-warped Voronoi partitioning
/// on the plate's own vertex adjacency graph.
///
/// Returns the new sub-plates. The caller replaces the original plate.
pub(super) fn rift_plate(plate: &Plate, seed: u64) -> Vec<Plate> {
    let n = plate.point_count();
    let sub_count = sub_plate_count(n, seed);

    let vert_adj = build_vertex_adjacency(plate);
    let seeds = pick_rift_seeds(plate, sub_count, seed);
    let partition = partition_plate(plate, &vert_adj, &seeds, seed);
    let axes = perturb_axes(plate.rotation_axis, plate.angular_speed, sub_count, seed);
    let boundary_set = find_rift_boundary(plate, &partition, &vert_adj);

    build_sub_plates(plate, &partition, &boundary_set, &axes, sub_count)
}

fn continental_fraction(plate: &Plate) -> f64 {
    let continental = plate
        .crust
        .iter()
        .filter(|c| c.crust_type == CrustType::Continental)
        .count();
    continental as f64 / plate.point_count() as f64
}

fn sub_plate_count(point_count: usize, seed: u64) -> usize {
    let hash = splitmix64(seed ^ 0xCAFE);
    let base = 2 + (hash % 3) as usize;
    if point_count < MIN_POINTS_FOR_MULTI_SPLIT {
        2
    } else {
        base.min(point_count / POINTS_PER_SUB_PLATE)
    }
}

/// Build vertex adjacency from the plate's triangle list.
fn build_vertex_adjacency(plate: &Plate) -> Vec<Vec<u32>> {
    let n = plate.reference_points.len();
    let mut adj: Vec<Vec<u32>> = vec![Vec::new(); n];
    for tri in &plate.triangles {
        for i in 0..3 {
            let a = tri[i];
            let b = tri[(i + 1) % 3];
            if !adj[a as usize].contains(&b) {
                adj[a as usize].push(b);
            }
            if !adj[b as usize].contains(&a) {
                adj[b as usize].push(a);
            }
        }
    }
    adj
}

/// Farthest-point sampling within the plate for well-separated seeds.
fn pick_rift_seeds(plate: &Plate, count: usize, seed: u64) -> Vec<usize> {
    let n = plate.point_count();
    let first = (splitmix64(seed ^ 0xBEEF) % n as u64) as usize;
    let mut seeds = vec![first];
    let mut min_dot = vec![f64::NEG_INFINITY; n];

    let first_pos = plate.reference_points[first];
    for (i, dot) in min_dot.iter_mut().enumerate() {
        *dot = plate.reference_points[i].dot(first_pos);
    }

    for _ in 1..count {
        let farthest = (0..n)
            .filter(|i| !seeds.contains(i))
            .min_by(|&a, &b| {
                min_dot[a]
                    .partial_cmp(&min_dot[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        seeds.push(farthest);
        let seed_pos = plate.reference_points[farthest];
        for (i, dot) in min_dot.iter_mut().enumerate() {
            let d = plate.reference_points[i].dot(seed_pos);
            if d > *dot {
                *dot = d;
            }
        }
    }

    seeds
}

/// Dijkstra flood-fill on the plate's vertex adjacency with noise-warped costs.
fn partition_plate(
    plate: &Plate,
    vert_adj: &[Vec<u32>],
    seeds: &[usize],
    seed: u64,
) -> Vec<u32> {
    let n = plate.point_count();
    let fbm: Fbm<Perlin> = Fbm::new(seed as u32)
        .set_octaves(RIFT_WARP_OCTAVES)
        .set_frequency(RIFT_WARP_FREQUENCY);

    let mut partition = vec![u32::MAX; n];
    let mut costs = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::with_capacity(n);

    for (sub_id, &local_seed) in seeds.iter().enumerate() {
        partition[local_seed] = sub_id as u32;
        costs[local_seed] = 0.0;
        heap.push(MinHeapEntry {
            cost: 0.0,
            index: local_seed as u32,
        });
    }

    while let Some(MinHeapEntry { cost, index }) = heap.pop() {
        let vi = index as usize;
        if cost > costs[vi] {
            continue;
        }

        let p = plate.reference_points[vi];

        for &ni in &vert_adj[vi] {
            let q = plate.reference_points[ni as usize];
            let edge_cost =
                super::plate_seed_placement::warped_edge_cost(p, q, RIFT_WARP_AMPLITUDE, &fbm);
            let new_cost = cost + edge_cost;

            if new_cost < costs[ni as usize] {
                costs[ni as usize] = new_cost;
                partition[ni as usize] = partition[vi];
                heap.push(MinHeapEntry {
                    cost: new_cost,
                    index: ni,
                });
            }
        }
    }

    partition
}

/// Vertices on the boundary between sub-plates get fresh oceanic crust.
fn find_rift_boundary(
    plate: &Plate,
    partition: &[u32],
    vert_adj: &[Vec<u32>],
) -> HashSet<usize> {
    let mut boundary = HashSet::new();
    for (vi, neighbors) in vert_adj.iter().enumerate() {
        let my_sub = partition[vi];
        if my_sub == u32::MAX {
            continue;
        }
        for &ni in neighbors {
            if partition[ni as usize] != u32::MAX && partition[ni as usize] != my_sub {
                boundary.insert(vi);
                break;
            }
        }
    }
    boundary
}

/// Build sub-plates from the partition. Each sub-plate gets its portion
/// of vertices + crust. Triangles whose 3 vertices all belong to the
/// same sub-plate are kept; cross-partition triangles are dropped.
/// The next resample rebuilds clean meshes.
fn build_sub_plates(
    plate: &Plate,
    partition: &[u32],
    boundary_set: &HashSet<usize>,
    axes: &[(DVec3, f64)],
    sub_count: usize,
) -> Vec<Plate> {
    // Map old vertex → new local index per sub-plate.
    let mut sub_verts: Vec<Vec<u32>> = vec![Vec::new(); sub_count];
    let mut old_to_new: Vec<(u32, u32)> = vec![(u32::MAX, u32::MAX); plate.reference_points.len()];

    for (vi, &sub) in partition.iter().enumerate() {
        if sub == u32::MAX || sub as usize >= sub_count {
            continue;
        }
        let local = sub_verts[sub as usize].len() as u32;
        old_to_new[vi] = (sub, local);
        sub_verts[sub as usize].push(vi as u32);
    }

    let mut sub_plates: Vec<Plate> = (0..sub_count)
        .map(|i| {
            let ref_pts: Vec<DVec3> = sub_verts[i]
                .iter()
                .map(|&vi| plate.reference_points[vi as usize])
                .collect();
            let crust: Vec<CrustData> = sub_verts[i]
                .iter()
                .map(|&vi| {
                    if boundary_set.contains(&(vi as usize))
                        && plate.crust[vi as usize].crust_type == CrustType::Continental
                    {
                        let tangent = arbitrary_tangent(plate.reference_points[vi as usize]);
                        CrustData::oceanic(RIFT_OCEAN_THICKNESS, RIFT_OCEAN_ELEVATION, 0.0, tangent)
                    } else {
                        plate.crust[vi as usize].clone()
                    }
                })
                .collect();

            // Collect triangles fully inside this sub-plate.
            let mut triangles: Vec<[u32; 3]> = Vec::new();
            let mut tri_global_indices: Vec<usize> = Vec::new();
            for (t, tri) in plate.triangles.iter().enumerate() {
                let subs: [u32; 3] = [
                    old_to_new[tri[0] as usize].0,
                    old_to_new[tri[1] as usize].0,
                    old_to_new[tri[2] as usize].0,
                ];
                if subs[0] == i as u32 && subs[1] == i as u32 && subs[2] == i as u32 {
                    triangles.push([
                        old_to_new[tri[0] as usize].1,
                        old_to_new[tri[1] as usize].1,
                        old_to_new[tri[2] as usize].1,
                    ]);
                    tri_global_indices.push(t);
                }
            }

            // Build adjacency from the parent's halfedge structure.
            let adjacency = build_sub_adjacency(&tri_global_indices, plate, &old_to_new, i);

            let mut p = Plate {
                reference_points: ref_pts,
                crust,
                triangles,
                adjacency,
                bounding_cap: BoundingCap {
                    center: DVec3::Y,
                    cos_radius: -1.0,
                },
                rotation: plate.rotation,
                rotation_axis: axes[i].0,
                angular_speed: axes[i].1,
            };
            p.recompute_bounding_cap();
            p
        })
        .collect();

    sub_plates.retain(|p| !p.reference_points.is_empty());
    sub_plates
}

fn build_sub_adjacency(
    tri_global_indices: &[usize],
    parent: &Plate,
    old_to_new: &[(u32, u32)],
    sub_id: usize,
) -> Vec<TriangleNeighbors> {
    // Map parent triangle index → local sub-plate triangle index.
    let mut parent_to_local: std::collections::HashMap<usize, u32> =
        std::collections::HashMap::with_capacity(tri_global_indices.len());
    for (local_t, &parent_t) in tri_global_indices.iter().enumerate() {
        parent_to_local.insert(parent_t, local_t as u32);
    }

    tri_global_indices
        .iter()
        .map(|&parent_t| {
            let mut neighbors: TriangleNeighbors = [None; 3];
            for edge in 0..3 {
                if let Some(parent_neighbor) = parent.adjacency[parent_t][edge] {
                    if let Some(&local_neighbor) =
                        parent_to_local.get(&(parent_neighbor as usize))
                    {
                        neighbors[edge] = Some(local_neighbor);
                    }
                }
            }
            neighbors
        })
        .collect()
}

/// Generate diverging rotation axes for sub-plates.
fn perturb_axes(
    parent_axis: DVec3,
    parent_speed: f64,
    count: usize,
    seed: u64,
) -> Vec<(DVec3, f64)> {
    let up = if parent_axis.y.abs() < TANGENT_AXIS_THRESHOLD {
        DVec3::Y
    } else {
        DVec3::X
    };
    let tangent_u = parent_axis.cross(up).normalize();
    let tangent_v = parent_axis.cross(tangent_u).normalize();

    let mut rng = seed ^ 0xD1CE;
    rng = splitmix64(rng);
    let phase = (rng as f64 / u64::MAX as f64) * std::f64::consts::TAU;

    (0..count)
        .map(|i| {
            let angle = phase + (i as f64 / count as f64) * std::f64::consts::TAU;
            let direction = tangent_u * angle.cos() + tangent_v * angle.sin();
            let new_axis = (parent_axis + AXIS_PERTURBATION * direction).normalize();

            rng = splitmix64(rng);
            let speed_factor =
                SPEED_FACTOR_BASE + SPEED_FACTOR_RANGE * (rng as f64 / u64::MAX as f64);
            (new_axis, parent_speed * speed_factor)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tectonic_simulation::fibonnaci_spiral::SphericalFibonacci;
    use crate::tectonic_simulation::plate_initializer::{initialize_plates, InitParams};
    use crate::tectonic_simulation::plate_seed_placement::assign_plates;
    use crate::tectonic_simulation::spherical_delaunay_triangulation::SphericalDelaunay;

    #[test]
    fn low_continental_never_rifts() {
        let fib = SphericalFibonacci::new(100);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, 8, 42);
        let (plates, _) = initialize_plates(
            &points,
            &del,
            &assignment,
            &InitParams {
                seed: 42,
                continental_fraction: 0.05,
            },
        );
        let plate = plates.iter().enumerate().find(|(_, p)| {
            let frac = p
                .crust
                .iter()
                .filter(|c| c.crust_type == CrustType::Continental)
                .count() as f64
                / p.point_count() as f64;
            frac < 0.3
        });
        if let Some((idx, plate)) = plate {
            for i in 0..100 {
                assert!(!should_rift(plate, idx, 100, 8, i as f64, 42));
            }
        }
    }

    #[test]
    fn high_continental_sometimes_rifts() {
        let fib = SphericalFibonacci::new(200);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, 2, 42);
        let (plates, _) = initialize_plates(
            &points,
            &del,
            &assignment,
            &InitParams {
                seed: 42,
                continental_fraction: 1.0,
            },
        );
        let plate = &plates[0];
        let rift_count = (0..10000)
            .filter(|&i| should_rift(plate, 0, 2000, 10, i as f64, i as u64))
            .count();
        assert!(rift_count > 0, "should rift at least once in 10000 tries");
        assert!(
            rift_count < 2000,
            "should not rift too often: {rift_count}/10000"
        );
    }

    #[test]
    fn rift_preserves_total_points() {
        let fib = SphericalFibonacci::new(500);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, 4, 42);
        let (plates, _) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        let plate = plates
            .iter()
            .max_by_key(|p| p.point_count())
            .unwrap();
        let original_count = plate.point_count();
        let sub_plates = rift_plate(plate, 42);
        let total: usize = sub_plates.iter().map(|p| p.point_count()).sum();
        assert_eq!(total, original_count);
    }

    #[test]
    fn rift_produces_multiple_sub_plates() {
        let fib = SphericalFibonacci::new(500);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, 4, 42);
        let (plates, _) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        let plate = plates
            .iter()
            .max_by_key(|p| p.point_count())
            .unwrap();
        let sub_plates = rift_plate(plate, 42);
        assert!(
            sub_plates.len() >= 2,
            "should produce at least 2 sub-plates, got {}",
            sub_plates.len()
        );
    }

    #[test]
    fn rift_sub_plates_have_different_axes() {
        let fib = SphericalFibonacci::new(500);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, 4, 42);
        let (plates, _) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        let plate = plates
            .iter()
            .max_by_key(|p| p.point_count())
            .unwrap();
        let sub_plates = rift_plate(plate, 42);
        for i in 0..sub_plates.len() {
            for j in (i + 1)..sub_plates.len() {
                let diff =
                    (sub_plates[i].rotation_axis - sub_plates[j].rotation_axis).length();
                assert!(diff > 1e-6, "sub-plates {i} and {j} have identical axes");
            }
        }
    }
}
