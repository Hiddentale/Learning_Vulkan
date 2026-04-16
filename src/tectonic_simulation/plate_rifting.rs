use std::collections::BinaryHeap;

use glam::DVec3;
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use super::plate_seed_placement::{warped_edge_cost, Adjacency, WARP_COST_FLOOR};
use super::plates::Plate;
use super::util::{splitmix64, MinHeapEntry};

/// Base Poisson rate λ_0 for rifting probability per check.
/// With RIFT_CHECK_INTERVAL=20 and area scaling, a 2x-average fully-continental
/// plate gets P≈18% per check, making supercontinent breakup near-certain.
const BASE_RIFT_RATE: f64 = 0.08;
/// Minimum simulation steps between rifting checks.
pub(super) const RIFT_CHECK_INTERVAL: usize = 20;
/// Minimum points a plate must have to be eligible for rifting.
const MIN_PLATE_POINTS: usize = 50;
/// Plates with continental fraction below this don't rift.
const MIN_CONTINENTAL_FRACTION: f64 = 0.3;
/// Angular perturbation applied to sub-plate rotation axes (radians).
/// ~30° ensures sub-plates diverge visibly after rifting.
const AXIS_PERTURBATION: f64 = 0.5;
/// Noise warp amplitude for fracture line irregularity.
const RIFT_WARP_AMPLITUDE: f64 = 0.7;
/// Noise frequency for fracture warp.
const RIFT_WARP_FREQUENCY: f64 = 3.0;
/// FBM octaves for fracture warp.
const RIFT_WARP_OCTAVES: usize = 3;
/// Minimum points for a plate to split into more than 2 sub-plates.
const MIN_POINTS_FOR_MULTI_SPLIT: usize = 100;
/// Minimum points per sub-plate when choosing sub-plate count.
const POINTS_PER_SUB_PLATE: usize = 25;
/// Dot product threshold for choosing a stable cross-product axis.
/// Avoids near-parallel vectors when building a tangent frame.
const TANGENT_AXIS_THRESHOLD: f64 = 0.9;
/// Speed factor range for sub-plate speed variation: [min, max] = [0.8, 1.2].
const SPEED_FACTOR_BASE: f64 = 0.8;
const SPEED_FACTOR_RANGE: f64 = 0.4;

/// Check whether a plate should rift this timestep.
///
/// Uses Poisson probability `P = λ·e^(-λ)` where `λ = λ_0 · f(xp) · A(p)/A_0`,
/// with `f(xp)` being the plate's continental fraction and `A(p)/A_0` the
/// plate's area relative to the average plate size. Larger continental plates
/// are much more likely to rift.
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

/// Split a plate into 2–4 sub-plates via noise-warped Voronoi partitioning.
///
/// Returns the new sub-plates. The caller must replace the original plate
/// and rebuild adjacency.
pub(super) fn rift_plate(
    plate: &Plate,
    points: &[DVec3],
    adjacency: &Adjacency,
    seed: u64,
) -> Vec<Plate> {
    let n = plate.point_count();
    let sub_count = sub_plate_count(n, seed);

    let seeds = pick_rift_seeds(plate, points, sub_count, seed);
    let partition = partition_plate(plate, points, adjacency, &seeds, seed);
    let axes = perturb_axes(plate.rotation_axis, plate.angular_speed, sub_count, seed);

    let mut sub_plates: Vec<Plate> = (0..sub_count)
        .map(|i| Plate {
            point_indices: Vec::new(),
            crust: Vec::new(),
            rotation_axis: axes[i].0,
            angular_speed: axes[i].1,
        })
        .collect();

    for (local, &global) in plate.point_indices.iter().enumerate() {
        let mut sub = partition[local];
        if sub == u32::MAX {
            let p = points[global as usize];
            sub = seeds
                .iter()
                .enumerate()
                .min_by(|(_, &a), (_, &b)| {
                    let da = p.dot(points[plate.point_indices[a] as usize]);
                    let db = p.dot(points[plate.point_indices[b] as usize]);
                    db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap()
                .0 as u32;
        }
        sub_plates[sub as usize].point_indices.push(global);
        sub_plates[sub as usize].crust.push(plate.crust[local].clone());
    }

    sub_plates.retain(|p| !p.point_indices.is_empty());
    sub_plates
}

fn continental_fraction(plate: &Plate) -> f64 {
    let continental = plate
        .crust
        .iter()
        .filter(|c| c.crust_type == super::plates::CrustType::Continental)
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

/// Farthest-point sampling within a single plate to pick well-separated seeds.
fn pick_rift_seeds(
    plate: &Plate,
    points: &[DVec3],
    count: usize,
    seed: u64,
) -> Vec<usize> {
    let n = plate.point_count();
    let first = (splitmix64(seed ^ 0xBEEF) % n as u64) as usize;
    let mut seeds = vec![first];
    let mut min_dot = vec![f64::NEG_INFINITY; n];

    let first_pos = points[plate.point_indices[first] as usize];
    for (i, dot) in min_dot.iter_mut().enumerate() {
        *dot = points[plate.point_indices[i] as usize].dot(first_pos);
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
        let seed_pos = points[plate.point_indices[farthest] as usize];
        for (i, dot) in min_dot.iter_mut().enumerate() {
            let d = points[plate.point_indices[i] as usize].dot(seed_pos);
            if d > *dot {
                *dot = d;
            }
        }
    }

    seeds
}

/// Dijkstra flood-fill restricted to the plate's points with noise-warped costs.
fn partition_plate(
    plate: &Plate,
    points: &[DVec3],
    adjacency: &Adjacency,
    seeds: &[usize],
    seed: u64,
) -> Vec<u32> {
    let n = plate.point_count();
    let fbm: Fbm<Perlin> = Fbm::new(seed as u32)
        .set_octaves(RIFT_WARP_OCTAVES)
        .set_frequency(RIFT_WARP_FREQUENCY);

    let mut global_to_local = vec![u32::MAX; points.len()];
    for (local, &global) in plate.point_indices.iter().enumerate() {
        global_to_local[global as usize] = local as u32;
    }

    let mut partition = vec![u32::MAX; n];
    let mut costs = vec![f64::INFINITY; n];
    let mut heap = BinaryHeap::with_capacity(n);

    for (sub_id, &local_seed) in seeds.iter().enumerate() {
        partition[local_seed] = sub_id as u32;
        costs[local_seed] = 0.0;
        heap.push(MinHeapEntry { cost: 0.0, index: local_seed as u32 });
    }

    while let Some(MinHeapEntry { cost, index }) = heap.pop() {
        let local = index as usize;
        if cost > costs[local] {
            continue;
        }

        let global = plate.point_indices[local];
        let p = points[global as usize];

        for &neighbor_global in adjacency.neighbors_of(global) {
            let neighbor_local = global_to_local[neighbor_global as usize];
            if neighbor_local == u32::MAX {
                continue;
            }
            let nl = neighbor_local as usize;
            let q = points[neighbor_global as usize];

            let edge_cost = warped_edge_cost(p, q, RIFT_WARP_AMPLITUDE, &fbm);
            let new_cost = cost + edge_cost;

            if new_cost < costs[nl] {
                costs[nl] = new_cost;
                partition[nl] = partition[local];
                heap.push(MinHeapEntry { cost: new_cost, index: neighbor_local });
            }
        }
    }

    partition
}

/// Generate diverging rotation axes for sub-plates.
fn perturb_axes(
    parent_axis: DVec3,
    parent_speed: f64,
    count: usize,
    seed: u64,
) -> Vec<(DVec3, f64)> {
    let up = if parent_axis.y.abs() < TANGENT_AXIS_THRESHOLD { DVec3::Y } else { DVec3::X };
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
            let speed_factor = SPEED_FACTOR_BASE + SPEED_FACTOR_RANGE * (rng as f64 / u64::MAX as f64);
            (new_axis, parent_speed * speed_factor)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tectonic_simulation::fibonnaci_spiral::SphericalFibonacci;
    use crate::tectonic_simulation::plate_initializer::{initialize_plates, InitParams};
    use crate::tectonic_simulation::plate_seed_placement::{assign_plates, WarpParams};
    use crate::tectonic_simulation::plates::CrustData;
    use crate::tectonic_simulation::spherical_delaunay_triangulation::SphericalDelaunay;

    fn make_continental_plate(count: usize) -> (Vec<DVec3>, Plate) {
        let fib = SphericalFibonacci::new(count as u32);
        let points = fib.all_points();
        let crust: Vec<CrustData> = points
            .iter()
            .map(|&p| {
                let tangent = if p.y.abs() < 0.9 {
                    p.cross(DVec3::Y).normalize()
                } else {
                    p.cross(DVec3::X).normalize()
                };
                CrustData::continental(
                    35.0,
                    0.3,
                    0.0,
                    tangent,
                    super::super::plates::OrogenyType::Andean,
                )
            })
            .collect();
        let point_indices: Vec<u32> = (0..count as u32).collect();
        let plate = Plate {
            point_indices,
            crust,
            rotation_axis: DVec3::Y,
            angular_speed: 0.01,
        };
        (points, plate)
    }

    fn setup(
        point_count: u32,
        plate_count: u32,
    ) -> (Vec<DVec3>, Vec<Plate>, Adjacency) {
        let fib = SphericalFibonacci::new(point_count);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment =
            assign_plates(&points, &fib, &del, plate_count, 42, &WarpParams::default());
        let plates = initialize_plates(&points, &del, &assignment, &InitParams::default());
        let adjacency = Adjacency::from_delaunay(points.len(), &del);
        (points, plates, adjacency)
    }

    #[test]
    fn zero_continental_never_rifts() {
        let (points, _) = make_continental_plate(100);
        let crust: Vec<CrustData> = points
            .iter()
            .map(|_| CrustData::oceanic(7.0, -4.0, 0.0, DVec3::X))
            .collect();
        let plate = Plate {
            point_indices: (0..100).collect(),
            crust,
            rotation_axis: DVec3::Y,
            angular_speed: 0.01,
        };
        for i in 0..1000 {
            assert!(!should_rift(&plate, 0, 1000, 10, i as f64, 42));
        }
    }

    #[test]
    fn too_small_plate_never_rifts() {
        let (_, plate) = make_continental_plate(10);
        for i in 0..1000 {
            assert!(!should_rift(&plate, 0, 1000, 10, i as f64, 42));
        }
    }

    #[test]
    fn high_continental_sometimes_rifts() {
        let (_, plate) = make_continental_plate(200);
        let rift_count = (0..10000)
            .filter(|&i| should_rift(&plate, 0, 2000, 10, i as f64, i as u64))
            .count();
        assert!(
            rift_count > 0,
            "should rift at least once in 10000 tries"
        );
        assert!(
            rift_count < 1000,
            "should not rift too often: {rift_count}/10000"
        );
    }

    #[test]
    fn rift_preserves_total_points() {
        let (points, plates, adjacency) = setup(500, 8);
        let plate = plates
            .iter()
            .max_by_key(|p| p.point_count())
            .unwrap();
        let original_count = plate.point_count();
        let sub_plates = rift_plate(plate, &points, &adjacency, 42);
        let total: usize = sub_plates.iter().map(|p| p.point_count()).sum();
        assert_eq!(total, original_count);
    }

    #[test]
    fn rift_produces_multiple_sub_plates() {
        let (points, plates, adjacency) = setup(500, 8);
        let plate = plates
            .iter()
            .max_by_key(|p| p.point_count())
            .unwrap();
        let sub_plates = rift_plate(plate, &points, &adjacency, 42);
        assert!(
            sub_plates.len() >= 2,
            "should produce at least 2 sub-plates, got {}",
            sub_plates.len()
        );
    }

    #[test]
    fn rift_preserves_crust_parallel() {
        let (points, plates, adjacency) = setup(500, 8);
        let plate = plates
            .iter()
            .max_by_key(|p| p.point_count())
            .unwrap();
        let sub_plates = rift_plate(plate, &points, &adjacency, 42);
        for sub in &sub_plates {
            assert_eq!(
                sub.point_indices.len(),
                sub.crust.len(),
                "indices and crust must be parallel"
            );
        }
    }

    #[test]
    fn sub_plates_have_different_axes() {
        let (points, plates, adjacency) = setup(500, 8);
        let plate = plates
            .iter()
            .max_by_key(|p| p.point_count())
            .unwrap();
        let sub_plates = rift_plate(plate, &points, &adjacency, 42);
        for i in 0..sub_plates.len() {
            for j in (i + 1)..sub_plates.len() {
                let diff = (sub_plates[i].rotation_axis - sub_plates[j].rotation_axis).length();
                assert!(
                    diff > 1e-6,
                    "sub-plates {i} and {j} have identical axes"
                );
            }
        }
    }

    #[test]
    fn sub_plate_axes_are_normalized() {
        let (points, plates, adjacency) = setup(500, 8);
        let plate = plates
            .iter()
            .max_by_key(|p| p.point_count())
            .unwrap();
        let sub_plates = rift_plate(plate, &points, &adjacency, 42);
        for sub in &sub_plates {
            let len = sub.rotation_axis.length();
            assert!(
                (len - 1.0).abs() < 1e-10,
                "axis not normalized: {len}"
            );
        }
    }
}
