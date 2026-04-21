use glam::DVec3;
use super::plates::{CrustType, Plate};
use super::util::splitmix64;

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

// TODO: rift_plate, pick_rift_seeds, partition_plate, find_rift_boundary
// all need rewriting for the per-plate mesh model. Rifting splits one
// plate's mesh into two by partitioning its triangles, then building
// new per-plate adjacency for each sub-mesh. The current implementation
// used global point indices and a global adjacency graph.

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
        let (plates, _) = initialize_plates(&points, &del, &assignment, &InitParams {
            seed: 42,
            continental_fraction: 0.05,
        });
        // Pick a plate that ended up with low continental fraction.
        let plate = plates
            .iter()
            .enumerate()
            .find(|(_, p)| {
                let frac = p.crust.iter().filter(|c| c.crust_type == CrustType::Continental).count()
                    as f64 / p.point_count() as f64;
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
        let (plates, _) = initialize_plates(&points, &del, &assignment, &InitParams {
            seed: 42,
            continental_fraction: 1.0,
        });
        let plate = &plates[0];
        let rift_count = (0..10000)
            .filter(|&i| should_rift(plate, 0, 2000, 10, i as f64, i as u64))
            .count();
        assert!(rift_count > 0, "should rift at least once in 10000 tries");
        assert!(rift_count < 2000, "should not rift too often: {rift_count}/10000");
    }
}
