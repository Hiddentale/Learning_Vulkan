use glam::DVec3;

use super::plates::{CrustData, Plate};
use super::simulate::BoundarySample;
use super::util::arbitrary_tangent;

/// Planet radius in km.
const PLANET_RADIUS: f64 = 6370.0;
/// Mid-ocean ridge axis depth in km.
const RIDGE_AXIS_DEPTH: f64 = -1.0;
/// Oceanic crust thickness at ridge in km.
const RIDGE_THICKNESS: f64 = 7.0;
/// Minimum divergence speed (km/Myr) to count as a ridge.
const MIN_DIVERGENCE_SPEED: f64 = 0.5;
/// Fastest plates → generate every 2 steps.
const MIN_GENERATION_INTERVAL: usize = 2;
/// Slowest plates → generate every 10 steps.
const MAX_GENERATION_INTERVAL: usize = 10;
/// Angular speed thresholds for the interval ramp (rad/Myr).
const FAST_ANGULAR_SPEED: f64 = 0.02;
const SLOW_ANGULAR_SPEED: f64 = 0.005;
/// Minimum angular separation between new ridge vertices (radians, ~1°).
const MIN_POINT_SEPARATION: f64 = 0.018;
/// Squared length below which a vector is treated as degenerate.
const DEGENERATE_LENGTH_SQ: f64 = 1e-20;

/// How many steps between oceanic crust generation passes.
pub(super) fn generation_interval(plates: &[Plate]) -> usize {
    let max_speed = plates
        .iter()
        .map(|p| p.angular_speed.abs())
        .fold(0.0_f64, f64::max);

    if max_speed >= FAST_ANGULAR_SPEED {
        return MIN_GENERATION_INTERVAL;
    }
    if max_speed <= SLOW_ANGULAR_SPEED {
        return MAX_GENERATION_INTERVAL;
    }

    let t = (max_speed - SLOW_ANGULAR_SPEED) / (FAST_ANGULAR_SPEED - SLOW_ANGULAR_SPEED);
    let interval = MAX_GENERATION_INTERVAL as f64
        - t * (MAX_GENERATION_INTERVAL - MIN_GENERATION_INTERVAL) as f64;
    interval.round() as usize
}

struct RidgePoint {
    world_pos: DVec3,
    plate_idx: usize,
    ridge_dir: DVec3,
}

/// Detect divergent boundaries and insert new oceanic crust vertices
/// into plate meshes at ridge positions.
pub(super) fn generate_oceanic_crust(
    plates: &mut [Plate],
    boundary: &[BoundarySample],
) {
    let ridge_points = find_ridge_points(boundary, plates);
    if ridge_points.is_empty() {
        return;
    }

    for rp in &ridge_points {
        let plate = &mut plates[rp.plate_idx];
        let ref_pos = plate.to_reference(rp.world_pos).normalize();

        let Some((tri, edge)) = plate.nearest_boundary_edge(ref_pos) else {
            continue;
        };

        let crust = CrustData::oceanic(RIDGE_THICKNESS, RIDGE_AXIS_DEPTH, 0.0, rp.ridge_dir);
        plate.insert_boundary_vertex(ref_pos, crust, tri, edge);
    }

    for plate in plates.iter_mut() {
        plate.recompute_bounding_cap();
    }
}

fn find_ridge_points(boundary: &[BoundarySample], plates: &[Plate]) -> Vec<RidgePoint> {
    let mut points: Vec<RidgePoint> = Vec::new();

    for edge in boundary {
        let vel_a = plates[edge.plate_a as usize].surface_velocity(edge.position);
        let vel_b = plates[edge.plate_b as usize].surface_velocity(edge.position);
        let relative = vel_a - vel_b;

        // Project relative velocity onto the sphere tangent plane at the boundary.
        let normal = edge.position.normalize();
        let tangent_vel = relative - normal * relative.dot(normal);
        if tangent_vel.length_squared() < DEGENERATE_LENGTH_SQ {
            continue;
        }

        // Divergence: positive component means plates are moving apart.
        // Use the direction from plate_a's centroid toward plate_b's to define "apart."
        let ca = plate_world_centroid(&plates[edge.plate_a as usize]);
        let cb = plate_world_centroid(&plates[edge.plate_b as usize]);
        let apart_dir = (cb - ca) - normal * (cb - ca).dot(normal);
        if apart_dir.length_squared() < DEGENERATE_LENGTH_SQ {
            continue;
        }
        let divergence = tangent_vel.dot(apart_dir.normalize()) * PLANET_RADIUS;
        if divergence < MIN_DIVERGENCE_SPEED {
            continue;
        }

        let ridge_pos = edge.position;

        // Minimum spacing between ridge points.
        let too_close = points.iter().any(|p| {
            p.world_pos.dot(ridge_pos).clamp(-1.0, 1.0).acos() < MIN_POINT_SEPARATION
        });
        if too_close {
            continue;
        }

        // Assign to the plate whose centroid is farther from the ridge
        // (the plate "behind" the ridge gets the new vertex on its leading edge).
        let dot_a = ridge_pos.dot(ca);
        let dot_b = ridge_pos.dot(cb);
        let plate_idx = if dot_a >= dot_b {
            edge.plate_a as usize
        } else {
            edge.plate_b as usize
        };

        let ridge_dir = compute_ridge_direction(ridge_pos, ca, cb);

        points.push(RidgePoint {
            world_pos: ridge_pos,
            plate_idx,
            ridge_dir,
        });
    }

    points
}

fn plate_world_centroid(plate: &Plate) -> DVec3 {
    plate
        .reference_points
        .iter()
        .map(|&p| plate.to_world(p))
        .sum::<DVec3>()
        .normalize_or_zero()
}

fn compute_ridge_direction(ridge_pos: DVec3, centroid_a: DVec3, centroid_b: DVec3) -> DVec3 {
    let spreading = (centroid_b - centroid_a).normalize_or_zero();
    let ridge_perp = spreading.cross(ridge_pos).normalize_or_zero();
    if ridge_perp.length_squared() < DEGENERATE_LENGTH_SQ {
        arbitrary_tangent(ridge_pos)
    } else {
        ridge_perp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_interval_fast_plates() {
        use crate::tectonic_simulation::plate_initializer::{initialize_plates, InitParams};
        use crate::tectonic_simulation::plate_seed_placement::assign_plates;
        use crate::tectonic_simulation::fibonnaci_spiral::SphericalFibonacci;
        use crate::tectonic_simulation::spherical_delaunay_triangulation::SphericalDelaunay;

        let fib = SphericalFibonacci::new(100);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, 2, 42);
        let (mut plates, _) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        plates[0].angular_speed = 0.025;
        assert_eq!(generation_interval(&plates), MIN_GENERATION_INTERVAL);
    }

    #[test]
    fn generation_interval_slow_plates() {
        use crate::tectonic_simulation::plate_initializer::{initialize_plates, InitParams};
        use crate::tectonic_simulation::plate_seed_placement::assign_plates;
        use crate::tectonic_simulation::fibonnaci_spiral::SphericalFibonacci;
        use crate::tectonic_simulation::spherical_delaunay_triangulation::SphericalDelaunay;

        let fib = SphericalFibonacci::new(100);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, 2, 42);
        let (mut plates, _) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        for p in &mut plates { p.angular_speed = 0.002; }
        assert_eq!(generation_interval(&plates), MAX_GENERATION_INTERVAL);
    }
}
