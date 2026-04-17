use glam::DVec3;

use super::boundary::BoundaryEdge;
use super::plates::{CrustData, Plate};
use super::util::{arbitrary_tangent, plate_centroid};

/// Planet radius in km.
const PLANET_RADIUS: f64 = 6370.0;
/// Mid-ocean ridge axis depth in km.
const RIDGE_AXIS_DEPTH: f64 = -1.0;
/// Abyssal plain depth floor in km.
const ABYSSAL_DEPTH: f64 = -6.0;
/// Oceanic crust thickness at ridge in km.
const RIDGE_THICKNESS: f64 = 7.0;
/// Half-space cooling coefficient.
/// Calibrated: at 80 Myr depth reaches ~-5.5 km.
/// -1.0 - k*sqrt(80) = -5.5 => k ≈ 0.503
const COOLING_COEFFICIENT: f64 = 0.503;
/// Minimum divergence speed (km/Myr) to count as a ridge.
const MIN_DIVERGENCE_SPEED: f64 = 0.5;
/// Fastest plates → generate every 10 steps.
const MIN_GENERATION_INTERVAL: usize = 10;
/// Slowest plates → generate every 60 steps.
const MAX_GENERATION_INTERVAL: usize = 60;
/// Angular speed threshold for the interval ramp (rad/Myr).
const FAST_ANGULAR_SPEED: f64 = 0.02;
const SLOW_ANGULAR_SPEED: f64 = 0.005;
/// Minimum angular separation between new ridge points (radians, ~2°).
const MIN_POINT_SEPARATION: f64 = 0.035;
/// Squared length below which a vector is treated as degenerate (effectively zero).
const DEGENERATE_LENGTH_SQ: f64 = 1e-20;
/// Value below which a scalar denominator is treated as zero.
const ZERO_THRESHOLD: f64 = 1e-10;
/// Each plate receives half the total spreading rate.
const HALF_SPREADING: f64 = 0.5;

pub(super) struct DivergentEdge {
    pub point_a: u32,
    pub point_b: u32,
    pub plate_a: u32,
    pub plate_b: u32,
    pub ridge_pos: DVec3,
    pub divergence_speed: f64,
    pub elevation_a: f64,
    pub elevation_b: f64,
}

pub(super) struct NewRidgePoint {
    pub position: DVec3,
    pub plate_index: u32,
    pub crust: CrustData,
}

/// How many steps between oceanic crust generation passes.
/// Faster max plate speed → shorter interval.
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

/// Find boundary edges where plates diverge.
pub(super) fn find_divergent_edges(
    boundary: &[BoundaryEdge],
    plates: &[Plate],
    points: &[DVec3],
) -> Vec<DivergentEdge> {
    let point_to_local = build_local_index_map(plates, points.len());
    let mut edges = Vec::new();

    for edge in boundary {
        if let Some(div) = classify_edge(edge, plates, points, &point_to_local) {
            edges.push(div);
        }
    }

    edges
}

fn build_local_index_map(plates: &[Plate], point_count: usize) -> Vec<usize> {
    let mut map = vec![0usize; point_count];
    for plate in plates {
        for (local, &global) in plate.point_indices.iter().enumerate() {
            map[global as usize] = local;
        }
    }
    map
}

fn classify_edge(
    edge: &BoundaryEdge,
    plates: &[Plate],
    points: &[DVec3],
    point_to_local: &[usize],
) -> Option<DivergentEdge> {
    let p = points[edge.point as usize];
    let vel_a = plates[edge.plate_a as usize].surface_velocity(p);
    let vel_b = plates[edge.plate_b as usize].surface_velocity(p);
    let relative = vel_a - vel_b;

    let neighbor_pos = points[edge.neighbor as usize];
    let edge_dir = neighbor_pos - p;
    let tangent = (edge_dir - p * edge_dir.dot(p)).normalize_or_zero();
    if tangent.length_squared() < DEGENERATE_LENGTH_SQ {
        return None;
    }

    let divergence = relative.dot(tangent) * PLANET_RADIUS;
    if divergence < MIN_DIVERGENCE_SPEED {
        return None;
    }

    let local_a = point_to_local[edge.point as usize];
    let local_b = point_to_local[edge.neighbor as usize];

    // After collision transfers or rifting, a boundary edge may reference
    // a stale local index. Skip these safely.
    let plate_a_len = plates[edge.plate_a as usize].crust.len();
    let plate_b_len = plates[edge.plate_b as usize].crust.len();
    if local_a >= plate_a_len || local_b >= plate_b_len {
        return None;
    }

    Some(DivergentEdge {
        point_a: edge.point,
        point_b: edge.neighbor,
        plate_a: edge.plate_a,
        plate_b: edge.plate_b,
        ridge_pos: (p + neighbor_pos).normalize(),
        divergence_speed: divergence,
        elevation_a: plates[edge.plate_a as usize].crust[local_a].elevation,
        elevation_b: plates[edge.plate_b as usize].crust[local_b].elevation,
    })
}

/// Generate new sample points along divergent ridges.
pub(super) fn generate_ridge_points(
    divergent_edges: &[DivergentEdge],
    plates: &[Plate],
    points: &[DVec3],
) -> Vec<NewRidgePoint> {
    let centroids: Vec<DVec3> = plates.iter().map(|p| plate_centroid(p, points)).collect();
    let mut new_points: Vec<NewRidgePoint> = Vec::new();

    for edge in divergent_edges {
        let ridge_pos = edge.ridge_pos;

        let too_close = new_points.iter().any(|np| {
            let dot = np.position.dot(ridge_pos).clamp(-1.0, 1.0);
            dot.acos() < MIN_POINT_SEPARATION
        });
        if too_close {
            continue;
        }

        let pa = points[edge.point_a as usize];
        let pb = points[edge.point_b as usize];
        let d_ridge = 0.0;
        let d_plate = ridge_pos.dot(pa).clamp(-1.0, 1.0).acos() * PLANET_RADIUS;

        let z_bar = HALF_SPREADING * (edge.elevation_a + edge.elevation_b);
        let spreading_rate = edge.divergence_speed * HALF_SPREADING;
        let z_gamma = ridge_profile(d_ridge, spreading_rate);
        let elevation = blend_elevation(d_ridge, d_plate, z_bar, z_gamma);

        let ridge_dir = compute_ridge_direction(ridge_pos, pa, pb);

        let dot_a = ridge_pos.dot(centroids[edge.plate_a as usize]);
        let dot_b = ridge_pos.dot(centroids[edge.plate_b as usize]);
        let plate_index = if dot_a >= dot_b { edge.plate_a } else { edge.plate_b };

        new_points.push(NewRidgePoint {
            position: ridge_pos,
            plate_index,
            crust: CrustData::oceanic(RIDGE_THICKNESS, elevation, 0.0, ridge_dir),
        });
    }

    new_points
}

fn compute_ridge_direction(ridge_pos: DVec3, pa: DVec3, pb: DVec3) -> DVec3 {
    // The paper defines r(p) = (p - q) × p where q is the ridge projection.
    // For points born on the ridge axis, p ≈ q so (p - q) ≈ 0. Instead compute
    // the direction perpendicular to the ridge line on the sphere surface:
    // the ridge runs roughly pa→pb, so cross with the surface normal.
    let ridge_perp = (pb - pa).cross(ridge_pos).normalize_or_zero();
    if ridge_perp.length_squared() < DEGENERATE_LENGTH_SQ {
        arbitrary_tangent(ridge_pos)
    } else {
        ridge_perp
    }
}

/// Template ridge profile: elevation as a function of distance from ridge axis.
/// Uses the half-space cooling model.
fn ridge_profile(distance_from_ridge: f64, spreading_rate: f64) -> f64 {
    if spreading_rate < ZERO_THRESHOLD || distance_from_ridge < ZERO_THRESHOLD {
        return RIDGE_AXIS_DEPTH;
    }
    let age = distance_from_ridge / spreading_rate;
    let depth = RIDGE_AXIS_DEPTH - COOLING_COEFFICIENT * age.sqrt();
    depth.max(ABYSSAL_DEPTH)
}

/// Blend interpolated plate-border elevation with ridge template.
/// α = d_ridge / (d_ridge + d_plate_edge)
/// z = α·z̄ + (1-α)·z_Γ
fn blend_elevation(d_ridge: f64, d_plate_edge: f64, z_bar: f64, z_gamma: f64) -> f64 {
    let denom = d_ridge + d_plate_edge;
    if denom < ZERO_THRESHOLD {
        return z_gamma;
    }
    let alpha = d_ridge / denom;
    alpha * z_bar + (1.0 - alpha) * z_gamma
}

/// Ridge direction perpendicular to the ridge line and tangent to the sphere.
fn ridge_direction(point: DVec3, ridge_projection: DVec3) -> DVec3 {
    (point - ridge_projection).cross(point).normalize_or_zero()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridge_profile_at_axis() {
        let z = ridge_profile(0.0, 50.0);
        assert!((z - RIDGE_AXIS_DEPTH).abs() < 1e-10);
    }

    #[test]
    fn ridge_profile_deepens_with_distance() {
        let z1 = ridge_profile(100.0, 50.0);
        let z2 = ridge_profile(500.0, 50.0);
        assert!(z2 < z1, "should deepen: z1={z1}, z2={z2}");
    }

    #[test]
    fn ridge_profile_clamps_to_abyssal() {
        let z = ridge_profile(100000.0, 1.0);
        assert!((z - ABYSSAL_DEPTH).abs() < 1e-10, "should clamp: z={z}");
    }

    #[test]
    fn blend_at_ridge_returns_template() {
        let z = blend_elevation(0.0, 100.0, -3.0, -2.5);
        assert!((z - (-2.5)).abs() < 1e-10);
    }

    #[test]
    fn blend_at_plate_edge_returns_interpolated() {
        let z = blend_elevation(100.0, 0.0, -3.0, -2.5);
        assert!((z - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn blend_midpoint_is_average() {
        let z = blend_elevation(50.0, 50.0, -4.0, -2.0);
        assert!((z - (-3.0)).abs() < 1e-10);
    }

    #[test]
    fn ridge_direction_tangent_to_sphere() {
        let p = DVec3::new(1.0, 0.0, 0.0);
        let q = DVec3::new(0.9, 0.1, 0.0).normalize();
        let r = ridge_direction(p, q);
        if r.length_squared() > 1e-10 {
            assert!(r.dot(p).abs() < 1e-10, "should be tangent: dot={}", r.dot(p));
        }
    }

    #[test]
    fn generation_interval_fast_plates() {
        let plates = vec![Plate {
            point_indices: vec![0],
            crust: vec![CrustData::oceanic(7.0, -4.0, 0.0, DVec3::X)],
            rotation_axis: DVec3::Y,
            angular_speed: 0.025,
        }];
        assert_eq!(generation_interval(&plates), MIN_GENERATION_INTERVAL);
    }

    #[test]
    fn generation_interval_slow_plates() {
        let plates = vec![Plate {
            point_indices: vec![0],
            crust: vec![CrustData::oceanic(7.0, -4.0, 0.0, DVec3::X)],
            rotation_axis: DVec3::Y,
            angular_speed: 0.002,
        }];
        assert_eq!(generation_interval(&plates), MAX_GENERATION_INTERVAL);
    }
}
