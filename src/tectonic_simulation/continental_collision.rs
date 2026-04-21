use std::f64::consts::PI;

use glam::DVec3;

use super::plates::{CrustType, Plate};

/// Planet radius R in km.
const PLANET_RADIUS: f64 = 6370.0;
/// Global maximum influence radius r_c in km.
const MAX_INFLUENCE_RADIUS: f64 = 4200.0;
/// Discrete collision coefficient Δ_c in km⁻¹.
const COLLISION_COEFFICIENT: f64 = 1.3e-5;
/// Reference plate speed v_0 in mm/yr.
const REFERENCE_SPEED: f64 = 100.0;
/// Distance below which a point is treated as coincident with the terrane centroid.
const COINCIDENT_THRESHOLD: f64 = 1e-12;

pub fn influence_radius(relative_speed: f64, terrane_area: f64, plate_count: usize) -> f64 {
    let reference_area = 4.0 * PI * PLANET_RADIUS * PLANET_RADIUS / plate_count as f64;
    let speed_factor = (relative_speed / REFERENCE_SPEED).max(0.0).sqrt();
    let area_factor = (terrane_area / reference_area).max(0.0).sqrt();
    MAX_INFLUENCE_RADIUS * speed_factor * area_factor
}

fn radial_falloff(distance: f64, radius: f64) -> f64 {
    if distance >= radius || radius <= 0.0 {
        return 0.0;
    }
    let t = 1.0 - (distance / radius) * (distance / radius);
    t * t
}

pub fn elevation_surge(terrane_area: f64, distance: f64, radius: f64) -> f64 {
    COLLISION_COEFFICIENT * terrane_area * radial_falloff(distance, radius)
}

pub fn fold_direction(point: DVec3, terrane_centroid: DVec3) -> DVec3 {
    let normal = point.normalize();
    let diff = point - terrane_centroid;
    if diff.length() < COINCIDENT_THRESHOLD {
        return DVec3::ZERO;
    }
    let radial = diff.normalize();
    normal.cross(radial).cross(normal).normalize_or_zero()
}

pub fn apply(
    elevation: f64,
    point: DVec3,
    terrane_centroid: DVec3,
    terrane_area: f64,
    distance: f64,
    radius: f64,
) -> (f64, DVec3) {
    let new_elevation = elevation + elevation_surge(terrane_area, distance, radius);
    let new_fold = fold_direction(point, terrane_centroid);
    (new_elevation, new_fold)
}

/// A connected region of continental crust within a plate.
pub struct Terrane {
    /// Local vertex indices within the plate's reference mesh.
    pub vertices: Vec<u32>,
    /// World-space centroid of the terrane.
    pub centroid: DVec3,
}

/// Find terranes (connected continental regions) within a plate.
///
/// Uses the plate's triangle adjacency to BFS over continental vertices.
pub fn find_terranes(plate: &Plate) -> Vec<Terrane> {
    // Build vertex adjacency from triangle list.
    let n = plate.reference_points.len();
    let mut vert_neighbors: Vec<Vec<u32>> = vec![Vec::new(); n];
    for tri in &plate.triangles {
        for i in 0..3 {
            let a = tri[i];
            let b = tri[(i + 1) % 3];
            if !vert_neighbors[a as usize].contains(&b) {
                vert_neighbors[a as usize].push(b);
            }
            if !vert_neighbors[b as usize].contains(&a) {
                vert_neighbors[b as usize].push(a);
            }
        }
    }

    let mut visited = vec![false; n];
    let mut terranes = Vec::new();

    for start in 0..n {
        if visited[start] || plate.crust[start].crust_type != CrustType::Continental {
            continue;
        }

        let mut component = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        visited[start] = true;
        queue.push_back(start as u32);

        while let Some(v) = queue.pop_front() {
            component.push(v);
            for &nb in &vert_neighbors[v as usize] {
                if !visited[nb as usize]
                    && plate.crust[nb as usize].crust_type == CrustType::Continental
                {
                    visited[nb as usize] = true;
                    queue.push_back(nb);
                }
            }
        }

        let centroid: DVec3 = component
            .iter()
            .map(|&v| plate.to_world(plate.reference_points[v as usize]))
            .sum::<DVec3>()
            .normalize_or_zero();

        terranes.push(Terrane {
            vertices: component,
            centroid,
        });
    }

    terranes
}

/// Transfer a terrane between plate meshes.
///
/// TODO: this requires mesh surgery — removing triangles from source,
/// adding vertices+triangles to target, rebuilding adjacency for both.
/// For now this is a placeholder.
pub fn transfer_terrane(
    _terrane: &Terrane,
    _source: &mut Plate,
    _target: &mut Plate,
) {
    todo!("transfer_terrane needs mesh surgery for per-plate sub-mesh model")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn radial_falloff_one_at_center() {
        assert!((radial_falloff(0.0, 100.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn radial_falloff_zero_at_boundary() {
        assert_eq!(radial_falloff(100.0, 100.0), 0.0);
        assert_eq!(radial_falloff(150.0, 100.0), 0.0);
    }

    #[test]
    fn radial_falloff_monotonically_decreasing() {
        let r = 500.0;
        let mut prev = 1.0;
        for i in 1..=10 {
            let x = r * i as f64 / 10.0;
            let val = radial_falloff(x, r);
            assert!(val <= prev, "not decreasing at x={x}");
            prev = val;
        }
    }

    #[test]
    fn influence_radius_scales_with_speed_and_area() {
        let r_slow = influence_radius(25.0, 1e7, 20);
        let r_fast = influence_radius(100.0, 1e7, 20);
        assert!(r_fast > r_slow);

        let r_small = influence_radius(50.0, 1e6, 20);
        let r_large = influence_radius(50.0, 1e8, 20);
        assert!(r_large > r_small);
    }

    #[test]
    fn influence_radius_equals_max_at_reference_values() {
        let reference_area = 4.0 * PI * PLANET_RADIUS * PLANET_RADIUS / 20.0;
        let r = influence_radius(REFERENCE_SPEED, reference_area, 20);
        assert!((r - MAX_INFLUENCE_RADIUS).abs() < 1e-6);
    }

    #[test]
    fn elevation_surge_zero_outside_radius() {
        assert_eq!(elevation_surge(1e7, 5000.0, 4000.0), 0.0);
    }

    #[test]
    fn fold_direction_tangent_to_sphere() {
        let p = DVec3::new(0.5, 0.5, 0.707).normalize();
        let q = DVec3::new(-0.3, 0.8, 0.5).normalize();
        let fold = fold_direction(p, q);
        assert!(fold.dot(p.normalize()).abs() < 1e-10);
        assert!((fold.length() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn apply_increases_elevation_within_radius() {
        let (new_z, _) = apply(0.5, DVec3::X, DVec3::Y, 1e7, 500.0, 4000.0);
        assert!(new_z > 0.5);
    }

    #[test]
    fn apply_no_change_outside_radius() {
        let (new_z, _) = apply(0.5, DVec3::X, DVec3::Y, 1e7, 5000.0, 4000.0);
        assert_eq!(new_z, 0.5);
    }

    #[test]
    fn find_terranes_on_initialized_plate() {
        use crate::tectonic_simulation::fibonnaci_spiral::SphericalFibonacci;
        use crate::tectonic_simulation::plate_initializer::{initialize_plates, InitParams};
        use crate::tectonic_simulation::plate_seed_placement::assign_plates;
        use crate::tectonic_simulation::spherical_delaunay_triangulation::SphericalDelaunay;

        let fib = SphericalFibonacci::new(500);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, 8, 42);
        let (plates, _) = initialize_plates(&points, &del, &assignment, &InitParams::default());

        for plate in &plates {
            let terranes = find_terranes(plate);
            for t in &terranes {
                assert!((t.centroid.length() - 1.0).abs() < 1e-6);
            }
        }
    }
}
