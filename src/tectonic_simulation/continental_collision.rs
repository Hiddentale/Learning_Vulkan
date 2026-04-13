use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::PI;

use glam::DVec3;

use super::plate_seed_placement::Adjacency;
use super::plates::{CrustType, OrogenyType, Plate};

/// Planet radius R in km.
const PLANET_RADIUS: f64 = 6370.0;
/// Global maximum influence radius r_c in km.
const MAX_INFLUENCE_RADIUS: f64 = 4200.0;
/// Discrete collision coefficient Δ_c in km⁻¹.
const COLLISION_COEFFICIENT: f64 = 1.3e-5;
/// Reference plate speed v_0 in mm/yr.
const REFERENCE_SPEED: f64 = 100.0;

/// Influence radius scales with convergence speed and terrane size so that
/// fast, large collisions deform a wider area than slow, small ones.
pub fn influence_radius(relative_speed: f64, terrane_area: f64, plate_count: usize) -> f64 {
    let reference_area = 4.0 * PI * PLANET_RADIUS * PLANET_RADIUS / plate_count as f64;
    let speed_factor = (relative_speed / REFERENCE_SPEED).max(0.0).sqrt();
    let area_factor = (terrane_area / reference_area).max(0.0).sqrt();
    MAX_INFLUENCE_RADIUS * speed_factor * area_factor
}

/// Compactly supported radial falloff: 1 at center, 0 at radius.
/// Ensures deformation stays local to the collision zone.
fn radial_falloff(distance: f64, radius: f64) -> f64 {
    if distance >= radius || radius <= 0.0 {
        return 0.0;
    }
    let t = 1.0 - (distance / radius) * (distance / radius);
    t * t
}

/// Discrete elevation surge from a collision event.
pub fn elevation_surge(terrane_area: f64, distance: f64, radius: f64) -> f64 {
    COLLISION_COEFFICIENT * terrane_area * radial_falloff(distance, radius)
}

/// Fold direction after collision: tangent to the sphere, pointing radially
/// away from the terrane centroid. This orients mountain ridges perpendicular
/// to the collision front.
pub fn fold_direction(point: DVec3, terrane_centroid: DVec3) -> DVec3 {
    let normal = point.normalize();
    let diff = point - terrane_centroid;
    if diff.length() < 1e-12 {
        return DVec3::ZERO;
    }
    let radial = diff.normalize();
    normal.cross(radial).cross(normal).normalize_or_zero()
}

/// Apply a collision event to a single point on the overriding plate.
/// Returns (new_elevation, new_fold_direction).
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
    /// Global point indices belonging to this terrane.
    pub points: Vec<u32>,
    /// Centroid of the terrane on the unit sphere.
    pub centroid: DVec3,
}

/// Find all terranes (connected continental regions) within a plate.
/// Each terrane is a connected component of continental-type points
/// using the Delaunay adjacency restricted to points within the plate.
pub fn find_terranes(plate: &Plate, positions: &[DVec3], adjacency: &Adjacency) -> Vec<Terrane> {
    let plate_set: HashSet<u32> = plate.point_indices.iter().copied().collect();
    let global_to_local: HashMap<u32, usize> = plate.point_indices.iter()
        .enumerate()
        .map(|(local, &global)| (global, local))
        .collect();
    let mut visited = HashSet::new();
    let mut terranes = Vec::new();

    for (local_idx, &global_idx) in plate.point_indices.iter().enumerate() {
        if plate.crust[local_idx].crust_type != CrustType::Continental {
            continue;
        }
        if visited.contains(&global_idx) {
            continue;
        }
        let points = flood_fill_continental(
            global_idx, plate, &plate_set, &global_to_local, adjacency, &mut visited,
        );
        let centroid = compute_centroid(&points, positions);
        terranes.push(Terrane { points, centroid });
    }

    terranes
}

fn flood_fill_continental(
    start: u32,
    plate: &Plate,
    plate_set: &HashSet<u32>,
    global_to_local: &HashMap<u32, usize>,
    adjacency: &Adjacency,
    visited: &mut HashSet<u32>,
) -> Vec<u32> {
    let mut component = Vec::new();
    let mut queue = VecDeque::new();
    visited.insert(start);
    queue.push_back(start);

    while let Some(current) = queue.pop_front() {
        component.push(current);
        for &neighbor in adjacency.neighbors_of(current) {
            if visited.contains(&neighbor) || !plate_set.contains(&neighbor) {
                continue;
            }
            let local = global_to_local[&neighbor];
            if plate.crust[local].crust_type != CrustType::Continental {
                continue;
            }
            visited.insert(neighbor);
            queue.push_back(neighbor);
        }
    }

    component
}

fn compute_centroid(points: &[u32], positions: &[DVec3]) -> DVec3 {
    let sum: DVec3 = points.iter().map(|&i| positions[i as usize]).sum();
    sum.normalize_or_zero()
}

/// Transfer a terrane from one plate to another.
/// Moves points and their crust data, setting orogeny to Himalayan.
pub fn transfer_terrane(
    terrane: &Terrane,
    source: &mut Plate,
    target: &mut Plate,
) {
    let terrane_set: HashSet<u32> = terrane.points.iter().copied().collect();

    // Extract from source, collecting crust data for transfer.
    let mut transferred_crust = Vec::with_capacity(terrane.points.len());
    let mut keep_indices = Vec::with_capacity(source.point_indices.len());
    let mut keep_crust = Vec::with_capacity(source.crust.len());

    for (i, &global) in source.point_indices.iter().enumerate() {
        if terrane_set.contains(&global) {
            let mut crust = source.crust[i].clone();
            crust.orogeny_type = Some(OrogenyType::Himalayan);
            transferred_crust.push((global, crust));
        } else {
            keep_indices.push(global);
            keep_crust.push(source.crust[i].clone());
        }
    }

    source.point_indices = keep_indices;
    source.crust = keep_crust;

    // Attach to target.
    for (global, crust) in transferred_crust {
        target.point_indices.push(global);
        target.crust.push(crust);
    }
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
    fn elevation_surge_strongest_at_center() {
        let area = 1e7;
        let at_center = elevation_surge(area, 0.0, 4000.0);
        let at_midpoint = elevation_surge(area, 2000.0, 4000.0);
        assert!(at_center > at_midpoint);
        assert!((at_center - COLLISION_COEFFICIENT * area).abs() < 1e-10);
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
    fn fold_direction_points_away_from_centroid() {
        let fold = fold_direction(DVec3::X, DVec3::Y);
        assert!(fold.dot(DVec3::Y) < 0.0);
    }

    #[test]
    fn fold_direction_zero_at_centroid() {
        assert!(fold_direction(DVec3::X, DVec3::X).length() < 1e-10);
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

    // --- Terrane detection and transfer tests ---

    use super::super::plates::CrustData;
    use super::super::fibonnaci_spiral::SphericalFibonacci;
    use super::super::spherical_delaunay_triangulation::SphericalDelaunay;

    fn make_plate(indices: Vec<u32>, crusts: Vec<CrustType>) -> Plate {
        let crust = crusts
            .into_iter()
            .map(|ct| match ct {
                CrustType::Continental => CrustData::continental(35.0, 0.3, 0.0, DVec3::X, OrogenyType::Andean),
                CrustType::Oceanic => CrustData::oceanic(7.0, -4.0, 0.0, DVec3::X),
            })
            .collect();
        Plate {
            point_indices: indices,
            crust,
            rotation_axis: DVec3::Z,
            angular_speed: 0.01,
        }
    }

    #[test]
    fn find_terranes_single_continental_region() {
        let fib = SphericalFibonacci::new(100);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let adj = Adjacency::from_delaunay(100, &del);

        // All points in one plate, first 30 continental, rest oceanic.
        let indices: Vec<u32> = (0..100).collect();
        let crusts: Vec<CrustType> = (0..100)
            .map(|i| if i < 30 { CrustType::Continental } else { CrustType::Oceanic })
            .collect();
        let plate = make_plate(indices, crusts);

        let terranes = find_terranes(&plate, &points, &adj);
        assert!(!terranes.is_empty());
        let total_continental: usize = terranes.iter().map(|t| t.points.len()).sum();
        assert_eq!(total_continental, 30);
    }

    #[test]
    fn find_terranes_returns_empty_for_all_oceanic() {
        let fib = SphericalFibonacci::new(50);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let adj = Adjacency::from_delaunay(50, &del);

        let indices: Vec<u32> = (0..50).collect();
        let crusts = vec![CrustType::Oceanic; 50];
        let plate = make_plate(indices, crusts);

        assert!(find_terranes(&plate, &points, &adj).is_empty());
    }

    #[test]
    fn find_terranes_centroid_is_normalized() {
        let fib = SphericalFibonacci::new(50);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let adj = Adjacency::from_delaunay(50, &del);

        let indices: Vec<u32> = (0..50).collect();
        let crusts = vec![CrustType::Continental; 50];
        let plate = make_plate(indices, crusts);

        let terranes = find_terranes(&plate, &points, &adj);
        for t in &terranes {
            assert!((t.centroid.length() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn transfer_terrane_moves_points() {
        let fib = SphericalFibonacci::new(20);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let adj = Adjacency::from_delaunay(20, &del);

        let source_indices: Vec<u32> = (0..10).collect();
        let source_crusts = vec![CrustType::Continental; 10];
        let mut source = make_plate(source_indices, source_crusts);

        let target_indices: Vec<u32> = (10..20).collect();
        let target_crusts = vec![CrustType::Continental; 10];
        let mut target = make_plate(target_indices, target_crusts);

        let terranes = find_terranes(&source, &points, &adj);
        assert!(!terranes.is_empty());
        let terrane = &terranes[0];
        let terrane_size = terrane.points.len();

        transfer_terrane(terrane, &mut source, &mut target);

        assert_eq!(source.point_indices.len(), 10 - terrane_size);
        assert_eq!(target.point_indices.len(), 10 + terrane_size);
        assert_eq!(source.crust.len(), source.point_indices.len());
        assert_eq!(target.crust.len(), target.point_indices.len());
    }

    #[test]
    fn transfer_terrane_sets_himalayan_orogeny() {
        let fib = SphericalFibonacci::new(20);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let adj = Adjacency::from_delaunay(20, &del);

        let mut source = make_plate((0..10).collect(), vec![CrustType::Continental; 10]);
        let mut target = make_plate((10..20).collect(), vec![CrustType::Oceanic; 10]);

        let terranes = find_terranes(&source, &points, &adj);
        let terrane = &terranes[0];
        transfer_terrane(terrane, &mut source, &mut target);

        // All transferred points should have Himalayan orogeny.
        for crust in &target.crust[10..] {
            assert_eq!(crust.orogeny_type, Some(OrogenyType::Himalayan));
        }
    }

    #[test]
    fn transfer_preserves_total_point_count() {
        let fib = SphericalFibonacci::new(20);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let adj = Adjacency::from_delaunay(20, &del);

        let mut source = make_plate((0..10).collect(), vec![CrustType::Continental; 10]);
        let mut target = make_plate((10..20).collect(), vec![CrustType::Continental; 10]);

        let terranes = find_terranes(&source, &points, &adj);
        transfer_terrane(&terranes[0], &mut source, &mut target);

        assert_eq!(source.point_indices.len() + target.point_indices.len(), 20);
    }
}
