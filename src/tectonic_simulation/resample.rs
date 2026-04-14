use glam::DVec3;

use super::fibonnaci_spiral::SphericalFibonacci;
use super::plate_seed_placement::Adjacency;
use super::plates::{CrustData, Plate};
use super::simulate::Simulation;
use super::spherical_delaunay_triangulation::SphericalDelaunay;

/// How many steps between resamples.
pub const RESAMPLE_INTERVAL: usize = 20;

/// Resample the simulation onto a fresh Fibonacci grid.
///
/// Builds a Delaunay triangulation on the old (drifted) points, then for each
/// new Fibonacci point locates the containing triangle and uses barycentric
/// interpolation for smooth crust parameter transfer. Discrete fields (plate
/// ownership, crust type, orogeny type) come from the dominant vertex.
pub fn resample(sim: &mut Simulation, point_count: u32) {
    let old_points = &sim.points;

    // Build per-point lookups from current plate state.
    let n = old_points.len();
    let mut point_plate = vec![0u32; n];
    let mut point_local = vec![0usize; n];
    for (plate_idx, plate) in sim.plates.iter().enumerate() {
        for (local, &global) in plate.point_indices.iter().enumerate() {
            point_plate[global as usize] = plate_idx as u32;
            point_local[global as usize] = local;
        }
    }

    // Delaunay on the drifted old points — used for barycentric interpolation.
    let old_delaunay = SphericalDelaunay::from_points(old_points);

    // Fresh Fibonacci grid.
    let fib = SphericalFibonacci::new(point_count);
    let new_points = fib.all_points();

    let plate_count = sim.plates.len();
    let mut new_plate_points: Vec<Vec<u32>> = vec![Vec::new(); plate_count];
    let mut new_plate_crust: Vec<Vec<CrustData>> = vec![Vec::new(); plate_count];

    let mut last_tri = 0;
    for (new_idx, &new_p) in new_points.iter().enumerate() {
        let (tri, b1, b2, b3) = old_delaunay.locate(new_p, old_points, last_tri);
        last_tri = tri;

        let base = tri * 3;
        let vi = [
            old_delaunay.triangles[base] as usize,
            old_delaunay.triangles[base + 1] as usize,
            old_delaunay.triangles[base + 2] as usize,
        ];
        let bary = [b1, b2, b3];
        let plates = [point_plate[vi[0]], point_plate[vi[1]], point_plate[vi[2]]];

        // Plate assignment: vertex with the largest barycentric weight.
        let dominant = if bary[0] >= bary[1] && bary[0] >= bary[2] { 0 }
                       else if bary[1] >= bary[2] { 1 }
                       else { 2 };
        let owner = plates[dominant];

        // Interpolate crust data. Only blend vertices belonging to the same plate
        // to avoid averaging across plate boundaries.
        let crust = interpolate_crust(
            &sim.plates, &point_plate, &point_local, vi, bary, owner,
        );

        new_plate_points[owner as usize].push(new_idx as u32);
        new_plate_crust[owner as usize].push(crust);
    }

    // Rebuild plates with new point assignments.
    for (plate_idx, plate) in sim.plates.iter_mut().enumerate() {
        plate.point_indices = new_plate_points[plate_idx].clone();
        plate.crust = new_plate_crust[plate_idx].clone();
    }

    // Replace simulation state with fresh Fibonacci grid + Delaunay.
    let new_delaunay = SphericalDelaunay::from_points(&new_points);
    sim.adjacency = Adjacency::from_delaunay(new_points.len(), &new_delaunay);
    sim.points = new_points;
}

/// Interpolate crust data from a triangle's three vertices.
/// Only blends vertices that belong to `owner_plate`; cross-plate vertices are excluded.
fn interpolate_crust(
    plates: &[Plate],
    point_plate: &[u32],
    point_local: &[usize],
    vi: [usize; 3],
    bary: [f64; 3],
    owner_plate: u32,
) -> CrustData {
    // Collect weights for same-plate vertices only.
    let mut w = [0.0_f64; 3];
    let mut total = 0.0;
    for i in 0..3 {
        if point_plate[vi[i]] == owner_plate {
            w[i] = bary[i].max(0.0);
            total += w[i];
        }
    }

    // If no same-plate vertex has positive weight (shouldn't happen), fall back to dominant.
    if total < 1e-30 {
        let dom = if bary[0] >= bary[1] && bary[0] >= bary[2] { 0 }
                  else if bary[1] >= bary[2] { 1 }
                  else { 2 };
        let local = point_local[vi[dom]];
        let plate = point_plate[vi[dom]] as usize;
        return plates[plate].crust[local].clone();
    }

    // Normalize weights.
    for i in 0..3 { w[i] /= total; }

    // Find the dominant same-plate vertex for discrete fields.
    let dom = if w[0] >= w[1] && w[0] >= w[2] { 0 }
              else if w[1] >= w[2] { 1 }
              else { 2 };

    // Gather crust data from vertices.
    let crusts: [&CrustData; 3] = std::array::from_fn(|i| {
        let plate = point_plate[vi[i]] as usize;
        let local = point_local[vi[i]];
        &plates[plate].crust[local]
    });
    let dom_crust = crusts[dom];

    // Blend continuous fields, take discrete from dominant vertex.
    let mut thickness = 0.0;
    let mut elevation = 0.0;
    let mut age = 0.0;
    let mut direction = DVec3::ZERO;
    for i in 0..3 {
        thickness += crusts[i].thickness * w[i];
        elevation += crusts[i].elevation * w[i];
        age += crusts[i].age * w[i];
        direction += crusts[i].local_direction * w[i];
    }

    CrustData {
        crust_type: dom_crust.crust_type,
        thickness,
        elevation,
        age,
        local_direction: direction.normalize_or_zero(),
        orogeny_type: dom_crust.orogeny_type,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::plate_initializer::{initialize_plates, InitParams};
    use super::super::plate_seed_placement::{assign_plates, WarpParams};

    fn setup(point_count: u32, plate_count: u32) -> Simulation {
        let fib = SphericalFibonacci::new(point_count);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, plate_count, 42, &WarpParams::default());
        let plates = initialize_plates(&points, &del, &assignment, &InitParams::default());
        Simulation::new(points, plates, &del)
    }

    #[test]
    fn resample_preserves_point_count() {
        let mut sim = setup(500, 8);
        sim.run(5);
        resample(&mut sim, 500);
        let total: usize = sim.plates.iter().map(|p| p.point_count()).sum();
        assert_eq!(total, 500);
    }

    #[test]
    fn resample_preserves_plate_count() {
        let mut sim = setup(500, 8);
        sim.run(5);
        let plate_count_before = sim.plates.len();
        resample(&mut sim, 500);
        assert_eq!(sim.plates.len(), plate_count_before);
    }

    #[test]
    fn resample_points_on_unit_sphere() {
        let mut sim = setup(500, 8);
        sim.run(5);
        resample(&mut sim, 500);
        for &p in &sim.points {
            assert!((p.length() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn resample_most_plates_have_points() {
        let mut sim = setup(500, 8);
        sim.run(5);
        resample(&mut sim, 500);
        let non_empty = sim.plates.iter().filter(|p| !p.point_indices.is_empty()).count();
        assert!(non_empty >= 6, "too many empty plates: only {non_empty}/8 have points");
    }

    #[test]
    fn resample_crust_parallel_to_indices() {
        let mut sim = setup(500, 8);
        sim.run(5);
        resample(&mut sim, 500);
        for plate in &sim.plates {
            assert_eq!(plate.crust.len(), plate.point_indices.len());
        }
    }

    #[test]
    fn resample_after_movement_still_works() {
        let mut sim = setup(1000, 12);
        sim.run(20);
        resample(&mut sim, 1000);
        sim.run(5);
        let total: usize = sim.plates.iter().map(|p| p.point_count()).sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn locate_finds_containing_triangle() {
        use super::super::fibonnaci_spiral::SphericalFibonacci;
        let fib = SphericalFibonacci::new(500);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);

        // Query 100 random points, verify barycentric coords are non-negative.
        let mut state = 77777u64;
        let next = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*s >> 11) as f64 / ((1u64 << 53) as f64)
        };
        let mut last_tri = 0;
        for _ in 0..100 {
            let z = next(&mut state) * 2.0 - 1.0;
            let theta = next(&mut state) * std::f64::consts::TAU;
            let r = (1.0 - z * z).sqrt();
            let p = DVec3::new(r * theta.cos(), r * theta.sin(), z).normalize();

            let (tri, b1, b2, b3) = del.locate(p, &points, last_tri);
            last_tri = tri;
            assert!(b1 >= 0.0 && b2 >= 0.0 && b3 >= 0.0,
                "negative barycentric: b1={b1:.4e} b2={b2:.4e} b3={b3:.4e}");
            assert!(b1 + b2 + b3 > 0.0, "zero total weight");
        }
    }

    #[test]
    fn locate_on_drifted_points() {
        // Simulate plate movement, build Delaunay on drifted points,
        // verify locate still works for all new Fibonacci points.
        let mut sim = setup(1000, 8);
        sim.run(10);

        let old_del = SphericalDelaunay::from_points(&sim.points);
        let fib = SphericalFibonacci::new(1000);
        let new_points = fib.all_points();

        let mut last_tri = 0;
        for &p in &new_points {
            let (tri, b1, b2, b3) = old_del.locate(p, &sim.points, last_tri);
            last_tri = tri;
            assert!(b1 >= -1e-10 && b2 >= -1e-10 && b3 >= -1e-10,
                "point outside triangle: b1={b1:.4e} b2={b2:.4e} b3={b3:.4e}");
        }
    }

    #[test]
    fn resample_crust_type_stable_for_interior_points() {
        // After resample, points deep inside a plate should keep their crust type.
        // Count how many points change crust type — should be a small fraction.
        let mut sim = setup(2000, 12);

        // Record initial crust types per point.
        let mut initial_crust = vec![super::super::plates::CrustType::Oceanic; 2000];
        for plate in &sim.plates {
            for (local, &global) in plate.point_indices.iter().enumerate() {
                initial_crust[global as usize] = plate.crust[local].crust_type;
            }
        }

        // Run and resample.
        sim.run(RESAMPLE_INTERVAL);

        // Count crust type changes.
        let mut changed = 0;
        for plate in &sim.plates {
            for (local, &global) in plate.point_indices.iter().enumerate() {
                if plate.crust[local].crust_type != initial_crust[global as usize] {
                    changed += 1;
                }
            }
        }

        // Less than 20% should change — this includes legitimate physical changes
        // (subduction, erosion, collision) over RESAMPLE_INTERVAL steps plus resample.
        let pct = changed as f64 / 2000.0 * 100.0;
        assert!(pct < 20.0,
            "too many crust type changes after resample: {changed}/2000 ({pct:.1}%)");
    }

    #[test]
    fn multiple_resample_cycles_stable() {
        // Run 3 resample cycles and verify no plate vanishes or fragments excessively.
        let mut sim = setup(1000, 8);
        for _ in 0..3 {
            sim.run(RESAMPLE_INTERVAL);
        }

        let total: usize = sim.plates.iter().map(|p| p.point_count()).sum();
        assert_eq!(total, 1000, "points lost after 3 resample cycles");

        let non_empty = sim.plates.iter().filter(|p| !p.point_indices.is_empty()).count();
        assert!(non_empty >= 5, "too many empty plates after 3 cycles: {non_empty}/8");
    }
}
