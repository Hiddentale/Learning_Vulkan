use glam::DVec3;

use super::fibonnaci_spiral::SphericalFibonacci;
use super::plate_seed_placement::{Adjacency, flood_fill_from_seeds_warped};
use super::plates::{CrustData, Plate};
use super::simulate::Simulation;
use super::spherical_delaunay_triangulation::SphericalDelaunay;

/// How many steps between resamples.
pub const RESAMPLE_INTERVAL: usize = 20;
/// Minimum points a plate needs for Delaunay triangulation.
/// Below this, nearest-point fallback is used instead.
const MIN_DELAUNAY_POINTS: usize = 10;
/// Barycentric weight sum below which all three vertices are weighted equally.
/// Guards against degenerate triangles where all coordinates are near zero.
const MIN_BARYCENTRIC_WEIGHT: f64 = 1e-30;
/// Noise seed for warped flood fill during plate reassignment.
const FLOOD_FILL_SEED: u32 = 42;

/// Resample the simulation onto a fresh Fibonacci grid.
///
/// Plate assignment: noise-warped flood fill from plate centroids (clean, contiguous,
/// organic boundaries). Crust interpolation: per-plate Delaunay on drifted points
/// (all triangle vertices same-plate, no cross-plate blending).
pub fn resample(sim: &mut Simulation, point_count: u32) {
    let plate_delaunays = build_plate_delaunays(sim);
    let plate_points = collect_plate_points(sim);

    let fib = SphericalFibonacci::new(point_count);
    let new_points = fib.all_points();
    let new_delaunay = SphericalDelaunay::from_points(&new_points);
    let new_adjacency = Adjacency::from_delaunay(new_points.len(), &new_delaunay);

    let new_plate_ids = assign_plates_by_flood_fill(sim, &fib, &new_points, &new_adjacency);
    let (new_plate_points, new_plate_crust) = interpolate_crust(
        sim, &new_points, &new_plate_ids, &plate_delaunays, &plate_points,
    );

    for (plate_idx, plate) in sim.plates.iter_mut().enumerate() {
        plate.point_indices = new_plate_points[plate_idx].clone();
        plate.crust = new_plate_crust[plate_idx].clone();
    }

    sim.adjacency = new_adjacency;
    sim.points = new_points;
}

fn collect_plate_points(sim: &Simulation) -> Vec<Vec<DVec3>> {
    sim.plates.iter().map(|plate| {
        plate.point_indices.iter().map(|&gi| sim.points[gi as usize]).collect()
    }).collect()
}

fn build_plate_delaunays(sim: &Simulation) -> Vec<Option<SphericalDelaunay>> {
    let plate_points = collect_plate_points(sim);
    plate_points.iter().map(|pts| {
        if pts.len() >= MIN_DELAUNAY_POINTS {
            // Drifted plates can have degenerate point distributions (near-coplanar,
            // clustered) that cause the incremental Delaunay builder to fail.
            // Fall back to nearest-point interpolation for those plates.
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                SphericalDelaunay::from_points(pts)
            })).ok()
        } else {
            None
        }
    }).collect()
}

fn assign_plates_by_flood_fill(
    sim: &Simulation,
    fib: &SphericalFibonacci,
    new_points: &[DVec3],
    new_adjacency: &Adjacency,
) -> Vec<u32> {
    let centroids: Vec<DVec3> = sim.plates.iter().map(|plate| {
        let sum: DVec3 = plate.point_indices.iter().map(|&gi| sim.points[gi as usize]).sum();
        if plate.point_indices.is_empty() { DVec3::ZERO } else { sum.normalize() }
    }).collect();

    let seeds: Vec<u32> = centroids.iter()
        .map(|&c| fib.nearest_index(c))
        .collect();

    flood_fill_from_seeds_warped(new_points, new_adjacency, &seeds, FLOOD_FILL_SEED)
}

fn interpolate_crust(
    sim: &Simulation,
    new_points: &[DVec3],
    new_plate_ids: &[u32],
    plate_delaunays: &[Option<SphericalDelaunay>],
    plate_points: &[Vec<DVec3>],
) -> (Vec<Vec<u32>>, Vec<Vec<CrustData>>) {
    let plate_count = sim.plates.len();
    let mut new_plate_points: Vec<Vec<u32>> = vec![Vec::new(); plate_count];
    let mut new_plate_crust: Vec<Vec<CrustData>> = vec![Vec::new(); plate_count];
    let mut last_tris = vec![0usize; plate_count];

    for (new_idx, &new_p) in new_points.iter().enumerate() {
        let owner = new_plate_ids[new_idx] as usize;

        let crust = if let Some(del) = &plate_delaunays[owner] {
            interpolate_from_delaunay(
                del, new_p, &plate_points[owner],
                &sim.plates[owner].crust, &mut last_tris[owner],
            )
        } else {
            nearest_crust(new_p, &plate_points[owner], &sim.plates[owner].crust)
        };

        new_plate_points[owner].push(new_idx as u32);
        new_plate_crust[owner].push(crust);
    }

    (new_plate_points, new_plate_crust)
}

fn interpolate_from_delaunay(
    del: &SphericalDelaunay,
    point: DVec3,
    plate_points: &[DVec3],
    plate_crust: &[CrustData],
    last_tri: &mut usize,
) -> CrustData {
    let (tri, b1, b2, b3) = del.locate(point, plate_points, *last_tri);
    *last_tri = tri;

    let base = tri * 3;
    let local_vi = [
        del.triangles[base] as usize,
        del.triangles[base + 1] as usize,
        del.triangles[base + 2] as usize,
    ];

    let total_w = b1.max(0.0) + b2.max(0.0) + b3.max(0.0);
    let w = if total_w > MIN_BARYCENTRIC_WEIGHT {
        [b1.max(0.0) / total_w, b2.max(0.0) / total_w, b3.max(0.0) / total_w]
    } else {
        [1.0 / 3.0; 3]
    };

    let dom = if w[0] >= w[1] && w[0] >= w[2] { 0 }
              else if w[1] >= w[2] { 1 }
              else { 2 };

    let crusts: [&CrustData; 3] = std::array::from_fn(|i| &plate_crust[local_vi[i]]);

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
        crust_type: crusts[dom].crust_type,
        thickness, elevation, age,
        local_direction: direction.normalize_or_zero(),
        orogeny_type: crusts[dom].orogeny_type,
    }
}

fn nearest_crust(point: DVec3, plate_points: &[DVec3], plate_crust: &[CrustData]) -> CrustData {
    let nearest_local = plate_points.iter().enumerate()
        .max_by(|(_, a), (_, b)| point.dot(**a).partial_cmp(&point.dot(**b)).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    plate_crust[nearest_local].clone()
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
        let mut sim = setup(2000, 12);

        let mut initial_crust = vec![super::super::plates::CrustType::Oceanic; 2000];
        for plate in &sim.plates {
            for (local, &global) in plate.point_indices.iter().enumerate() {
                initial_crust[global as usize] = plate.crust[local].crust_type;
            }
        }

        sim.run(RESAMPLE_INTERVAL);

        let mut changed = 0;
        for plate in &sim.plates {
            for (local, &global) in plate.point_indices.iter().enumerate() {
                if plate.crust[local].crust_type != initial_crust[global as usize] {
                    changed += 1;
                }
            }
        }

        let pct = changed as f64 / 2000.0 * 100.0;
        assert!(pct < 20.0,
            "too many crust type changes after resample: {changed}/2000 ({pct:.1}%)");
    }

    #[test]
    fn multiple_resample_cycles_stable() {
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
