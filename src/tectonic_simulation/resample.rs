use std::io::Write;
use std::time::Instant;

use glam::DVec3;

use super::fibonnaci_spiral::SphericalFibonacci;
use super::plate_seed_placement::{Adjacency, flood_fill_from_seeds_warped};
use super::plates::{CrustData, Plate};
use super::simulate::Simulation;
use super::spherical_delaunay_triangulation::SphericalDelaunay;

/// How many steps between resamples.
pub const RESAMPLE_INTERVAL: usize = 20;

/// Resample the simulation onto a fresh Fibonacci grid.
///
/// Plate assignment: noise-warped flood fill from plate centroids (clean, contiguous,
/// organic boundaries). Crust interpolation: per-plate Delaunay on drifted points
/// (all triangle vertices same-plate, no cross-plate blending).
pub fn resample(sim: &mut Simulation, point_count: u32) {
    let mut log = std::fs::OpenOptions::new()
        .create(true).append(true)
        .open("sim_profile.log").ok();
    let resample_start = Instant::now();

    let old_points = &sim.points;
    let plate_count = sim.plates.len();

    // Build per-plate Delaunay triangulations for crust interpolation.
    let t0 = Instant::now();
    let plate_points: Vec<Vec<DVec3>> = sim.plates.iter().map(|plate| {
        plate.point_indices.iter().map(|&gi| old_points[gi as usize]).collect()
    }).collect();
    let plate_delaunays: Vec<Option<SphericalDelaunay>> = plate_points.iter().map(|pts| {
        if pts.len() >= 10 {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                SphericalDelaunay::from_points(pts)
            })).ok()
        } else {
            None
        }
    }).collect();
    let plate_del_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Fresh Fibonacci grid.
    let fib = SphericalFibonacci::new(point_count);
    let new_points = fib.all_points();
    let t0 = Instant::now();
    let new_delaunay = SphericalDelaunay::from_points(&new_points);
    let new_del_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let t0 = Instant::now();
    let new_adjacency = Adjacency::from_delaunay(new_points.len(), &new_delaunay);
    let adjacency_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Step 1: Plate assignment via noise-warped flood fill from centroids.
    let t0 = Instant::now();
    let mut centroids: Vec<DVec3> = vec![DVec3::ZERO; plate_count];
    for (plate_idx, plate) in sim.plates.iter().enumerate() {
        for &global in &plate.point_indices {
            centroids[plate_idx] += old_points[global as usize];
        }
        if !plate.point_indices.is_empty() {
            centroids[plate_idx] = centroids[plate_idx].normalize();
        }
    }
    let seeds: Vec<u32> = centroids.iter()
        .map(|&c| fib.nearest_index(c))
        .collect();
    let new_plate_ids = flood_fill_from_seeds_warped(&new_points, &new_adjacency, &seeds, 42);
    let flood_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Step 2: Crust interpolation via per-plate Delaunay.
    let t0 = Instant::now();
    let mut new_plate_points: Vec<Vec<u32>> = vec![Vec::new(); plate_count];
    let mut new_plate_crust: Vec<Vec<CrustData>> = vec![Vec::new(); plate_count];
    let mut last_tris = vec![0usize; plate_count];

    for (new_idx, &new_p) in new_points.iter().enumerate() {
        let owner = new_plate_ids[new_idx] as usize;

        let crust = if let Some(del) = &plate_delaunays[owner] {
            let (tri, b1, b2, b3) = del.locate(new_p, &plate_points[owner], last_tris[owner]);
            last_tris[owner] = tri;

            let base = tri * 3;
            let local_vi = [
                del.triangles[base] as usize,
                del.triangles[base + 1] as usize,
                del.triangles[base + 2] as usize,
            ];

            let total_w = b1.max(0.0) + b2.max(0.0) + b3.max(0.0);
            let w = if total_w > 1e-30 {
                [b1.max(0.0) / total_w, b2.max(0.0) / total_w, b3.max(0.0) / total_w]
            } else {
                [1.0 / 3.0; 3]
            };

            let dom = if w[0] >= w[1] && w[0] >= w[2] { 0 }
                      else if w[1] >= w[2] { 1 }
                      else { 2 };

            let crusts: [&CrustData; 3] = std::array::from_fn(|i| {
                &sim.plates[owner].crust[local_vi[i]]
            });

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
        } else {
            // Small plate: nearest point's crust.
            let nearest_local = plate_points[owner].iter().enumerate()
                .max_by(|(_, a), (_, b)| new_p.dot(**a).partial_cmp(&new_p.dot(**b)).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            sim.plates[owner].crust[nearest_local].clone()
        };

        new_plate_points[owner].push(new_idx as u32);
        new_plate_crust[owner].push(crust);
    }
    let interp_ms = t0.elapsed().as_secs_f64() * 1000.0;

    for (plate_idx, plate) in sim.plates.iter_mut().enumerate() {
        plate.point_indices = new_plate_points[plate_idx].clone();
        plate.crust = new_plate_crust[plate_idx].clone();
    }

    sim.adjacency = new_adjacency;
    sim.points = new_points;

    let total_ms = resample_start.elapsed().as_secs_f64() * 1000.0;
    if let Some(ref mut f) = log {
        let _ = writeln!(f,
            "  RESAMPLE: total={:.0}ms | plate_del={:.0} new_del={:.0} adj={:.0} flood={:.0} interp={:.0}",
            total_ms, plate_del_ms, new_del_ms, adjacency_ms, flood_ms, interp_ms
        );
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
