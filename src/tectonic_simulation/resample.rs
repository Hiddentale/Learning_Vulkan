use std::io::Write;

use glam::DVec3;

use super::fibonnaci_spiral::SphericalFibonacci;
use super::plate_seed_placement::Adjacency;
use super::plates::{CrustData, CrustType};
use super::simulate::Simulation;
use super::sphere_grid::SphereGrid;
use super::spherical_delaunay_triangulation::SphericalDelaunay;
use super::subduction::{resolve_subduction, SubductionResult};

/// How many steps between resamples.
pub const RESAMPLE_INTERVAL: usize = 20;
/// Barycentric weight sum below which all three vertices are weighted equally.
/// Guards against degenerate triangles where all coordinates are near zero.
const MIN_BARYCENTRIC_WEIGHT: f64 = 1e-30;
/// Nearest-neighbor count used by the ambiguity diagnostic. Each new point
/// queries this many nearest old points and counts how many distinct plates
/// appear among them. K=6 matches the IDW neighborhood used elsewhere.
const AMBIGUITY_K: usize = 6;

/// Resample the simulation onto a fresh Fibonacci grid.
///
/// Plate assignment: each new point inherits the plate of the nearest old
/// point (via the old Delaunay triangulation), preserving plate shapes
/// through resamples instead of re-partitioning from centroids.
///
/// Crust interpolation: barycentric interpolation from the enclosing old
/// triangle for smooth crust data transfer.
pub fn resample(sim: &mut Simulation, point_count: u32) {
    let (global_crust, global_plate) = build_global_lookups(sim);
    let global_delaunay = build_global_delaunay(sim);

    let fib = SphericalFibonacci::new(point_count);
    let new_points = fib.all_points();
    let new_delaunay = SphericalDelaunay::from_points(&new_points);
    let new_adjacency = Adjacency::from_delaunay(new_points.len(), &new_delaunay);

    log_plate_ambiguity(sim, &new_points, &global_plate);

    let new_plate_ids = assign_plates_by_triangle_owner(
        &new_points, &sim.points, &global_plate, &global_crust, &global_delaunay,
    );
    // let new_plate_ids = assign_plates_by_nearest(
    //     &new_points, &sim.points, &global_plate, &global_delaunay,
    // );
    let (new_plate_points, new_plate_crust) = interpolate_crust(
        sim, &new_points, &new_plate_ids, &global_delaunay, &global_crust,
    );

    for (plate_idx, plate) in sim.plates.iter_mut().enumerate() {
        plate.point_indices = new_plate_points[plate_idx].clone();
        plate.crust = new_plate_crust[plate_idx].clone();
    }

    sim.adjacency = new_adjacency;
    sim.points = new_points;
}

/// For each new Fibonacci point, count distinct plate IDs among its
/// `AMBIGUITY_K` nearest old points. Histogram the counts and log to the
/// sim's diagnostic file. A histogram dominated by "1 distinct plate" means
/// resample assignment is well-defined; mass at "2+" marks overlap zones
/// where nearest-neighbor is noisy.
fn log_plate_ambiguity(sim: &mut Simulation, new_points: &[DVec3], global_plate: &[u32]) {
    let diag = match sim.diagnostics.as_mut() {
        Some(d) => d,
        None => return,
    };

    let grid = SphereGrid::build(&sim.points);
    let mut histogram = [0usize; AMBIGUITY_K + 1];
    let mut worst_pos = DVec3::ZERO;
    let mut worst_count = 0usize;

    for &p in new_points {
        let neighbors = grid.find_nearest_k(p, &sim.points, AMBIGUITY_K);
        let mut seen = [u32::MAX; AMBIGUITY_K];
        let mut distinct = 0usize;
        for (idx, _) in &neighbors {
            let plate = global_plate[*idx as usize];
            if !seen[..distinct].contains(&plate) {
                seen[distinct] = plate;
                distinct += 1;
            }
        }
        histogram[distinct.min(AMBIGUITY_K)] += 1;
        if distinct > worst_count {
            worst_count = distinct;
            worst_pos = p;
        }
    }

    let total = new_points.len() as f64;
    let _ = writeln!(
        diag.file,
        "=== RESAMPLE AMBIGUITY step={} t={:.0} Myr new_points={} K={} ===",
        sim.step_count, sim.time, new_points.len(), AMBIGUITY_K,
    );
    for (k, count) in histogram.iter().enumerate().skip(1) {
        if *count == 0 {
            continue;
        }
        let _ = writeln!(
            diag.file,
            "  distinct_plates_in_{}nn={}: {} ({:.2}%)",
            AMBIGUITY_K, k, count, (*count as f64 / total) * 100.0,
        );
    }
    let _ = writeln!(
        diag.file,
        "  worst: {} distinct plates at pos=({:.3},{:.3},{:.3})",
        worst_count, worst_pos.x, worst_pos.y, worst_pos.z,
    );
    let _ = diag.file.flush();
}

/// Map global point index -> (CrustData, plate_index).
fn build_global_lookups(sim: &Simulation) -> (Vec<CrustData>, Vec<u32>) {
    let mut crust = vec![CrustData::oceanic(7.0, -4.0, 0.0, DVec3::X); sim.points.len()];
    let mut plate = vec![0u32; sim.points.len()];
    for (plate_idx, p) in sim.plates.iter().enumerate() {
        for (local, &global) in p.point_indices.iter().enumerate() {
            crust[global as usize] = p.crust[local].clone();
            plate[global as usize] = plate_idx as u32;
        }
    }
    (crust, plate)
}

/// Build one Delaunay from all old simulation points.
fn build_global_delaunay(sim: &Simulation) -> Option<SphericalDelaunay> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        SphericalDelaunay::from_points(&sim.points)
    })).ok()
}

/// Assign each new point the plate that owns the Delaunay triangle it lands in.
///
/// The owner of each triangle is the majority plate among its three vertices.
/// For triple junctions (three distinct plates on one triangle) we resolve
/// pairwise using `subduction::resolve_subduction`: whichever plate would not
/// subduct wins. This mirrors the paper's "partition the triangulation"
/// language and is stable against the per-vertex flipping that produced
/// boundary fragmentation with the nearest-vertex approach.
fn assign_plates_by_triangle_owner(
    new_points: &[DVec3],
    old_points: &[DVec3],
    global_plate: &[u32],
    global_crust: &[CrustData],
    global_delaunay: &Option<SphericalDelaunay>,
) -> Vec<u32> {
    let del = match global_delaunay {
        Some(d) => d,
        None => {
            return new_points
                .iter()
                .map(|&p| {
                    let nearest = old_points
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| p.dot(**a).partial_cmp(&p.dot(**b)).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    global_plate[nearest]
                })
                .collect();
        }
    };

    let triangle_owner = compute_triangle_ownership(del, global_plate, global_crust);

    let mut plate_ids = vec![0u32; new_points.len()];
    let mut last_tri = 0usize;
    for (i, &p) in new_points.iter().enumerate() {
        let (tri, _, _, _) = del.locate(p, old_points, last_tri);
        last_tri = tri;
        plate_ids[i] = triangle_owner[tri];
    }
    plate_ids
}

/// Compute one plate owner per Delaunay triangle.
pub(super) fn compute_triangle_ownership(
    del: &SphericalDelaunay,
    global_plate: &[u32],
    global_crust: &[CrustData],
) -> Vec<u32> {
    let n_tri = del.triangles.len() / 3;
    let mut owner = Vec::with_capacity(n_tri);
    for t in 0..n_tri {
        let base = t * 3;
        let v = [
            del.triangles[base] as usize,
            del.triangles[base + 1] as usize,
            del.triangles[base + 2] as usize,
        ];
        let p = [
            global_plate[v[0]],
            global_plate[v[1]],
            global_plate[v[2]],
        ];

        let label = if p[0] == p[1] && p[1] == p[2] {
            p[0]
        } else if p[0] == p[1] || p[0] == p[2] {
            p[0]
        } else if p[1] == p[2] {
            p[1]
        } else {
            resolve_triple_junction(v, p, global_crust)
        };
        owner.push(label);
    }
    owner
}

/// Three distinct plates meet at this triangle. Apply `resolve_subduction`
/// pairwise to pick the plate that does not subduct. Deterministic in the
/// rare continental-continental case: the lower plate id wins.
fn resolve_triple_junction(
    verts: [usize; 3],
    plates: [u32; 3],
    global_crust: &[CrustData],
) -> u32 {
    let winner_01 = pairwise_winner(plates[0], plates[1], verts[0], verts[1], global_crust);
    let winner_2 = match winner_01 {
        Some(p) => {
            let v_winner = if p == plates[0] { verts[0] } else { verts[1] };
            pairwise_winner(p, plates[2], v_winner, verts[2], global_crust)
        }
        None => None,
    };
    winner_2.or(winner_01).unwrap_or(plates[0].min(plates[1]).min(plates[2]))
}

/// Winner in a two-plate interaction: the non-subducting plate. `None`
/// means continental collision (no subduction) — caller picks a fallback.
fn pairwise_winner(
    plate_i: u32,
    plate_j: u32,
    vert_i: usize,
    vert_j: usize,
    global_crust: &[CrustData],
) -> Option<u32> {
    let ci = &global_crust[vert_i];
    let cj = &global_crust[vert_j];
    match resolve_subduction(plate_i, plate_j, ci.crust_type, ci.age, cj.crust_type, cj.age) {
        SubductionResult::PlateSubducts(s) => {
            Some(if s == plate_i { plate_j } else { plate_i })
        }
        SubductionResult::ContinentalCollision => {
            // Both are continental; neither subducts. Pick lower-id continental
            // deterministically so the same triangle always labels the same way.
            let lower = plate_i.min(plate_j);
            if ci.crust_type == CrustType::Continental && cj.crust_type == CrustType::Continental {
                Some(lower)
            } else {
                None
            }
        }
    }
}

/// Assign each new point to the plate of the nearest old point.
/// Uses the old Delaunay to locate the enclosing triangle, then picks the
/// nearest vertex — preserving plate shapes through resamples.
#[allow(dead_code)]
fn assign_plates_by_nearest(
    new_points: &[DVec3],
    old_points: &[DVec3],
    global_plate: &[u32],
    global_delaunay: &Option<SphericalDelaunay>,
) -> Vec<u32> {
    let mut plate_ids = vec![0u32; new_points.len()];
    let mut last_tri = 0usize;

    for (i, &p) in new_points.iter().enumerate() {
        plate_ids[i] = if let Some(del) = global_delaunay {
            let (tri, b1, b2, b3) = del.locate(p, old_points, last_tri);
            last_tri = tri;
            let base = tri * 3;
            let vi = [
                del.triangles[base] as usize,
                del.triangles[base + 1] as usize,
                del.triangles[base + 2] as usize,
            ];
            // Pick the vertex with the largest barycentric coordinate (nearest).
            let bary = [b1, b2, b3];
            let best = if bary[0] >= bary[1] && bary[0] >= bary[2] { 0 }
                       else if bary[1] >= bary[2] { 1 }
                       else { 2 };
            global_plate[vi[best]]
        } else {
            // Fallback: brute-force nearest.
            let nearest = old_points.iter().enumerate()
                .max_by(|(_, a), (_, b)| p.dot(**a).partial_cmp(&p.dot(**b)).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            global_plate[nearest]
        };
    }

    plate_ids
}

fn interpolate_crust(
    sim: &Simulation,
    new_points: &[DVec3],
    new_plate_ids: &[u32],
    global_delaunay: &Option<SphericalDelaunay>,
    global_crust: &[CrustData],
) -> (Vec<Vec<u32>>, Vec<Vec<CrustData>>) {
    let plate_count = sim.plates.len();
    let mut new_plate_points: Vec<Vec<u32>> = vec![Vec::new(); plate_count];
    let mut new_plate_crust: Vec<Vec<CrustData>> = vec![Vec::new(); plate_count];
    let mut last_tri = 0usize;

    for (new_idx, &new_p) in new_points.iter().enumerate() {
        let owner = new_plate_ids[new_idx] as usize;

        let crust = if let Some(del) = global_delaunay {
            interpolate_from_global(
                del, new_p, &sim.points, global_crust, &mut last_tri,
            )
        } else {
            nearest_crust_global(new_p, &sim.points, global_crust)
        };

        new_plate_points[owner].push(new_idx as u32);
        new_plate_crust[owner].push(crust);
    }

    (new_plate_points, new_plate_crust)
}

/// Interpolate crust data from a global Delaunay triangle.
/// All three vertices are spatially nearby — no back-side triangle problem.
/// Cross-plate blending at boundaries creates organic coastline irregularity.
fn interpolate_from_global(
    del: &SphericalDelaunay,
    point: DVec3,
    old_points: &[DVec3],
    global_crust: &[CrustData],
    last_tri: &mut usize,
) -> CrustData {
    let (tri, b1, b2, b3) = del.locate(point, old_points, *last_tri);
    *last_tri = tri;

    let base = tri * 3;
    let gi = [
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

    let crusts: [&CrustData; 3] = std::array::from_fn(|i| &global_crust[gi[i]]);

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
        subducted_distance: 0.0,
    }
}

fn nearest_crust_global(point: DVec3, old_points: &[DVec3], global_crust: &[CrustData]) -> CrustData {
    let nearest = old_points.iter().enumerate()
        .max_by(|(_, a), (_, b)| point.dot(**a).partial_cmp(&point.dot(**b)).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);
    global_crust[nearest].clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::plate_initializer::{initialize_plates, InitParams};
    use super::super::plate_seed_placement::assign_plates;

    fn setup(point_count: u32, plate_count: u32) -> Simulation {
        let fib = SphericalFibonacci::new(point_count);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, plate_count, 42);
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
        let after_resample: usize = sim.plates.iter().map(|p| p.point_count()).sum();
        assert_eq!(after_resample, 1000, "resample should restore full point count");
        sim.run(5);
        let total: usize = sim.plates.iter().map(|p| p.point_count()).sum();
        // Subduction consumption can remove vertices between resamples.
        assert!(total > 0 && total <= 1000, "total out of expected range: {total}");
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
