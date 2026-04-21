use std::collections::HashMap;

use glam::DVec3;

use super::fibonnaci_spiral::SphericalFibonacci;
use super::plate_seed_placement::Adjacency;
use super::plates::{
    BoundingCap, CrustData, CrustType, Plate, SampleCache, TriangleNeighbors, WalkResult,
};
use super::simulate::{Simulation, NO_PLATE};
use super::spherical_delaunay_triangulation::SphericalDelaunay;

/// Resample interval bounds (steps). Faster plates resample sooner.
const MIN_RESAMPLE_INTERVAL: usize = 10;
const MAX_RESAMPLE_INTERVAL: usize = 60;
/// Angular speed thresholds for the interval ramp (rad/Myr).
const FAST_SPEED: f64 = 0.02;
const SLOW_SPEED: f64 = 0.005;

/// Adaptive resample interval: fast plates → 10 steps, slow → 60.
pub fn resample_interval(plates: &[Plate]) -> usize {
    let max_speed = plates
        .iter()
        .map(|p| p.angular_speed.abs())
        .fold(0.0_f64, f64::max);

    if max_speed >= FAST_SPEED {
        return MIN_RESAMPLE_INTERVAL;
    }
    if max_speed <= SLOW_SPEED {
        return MAX_RESAMPLE_INTERVAL;
    }

    let t = (max_speed - SLOW_SPEED) / (FAST_SPEED - SLOW_SPEED);
    let interval = MAX_RESAMPLE_INTERVAL as f64
        - t * (MAX_RESAMPLE_INTERVAL - MIN_RESAMPLE_INTERVAL) as f64;
    interval.round() as usize
}
/// Maximum walk steps when locating a new point in an old plate mesh.
const MAX_WALK_STEPS: u32 = 256;
const NO_TRIANGLE: u32 = u32::MAX;
/// K-nearest neighbors for fallback crust type majority vote.
const FALLBACK_VOTE_K: usize = 6;

struct PointAssignment {
    plate: u32,
    crust: CrustData,
}

/// Resample the simulation onto a fresh Fibonacci grid.
///
/// 1. Generate new Fibonacci sample points + global Delaunay.
/// 2. Locate each new point in the old plate meshes (walk-based).
/// 3. Barycentric-interpolate crust from the enclosing old triangle.
/// 4. Partition the new Delaunay into per-plate sub-meshes.
/// 5. Store reference points in each plate's frame (R_k⁻¹ · world).
/// 6. Build per-plate triangle adjacency and sample warm-start cache.
pub fn resample(sim: &mut Simulation) {
    let fib = SphericalFibonacci::new(sim.target_sample_count);
    let new_points = fib.all_points();
    let del = SphericalDelaunay::from_points(&new_points);
    let sample_adj = Adjacency::from_delaunay(new_points.len(), &del);

    let assignments = assign_from_old_plates(&sim.plates, &new_points);

    // Log assignment gaps.
    let gap_count = assignments.iter().filter(|a| a.plate == NO_PLATE).count();
    if let Some(diag) = sim.diagnostics.as_mut() {
        if diag.active(sim.step_count) {
            use std::io::Write;
            let _ = writeln!(
                diag.file,
                "=== RESAMPLE step={} t={:.0} Myr: {}/{} assigned, {} gap, {} plates ===",
                sim.step_count, sim.time,
                assignments.len() - gap_count, assignments.len(),
                gap_count, sim.plates.len(),
            );
            for (k, plate) in sim.plates.iter().enumerate() {
                let assigned_to_k = assignments.iter().filter(|a| a.plate == k as u32).count();
                let _ = writeln!(
                    diag.file,
                    "  plate[{}]: {} pts assigned (was {} verts, {} tris)",
                    k, assigned_to_k, plate.point_count(), plate.triangle_count(),
                );
            }
        }
    }

    let tri_owner = assign_triangle_ownership(&del, &assignments);
    let (new_plates, global_indices) =
        build_plate_meshes(&new_points, &del, &tri_owner, &assignments, &sim.plates);
    let cache = build_sample_cache(&new_plates, &global_indices, new_points.len());

    // Log post-resample plate sizes.
    if let Some(diag) = sim.diagnostics.as_mut() {
        if diag.active(sim.step_count) {
            use std::io::Write;
            for (k, plate) in new_plates.iter().enumerate() {
                let _ = writeln!(
                    diag.file,
                    "  new plate[{}]: {} verts, {} tris",
                    k, plate.point_count(), plate.triangle_count(),
                );
            }
            let cached = cache.iter().filter(|c| c.plate != NO_PLATE).count();
            let _ = writeln!(diag.file, "  cache: {}/{} assigned", cached, cache.len());
        }
    }

    sim.sample_points = new_points;
    sim.sample_cache = cache;
    sim.sample_adjacency = sample_adj;
    sim.plates = new_plates;
}

// ── Step 1-2: locate new points in old plate meshes ───────────────────

/// For each new Fibonacci point, walk the old plate meshes to find which
/// plate covers it and barycentric-interpolate crust. Uses a per-plate
/// "last successful triangle" as a spatial locality hint for the walk.
fn assign_from_old_plates(plates: &[Plate], new_points: &[DVec3]) -> Vec<PointAssignment> {
    let mut last_tri = vec![0u32; plates.len()];

    // Build a spatial index of all plate vertices for fast nearest-vertex fallback.
    let (world_verts, vert_plate, vert_local) = build_vertex_index(plates);
    let grid = super::sphere_grid::SphereGrid::build(&world_verts);

    new_points
        .iter()
        .map(|&p| {
            // Try walk-based assignment (barycentric interpolation).
            for (k, plate) in plates.iter().enumerate() {
                if !plate.world_point_in_cap(p) {
                    continue;
                }
                let q = plate.to_reference(p);
                if let WalkResult::Found { triangle, bary } =
                    plate.walk_to_point(q, last_tri[k], MAX_WALK_STEPS)
                {
                    last_tri[k] = triangle;
                    let [vi, vj, vk] = plate.triangles[triangle as usize];
                    let crust = CrustData::barycentric_blend(
                        &plate.crust[vi as usize],
                        &plate.crust[vj as usize],
                        &plate.crust[vk as usize],
                        bary,
                    );
                    return PointAssignment {
                        plate: k as u32,
                        crust,
                    };
                }
            }

            // Walk failed — assign to nearest plate vertex. Use K-nearest
            // majority vote for crust type to prevent speckle noise at
            // continent-ocean boundaries where the nearest single vertex
            // might be across a plate boundary.
            let neighbors = grid.find_nearest_k(p, &world_verts, FALLBACK_VOTE_K);
            let gi = neighbors[0].0 as usize;
            let assigned_plate = vert_plate[gi];

            // Continuous fields from the nearest vertex on the assigned plate.
            let nearest_crust = &plates[assigned_plate as usize].crust[vert_local[gi]];

            // Majority vote on crust type among K neighbors on the same plate.
            let same_plate_cont = neighbors
                .iter()
                .filter(|(idx, _)| vert_plate[*idx as usize] == assigned_plate)
                .filter(|(idx, _)| {
                    plates[assigned_plate as usize].crust[vert_local[*idx as usize]].crust_type
                        == CrustType::Continental
                })
                .count();
            let same_plate_total = neighbors
                .iter()
                .filter(|(idx, _)| vert_plate[*idx as usize] == assigned_plate)
                .count()
                .max(1);
            let voted_type = if same_plate_cont * 2 >= same_plate_total {
                CrustType::Continental
            } else {
                CrustType::Oceanic
            };

            let mut crust = nearest_crust.clone();
            crust.crust_type = voted_type;
            if voted_type == CrustType::Oceanic {
                crust.orogeny_type = None;
            }
            PointAssignment {
                plate: assigned_plate,
                crust,
            }
        })
        .collect()
}

/// Build a flat array of all plate vertices in world space, with
/// parallel arrays mapping back to (plate_index, local_vertex_index).
/// Used by SphereGrid for O(1) nearest-vertex lookups during fallback.
fn build_vertex_index(plates: &[Plate]) -> (Vec<DVec3>, Vec<u32>, Vec<usize>) {
    let total: usize = plates.iter().map(|p| p.point_count()).sum();
    let mut world_verts = Vec::with_capacity(total);
    let mut vert_plate = Vec::with_capacity(total);
    let mut vert_local = Vec::with_capacity(total);

    for (k, plate) in plates.iter().enumerate() {
        for (i, &ref_p) in plate.reference_points.iter().enumerate() {
            world_verts.push(plate.to_world(ref_p));
            vert_plate.push(k as u32);
            vert_local.push(i);
        }
    }

    (world_verts, vert_plate, vert_local)
}

// ── Step 3: partition the new Delaunay into plates ────────────────────

/// Assign each triangle in the new Delaunay to a plate via majority vote
/// on its vertex assignments. Triple junctions resolved by crust density.
fn assign_triangle_ownership(
    del: &SphericalDelaunay,
    assignments: &[PointAssignment],
) -> Vec<u32> {
    let n_tri = del.triangles.len() / 3;
    (0..n_tri)
        .map(|t| {
            let base = t * 3;
            let p = [
                assignments[del.triangles[base] as usize].plate,
                assignments[del.triangles[base + 1] as usize].plate,
                assignments[del.triangles[base + 2] as usize].plate,
            ];

            if p[0] == p[1] && p[1] == p[2] {
                p[0]
            } else if p[0] == p[1] || p[0] == p[2] {
                p[0]
            } else if p[1] == p[2] {
                p[1]
            } else {
                resolve_triple_junction(base, del, assignments, p)
            }
        })
        .collect()
}

/// Three distinct plates meet: pick the one with lowest-density crust
/// (continental over oceanic, younger oceanic over older).
fn resolve_triple_junction(
    base: usize,
    del: &SphericalDelaunay,
    assignments: &[PointAssignment],
    plates: [u32; 3],
) -> u32 {
    let crusts: [&CrustData; 3] = std::array::from_fn(|i| {
        &assignments[del.triangles[base + i] as usize].crust
    });
    let mut best = 0;
    for i in 1..3 {
        let wins = match (crusts[i].crust_type, crusts[best].crust_type) {
            (CrustType::Continental, CrustType::Oceanic) => true,
            (CrustType::Oceanic, CrustType::Continental) => false,
            _ => crusts[i].age < crusts[best].age,
        };
        if wins {
            best = i;
        }
    }
    plates[best]
}

// ── Step 4-5: build per-plate sub-meshes ──────────────────────────────

/// Build per-plate reference meshes from the partitioned global Delaunay.
/// Returns (plates, global_indices) where `global_indices[k][local]` is the
/// Fibonacci index of plate k's local vertex — needed for sample cache.
fn build_plate_meshes(
    new_points: &[DVec3],
    del: &SphericalDelaunay,
    tri_owner: &[u32],
    assignments: &[PointAssignment],
    old_plates: &[Plate],
) -> (Vec<Plate>, Vec<Vec<u32>>) {
    let plate_count = old_plates.len();
    let n_tri = del.triangles.len() / 3;

    let mut plate_tris: Vec<Vec<usize>> = vec![Vec::new(); plate_count];
    for t in 0..n_tri {
        let owner = tri_owner[t] as usize;
        if owner < plate_count {
            plate_tris[owner].push(t);
        }
    }

    let mut plates = Vec::with_capacity(plate_count);
    let mut all_global_indices = Vec::with_capacity(plate_count);

    for (k, old_plate) in old_plates.iter().enumerate() {
        let (local_verts, global_to_local) = collect_plate_vertices(&plate_tris[k], del);

        let inv_rot = old_plate.rotation.inverse();
        let reference_points: Vec<DVec3> = local_verts
            .iter()
            .map(|&gi| inv_rot.mul_vec3(new_points[gi as usize]).normalize())
            .collect();

        let crust: Vec<CrustData> = local_verts
            .iter()
            .map(|&gi| assignments[gi as usize].crust.clone())
            .collect();

        let triangles: Vec<[u32; 3]> = plate_tris[k]
            .iter()
            .map(|&t| {
                let base = t * 3;
                [
                    global_to_local[&del.triangles[base]],
                    global_to_local[&del.triangles[base + 1]],
                    global_to_local[&del.triangles[base + 2]],
                ]
            })
            .collect();

        let adjacency = build_triangle_adjacency(&plate_tris[k], del);

        let mut plate = Plate {
            reference_points,
            crust,
            triangles,
            adjacency,
            bounding_cap: BoundingCap {
                center: DVec3::Y,
                cos_radius: -1.0,
            },
            rotation: old_plate.rotation,
            rotation_axis: old_plate.rotation_axis,
            angular_speed: old_plate.angular_speed,
        };
        plate.recompute_bounding_cap();

        plates.push(plate);
        all_global_indices.push(local_verts);
    }

    (plates, all_global_indices)
}

/// Collect unique vertices from this plate's triangles, returning
/// (global_indices, global_to_local_map).
fn collect_plate_vertices(
    plate_tris: &[usize],
    del: &SphericalDelaunay,
) -> (Vec<u32>, HashMap<u32, u32>) {
    let mut global_to_local: HashMap<u32, u32> = HashMap::new();
    let mut local_verts: Vec<u32> = Vec::new();

    for &t in plate_tris {
        let base = t * 3;
        for i in 0..3 {
            let gi = del.triangles[base + i];
            global_to_local.entry(gi).or_insert_with(|| {
                let idx = local_verts.len() as u32;
                local_verts.push(gi);
                idx
            });
        }
    }

    (local_verts, global_to_local)
}

/// Build per-triangle adjacency from the global Delaunay halfedge structure.
/// Edges whose neighbor triangle belongs to a different plate become `None`
/// (plate boundary), enabling step 3's exit detection.
fn build_triangle_adjacency(
    global_tri_indices: &[usize],
    del: &SphericalDelaunay,
) -> Vec<TriangleNeighbors> {
    let mut global_to_local_tri: HashMap<usize, u32> =
        HashMap::with_capacity(global_tri_indices.len());
    for (local_t, &global_t) in global_tri_indices.iter().enumerate() {
        global_to_local_tri.insert(global_t, local_t as u32);
    }

    global_tri_indices
        .iter()
        .map(|&global_t| {
            let mut neighbors: TriangleNeighbors = [None; 3];
            for edge in 0..3 {
                let he = del.halfedges[global_t * 3 + edge];
                if he != u32::MAX {
                    let neighbor_global = (he / 3) as usize;
                    if let Some(&neighbor_local) = global_to_local_tri.get(&neighbor_global) {
                        neighbors[edge] = Some(neighbor_local);
                    }
                }
            }
            neighbors
        })
        .collect()
}

// ── Step 6: build sample warm-start cache ─────────────────────────────

/// Populate the sample cache so the first step after resample has warm-start
/// data. Each sample vertex sits exactly on a plate mesh vertex, so the
/// barycentric weight is 1.0 at that vertex and 0.0 at the other two.
fn build_sample_cache(
    plates: &[Plate],
    global_indices: &[Vec<u32>],
    sample_count: usize,
) -> Vec<SampleCache> {
    let mut cache = vec![
        SampleCache {
            plate: NO_PLATE,
            triangle: NO_TRIANGLE,
            bary: [0.0; 3],
        };
        sample_count
    ];

    for (k, globals) in global_indices.iter().enumerate() {
        let incident = vertex_incident_triangles(&plates[k].triangles, globals.len());

        for (local, &global) in globals.iter().enumerate() {
            let tri = incident[local];
            if tri == NO_TRIANGLE {
                continue;
            }

            let tri_verts = plates[k].triangles[tri as usize];
            let local_u32 = local as u32;
            let bary = if tri_verts[0] == local_u32 {
                [1.0, 0.0, 0.0]
            } else if tri_verts[1] == local_u32 {
                [0.0, 1.0, 0.0]
            } else {
                [0.0, 0.0, 1.0]
            };

            cache[global as usize] = SampleCache {
                plate: k as u32,
                triangle: tri,
                bary,
            };
        }
    }

    cache
}

/// For each local vertex, find one incident triangle (any will do).
fn vertex_incident_triangles(triangles: &[[u32; 3]], vertex_count: usize) -> Vec<u32> {
    let mut incident = vec![NO_TRIANGLE; vertex_count];
    for (t, tri) in triangles.iter().enumerate() {
        for &v in tri {
            if incident[v as usize] == NO_TRIANGLE {
                incident[v as usize] = t as u32;
            }
        }
    }
    incident
}

#[cfg(test)]
mod tests {
    use super::super::fibonnaci_spiral::SphericalFibonacci;
    use super::super::plate_initializer::{initialize_plates, InitParams};
    use super::super::plate_seed_placement::{assign_plates, Adjacency};
    use super::super::plates::CrustType;
    use super::super::simulate::Simulation;
    use super::super::spherical_delaunay_triangulation::SphericalDelaunay;
    use super::*;

    fn make_sim(point_count: u32, plate_count: u32) -> Simulation {
        let fib = SphericalFibonacci::new(point_count);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, plate_count, 42);
        let (plates, cache) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        let adj = Adjacency::from_delaunay(points.len(), &del);
        Simulation::new(points, cache, adj, plates)
    }

    #[test]
    fn resample_preserves_sample_count() {
        let mut sim = make_sim(500, 8);
        sim.run(5);
        resample(&mut sim);
        assert_eq!(sim.sample_points.len(), 500);
        assert_eq!(sim.sample_cache.len(), 500);
    }

    #[test]
    fn resample_preserves_plate_count() {
        let mut sim = make_sim(500, 8);
        let plates_before = sim.plates.len();
        sim.run(5);
        resample(&mut sim);
        assert_eq!(sim.plates.len(), plates_before);
    }

    #[test]
    fn resample_samples_on_unit_sphere() {
        let mut sim = make_sim(500, 8);
        sim.run(5);
        resample(&mut sim);
        for &p in &sim.sample_points {
            assert!((p.length() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn resample_most_samples_assigned() {
        let mut sim = make_sim(500, 8);
        sim.run(5);
        resample(&mut sim);
        let assigned = sim
            .sample_cache
            .iter()
            .filter(|c| c.plate != NO_PLATE)
            .count();
        // Boundary gaps between plates leave ~10-20% unassigned.
        assert!(
            assigned >= 350,
            "only {assigned}/500 assigned after resample"
        );
    }

    #[test]
    fn resample_plates_have_triangles() {
        let mut sim = make_sim(500, 8);
        sim.run(5);
        resample(&mut sim);
        for (i, plate) in sim.plates.iter().enumerate() {
            if plate.reference_points.is_empty() {
                continue;
            }
            assert!(
                !plate.triangles.is_empty(),
                "plate {i} has vertices but no triangles"
            );
        }
    }

    #[test]
    fn resample_adjacency_parallel_to_triangles() {
        let mut sim = make_sim(500, 8);
        sim.run(5);
        resample(&mut sim);
        for plate in &sim.plates {
            assert_eq!(plate.adjacency.len(), plate.triangles.len());
        }
    }

    #[test]
    fn resample_crust_parallel_to_reference_points() {
        let mut sim = make_sim(500, 8);
        sim.run(5);
        resample(&mut sim);
        for plate in &sim.plates {
            assert_eq!(plate.crust.len(), plate.reference_points.len());
        }
    }

    #[test]
    fn resample_bounding_caps_cover_vertices() {
        let mut sim = make_sim(500, 8);
        sim.run(5);
        resample(&mut sim);
        for plate in &sim.plates {
            for &ref_p in &plate.reference_points {
                let dot = ref_p.dot(plate.bounding_cap.center);
                assert!(
                    dot >= plate.bounding_cap.cos_radius - 1e-6,
                    "vertex outside bounding cap"
                );
            }
        }
    }

    #[test]
    fn resample_cache_bary_sums_to_one() {
        let mut sim = make_sim(500, 8);
        sim.run(5);
        resample(&mut sim);
        for cache in &sim.sample_cache {
            if cache.plate == NO_PLATE {
                continue;
            }
            let sum: f64 = cache.bary.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "bary sum {sum} != 1.0"
            );
        }
    }

    #[test]
    fn resample_continental_fraction_roughly_preserved() {
        let mut sim = make_sim(1000, 8);
        let before: usize = sim
            .plates
            .iter()
            .flat_map(|p| &p.crust)
            .filter(|c| c.crust_type == CrustType::Continental)
            .count();
        let total_before: usize = sim.plates.iter().map(|p| p.point_count()).sum();
        let frac_before = before as f64 / total_before as f64;

        sim.run(5);
        resample(&mut sim);

        let after: usize = sim
            .plates
            .iter()
            .flat_map(|p| &p.crust)
            .filter(|c| c.crust_type == CrustType::Continental)
            .count();
        let total_after: usize = sim.plates.iter().map(|p| p.point_count()).sum();
        let frac_after = after as f64 / total_after as f64;

        assert!(
            (frac_after - frac_before).abs() < 0.15,
            "continental fraction changed too much: {frac_before:.2} → {frac_after:.2}"
        );
    }

    #[test]
    fn multiple_resample_cycles_stable() {
        let mut sim = make_sim(500, 8);
        for _ in 0..3 {
            sim.run(resample_interval(&sim.plates));
        }
        assert_eq!(sim.sample_points.len(), 500);
        let non_empty = sim
            .plates
            .iter()
            .filter(|p| !p.reference_points.is_empty())
            .count();
        assert!(non_empty >= 2, "too few plates survived: {non_empty}");
    }

    #[test]
    fn walk_finds_vertices_from_incident_triangle_after_resample() {
        use super::super::plates::WalkResult;

        let mut sim = make_sim(500, 8);
        sim.run(5);
        resample(&mut sim);

        for plate in &sim.plates {
            if plate.point_count() == 0 {
                continue;
            }
            let mut found = 0;
            for (i, &ref_p) in plate.reference_points.iter().enumerate() {
                let start = plate
                    .triangles
                    .iter()
                    .position(|tri| tri.contains(&(i as u32)))
                    .unwrap_or(0);
                if let WalkResult::Found { .. } =
                    plate.walk_to_point(ref_p, start as u32, 256)
                {
                    found += 1;
                }
            }
            assert_eq!(
                found,
                plate.point_count(),
                "walk missed {} vertices after resample",
                plate.point_count() - found
            );
        }
    }
}
