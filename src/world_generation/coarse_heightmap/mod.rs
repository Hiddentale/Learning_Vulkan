mod boundary;
mod climate;
mod continental_selection;
mod elevation;
mod hotspots;
mod stress;
mod tectonic_features;

use glam::DVec3;
use noise::{Fbm, Perlin};

use crate::world_generation::sphere_geometry::plate_seed_placement::{Adjacency, PlateAssignment};

pub struct CoarseHeightmap {
    /// Per-point elevation in km (positive = above sea level).
    pub elevation: Vec<f32>,
    /// Per-point mean temperature in degrees Celsius.
    pub temperature: Vec<f32>,
    /// Per-point annual precipitation in mm.
    pub precipitation: Vec<f32>,
    /// Per-point flag: true if this point's plate is continental.
    pub is_continental: Vec<bool>,
}

pub fn generate(
    points: &[DVec3],
    assignment: &PlateAssignment,
    adjacency: &Adjacency,
    seed: u64,
) -> CoarseHeightmap {
    let n = points.len();
    let plate_count = assignment.plate_count() as usize;

    let is_continental = select_continents(points, assignment, adjacency, plate_count, n, seed);
    let boundaries = boundary::classify(adjacency, assignment, &continental_set(&assignment.plate_ids, &is_continental), seed);
    let dist_fields = compute_distance_fields(n, adjacency, &boundaries);
    let (stress, stress_dir) = stress::propagate(
        n, points, adjacency, &assignment.plate_ids,
        &boundaries.mountain_seeds, &boundaries.subduction_seeds,
    );
    let terrain_class = stress::classify_interior_terrain(n, points, seed);

    let fbm_elevation: Fbm<Perlin> = Fbm::new(seed as u32);
    let fbm_mountain: Fbm<Perlin> = Fbm::new(seed.wrapping_add(1) as u32);
    let fbm_arc: Fbm<Perlin> = Fbm::new(seed.wrapping_add(4) as u32);
    let fbm_basin: Fbm<Perlin> = Fbm::new(seed.wrapping_add(3) as u32);
    let fbm_coast: Fbm<Perlin> = Fbm::new(seed.wrapping_add(5) as u32);
    let fbm_rift: Fbm<Perlin> = Fbm::new(seed.wrapping_add(6) as u32);
    let fbm_fold: Fbm<Perlin> = Fbm::new(seed.wrapping_add(7) as u32);

    let mut elev = compute_base_elevation(
        n, points, &is_continental, &dist_fields, &stress, &terrain_class,
        &fbm_elevation, &fbm_mountain, &fbm_arc, &fbm_basin,
    );

    tectonic_features::apply_fold_ridges(
        n, points, &dist_fields.mountain, &is_continental, &stress, &stress_dir,
        &fbm_fold, &mut elev,
    );
    tectonic_features::apply_rift_valleys(
        n, points, &dist_fields.rift, &is_continental, &fbm_rift, &mut elev,
    );
    tectonic_features::apply_back_arc_basins(
        n, &dist_fields.subduction, &dist_fields.mountain, &is_continental, &stress, &mut elev,
    );
    tectonic_features::apply_plateaus(
        n, &dist_fields.mountain, &dist_fields.coast, &is_continental, &stress, &mut elev,
    );
    tectonic_features::apply_coastal_roughening(
        n, points, &dist_fields.coast, &is_continental, &stress, &fbm_coast, &mut elev,
    );
    hotspots::generate_and_apply(points, &is_continental, seed, &mut elev);

    let (temperature, precipitation) = compute_climate(n, points, &elev, &dist_fields.coast);

    CoarseHeightmap {
        elevation: elev,
        temperature,
        precipitation,
        is_continental,
    }
}

// ── Internal helpers ────────────────────────────────────────────────────────

struct DistanceFields {
    coast: Vec<u32>,
    mountain: Vec<u32>,
    ridge: Vec<u32>,
    arc: Vec<u32>,
    subduction: Vec<u32>,
    rift: Vec<u32>,
}

fn select_continents(
    points: &[DVec3],
    assignment: &PlateAssignment,
    adjacency: &Adjacency,
    plate_count: usize,
    n: usize,
    seed: u64,
) -> Vec<bool> {
    let plate_adj = continental_selection::build_plate_adjacency(adjacency, &assignment.plate_ids, plate_count);
    let plate_sizes = continental_selection::compute_plate_sizes(&assignment.plate_ids, plate_count);
    let plate_centroids = continental_selection::compute_plate_centroids(points, &assignment.plate_ids, plate_count);
    let continental = continental_selection::pick_continental_plates(
        &plate_adj, &plate_sizes, &plate_centroids, n, seed,
    );
    assignment.plate_ids.iter().map(|pid| continental.contains(pid)).collect()
}

fn continental_set(plate_ids: &[u32], is_continental: &[bool]) -> std::collections::HashSet<u32> {
    plate_ids.iter().zip(is_continental).filter(|(_, &c)| c).map(|(&pid, _)| pid).collect()
}

fn compute_distance_fields(
    n: usize,
    adjacency: &Adjacency,
    boundaries: &boundary::BoundarySeeds,
) -> DistanceFields {
    DistanceFields {
        coast: boundary::bfs_distance(n, adjacency, &boundaries.coast_seeds),
        mountain: boundary::bfs_distance(n, adjacency, &boundaries.mountain_seeds),
        ridge: boundary::bfs_distance(n, adjacency, &boundaries.ridge_seeds),
        arc: boundary::bfs_distance(n, adjacency, &boundaries.arc_seeds),
        subduction: boundary::bfs_distance(n, adjacency, &boundaries.subduction_seeds),
        rift: boundary::bfs_distance_capped(
            n, adjacency, &boundaries.rift_seeds, tectonic_features::RIFT_HALF_WIDTH,
        ),
    }
}

fn compute_base_elevation(
    n: usize,
    points: &[DVec3],
    is_continental: &[bool],
    df: &DistanceFields,
    stress: &[f32],
    terrain_class: &[f32],
    fbm_elevation: &Fbm<Perlin>,
    fbm_mountain: &Fbm<Perlin>,
    fbm_arc: &Fbm<Perlin>,
    fbm_basin: &Fbm<Perlin>,
) -> Vec<f32> {
    use noise::NoiseFn;
    let mut elev = vec![0.0f32; n];
    for i in 0..n {
        let p = points[i];
        let noise_val = fbm_elevation.get([p.x * 4.0, p.y * 4.0, p.z * 4.0]);
        elev[i] = if is_continental[i] {
            elevation::continental(
                df.coast[i] as f64, df.mountain[i] as f64, df.subduction[i] as f64,
                p, noise_val, stress[i] as f64, terrain_class[i], fbm_mountain, fbm_basin,
            )
        } else {
            elevation::oceanic(
                df.coast[i] as f64, df.ridge[i] as f64, df.arc[i] as f64,
                df.subduction[i] as f64, p, noise_val, fbm_arc,
            )
        } as f32;
    }
    elev
}

fn compute_climate(
    n: usize,
    points: &[DVec3],
    elevation: &[f32],
    dist_coast: &[u32],
) -> (Vec<f32>, Vec<f32>) {
    let mut temp = vec![0.0f32; n];
    let mut precip = vec![0.0f32; n];
    for i in 0..n {
        temp[i] = climate::temperature(points[i], elevation[i] as f64) as f32;
        precip[i] = climate::precipitation(points[i], dist_coast[i] as f64) as f32;
    }
    (temp, precip)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;
    use crate::world_generation::sphere_geometry::fibonnaci_spiral::SphericalFibonacci;
    use crate::world_generation::sphere_geometry::plate_seed_placement::assign_plates;
    use crate::world_generation::sphere_geometry::spherical_delaunay_triangulation::SphericalDelaunay;

    fn make_heightmap(point_count: u32, plate_count: u32, seed: u64) -> (Vec<DVec3>, CoarseHeightmap) {
        let fib = SphericalFibonacci::new(point_count);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, plate_count, seed);
        let adjacency = Adjacency::from_delaunay(points.len(), &del);
        let heightmap = generate(&points, &assignment, &adjacency, seed);
        (points, heightmap)
    }

    #[test]
    fn multiple_continents() {
        let (points, hm) = make_heightmap(5000, 20, 42);
        let del = SphericalDelaunay::from_points(&points);
        let adj = Adjacency::from_delaunay(points.len(), &del);

        let mut visited = vec![false; points.len()];
        let mut continent_count = 0;
        for start in 0..points.len() {
            if !hm.is_continental[start] || visited[start] {
                continue;
            }
            continent_count += 1;
            let mut queue = VecDeque::new();
            visited[start] = true;
            queue.push_back(start as u32);
            while let Some(current) = queue.pop_front() {
                for &nb in adj.neighbors_of(current) {
                    if !visited[nb as usize] && hm.is_continental[nb as usize] {
                        visited[nb as usize] = true;
                        queue.push_back(nb);
                    }
                }
            }
        }
        assert!(continent_count >= 2, "expected ≥2 continents, got {continent_count}");
    }

    #[test]
    fn continental_fraction_near_30_percent() {
        let (_, hm) = make_heightmap(5000, 20, 42);
        let count = hm.is_continental.iter().filter(|&&c| c).count();
        let fraction = count as f64 / hm.is_continental.len() as f64;
        assert!(fraction > 0.15 && fraction < 0.45, "continental fraction {fraction:.2}");
    }

    #[test]
    fn ocean_negative_land_positive() {
        let (_, hm) = make_heightmap(5000, 20, 42);
        let ocean_neg = hm.elevation.iter().zip(&hm.is_continental)
            .filter(|(_, &c)| !c).filter(|(&e, _)| e < 0.0).count();
        let ocean_total = hm.is_continental.iter().filter(|&&c| !c).count();
        assert!((ocean_neg as f64 / ocean_total as f64) > 0.7);

        let land_pos = hm.elevation.iter().zip(&hm.is_continental)
            .filter(|(_, &c)| c).filter(|(&e, _)| e > 0.0).count();
        let land_total = hm.is_continental.iter().filter(|&&c| c).count();
        assert!((land_pos as f64 / land_total as f64) > 0.7);
    }

    #[test]
    fn islands_exist() {
        let (_, hm) = make_heightmap(10_000, 30, 42);
        let count = hm.elevation.iter().zip(&hm.is_continental)
            .filter(|(_, &c)| !c).filter(|(&e, _)| e > 0.0).count();
        assert!(count > 0, "no islands found");
    }

    #[test]
    fn temperature_decreases_poleward() {
        let (points, hm) = make_heightmap(5000, 20, 42);
        let avg = |filter: fn(&DVec3) -> bool| -> f64 {
            let vals: Vec<f64> = points.iter().zip(&hm.temperature)
                .filter(|(p, _)| filter(p)).map(|(_, &t)| t as f64).collect();
            vals.iter().sum::<f64>() / vals.len() as f64
        };
        let eq = avg(|p| p.y.abs() < 0.2);
        let pole = avg(|p| p.y.abs() > 0.8);
        assert!(eq > pole + 10.0, "equator {eq:.1} not warmer than polar {pole:.1}");
    }

    #[test]
    fn precipitation_positive() {
        let (_, hm) = make_heightmap(5000, 20, 42);
        for (i, &p) in hm.precipitation.iter().enumerate() {
            assert!(p >= 49.0, "point {i} has precipitation {p}");
        }
    }

    #[test]
    fn elevation_range_reasonable() {
        let (_, hm) = make_heightmap(5000, 20, 42);
        for (i, &e) in hm.elevation.iter().enumerate() {
            assert!(e > -7.0 && e < 10.0, "point {i} has elevation {e}");
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let (_, a) = make_heightmap(2000, 12, 77);
        let (_, b) = make_heightmap(2000, 12, 77);
        assert_eq!(a.elevation, b.elevation);
        assert_eq!(a.temperature, b.temperature);
        assert_eq!(a.precipitation, b.precipitation);
        assert_eq!(a.is_continental, b.is_continental);
    }

    /// Run with: cargo test --release -- export_globe --ignored --nocapture
    #[test]
    #[ignore]
    fn export_globe() {
        let (points, hm) = make_heightmap(20_000, 30, 42);
        let del = SphericalDelaunay::from_points(&points);

        let mut indices = String::new();
        for t in 0..del.triangle_count() {
            let base = t * 3;
            if t > 0 { indices.push(','); }
            indices.push_str(&format!(
                "{},{},{}", del.triangles[base], del.triangles[base + 1], del.triangles[base + 2]
            ));
        }

        let mut positions = String::new();
        let mut colors = String::new();
        for (i, &p) in points.iter().enumerate() {
            let scale = 1.0 + (hm.elevation[i] as f64) * 0.01;
            if i > 0 { positions.push(','); colors.push(','); }
            positions.push_str(&format!("{:.5},{:.5},{:.5}", p.x * scale, p.y * scale, p.z * scale));
            let (r, g, b) = elevation_color(hm.elevation[i]);
            colors.push_str(&format!("{r:.3},{g:.3},{b:.3}"));
        }

        let info = format!(
            "{} points, {} triangles — drag to orbit, scroll to zoom, W for wireframe",
            points.len(), del.triangle_count(),
        );
        let html = include_str!("globe_template.html")
            .replace("POSITIONS_DATA", &positions)
            .replace("COLORS_DATA", &colors)
            .replace("INDICES_DATA", &indices)
            .replace("INFO_TEXT", &info);

        let path = std::path::Path::new("src/world_generation/coarse_heightmap/globe.html");
        std::fs::write(path, html).expect("failed to write globe.html");
        println!("\n  Wrote {}\n", path.display());
    }

    fn elevation_color(elev: f32) -> (f32, f32, f32) {
        if elev < -3.0 {
            (0.05, 0.08, 0.35)
        } else if elev < -0.5 {
            let t = (elev + 3.0) / 2.5;
            (0.05 + t * 0.1, 0.08 + t * 0.2, 0.35 + t * 0.25)
        } else if elev < 0.0 {
            let t = (elev + 0.5) / 0.5;
            (0.15 + t * 0.15, 0.28 + t * 0.25, 0.60 - t * 0.05)
        } else if elev < 0.3 {
            let t = elev / 0.3;
            (0.30 + t * 0.05, 0.53 + t * 0.1, 0.55 - t * 0.25)
        } else if elev < 1.5 {
            let t = (elev - 0.3) / 1.2;
            (0.35 - t * 0.1, 0.63 - t * 0.2, 0.30 - t * 0.05)
        } else if elev < 3.0 {
            let t = (elev - 1.5) / 1.5;
            (0.45 + t * 0.15, 0.35 - t * 0.05, 0.15 + t * 0.05)
        } else if elev < 5.0 {
            let t = (elev - 3.0) / 2.0;
            (0.60 + t * 0.1, 0.30 + t * 0.2, 0.20 + t * 0.25)
        } else {
            let t = ((elev - 5.0) / 3.0).min(1.0);
            (0.70 + t * 0.25, 0.50 + t * 0.45, 0.45 + t * 0.50)
        }
    }
}
