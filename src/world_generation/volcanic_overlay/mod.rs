mod atlas;
mod stamp;

use glam::DVec3;

use super::coarse_heightmap::CoarseHeightmap;
use super::terrain_amplification::AmplifiedTerrain;

/// Apply volcanic features (hotspot islands, seamounts, archipelagos) onto the
/// amplified terrain. Patches from the volcanic atlas are stamped with feathered
/// blending so they rise naturally from the diffusion-generated ocean floor.
pub fn overlay(
    terrain: &mut AmplifiedTerrain,
    coarse: &CoarseHeightmap,
    points: &[DVec3],
    seed: u64,
) {
    let atlas = match atlas::VolcanicAtlas::load() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[volcanic_overlay] failed to load atlas: {e}");
            return;
        }
    };

    let placements = generate_placements(coarse, points, seed, &atlas);

    stamp::stamp_all(
        &mut terrain.cross_elevation,
        terrain.cross_width,
        terrain.cross_height,
        &placements,
        &atlas,
    );

    eprintln!(
        "[volcanic_overlay] stamped {} volcanic features",
        placements.len()
    );
}

// ── Placement generation ───────────────────────────────────────────────────

/// A volcanic feature to stamp onto the terrain.
pub(crate) struct Placement {
    /// Unit direction on the sphere.
    pub center: DVec3,
    /// Which atlas category (0-3).
    pub category: u32,
    /// Which patch within the category.
    pub patch_idx: u32,
    /// Angular radius on the sphere (radians).
    pub angular_radius: f64,
    /// Rotation in radians (0, pi/2, pi, 3pi/2).
    pub rotation: u8,
    /// Mirror horizontally.
    pub flip: bool,
    /// Vertical scale factor for the feature (1.0 = original height).
    pub height_scale: f32,
}

const HOTSPOT_COUNT: usize = 15;
const TRAIL_STEPS: usize = 4;
const TRAIL_SPACING: f64 = 0.04;
const TRAIL_DECAY: f64 = 0.55;

/// Maximum BFS hop distance to be considered "near" a ridge for seamount clustering.
const NEAR_RIDGE_HOPS: u32 = 12;
/// Maximum BFS hop distance for island arc placement from subduction zone.
const ARC_MIN_HOPS: u32 = 3;
const ARC_MAX_HOPS: u32 = 8;

fn generate_placements(
    coarse: &CoarseHeightmap,
    points: &[DVec3],
    seed: u64,
    atlas: &atlas::VolcanicAtlas,
) -> Vec<Placement> {
    let mut placements = Vec::new();
    let mut rng = splitmix64(seed.wrapping_add(55555));

    let n_shield = atlas.patches_per_cat.min(10) as u64;
    let n_strato = atlas.patches_per_cat.min(8) as u64;
    let n_seamount = atlas.patches_per_cat.min(8) as u64;

    // ── Hotspot chains (mantle plumes — random placement, no plate boundary relation) ──
    for _ in 0..HOTSPOT_COUNT {
        rng = splitmix64(rng);
        let center = random_sphere_point(&mut rng);

        let nearest = nearest_point(points, center);
        if coarse.is_continental[nearest] {
            continue;
        }

        rng = splitmix64(rng);
        let patch_idx = (rng % n_shield) as u32;
        rng = splitmix64(rng);
        let rotation = (rng % 4) as u8;
        rng = splitmix64(rng);
        let flip = rng % 2 == 0;

        placements.push(Placement {
            center,
            category: 0,
            patch_idx,
            angular_radius: 0.035,
            rotation,
            flip,
            height_scale: 1.0,
        });

        // Trail of older, subsiding islands along a random "plate motion" direction
        rng = splitmix64(rng);
        let trail_dir = random_tangent(center, rng);
        let mut trail_pos = center;
        let mut trail_height = 1.0f32;

        for step in 0..TRAIL_STEPS {
            trail_height *= TRAIL_DECAY as f32;
            trail_pos = (trail_pos + trail_dir * TRAIL_SPACING).normalize();

            rng = splitmix64(rng);
            let trail_rot = (rng % 4) as u8;
            rng = splitmix64(rng);
            let trail_flip = rng % 2 == 0;

            // Younger trail islands use shield patches, older ones use seamount/atoll
            let (cat, pidx) = if step < 2 {
                rng = splitmix64(rng);
                (0, (rng % n_shield) as u32)
            } else {
                rng = splitmix64(rng);
                (2, (rng % n_seamount) as u32)
            };

            placements.push(Placement {
                center: trail_pos,
                category: cat,
                patch_idx: pidx,
                angular_radius: 0.03 - step as f64 * 0.004,
                rotation: trail_rot,
                flip: trail_flip,
                height_scale: trail_height,
            });
        }
    }

    // ── Island arcs — stratovolcanoes along oceanic convergent boundaries ──
    // Place at oceanic points near ocean-ocean convergent boundaries (arc seeds).
    // These form chains like Japan, Philippines, and the Aleutians.
    for i in 0..points.len() {
        if coarse.is_continental[i] {
            continue;
        }
        let da = coarse.dist_arc[i];
        if da == u32::MAX || da < ARC_MIN_HOPS || da > ARC_MAX_HOPS {
            continue;
        }

        // Sparse: ~1 in 12 qualifying points gets a volcano
        rng = splitmix64(rng.wrapping_add(i as u64));
        if rng % 12 != 0 {
            continue;
        }

        rng = splitmix64(rng);
        let patch_idx = (rng % n_strato) as u32;
        rng = splitmix64(rng);
        let rotation = (rng % 4) as u8;
        rng = splitmix64(rng);
        let flip = rng % 2 == 0;
        rng = splitmix64(rng);
        let height_scale = 0.6 + (rng as f64 / u64::MAX as f64) as f32 * 0.4;

        placements.push(Placement {
            center: points[i],
            category: 1,
            patch_idx,
            angular_radius: 0.025,
            rotation,
            flip,
            height_scale,
        });
    }

    // ── Seamounts — clustered near mid-ocean ridges ──
    // Higher density close to ridges (young oceanic crust), sparser further away.
    for i in 0..points.len() {
        if coarse.is_continental[i] {
            continue;
        }
        let dr = coarse.dist_ridge[i];
        if dr == u32::MAX {
            continue;
        }

        // Probability decreases with distance from ridge
        let prob = if dr <= NEAR_RIDGE_HOPS {
            40u64 // 1 in 40 near ridge
        } else {
            200u64 // 1 in 200 far from ridge
        };

        rng = splitmix64(rng.wrapping_add(i as u64));
        if rng % prob != 0 {
            continue;
        }

        rng = splitmix64(rng);
        let patch_idx = (rng % n_seamount) as u32;
        rng = splitmix64(rng);
        let rotation = (rng % 4) as u8;
        rng = splitmix64(rng);
        let flip = rng % 2 == 0;
        rng = splitmix64(rng);
        let height_scale = 0.2 + (rng as f64 / u64::MAX as f64) as f32 * 0.6;

        placements.push(Placement {
            center: points[i],
            category: 2,
            patch_idx,
            angular_radius: 0.012 + (rng as f64 / u64::MAX as f64) * 0.008,
            rotation,
            flip,
            height_scale,
        });
    }

    placements
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

fn random_sphere_point(rng: &mut u64) -> DVec3 {
    *rng = splitmix64(*rng);
    let z = (*rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
    *rng = splitmix64(*rng);
    let theta = (*rng as f64 / u64::MAX as f64) * std::f64::consts::TAU;
    let r = (1.0 - z * z).sqrt();
    DVec3::new(r * theta.cos(), r * theta.sin(), z)
}

fn random_tangent(normal: DVec3, rng: u64) -> DVec3 {
    let theta = (rng as f64 / u64::MAX as f64) * std::f64::consts::TAU;
    let up = if normal.y.abs() < 0.9 {
        DVec3::Y
    } else {
        DVec3::X
    };
    let u = normal.cross(up).normalize();
    let v = normal.cross(u);
    (u * theta.cos() + v * theta.sin()).normalize()
}

fn nearest_point(points: &[DVec3], target: DVec3) -> usize {
    points
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.dot(target).partial_cmp(&b.dot(target)).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_generation::coarse_heightmap;
    use crate::world_generation::sphere_geometry::fibonnaci_spiral::SphericalFibonacci;
    use crate::world_generation::sphere_geometry::plate_seed_placement::{
        assign_plates, Adjacency,
    };
    use crate::world_generation::sphere_geometry::spherical_delaunay_triangulation::SphericalDelaunay;
    use crate::world_generation::terrain_amplification;

    /// Run with: cargo test --release -- volcanic_overlay_ply --ignored --nocapture
    #[test]
    #[ignore]
    fn volcanic_overlay_ply() {
        use std::io::Write;

        let seed = 42u64;
        let fib = SphericalFibonacci::new(10_000);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, 40, seed);
        let adjacency = Adjacency::from_delaunay(points.len(), &del);
        let coarse = coarse_heightmap::generate(&points, &assignment, &adjacency, seed);

        let model_dir = std::path::Path::new("data/models/terrain-diffusion-30m");
        let mut terrain = terrain_amplification::amplify(&coarse, &points, &fib, seed, model_dir)
            .expect("amplification failed");

        println!("\n  Applying volcanic overlay...");
        overlay(&mut terrain, &coarse, &points, seed);

        // ── PLY export (copied from amplify_obj_export to guarantee identical format) ──
        let res = terrain.faces[0].resolution;
        let mut elev_grids: Vec<Vec<f32>> = terrain.faces.iter().map(|f| f.elevation.clone()).collect();

        // Seam blending — exact same edge table and feather logic as amplify_obj_export
        let se_edges: [(usize, u8, usize, u8, bool); 12] = [
            (0, 0, 2, 1, true),  (0, 2, 3, 1, false), (0, 3, 4, 1, false), (0, 1, 5, 3, false),
            (1, 0, 2, 3, false), (1, 2, 3, 3, true),  (1, 3, 5, 1, false), (1, 1, 4, 3, false),
            (2, 0, 5, 0, true),  (2, 2, 4, 0, false), (3, 0, 4, 2, false), (3, 2, 5, 2, true),
        ];
        let feather = 8u32;
        for &(fa, ba, fb, bb, rev) in &se_edges {
            let mut mismatch = vec![0.0f32; res as usize];
            for i in 0..res {
                let j = if rev { res - 1 - i } else { i };
                let (ra, ca) = border_pixel(ba, 0, i, res);
                let (rb, cb) = border_pixel(bb, 0, j, res);
                mismatch[i as usize] = elev_grids[fa][(ra * res + ca) as usize]
                    - elev_grids[fb][(rb * res + cb) as usize];
            }
            for depth in 0..feather.min(res / 4) {
                let alpha = 1.0 - depth as f32 / feather as f32;
                for i in 0..res {
                    let j = if rev { res - 1 - i } else { i };
                    let (ra, ca) = border_pixel(ba, depth, i, res);
                    let (rb, cb) = border_pixel(bb, depth, j, res);
                    let corr = mismatch[i as usize] * 0.5 * alpha;
                    elev_grids[fa][(ra * res + ca) as usize] -= corr;
                    elev_grids[fb][(rb * res + cb) as usize] += corr;
                }
            }
        }

        // Vertex generation
        let radius = 1.0f64;
        const EARTH_RADIUS_M: f64 = 6_371_000.0;
        let elev_exaggeration = radius / EARTH_RADIUS_M * 10.0;
        let total_verts = (6 * res * res) as usize;
        let mut positions: Vec<[f32; 3]> = Vec::with_capacity(total_verts);
        let mut colors: Vec<[u8; 3]> = Vec::with_capacity(total_verts);

        for fi in 0..6usize {
            for r in 0..res {
                for c in 0..res {
                    let vr = r as f64 / (res - 1).max(1) as f64;
                    let uc = c as f64 / (res - 1).max(1) as f64;
                    let (v_param, u_param) = if fi == 0 || fi == 1 || fi == 4 || fi == 5 {
                        (1.0 - vr, uc)
                    } else { (vr, uc) };
                    let (x, y, z) = match fi {
                        0 => (1.0, 2.0 * v_param - 1.0, 1.0 - 2.0 * u_param),
                        1 => (-1.0, 2.0 * v_param - 1.0, 2.0 * u_param - 1.0),
                        2 => (2.0 * u_param - 1.0, 1.0, 2.0 * v_param - 1.0),
                        3 => (2.0 * u_param - 1.0, -1.0, -(2.0 * v_param - 1.0)),
                        4 => (2.0 * u_param - 1.0, 2.0 * v_param - 1.0, 1.0),
                        _ => (1.0 - 2.0 * u_param, 2.0 * v_param - 1.0, -1.0),
                    };
                    let len = (x * x + y * y + z * z).sqrt();
                    let dir = DVec3::new(x / len, y / len, z / len);
                    let elev = elev_grids[fi][(r * res + c) as usize];
                    let displaced = radius + elev as f64 * elev_exaggeration;
                    positions.push([(dir.x * displaced) as f32, (dir.y * displaced) as f32, (dir.z * displaced) as f32]);
                    let (cr, cg, cb) = elevation_color_meters(elev);
                    colors.push([(cr * 255.0) as u8, (cg * 255.0) as u8, (cb * 255.0) as u8]);
                }
            }
        }

        // Merge border vertices
        let mut remap: Vec<u32> = (0..total_verts as u32).collect();
        let mut dir_map: std::collections::HashMap<(i64, i64, i64), u32> = std::collections::HashMap::new();
        let n_face = (res * res) as usize;
        for face_id in 0..6 {
            let base = face_id * n_face;
            for i in 0..res as usize {
                for &idx in &[base + i, base + (res as usize - 1) * res as usize + i,
                              base + i * res as usize, base + i * res as usize + (res as usize - 1)] {
                    let p = positions[idx];
                    let len = (p[0] as f64 * p[0] as f64 + p[1] as f64 * p[1] as f64 + p[2] as f64 * p[2] as f64).sqrt();
                    let key = ((p[0] as f64 / len * 1e4).round() as i64,
                               (p[1] as f64 / len * 1e4).round() as i64,
                               (p[2] as f64 / len * 1e4).round() as i64);
                    if let Some(&canonical) = dir_map.get(&key) {
                        let cp = &positions[canonical as usize];
                        positions[canonical as usize] = [(cp[0]+p[0])*0.5, (cp[1]+p[1])*0.5, (cp[2]+p[2])*0.5];
                        remap[idx] = canonical;
                    } else { dir_map.insert(key, idx as u32); }
                }
            }
        }

        // Triangles
        let mut tris: Vec<[i32; 3]> = Vec::new();
        for fi in 0..6u32 {
            let offset = fi * res * res;
            for r in 0..(res - 1) { for c in 0..(res - 1) {
                let v00 = remap[(offset + r * res + c) as usize] as i32;
                let v01 = remap[(offset + r * res + c + 1) as usize] as i32;
                let v10 = remap[(offset + (r + 1) * res + c) as usize] as i32;
                let v11 = remap[(offset + (r + 1) * res + c + 1) as usize] as i32;
                tris.push([v00, v10, v01]);
                tris.push([v01, v10, v11]);
            }}
        }

        // Write PLY
        let ply_path = "src/world_generation/volcanic_overlay/volcanic_planet.ply";
        let mut file = std::fs::File::create(ply_path).expect("failed to create PLY");
        let nv = total_verts;
        let nf = tris.len();
        write!(file, "ply\nformat binary_little_endian 1.0\n\
            element vertex {nv}\nproperty float x\nproperty float y\nproperty float z\n\
            property uchar red\nproperty uchar green\nproperty uchar blue\n\
            element face {nf}\nproperty list uchar int vertex_indices\nend_header\n").unwrap();
        for i in 0..nv {
            file.write_all(&positions[i][0].to_le_bytes()).unwrap();
            file.write_all(&positions[i][1].to_le_bytes()).unwrap();
            file.write_all(&positions[i][2].to_le_bytes()).unwrap();
            file.write_all(&colors[i]).unwrap();
        }
        for tri in &tris {
            file.write_all(&[3u8]).unwrap();
            file.write_all(&tri[0].to_le_bytes()).unwrap();
            file.write_all(&tri[1].to_le_bytes()).unwrap();
            file.write_all(&tri[2].to_le_bytes()).unwrap();
        }
        drop(file);
        let size = std::fs::metadata(ply_path).unwrap().len() as f64 / 1e6;
        println!("\n  Wrote {} ({:.1} MB, {} verts, {} faces)\n", ply_path, size, nv, nf);
    }

    fn border_pixel(border: u8, depth: u32, i: u32, res: u32) -> (u32, u32) {
        match border {
            0 => (depth, i),
            1 => (i, res - 1 - depth),
            2 => (res - 1 - depth, i),
            3 => (i, depth),
            _ => unreachable!(),
        }
    }

    fn elevation_color_meters(elev: f32) -> (f32, f32, f32) {
        if elev < -3000.0 {
            (0.04, 0.07, 0.30)
        } else if elev < -200.0 {
            let t = (elev + 3000.0) / 2800.0;
            (0.04 + t * 0.05, 0.07 + t * 0.12, 0.30 + t * 0.20)
        } else if elev < 0.0 {
            let t = (elev + 200.0) / 200.0;
            (0.09 + t * 0.06, 0.19 + t * 0.11, 0.50 + t * 0.10)
        } else if elev < 50.0 {
            let t = elev / 50.0;
            (0.55 + t * 0.10, 0.72 + t * 0.03, 0.45 - t * 0.10)
        } else if elev < 500.0 {
            let t = (elev - 50.0) / 450.0;
            (0.35 - t * 0.05, 0.62 - t * 0.10, 0.28 - t * 0.03)
        } else if elev < 1500.0 {
            let t = (elev - 500.0) / 1000.0;
            (0.30 + t * 0.25, 0.52 - t * 0.15, 0.25 - t * 0.05)
        } else if elev < 3000.0 {
            let t = (elev - 1500.0) / 1500.0;
            (0.55 + t * 0.15, 0.37 + t * 0.10, 0.20 + t * 0.15)
        } else {
            let t = ((elev - 3000.0) / 3000.0).min(1.0);
            (0.70 + t * 0.25, 0.47 + t * 0.48, 0.35 + t * 0.60)
        }
    }
}
