use std::collections::HashMap;
use std::io::Write;
use std::path::Path;

use glam::DVec3;

use super::terrain_amplification::AmplifiedTerrain;

const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Sphere-export edge table: (face_a, border_a, face_b, border_b, reversed).
const SE_EDGES: [(usize, u8, usize, u8, bool); 12] = [
    (0, 0, 2, 1, true),
    (0, 2, 3, 1, false),
    (0, 3, 4, 1, false),
    (0, 1, 5, 3, false),
    (1, 0, 2, 3, false),
    (1, 2, 3, 3, true),
    (1, 3, 5, 1, false),
    (1, 1, 4, 3, false),
    (2, 0, 5, 0, true),
    (2, 2, 4, 0, false),
    (3, 0, 4, 2, false),
    (3, 2, 5, 2, true),
];

/// Export amplified terrain as a binary PLY sphere with seam blending and
/// vertex merging at face borders.
pub fn export_planet_ply(terrain: &AmplifiedTerrain, ply_path: &Path) {
    let res = terrain.faces[0].resolution;
    let radius = 1.0f64;
    let elev_exaggeration = radius / EARTH_RADIUS_M * 10.0;

    // Clone elevation for seam correction
    let mut elev_grids: Vec<Vec<f32>> = terrain.faces.iter().map(|f| f.elevation.clone()).collect();
    blend_seams(&mut elev_grids, res);

    // Generate vertices
    let total_verts = (6 * res * res) as usize;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(total_verts);
    let mut colors: Vec<[u8; 3]> = Vec::with_capacity(total_verts);

    for fi in 0..6usize {
        for r in 0..res {
            for c in 0..res {
                let dir = pixel_to_sphere_dir(fi, r, c, res);
                let elev = elev_grids[fi][(r * res + c) as usize];
                let displaced = radius + elev as f64 * elev_exaggeration;

                positions.push([
                    (dir.x * displaced) as f32,
                    (dir.y * displaced) as f32,
                    (dir.z * displaced) as f32,
                ]);
                let (cr, cg, cb) = elevation_color(elev);
                colors.push([
                    (cr * 255.0) as u8,
                    (cg * 255.0) as u8,
                    (cb * 255.0) as u8,
                ]);
            }
        }
    }

    // Merge border vertices
    let mut remap: Vec<u32> = (0..total_verts as u32).collect();
    let mut dir_map: HashMap<(i64, i64, i64), u32> = HashMap::new();
    let n_face = (res * res) as usize;

    for face_id in 0..6 {
        let base = face_id * n_face;
        for i in 0..res as usize {
            let border_indices = [
                base + i,
                base + (res as usize - 1) * res as usize + i,
                base + i * res as usize,
                base + i * res as usize + (res as usize - 1),
            ];
            for &idx in &border_indices {
                let p = positions[idx];
                let len = (p[0] as f64 * p[0] as f64
                    + p[1] as f64 * p[1] as f64
                    + p[2] as f64 * p[2] as f64)
                    .sqrt();
                let key = (
                    (p[0] as f64 / len * 1e4).round() as i64,
                    (p[1] as f64 / len * 1e4).round() as i64,
                    (p[2] as f64 / len * 1e4).round() as i64,
                );
                if let Some(&canonical) = dir_map.get(&key) {
                    let cp = &positions[canonical as usize];
                    positions[canonical as usize] = [
                        (cp[0] + p[0]) * 0.5,
                        (cp[1] + p[1]) * 0.5,
                        (cp[2] + p[2]) * 0.5,
                    ];
                    remap[idx] = canonical;
                } else {
                    dir_map.insert(key, idx as u32);
                }
            }
        }
    }

    // Build triangles
    let mut tris: Vec<[i32; 3]> = Vec::with_capacity(6 * 2 * (res - 1) as usize * (res - 1) as usize);
    for fi in 0..6u32 {
        let offset = fi * res * res;
        for r in 0..(res - 1) {
            for c in 0..(res - 1) {
                let v00 = remap[(offset + r * res + c) as usize] as i32;
                let v01 = remap[(offset + r * res + c + 1) as usize] as i32;
                let v10 = remap[(offset + (r + 1) * res + c) as usize] as i32;
                let v11 = remap[(offset + (r + 1) * res + c + 1) as usize] as i32;
                tris.push([v00, v10, v01]);
                tris.push([v01, v10, v11]);
            }
        }
    }

    // Write binary PLY
    let mut file = std::fs::File::create(ply_path).expect("failed to create PLY");
    let nv = total_verts;
    let nf = tris.len();
    write!(
        file,
        "ply\nformat binary_little_endian 1.0\n\
        element vertex {nv}\nproperty float x\nproperty float y\nproperty float z\n\
        property uchar red\nproperty uchar green\nproperty uchar blue\n\
        element face {nf}\nproperty list uchar int vertex_indices\nend_header\n"
    )
    .unwrap();

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

    let merged = remap.iter().enumerate().filter(|(i, &r)| r != *i as u32).count();
    let size = std::fs::metadata(ply_path).unwrap().len() as f64 / 1e6;
    println!(
        "\n  Wrote {} ({:.1} MB, {} verts ({} merged), {} faces, {}x{} per face)\n",
        ply_path.display(), size, nv, merged, nf, res, res
    );
}

// ── Internals ──────────────────────────────────────────────────────────────

fn blend_seams(elev_grids: &mut [Vec<f32>], res: u32) {
    let feather = 8u32;
    for &(fa, ba, fb, bb, rev) in &SE_EDGES {
        let mut mismatch = vec![0.0f32; res as usize];
        for i in 0..res {
            let j = if rev { res - 1 - i } else { i };
            let (ra, ca) = border_pixel(ba, 0, i, res);
            let (rb, cb) = border_pixel(bb, 0, j, res);
            let ea = elev_grids[fa][(ra * res + ca) as usize];
            let eb = elev_grids[fb][(rb * res + cb) as usize];
            mismatch[i as usize] = ea - eb;
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

fn pixel_to_sphere_dir(fi: usize, r: u32, c: u32, res: u32) -> DVec3 {
    let vr = r as f64 / (res - 1).max(1) as f64;
    let uc = c as f64 / (res - 1).max(1) as f64;
    let (v_param, u_param) = if fi == 0 || fi == 1 || fi == 4 || fi == 5 {
        (1.0 - vr, uc)
    } else {
        (vr, uc)
    };
    let (x, y, z) = match fi {
        0 => (1.0, 2.0 * v_param - 1.0, 1.0 - 2.0 * u_param),
        1 => (-1.0, 2.0 * v_param - 1.0, 2.0 * u_param - 1.0),
        2 => (2.0 * u_param - 1.0, 1.0, 2.0 * v_param - 1.0),
        3 => (2.0 * u_param - 1.0, -1.0, -(2.0 * v_param - 1.0)),
        4 => (2.0 * u_param - 1.0, 2.0 * v_param - 1.0, 1.0),
        _ => (1.0 - 2.0 * u_param, 2.0 * v_param - 1.0, -1.0),
    };
    let len = (x * x + y * y + z * z).sqrt();
    DVec3::new(x / len, y / len, z / len)
}

fn elevation_color(elev: f32) -> (f32, f32, f32) {
    if elev < -3000.0 {
        (0.05, 0.08, 0.35)
    } else if elev < -500.0 {
        let t = (elev + 3000.0) / 2500.0;
        (0.05 + t * 0.1, 0.08 + t * 0.2, 0.35 + t * 0.25)
    } else if elev < 0.0 {
        let t = (elev + 500.0) / 500.0;
        (0.15 + t * 0.15, 0.28 + t * 0.25, 0.60 - t * 0.05)
    } else if elev < 300.0 {
        let t = elev / 300.0;
        (0.30 + t * 0.05, 0.53 + t * 0.1, 0.55 - t * 0.25)
    } else if elev < 1500.0 {
        let t = (elev - 300.0) / 1200.0;
        (0.35 - t * 0.1, 0.63 - t * 0.2, 0.30 - t * 0.05)
    } else if elev < 3000.0 {
        let t = (elev - 1500.0) / 1500.0;
        (0.45 + t * 0.15, 0.35 - t * 0.05, 0.15 + t * 0.05)
    } else if elev < 5000.0 {
        let t = (elev - 3000.0) / 2000.0;
        (0.60 + t * 0.1, 0.30 + t * 0.2, 0.20 + t * 0.25)
    } else {
        let t = ((elev - 5000.0) / 3000.0).min(1.0);
        (0.70 + t * 0.25, 0.50 + t * 0.45, 0.45 + t * 0.50)
    }
}
