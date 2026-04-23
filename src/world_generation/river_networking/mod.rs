mod depression_fill;
mod flow;
mod smooth;
mod valley;

/// Minimum flow accumulation to be considered a river pixel.
const RIVER_THRESHOLD: f32 = 50.0;

/// Run hydrological post-processing on the cross-layout elevation grid.
/// Modifies elevation in-place: fills depressions, smooths river bumps,
/// computes flow routing, and carves river valleys.
pub fn process(elevation: &mut [f32], width: u32, height: u32) -> FlowData {
    let w = width as usize;
    let h = height as usize;

    // Step 1: Fill depressions so water can drain to ocean
    depression_fill::fill(elevation, w, h, 1e-3, Some(500.0));

    // Step 2: Smooth small upslope bumps in flat/river areas
    smooth::smooth_river_bumps(elevation, w, h, 50.0, 0.3, 3);

    // Step 3: Compute D8 flow directions and accumulation
    let (flow_dir, is_sink) = flow::d8_flow(elevation, w, h);
    let accumulation = flow::accumulate(elevation, &flow_dir, &is_sink, w, h);

    // Step 4: Strahler stream ordering
    let stream_order = flow::strahler_order(&flow_dir, &is_sink, &accumulation, elevation, w, h, RIVER_THRESHOLD);

    // Step 5: Carve river valleys into the elevation
    valley::carve(elevation, &stream_order, &accumulation, w, h, RIVER_THRESHOLD);

    FlowData {
        flow_dir,
        accumulation,
        stream_order,
        is_sink,
        width,
        height,
    }
}

pub struct FlowData {
    /// Per-pixel D8 flow direction (0-7, or u8::MAX for sinks).
    pub flow_dir: Vec<u8>,
    /// Per-pixel flow accumulation count.
    pub accumulation: Vec<f32>,
    /// Strahler stream order (0 = not a river, 1+ = stream order).
    pub stream_order: Vec<u8>,
    /// True if pixel is a sink (ocean or internal basin).
    pub is_sink: Vec<bool>,
    pub width: u32,
    pub height: u32,
}

#[cfg(test)]
mod tests {
    use crate::world_generation::coarse_heightmap;
    use crate::world_generation::sphere_geometry::fibonnaci_spiral::SphericalFibonacci;
    use crate::world_generation::sphere_geometry::plate_seed_placement::{assign_plates, Adjacency};
    use crate::world_generation::sphere_geometry::spherical_delaunay_triangulation::SphericalDelaunay;
    use crate::world_generation::terrain_amplification;
    use crate::world_generation::volcanic_overlay;

    /// Full pipeline: amplify → volcanic overlay → river networking → resample faces → PLY.
    /// Everything runs on the cross-layout grid (continuous 2D), faces resampled at the end.
    /// Run with: cargo test --release -- river_networking_ply --ignored --nocapture
    #[test]
    #[ignore]
    fn river_networking_ply() {
        use glam::DVec3;
        use std::io::Write;

        let seed = 137u64;
        let fib = SphericalFibonacci::new(10_000);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, 40, seed);
        let adjacency = Adjacency::from_delaunay(points.len(), &del);
        let coarse = coarse_heightmap::generate(&points, &assignment, &adjacency, seed);

        let model_dir = std::path::Path::new("data/models/terrain-diffusion-30m");
        let mut terrain = terrain_amplification::amplify(&coarse, &points, &fib, seed, model_dir).expect("amplification failed");

        // Step 1: Volcanic overlay on cross grid
        println!("\n  Applying volcanic overlay on cross grid...");
        volcanic_overlay::overlay(&mut terrain, &coarse, &points, seed);

        // Step 2: River networking on cross grid (with volcanic islands)
        println!(
            "  Running river networking on cross-layout ({}×{})...",
            terrain.cross_width, terrain.cross_height
        );
        let flow = super::process(&mut terrain.cross_elevation, terrain.cross_width, terrain.cross_height);

        let max_accum = flow.accumulation.iter().cloned().fold(0.0f32, f32::max);
        let river_count = flow.accumulation.iter().filter(|&&a| a > 100.0).count();
        println!("  Flow: max_accumulation={max_accum:.0}, river_pixels(>100)={river_count}");

        // Export cross-layout elevation as PNG (full native resolution)
        println!("  Exporting cross-layout PNG...");
        export_cross_png(
            &terrain.cross_elevation,
            terrain.cross_width as usize,
            terrain.cross_height as usize,
            "src/world_generation/river_networking/cross_elevation.png",
        );

        // Export flow accumulation as PNG (rivers visible as bright lines)
        export_flow_png(
            &flow.accumulation,
            &terrain.cross_elevation,
            terrain.cross_width as usize,
            terrain.cross_height as usize,
            "src/world_generation/river_networking/cross_rivers.png",
        );

        // Step 3: Resample faces from the final cross grid
        println!("  Resampling faces from processed cross...");
        terrain.resample_faces_from_cross();

        // ── PLY export (same format as amplify_obj_export) ──
        let res = terrain.faces[0].resolution;
        let mut elev_grids: Vec<Vec<f32>> = terrain.faces.iter().map(|f| f.elevation.clone()).collect();

        let se_edges: [(usize, u8, usize, u8, bool); 12] = [
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
        let feather = 8u32;
        for &(fa, ba, fb, bb, rev) in &se_edges {
            let mut mismatch = vec![0.0f32; res as usize];
            for i in 0..res {
                let j = if rev { res - 1 - i } else { i };
                let (ra, ca) = border_pixel(ba, 0, i, res);
                let (rb, cb) = border_pixel(bb, 0, j, res);
                mismatch[i as usize] = elev_grids[fa][(ra * res + ca) as usize] - elev_grids[fb][(rb * res + cb) as usize];
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
                    let dir = DVec3::new(x / len, y / len, z / len);
                    let elev = elev_grids[fi][(r * res + c) as usize];
                    let displaced = radius + elev as f64 * elev_exaggeration;
                    positions.push([(dir.x * displaced) as f32, (dir.y * displaced) as f32, (dir.z * displaced) as f32]);
                    let (cr, cg, cb) = elevation_color_meters(elev);
                    colors.push([(cr * 255.0) as u8, (cg * 255.0) as u8, (cb * 255.0) as u8]);
                }
            }
        }

        let mut remap: Vec<u32> = (0..total_verts as u32).collect();
        let mut dir_map: std::collections::HashMap<(i64, i64, i64), u32> = std::collections::HashMap::new();
        let n_face = (res * res) as usize;
        for face_id in 0..6 {
            let base = face_id * n_face;
            for i in 0..res as usize {
                for &idx in &[
                    base + i,
                    base + (res as usize - 1) * res as usize + i,
                    base + i * res as usize,
                    base + i * res as usize + (res as usize - 1),
                ] {
                    let p = positions[idx];
                    let len = (p[0] as f64 * p[0] as f64 + p[1] as f64 * p[1] as f64 + p[2] as f64 * p[2] as f64).sqrt();
                    let key = (
                        (p[0] as f64 / len * 1e4).round() as i64,
                        (p[1] as f64 / len * 1e4).round() as i64,
                        (p[2] as f64 / len * 1e4).round() as i64,
                    );
                    if let Some(&canonical) = dir_map.get(&key) {
                        let cp = &positions[canonical as usize];
                        positions[canonical as usize] = [(cp[0] + p[0]) * 0.5, (cp[1] + p[1]) * 0.5, (cp[2] + p[2]) * 0.5];
                        remap[idx] = canonical;
                    } else {
                        dir_map.insert(key, idx as u32);
                    }
                }
            }
        }

        let mut tris: Vec<[i32; 3]> = Vec::new();
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

        let ply_path = "src/world_generation/river_networking/river_planet.ply";
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

    fn elevation_to_rgb(elev: f32) -> [u8; 3] {
        let (r, g, b) = elevation_color_meters(elev);
        [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
    }

    fn export_cross_png(elevation: &[f32], w: usize, h: usize, path: &str) {
        let mut img = image::RgbImage::new(w as u32, h as u32);
        for r in 0..h {
            for c in 0..w {
                let elev = elevation[r * w + c];
                let rgb = elevation_to_rgb(elev);
                img.put_pixel(c as u32, r as u32, image::Rgb(rgb));
            }
        }
        img.save(path).expect("failed to save PNG");
        let size = std::fs::metadata(path).unwrap().len() as f64 / 1024.0;
        println!("  Wrote {} ({:.0} KB, {}×{})", path, size, w, h);
    }

    fn export_flow_png(accumulation: &[f32], elevation: &[f32], w: usize, h: usize, path: &str) {
        let mut img = image::RgbImage::new(w as u32, h as u32);
        for r in 0..h {
            for c in 0..w {
                let idx = r * w + c;
                let elev = elevation[idx];
                let acc = accumulation[idx];

                if elev <= 0.0 || elev.is_nan() {
                    // Ocean: dark blue
                    let t = (elev.max(-5000.0) + 5000.0) / 5000.0;
                    let b = (80.0 + t * 100.0) as u8;
                    img.put_pixel(c as u32, r as u32, image::Rgb([10, 20, b]));
                } else if acc > 50.0 {
                    // River: blue intensity by accumulation
                    let t = ((acc.ln() - 50.0f32.ln()) / (10000.0f32.ln() - 50.0f32.ln())).clamp(0.0, 1.0);
                    let b = (150.0 + t * 105.0) as u8;
                    let g = (80.0 + t * 50.0) as u8;
                    img.put_pixel(c as u32, r as u32, image::Rgb([20, g, b]));
                } else {
                    // Land: terrain color
                    let rgb = elevation_to_rgb(elev);
                    img.put_pixel(c as u32, r as u32, image::Rgb(rgb));
                }
            }
        }
        img.save(path).expect("failed to save PNG");
        let size = std::fs::metadata(path).unwrap().len() as f64 / 1024.0;
        println!("  Wrote {} ({:.0} KB, {}×{})", path, size, w, h);
    }
}
