mod coarse_stage;
mod decoder_stage;
mod latent_stage;
pub(crate) mod rasterize;
mod scheduler;
mod session;
mod tiling;

use std::path::Path;

use anyhow::{Context, Result};
use glam::DVec3;
use log::info;

use crate::world_generation::coarse_heightmap::CoarseHeightmap;
use crate::world_generation::sphere_geometry::fibonnaci_spiral::SphericalFibonacci;

/// Amplified terrain: 6 cube-face heightmaps at high resolution.
pub struct AmplifiedTerrain {
    pub faces: [FaceHeightmap; 6],
}

#[derive(Debug)]
pub struct FaceHeightmap {
    /// Elevation in meters, row-major.
    pub elevation: Vec<f32>,
    /// Mean temperature in degrees C.
    pub temperature: Vec<f32>,
    /// Annual precipitation in mm.
    pub precipitation: Vec<f32>,
    /// Pixels per side.
    pub resolution: u32,
}

/// Target resolution per cube face for the diffusion output.
/// For a 100km planet at 30m: 100_000 / 6 / 30 ≈ 556.
const TARGET_FACE_RESOLUTION: u32 = 556;

/// Coarse conditioning resolution. At 64, fills the model tile exactly (no padding).
/// 10,000 Fibonacci points give ~1,667 per face → ~41 effective pixels per edge,
/// so 64×64 is well-sampled by IDW with 6 neighbors.
const COARSE_FACE_RESOLUTION: u32 = 64;

/// Run the full terrain-diffusion amplification pipeline.
///
/// Requires ONNX model files in `model_dir`:
///   coarse_model.onnx, base_model.onnx, decoder_model.onnx
pub fn amplify(
    coarse: &CoarseHeightmap,
    points: &[DVec3],
    fibonacci: &SphericalFibonacci,
    seed: u64,
    model_dir: &Path,
) -> Result<AmplifiedTerrain> {
    let t_total = std::time::Instant::now();
    session::log_to_file("[pipeline] === amplify() start ===");

    let t0 = std::time::Instant::now();
    let face_grids = rasterize::rasterize(coarse, points, fibonacci, COARSE_FACE_RESOLUTION);
    session::log_to_file(&format!("[pipeline] rasterize: {:.3}s", t0.elapsed().as_secs_f64()));

    let mut coarse_outputs = Vec::with_capacity(6);
    let mut latent_outputs = Vec::with_capacity(6);

    // Stage 1: load coarse model once, run all 6 faces
    {
        let mut session = session::load_model(model_dir, "coarse_model")
            .context("loading coarse model")?;
        for (i, grid) in face_grids.iter().enumerate() {
            let t0 = std::time::Instant::now();
            let padded = rasterize::pad_to_size(grid, 64);
            coarse_outputs.push(coarse_stage::run(&mut session, &padded, seed)?);
            session::log_to_file(&format!("[pipeline] coarse face {i}: {:.3}s", t0.elapsed().as_secs_f64()));
        }
    }

    // Stage 2: load latent model once, run all 6 faces
    {
        let mut session = session::load_model(model_dir, "base_model")
            .context("loading base model")?;
        for (i, coarse_out) in coarse_outputs.iter().enumerate() {
            let t0 = std::time::Instant::now();
            latent_outputs.push(latent_stage::run(&mut session, coarse_out, 64, 64, seed)?);
            session::log_to_file(&format!("[pipeline] latent face {i}: {:.3}s", t0.elapsed().as_secs_f64()));
        }
    }

    // Stage 3: load decoder model once, run all 6 faces
    let mut faces: Vec<FaceHeightmap> = Vec::with_capacity(6);
    {
        let mut session = session::load_model(model_dir, "decoder_model")
            .context("loading decoder model")?;
        for (i, (coarse_out, latent_out)) in
            coarse_outputs.iter().zip(&latent_outputs).enumerate()
        {
            let t0 = std::time::Instant::now();
            let residual = decoder_stage::run(
                &mut session, latent_out, 64, TARGET_FACE_RESOLUTION, seed,
            )?;
            faces.push(combine_output(coarse_out, latent_out, &residual, TARGET_FACE_RESOLUTION));
            session::log_to_file(&format!("[pipeline] decoder face {i}: {:.3}s", t0.elapsed().as_secs_f64()));
        }
    }
    // Seam blending: smooth elevation discontinuities at cube face borders
    blend_face_seams(&mut faces);
    session::log_to_file(&format!("[pipeline] === total: {:.3}s ===", t_total.elapsed().as_secs_f64()));

    Ok(AmplifiedTerrain {
        faces: faces.try_into().unwrap(),
    })
}

// ── Seam blending ───────────────────────────────────────────────────────────

/// Which border of face A connects to which border of face B.
/// Each edge: (face_a, border_a, face_b, border_b, reversed)
/// Borders: 0=top(row=0), 1=right(col=max), 2=bottom(row=max), 3=left(col=0)
///
/// Derived from voxel::sphere::face_basis() — each face's border traces a cube
/// edge in the -1→+1 direction along the varying coordinate, so all traversals
/// are aligned (no reversal needed).
const CUBE_EDGES: [(usize, u8, usize, u8, bool); 12] = [
    (0, 0, 5, 2, false), // PosX top    ↔ NegZ bottom  (X=1, Z=-1 edge, Y varies)
    (0, 1, 2, 2, false), // PosX right  ↔ PosY bottom  (X=1, Y=+1 edge, Z varies)
    (0, 2, 4, 1, false), // PosX bottom ↔ PosZ right   (X=1, Z=+1 edge, Y varies)
    (0, 3, 3, 1, false), // PosX left   ↔ NegY right   (X=1, Y=-1 edge, Z varies)
    (1, 0, 3, 3, false), // NegX top    ↔ NegY left    (X=-1, Y=-1 edge, Z varies)
    (1, 1, 4, 3, false), // NegX right  ↔ PosZ left    (X=-1, Z=+1 edge, Y varies)
    (1, 2, 2, 0, false), // NegX bottom ↔ PosY top     (X=-1, Y=+1 edge, Z varies)
    (1, 3, 5, 0, false), // NegX left   ↔ NegZ top     (X=-1, Z=-1 edge, Y varies)
    (2, 3, 5, 1, false), // PosY left   ↔ NegZ right   (Y=1, Z=-1 edge, X varies)
    (2, 1, 4, 2, false), // PosY right  ↔ PosZ bottom  (Y=1, Z=+1 edge, X varies)
    (3, 0, 5, 3, false), // NegY top    ↔ NegZ left    (Y=-1, Z=-1 edge, X varies)
    (3, 2, 4, 0, false), // NegY bottom ↔ PosZ top     (Y=-1, Z=+1 edge, X varies)
];

const BLEND_WIDTH: u32 = 16;

fn blend_face_seams(faces: &mut [FaceHeightmap]) {
    for &(fa, ba, fb, bb, rev) in &CUBE_EDGES {
        let res = faces[fa].resolution;

        for depth in 0..BLEND_WIDTH {
            // Blend weight: 1.0 at border, 0.0 at BLEND_WIDTH
            let alpha = 1.0 - depth as f32 / BLEND_WIDTH as f32;
            let alpha = alpha * alpha * (3.0 - 2.0 * alpha); // smoothstep

            for i in 0..res {
                let j = if rev { res - 1 - i } else { i };

                let (ra, ca) = border_pixel(ba, depth, i, res);
                let (rb, cb) = border_pixel(bb, depth, j, res);

                let idx_a = (ra * res + ca) as usize;
                let idx_b = (rb * res + cb) as usize;

                let ea = faces[fa].elevation[idx_a];
                let eb = faces[fb].elevation[idx_b];
                let avg = (ea + eb) * 0.5;

                // Each face moves halfway toward the average
                faces[fa].elevation[idx_a] = ea + alpha * (avg - ea);
                faces[fb].elevation[idx_b] = eb + alpha * (avg - eb);
            }
        }
    }
}

/// Get (row, col) for a pixel at `depth` inward from `border`, at position `i` along the border.
fn border_pixel(border: u8, depth: u32, i: u32, res: u32) -> (u32, u32) {
    match border {
        0 => (depth, i),             // top: row=depth, col=i
        1 => (i, res - 1 - depth),   // right: row=i, col=max-depth
        2 => (res - 1 - depth, i),   // bottom: row=max-depth, col=i
        3 => (i, depth),             // left: row=i, col=depth
        _ => unreachable!(),
    }
}

// Denormalization constants
const LOWFREQ_MEAN: f32 = -31.4;
const LOWFREQ_STD: f32 = 38.6;
const RESIDUAL_MEAN: f32 = 0.0;
const RESIDUAL_STD: f32 = 1.1678;

/// Combine latent low-frequency elevation + decoder residual into final output.
/// Follows the Java WorldPipeline.computeElev() logic including Laplacian denoise.
fn combine_output(
    coarse: &[f32],
    latent: &[f32],
    residual: &[f32],
    resolution: u32,
) -> FaceHeightmap {
    let coarse_res = 64u32;
    let latent_res = 64u32;
    let latent_px = (latent_res * latent_res) as usize;
    let hi_px = (resolution * resolution) as usize;

    let lowfreq_raw = &latent[4 * latent_px..5 * latent_px];

    let lf_min = lowfreq_raw.iter().cloned().fold(f32::INFINITY, f32::min);
    let lf_max = lowfreq_raw.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let res_min = residual.iter().cloned().fold(f32::INFINITY, f32::min);
    let res_max = residual.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    session::log_to_file(&format!(
        "[combine] latent ch4 raw: [{lf_min:.2}, {lf_max:.2}], residual raw: [{res_min:.2}, {res_max:.2}]"
    ));

    // Denormalize low-frequency (64×64) and residual (resolution×resolution)
    let mut lowfreq = vec![0.0f32; latent_px];
    for i in 0..latent_px {
        lowfreq[i] = lowfreq_raw[i] * LOWFREQ_STD + LOWFREQ_MEAN;
    }
    let residual_denorm: Vec<f32> = residual
        .iter()
        .map(|&v| v * RESIDUAL_STD + RESIDUAL_MEAN)
        .collect();

    // Laplacian denoise: smooth lowfreq to remove artifacts before combining
    // (Java: LaplacianUtils.laplacianDenoise + laplacianDecode)
    let denoised_lowfreq = laplacian_denoise(
        &residual_denorm, resolution,
        &lowfreq, latent_res,
        LAPLACIAN_SIGMA,
    );
    let elevation_sqrt = laplacian_decode(
        &residual_denorm, resolution,
        &denoised_lowfreq, latent_res,
    );

    let es_min = elevation_sqrt.iter().cloned().fold(f32::INFINITY, f32::min);
    let es_max = elevation_sqrt.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    session::log_to_file(&format!(
        "[combine] after laplacian: sqrt-encoded [{es_min:.2}, {es_max:.2}]"
    ));

    // Reverse sqrt encoding: sign(x) * x²
    let elevation: Vec<f32> = elevation_sqrt
        .iter()
        .map(|&es| es.signum() * es * es)
        .collect();

    let e_min = elevation.iter().cloned().fold(f32::INFINITY, f32::min);
    let e_max = elevation.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    session::log_to_file(&format!("[combine] final elevation: [{e_min:.1}, {e_max:.1}] m"));

    // Climate from coarse output
    let coarse_px = (coarse_res * coarse_res) as usize;
    let temperature = bilinear_upsample(
        &coarse[2 * coarse_px..3 * coarse_px],
        coarse_res,
        resolution,
    );
    let precipitation = bilinear_upsample(
        &coarse[4 * coarse_px..5 * coarse_px],
        coarse_res,
        resolution,
    );

    FaceHeightmap {
        elevation,
        temperature,
        precipitation,
        resolution,
    }
}

// ── Laplacian denoise/decode ───────────────────────────────────────────────

const LAPLACIAN_SIGMA: f32 = 5.0;

/// Laplacian denoise: smooth lowfreq to reduce artifacts.
/// Port of Java LaplacianUtils.laplacianDenoise.
fn laplacian_denoise(
    residual: &[f32], res_size: u32,
    lowfreq: &[f32], low_size: u32,
    sigma: f32,
) -> Vec<f32> {
    let hi = res_size as usize;
    let lo = low_size as usize;

    // Step 1: decode with extrapolation — upsample lowfreq (extrapolated) + residual
    let lowfreq_up = bilinear_upsample_extrapolated(lowfreq, low_size, res_size);
    let mut decoded = vec![0.0f32; hi * hi];
    for i in 0..decoded.len() {
        decoded[i] = residual[i] + lowfreq_up[i];
    }

    // Step 2: downsample decoded back to lowfreq resolution
    let downsampled = bilinear_downsample(&decoded, res_size, low_size);

    // Step 3: Gaussian blur
    gaussian_blur_2d(&downsampled, lo, lo, sigma)
}

/// Laplacian decode: upsample lowfreq + residual.
fn laplacian_decode(
    residual: &[f32], res_size: u32,
    lowfreq: &[f32], low_size: u32,
) -> Vec<f32> {
    let lowfreq_up = bilinear_upsample(lowfreq, low_size, res_size);
    residual.iter().zip(&lowfreq_up).map(|(&r, &l)| r + l).collect()
}

/// Bilinear upsample with linear extrapolation padding (1 pixel each side).
/// Matches Java LaplacianUtils.bilinearResizeExtrapolated.
fn bilinear_upsample_extrapolated(src: &[f32], src_size: u32, dst_size: u32) -> Vec<f32> {
    let s = src_size as usize;
    let padded_size = s + 2;
    let mut padded = vec![0.0f32; padded_size * padded_size];

    // Copy interior
    for r in 0..s {
        for c in 0..s {
            padded[(r + 1) * padded_size + (c + 1)] = src[r * s + c];
        }
    }

    // Extrapolate rows (top/bottom)
    for c in 1..=s {
        padded[c] = if s > 1 {
            2.0 * padded[padded_size + c] - padded[2 * padded_size + c]
        } else {
            padded[padded_size + c]
        };
        padded[(s + 1) * padded_size + c] = if s > 1 {
            2.0 * padded[s * padded_size + c] - padded[(s - 1) * padded_size + c]
        } else {
            padded[s * padded_size + c]
        };
    }

    // Extrapolate cols (left/right, on already-padded rows)
    for r in 0..padded_size {
        padded[r * padded_size] = if s > 1 {
            2.0 * padded[r * padded_size + 1] - padded[r * padded_size + 2]
        } else {
            padded[r * padded_size + 1]
        };
        padded[r * padded_size + s + 1] = if s > 1 {
            2.0 * padded[r * padded_size + s] - padded[r * padded_size + s - 1]
        } else {
            padded[r * padded_size + s]
        };
    }

    // Resize padded to include the padding pixels proportionally
    let scale = dst_size as f64 / src_size as f64;
    let pad_px = scale.round() as u32;
    let new_size = dst_size + 2 * pad_px;
    let resized = bilinear_upsample(&padded, padded_size as u32, new_size);

    // Crop center
    let mut result = vec![0.0f32; (dst_size * dst_size) as usize];
    for r in 0..dst_size {
        for c in 0..dst_size {
            result[(r * dst_size + c) as usize] =
                resized[((r + pad_px) * new_size + (c + pad_px)) as usize];
        }
    }
    result
}

/// Bilinear downsample (align_corners=False, PyTorch style).
fn bilinear_downsample(src: &[f32], src_size: u32, dst_size: u32) -> Vec<f32> {
    bilinear_upsample(src, src_size, dst_size)
}

/// Separable Gaussian blur with clamp-to-edge padding.
fn gaussian_blur_2d(src: &[f32], h: usize, w: usize, sigma: f32) -> Vec<f32> {
    let ks = ((sigma * 2.0) as usize / 2) * 2 + 1;
    let half = ks / 2;

    // Build 1D kernel
    let mut kernel = vec![0.0f32; ks];
    let mut sum = 0.0f32;
    for i in 0..ks {
        let x = i as f32 - half as f32;
        kernel[i] = (-0.5 * x * x / (sigma * sigma)).exp();
        sum += kernel[i];
    }
    for k in &mut kernel {
        *k /= sum;
    }

    // Horizontal pass
    let mut tmp = vec![0.0f32; h * w];
    for r in 0..h {
        for c in 0..w {
            let mut v = 0.0f32;
            for ki in 0..ks {
                let cc = (c as i32 + ki as i32 - half as i32).clamp(0, w as i32 - 1) as usize;
                v += src[r * w + cc] * kernel[ki];
            }
            tmp[r * w + c] = v;
        }
    }

    // Vertical pass
    let mut result = vec![0.0f32; h * w];
    for r in 0..h {
        for c in 0..w {
            let mut v = 0.0f32;
            for ki in 0..ks {
                let rr = (r as i32 + ki as i32 - half as i32).clamp(0, h as i32 - 1) as usize;
                v += tmp[rr * w + c] * kernel[ki];
            }
            result[r * w + c] = v;
        }
    }
    result
}

fn bilinear_upsample(src: &[f32], src_res: u32, dst_res: u32) -> Vec<f32> {
    let mut dst = vec![0.0f32; (dst_res * dst_res) as usize];
    let scale = src_res as f64 / dst_res as f64;

    for r in 0..dst_res {
        for c in 0..dst_res {
            let sy = r as f64 * scale;
            let sx = c as f64 * scale;
            let y0 = (sy as u32).min(src_res - 1);
            let y1 = (y0 + 1).min(src_res - 1);
            let x0 = (sx as u32).min(src_res - 1);
            let x1 = (x0 + 1).min(src_res - 1);
            let fy = sy - y0 as f64;
            let fx = sx - x0 as f64;

            let v00 = src[(y0 * src_res + x0) as usize];
            let v10 = src[(y0 * src_res + x1) as usize];
            let v01 = src[(y1 * src_res + x0) as usize];
            let v11 = src[(y1 * src_res + x1) as usize];

            let v = v00 as f64 * (1.0 - fx) * (1.0 - fy)
                + v10 as f64 * fx * (1.0 - fy)
                + v01 as f64 * (1.0 - fx) * fy
                + v11 as f64 * fx * fy;

            dst[(r * dst_res + c) as usize] = v as f32;
        }
    }
    dst
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_generation::coarse_heightmap;
    use crate::world_generation::sphere_geometry::fibonnaci_spiral::SphericalFibonacci;
    use crate::world_generation::sphere_geometry::plate_seed_placement::{
        assign_plates, Adjacency,
    };
    use crate::world_generation::sphere_geometry::spherical_delaunay_triangulation::SphericalDelaunay;

    /// Run with: cargo test --release -- amplify_smoke --ignored --nocapture
    /// Requires ONNX models in data/models/terrain-diffusion-30m/
    #[test]
    #[ignore]
    fn amplify_smoke() {
        let seed = 42u64;
        let point_count = 10_000u32;
        let plate_count = 20u32;

        // Generate coarse heightmap
        let fib = SphericalFibonacci::new(point_count);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, plate_count, seed);
        let adjacency = Adjacency::from_delaunay(points.len(), &del);
        let coarse = coarse_heightmap::generate(&points, &assignment, &adjacency, seed);

        println!("Coarse heightmap generated: {} points", points.len());

        // Amplify
        let model_dir = std::path::Path::new("data/models/terrain-diffusion-30m");
        assert!(
            model_dir.join("coarse_model.onnx").exists(),
            "ONNX models not found in {}", model_dir.display()
        );

        let result = amplify(&coarse, &points, &fib, seed, model_dir);
        match result {
            Ok(terrain) => {
                println!("\nAmplification succeeded!");
                for (i, face) in terrain.faces.iter().enumerate() {
                    let min = face.elevation.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = face.elevation.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    println!(
                        "  Face {i}: {}x{} pixels, elevation [{min:.1}, {max:.1}] m",
                        face.resolution, face.resolution,
                    );
                }
            }
            Err(e) => {
                println!("\nAmplification failed: {e:#}");
                panic!("{e:#}");
            }
        }
    }

    /// Run with: cargo test --release -- amplify_globe_export --ignored --nocapture
    #[test]
    #[ignore]
    fn amplify_globe_export() {
        use crate::voxel::sphere::{self, Face};

        let seed = 42u64;
        let fib = SphericalFibonacci::new(10_000);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, 20, seed);
        let adjacency = Adjacency::from_delaunay(points.len(), &del);
        let coarse = coarse_heightmap::generate(&points, &assignment, &adjacency, seed);

        let model_dir = std::path::Path::new("data/models/terrain-diffusion-30m");
        let terrain = amplify(&coarse, &points, &fib, seed, model_dir)
            .expect("amplification failed");

        let faces = [
            Face::PosX, Face::NegX, Face::PosY, Face::NegY, Face::PosZ, Face::NegZ,
        ];
        let vis_res = 256u32;
        let mut all_positions = String::new();
        let mut all_colors = String::new();
        let mut all_indices = String::new();
        let mut vertex_offset = 0u32;

        for (fi, &face) in faces.iter().enumerate() {
            let fh = &terrain.faces[fi];
            let full_res = fh.resolution;
            let (tu, tv, normal) = sphere::face_basis(face);
            let tu = DVec3::new(tu.x as f64, tu.y as f64, tu.z as f64);
            let tv = DVec3::new(tv.x as f64, tv.y as f64, tv.z as f64);
            let normal = DVec3::new(normal.x as f64, normal.y as f64, normal.z as f64);

            for r in 0..vis_res {
                for c in 0..vis_res {
                    let u = (c as f64 + 0.5) / vis_res as f64 * 2.0 - 1.0;
                    let v = (r as f64 + 0.5) / vis_res as f64 * 2.0 - 1.0;
                    let dir = sphere::cube_to_sphere_unit(tu * u + tv * v + normal).normalize();

                    let fr = (r as f64 / vis_res as f64 * full_res as f64) as u32;
                    let fc = (c as f64 / vis_res as f64 * full_res as f64) as u32;
                    let elev = fh.elevation
                        [(fr.min(full_res - 1) * full_res + fc.min(full_res - 1)) as usize];

                    let scale = 1.0 + (elev as f64) * 0.00001;
                    if vertex_offset > 0 || r > 0 || c > 0 {
                        all_positions.push(',');
                        all_colors.push(',');
                    }
                    all_positions.push_str(&format!(
                        "{:.5},{:.5},{:.5}",
                        dir.x * scale, dir.y * scale, dir.z * scale
                    ));
                    let (cr, cg, cb) = elevation_color_meters(elev);
                    all_colors.push_str(&format!("{cr:.3},{cg:.3},{cb:.3}"));
                }
            }

            for r in 0..vis_res - 1 {
                for c in 0..vis_res - 1 {
                    let tl = vertex_offset + r * vis_res + c;
                    let tr = tl + 1;
                    let bl = tl + vis_res;
                    let br = bl + 1;
                    if !all_indices.is_empty() {
                        all_indices.push(',');
                    }
                    all_indices.push_str(&format!("{tl},{tr},{bl},{tr},{br},{bl}"));
                }
            }
            vertex_offset += vis_res * vis_res;
        }

        // Print ocean/land stats
        let mut total_px = 0u64;
        let mut ocean_px = 0u64;
        for fh in &terrain.faces {
            for &e in &fh.elevation {
                total_px += 1;
                if e < 0.0 { ocean_px += 1; }
            }
            let min = fh.elevation.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = fh.elevation.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let face_ocean = fh.elevation.iter().filter(|&&e| e < 0.0).count();
            let pct = 100.0 * face_ocean as f64 / fh.elevation.len() as f64;
            println!("  Face: [{min:.0}, {max:.0}]m, ocean: {pct:.1}%");
        }
        println!("  Total ocean: {:.1}%", 100.0 * ocean_px as f64 / total_px as f64);

        let info = format!(
            "{vertex_offset} vertices, 6 faces @ {vis_res}x{vis_res} — drag to orbit, scroll to zoom, W for wireframe",
        );
        let html = include_str!("../coarse_heightmap/globe_template.html")
            .replace("POSITIONS_DATA", &all_positions)
            .replace("COLORS_DATA", &all_colors)
            .replace("INDICES_DATA", &all_indices)
            .replace("INFO_TEXT", &info);

        let path = std::path::Path::new(
            "src/world_generation/terrain_amplification/amplified_globe.html",
        );
        std::fs::write(path, &html).expect("failed to write");
        println!(
            "\n  Wrote {} ({:.1} MB)\n",
            path.display(),
            std::fs::metadata(path).unwrap().len() as f64 / 1e6
        );
    }

    fn elevation_color_meters(elev: f32) -> (f32, f32, f32) {
        if elev < -3000.0 {
            (0.04, 0.07, 0.30)               // deep ocean
        } else if elev < -200.0 {
            let t = (elev + 3000.0) / 2800.0;
            (0.04 + t * 0.05, 0.07 + t * 0.12, 0.30 + t * 0.20) // mid ocean
        } else if elev < 0.0 {
            let t = (elev + 200.0) / 200.0;
            (0.09 + t * 0.06, 0.19 + t * 0.11, 0.50 + t * 0.10) // shallow water
        } else if elev < 50.0 {
            let t = elev / 50.0;
            (0.55 + t * 0.10, 0.72 + t * 0.03, 0.45 - t * 0.10) // beach/lowland
        } else if elev < 500.0 {
            let t = (elev - 50.0) / 450.0;
            (0.35 - t * 0.05, 0.62 - t * 0.10, 0.28 - t * 0.03) // plains → hills
        } else if elev < 1500.0 {
            let t = (elev - 500.0) / 1000.0;
            (0.30 + t * 0.25, 0.52 - t * 0.15, 0.25 - t * 0.05) // hills → mountains
        } else if elev < 3000.0 {
            let t = (elev - 1500.0) / 1500.0;
            (0.55 + t * 0.15, 0.37 + t * 0.10, 0.20 + t * 0.15) // high mountains
        } else {
            let t = ((elev - 3000.0) / 3000.0).min(1.0);
            (0.70 + t * 0.25, 0.47 + t * 0.48, 0.35 + t * 0.60) // peaks → snow
        }
    }
}
