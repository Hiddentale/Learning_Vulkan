/// Stage 2: Latent model — 2-step flow-matching.
/// Takes coarse output, constructs a 58-dim conditioning vector, and produces
/// a 5-channel latent representation at 64×64 per tile.

use anyhow::Result;

use super::scheduler;
use super::session::ModelSession;
use super::tiling::{self, BlendGrid};

const TILE_SIZE: u32 = 64;
const TILE_STRIDE: u32 = 32;
const OUTPUT_CHANNELS: u32 = 5;
const COND_DIM: usize = 58;

const COND_MEANS: [f64; 7] = [14.99, 11.65, 15.87, 619.26, 833.12, 69.40, 0.66];
const COND_STDS: [f64; 7] = [21.72, 21.78, 10.40, 452.29, 738.09, 34.59, 0.47];
const COND_DIMS: [usize; 6] = [16, 16, 4, 16, 5, 1];
const T_INTER: f64 = 0.611;

pub(super) fn run(
    model: &mut ModelSession,
    coarse_output: &[f32],
    coarse_res: u32,
    output_res: u32,
    seed: u64,
) -> Result<Vec<f32>> {
    let positions = tiling::tile_positions(output_res, TILE_SIZE, TILE_STRIDE);
    let window = tiling::linear_weight_window(TILE_SIZE);
    let mut blend = BlendGrid::new(OUTPUT_CHANNELS, output_res, output_res);

    for &(tx, ty) in &positions {
        let cond = build_conditioning(coarse_output, coarse_res, tx, ty);
        let tile = run_tile(model, &cond, seed, tx, ty)?;
        blend.blend_tile(&tile, TILE_SIZE, tx, ty, &window);
    }

    blend.finalize();
    Ok(blend.data)
}

fn run_tile(
    model: &mut ModelSession,
    cond_58: &[f32],
    seed: u64,
    tx: u32,
    ty: u32,
) -> Result<Vec<f32>> {
    let s = TILE_SIZE as usize;
    let channels = OUTPUT_CHANNELS as usize;
    let pixel_count = s * s;

    let mut rng = super::splitmix64(seed ^ 0xBEEF ^ ((tx as u64) << 32 | ty as u64));
    let noise: Vec<f32> = (0..channels * pixel_count)
        .map(|_| {
            rng = super::splitmix64(rng);
            normal_from_rng(rng)
        })
        .collect();

    let mut sample = scheduler::flow_matching_init(&noise);
    let t_init = scheduler::flow_matching_t_init();
    let sigma_data = scheduler::sigma_data() as f32;

    // Step 1: t_init → t_inter
    // model_in = xT / sigma_data
    {
        let x_in: Vec<f32> = sample.iter().map(|&v| v / sigma_data).collect();
        let model_output = model.run(vec![
            ("x", vec![1, channels, s, s], x_in),
            ("noise_labels", vec![1], vec![t_init as f32]),
            ("cond_0", vec![1, COND_DIM], cond_58.to_vec()),
        ])?;
        sample = scheduler::flow_matching_step(&sample, &model_output, t_init);
    }

    // Reinitialize at t_inter:
    // Java: xT[k] = cosT * sample[k] + sinT * (noise[k] * SIGMA_DATA)
    // σ_data multiplies noise only, NOT the sample
    {
        let t = T_INTER;
        let cos_t = t.cos() as f32;
        let sin_t = t.sin() as f32;
        let mut rng2 = super::splitmix64(rng);
        for i in 0..sample.len() {
            rng2 = super::splitmix64(rng2);
            let n = normal_from_rng(rng2);
            sample[i] = cos_t * sample[i] + sin_t * sigma_data * n;
        }
    }

    // Step 2: t_inter → 0
    {
        let x_in: Vec<f32> = sample.iter().map(|&v| v / sigma_data).collect();
        let model_output = model.run(vec![
            ("x", vec![1, channels, s, s], x_in),
            ("noise_labels", vec![1], vec![T_INTER as f32]),
            ("cond_0", vec![1, COND_DIM], cond_58.to_vec()),
        ])?;
        sample = scheduler::flow_matching_step(&sample, &model_output, T_INTER);
    }

    Ok(sample)
}

fn build_conditioning(coarse: &[f32], res: u32, tx: u32, ty: u32) -> Vec<f32> {
    let s = TILE_SIZE;
    let cx = tx + s / 2 - 2;
    let cy = ty + s / 2 - 2;

    let mut raw = vec![0.0f64; 7 * 16];
    for ch in 0..6u32 {
        for r in 0..4u32 {
            for c in 0..4u32 {
                let sy = (cy + r).min(res - 1);
                let sx = (cx + c).min(res - 1);
                let src = (ch * res * res + sy * res + sx) as usize;
                raw[(ch as usize) * 16 + (r * 4 + c) as usize] = coarse[src] as f64;
            }
        }
    }
    // Mask channel: pre-normalized value (Java: maskNorm = (1 - COND_MEANS[6]) / COND_STDS[6])
    let mask_norm = (1.0 - COND_MEANS[6]) / COND_STDS[6];
    for i in 0..16 {
        raw[6 * 16 + i] = mask_norm;
    }

    // Normalize channels 0-5 with COND_MEANS/STDS; channel 6 already normalized above
    let mut normalized = vec![0.0f64; 7 * 16];
    for ch in 0..6 {
        for px in 0..16 {
            let v = (raw[ch * 16 + px] - COND_MEANS[ch]) / COND_STDS[ch];
            normalized[ch * 16 + px] = if v.is_nan() { 0.0 } else { v };
        }
    }
    // Copy pre-normalized mask channel
    for px in 0..16 {
        normalized[6 * 16 + px] = raw[6 * 16 + px];
    }

    let mut climate_means = [0.0f64; 4];
    for ch in 0..4 {
        let base = (ch + 2) * 16;
        climate_means[ch] = (normalized[base + 5]
            + normalized[base + 6]
            + normalized[base + 9]
            + normalized[base + 10])
            / 4.0;
    }

    let noise_level_norm = (0.0 - 0.5) * 12.0_f64.sqrt();
    let total_dim: usize = COND_DIMS.iter().sum();
    let c_global = ((total_dim * COND_DIMS.len()) as f64).sqrt();
    let scales: Vec<f64> = COND_DIMS
        .iter()
        .map(|&d| c_global / (d as f64).sqrt() / COND_DIMS.len() as f64)
        .collect();

    let mut cond = vec![0.0f32; COND_DIM];
    let mut o = 0;

    for i in 0..16 {
        cond[o + i] = (normalized[i] * scales[0]) as f32;
    }
    o += 16;
    for i in 0..16 {
        cond[o + i] = (normalized[16 + i] * scales[1]) as f32;
    }
    o += 16;
    for i in 0..4 {
        cond[o + i] = (climate_means[i] * scales[2]) as f32;
    }
    o += 4;
    for i in 0..16 {
        cond[o + i] = (normalized[6 * 16 + i] * scales[3]) as f32;
    }
    o += 16;
    // 52-56: histogram (zeros)
    o += 5;
    cond[o] = (noise_level_norm * scales[5]) as f32;

    cond
}

fn normal_from_rng(state: u64) -> f32 {
    let u1 = (state & 0xFFFFFFFF) as f64 / 0xFFFFFFFF_u64 as f64;
    let u2 = (state >> 32) as f64 / 0xFFFFFFFF_u64 as f64;
    let u1 = u1.max(1e-10);
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}
