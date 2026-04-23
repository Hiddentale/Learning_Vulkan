/// Stage 3: Decoder model — single flow-matching step.
/// Upsamples latent channels from 64×64 to 256×256, runs one inference step,
/// and produces the final elevation residual.

use anyhow::Result;

use super::scheduler;
use super::session::ModelSession;
use super::tiling::{self, BlendGrid};

const TILE_SIZE: u32 = 512;
const TILE_STRIDE: u32 = 384;
const LATENT_COMPRESSION: u32 = 8;
const INPUT_CHANNELS: usize = 5;
const OUTPUT_CHANNELS: u32 = 1;

pub(super) fn run(
    model: &mut ModelSession,
    latent: &[f32],
    latent_w: u32,
    latent_h: u32,
    output_w: u32,
    output_h: u32,
    seed: u64,
) -> Result<Vec<f32>> {
    let positions = tiling::tile_positions_rect(output_w, output_h, TILE_SIZE, TILE_STRIDE);
    let window = tiling::linear_weight_window(TILE_SIZE);
    let mut blend = BlendGrid::new(OUTPUT_CHANNELS, output_w, output_h);

    for &(tx, ty) in &positions {
        let tile = run_tile(model, latent, latent_w, latent_h, tx, ty, seed)?;
        blend.blend_tile(&tile, TILE_SIZE, tx, ty, &window);
    }

    blend.finalize();
    Ok(blend.data)
}

fn run_tile(
    model: &mut ModelSession,
    latent: &[f32],
    latent_w: u32,
    latent_h: u32,
    tx: u32,
    ty: u32,
    seed: u64,
) -> Result<Vec<f32>> {
    let s = TILE_SIZE as usize;
    let pixel_count = s * s;

    let mut rng = super::splitmix64(seed ^ 0xDEAD ^ ((tx as u64) << 32 | ty as u64));
    let noise: Vec<f32> = (0..pixel_count)
        .map(|_| {
            rng = super::splitmix64(rng);
            normal_from_rng(rng)
        })
        .collect();

    let t_init = scheduler::flow_matching_t_init();
    let sigma_data = scheduler::sigma_data() as f32;

    // x_T = sin(t) * σ_data * noise
    let init_scale = (t_init.sin() * scheduler::sigma_data()) as f32;
    let init_noise: Vec<f32> = noise.iter().map(|&n| init_scale * n).collect();

    let upsampled = upsample_latent(latent, latent_w, latent_h, tx, ty);

    // Stack: [noise/σ_data, upsampled_latent(4ch)] → (1, 5, S, S)
    let mut input = vec![0.0f32; INPUT_CHANNELS * pixel_count];
    for i in 0..pixel_count {
        input[i] = init_noise[i] / sigma_data;
    }
    for ch in 0..4 {
        let src_off = ch * pixel_count;
        let dst_off = (ch + 1) * pixel_count;
        input[dst_off..dst_off + pixel_count]
            .copy_from_slice(&upsampled[src_off..src_off + pixel_count]);
    }

    let model_output = model.run(vec![
        ("x", vec![1, INPUT_CHANNELS, s, s], input),
        ("noise_labels", vec![1], vec![t_init as f32]),
    ])?;

    Ok(scheduler::flow_matching_step(&init_noise, &model_output, t_init))
}

/// Nearest-neighbor upsample latent to decoder tile size.
fn upsample_latent(latent: &[f32], lat_w: u32, lat_h: u32, tx: u32, ty: u32) -> Vec<f32> {
    let s = TILE_SIZE as usize;
    let lc = LATENT_COMPRESSION as usize;
    let slc = s / lc;
    let pixel_count = s * s;
    let lat_plane = (lat_w * lat_h) as usize;
    let mut upsampled = vec![0.0f32; 4 * pixel_count];

    let lx = tx as usize / lc;
    let ly = ty as usize / lc;

    for ch in 0..4usize {
        let lat_ch_offset = ch * lat_plane;
        for r in 0..s {
            let sr = r * slc / s;
            let src_r = (ly + sr).min(lat_h as usize - 1);
            for c in 0..s {
                let sc = c * slc / s;
                let src_c = (lx + sc).min(lat_w as usize - 1);
                let src = lat_ch_offset + src_r * lat_w as usize + src_c;
                upsampled[ch * pixel_count + r * s + c] = latent[src];
            }
        }
    }

    upsampled
}

fn normal_from_rng(state: u64) -> f32 {
    let u1 = (state & 0xFFFFFFFF) as f64 / 0xFFFFFFFF_u64 as f64;
    let u2 = (state >> 32) as f64 / 0xFFFFFFFF_u64 as f64;
    let u1 = u1.max(1e-10);
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}
