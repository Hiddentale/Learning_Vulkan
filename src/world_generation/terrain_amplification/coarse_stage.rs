/// Stage 1: Coarse model — 20-step DPM-Solver++ denoising.
/// Takes conditioning from the rasterized coarse heightmap, produces 6-channel
/// coarse climate+elevation output at 64×64 per tile.

use anyhow::Result;

use super::rasterize::FaceGrid;
use super::scheduler::{self, DpmSolverState};
use super::session::ModelSession;
use super::synthetic_cond::SyntheticConditioner;
use super::tiling::{self, BlendGrid};

const TILE_SIZE: u32 = 64;
const TILE_STRIDE: u32 = 48;
const NUM_STEPS: usize = 20;
const OUTPUT_CHANNELS: u32 = 6;

/// Conditioning normalization uses MODEL output stats at indices [0,2,3,4,5],
/// matching Java WorldPipeline.coarseTile() — NOT the COND_MEANS used in latent stage.
const COND_NORM_MEANS: [f64; 5] = [-37.679, 18.030, 333.844, 1350.126, 52.444];
const COND_NORM_STDS: [f64; 5] = [39.685, 8.940, 322.252, 856.343, 30.983];
const COND_SNR: [f64; 5] = [0.3, 0.1, 1.0, 0.1, 1.0];

// Coarse model output denormalization (6 channels)
const MODEL_MEANS: [f32; 6] = [-37.679, 2.226, 18.030, 333.844, 1350.126, 52.444];
const MODEL_STDS: [f32; 6] = [39.685, 3.098, 8.940, 322.252, 856.343, 30.983];

pub(super) fn run(
    model: &mut ModelSession,
    face: &FaceGrid,
    cond: &SyntheticConditioner,
    seed: u64,
) -> Result<Vec<f32>> {
    let s = TILE_SIZE;
    let positions = tiling::tile_positions_rect(face.resolution, face.height, s, TILE_STRIDE);
    let window = tiling::linear_weight_window(s);
    let mut blend = BlendGrid::new(OUTPUT_CHANNELS, face.resolution, face.height);

    for &(tx, ty) in &positions {
        let tile_output = run_tile(model, face, cond, tx, ty, seed)?;
        blend.blend_tile(&tile_output, s, tx, ty, &window);
    }

    blend.finalize();
    Ok(blend.data)
}

fn run_tile(
    model: &mut ModelSession,
    face: &FaceGrid,
    cond: &SyntheticConditioner,
    tx: u32,
    ty: u32,
    seed: u64,
) -> Result<Vec<f32>> {
    let s = TILE_SIZE as usize;
    let conditioning = extract_conditioning(face, cond, tx, ty, seed);

    let noise_channels = 6;
    let sample_size = noise_channels * s * s;
    let mut rng = super::splitmix64(seed ^ ((tx as u64) << 32 | ty as u64));
    let mut sample: Vec<f32> = (0..sample_size)
        .map(|_| {
            rng = super::splitmix64(rng);
            normal_from_rng(rng)
        })
        .collect();

    let sigma_max = 80.0f32;
    for v in &mut sample {
        *v *= sigma_max;
    }

    let cond_scalars: Vec<f32> = COND_SNR
        .iter()
        .map(|&snr| (snr / 8.0).ln() as f32)
        .collect();

    let mut solver = DpmSolverState::new(NUM_STEPS);
    while !solver.is_done() {
        let pc = scheduler::precondition(solver.current_sigma());
        let c_in = pc.c_in as f32;
        let scaled: Vec<f32> = sample.iter().map(|&v| v * c_in).collect();

        // Stack: [scaled_sample(6ch), conditioning(5ch)] → (1, 11, S, S)
        let mut input = vec![0.0f32; 11 * s * s];
        input[..6 * s * s].copy_from_slice(&scaled);
        input[6 * s * s..].copy_from_slice(&conditioning);

        let mut inputs = vec![
            ("x", vec![1, 11, s, s], input),
            ("noise_labels", vec![1], vec![pc.c_noise]),
        ];
        for (i, &val) in cond_scalars.iter().enumerate() {
            inputs.push((
                match i {
                    0 => "cond_0",
                    1 => "cond_1",
                    2 => "cond_2",
                    3 => "cond_3",
                    _ => "cond_4",
                },
                vec![1],
                vec![val],
            ));
        }

        let model_output = model.run(inputs)?;
        sample = solver.step(&sample, &model_output);
    }

    // Denormalize: sample / σ_data * MODEL_STDS + MODEL_MEANS per channel
    let sigma_data = scheduler::sigma_data() as f32;
    let pixels = s * s;
    for ch in 0..6 {
        let offset = ch * pixels;
        for px in 0..pixels {
            let raw = sample[offset + px] / sigma_data;
            sample[offset + px] = raw * MODEL_STDS[ch] + MODEL_MEANS[ch];
        }
    }

    // p5 transformation: ch1 = ch0 - ch1 (Java WorldPipeline.java line 205)
    for px in 0..pixels {
        sample[pixels + px] = sample[px] - sample[pixels + px];
    }

    Ok(sample)
}

fn extract_conditioning(
    face: &FaceGrid,
    synth: &SyntheticConditioner,
    tx: u32,
    ty: u32,
    seed: u64,
) -> Vec<f32> {
    let s = TILE_SIZE as usize;
    let pixels = s * s;

    // Generate synthetic conditioning merged with coarse heightmap data.
    // Returns [elev_sqrt, temp, temp_std, precip, precip_cv] in physical units.
    let raw = synth.generate_conditioning(face, tx, ty, TILE_SIZE, TILE_SIZE);

    // Normalize with MODEL_MEANS/MODEL_STDS at indices [0,2,3,4,5]
    let mut cond = vec![0.0f32; 5 * pixels];
    for px in 0..pixels {
        for ch in 0..5 {
            cond[ch * pixels + px] =
                ((raw[ch * pixels + px] as f64 - COND_NORM_MEANS[ch]) / COND_NORM_STDS[ch]) as f32;
        }
    }

    // Log conditioning stats before noise mixing
    for ch in 0..5 {
        let slice = &cond[ch * pixels..(ch + 1) * pixels];
        let mn = slice.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        super::session::log_to_file(&format!(
            "[cond] ch{ch} normalized range: [{mn:.3}, {mx:.3}]"
        ));
    }

    // Mix conditioning noise: cond_mixed = cos(atan(SNR)) * cond + sin(atan(SNR)) * noise
    let mut noise_rng = super::splitmix64(seed ^ ((tx as u64) << 16 | ty as u64 | 0xCAFE_0000));
    for ch in 0..5usize {
        let cos_t = (COND_SNR[ch].atan()).cos() as f32;
        let sin_t = (COND_SNR[ch].atan()).sin() as f32;
        for px in 0..pixels {
            noise_rng = super::splitmix64(noise_rng);
            let noise = normal_from_rng(noise_rng);
            let idx = ch * pixels + px;
            cond[idx] = cos_t * cond[idx] + sin_t * noise;
        }
    }

    cond
}

fn normal_from_rng(state: u64) -> f32 {
    let u1 = (state & 0xFFFFFFFF) as f64 / 0xFFFFFFFF_u64 as f64;
    let u2 = (state >> 32) as f64 / 0xFFFFFFFF_u64 as f64;
    let u1 = u1.max(1e-10);
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}
