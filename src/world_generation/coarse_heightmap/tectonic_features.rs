use std::f64::consts::PI;

use glam::DVec3;
use noise::{Fbm, NoiseFn, Perlin};

use super::elevation::{gaussian, smoothstep};

// Fold ridges
const FOLD_FREQ_PRIMARY: f64 = 50.0;
const FOLD_FREQ_SECONDARY: f64 = 125.0;
const FOLD_MEAN_OFFSET: f64 = 0.36;
const FOLD_PHASE_WARP_AMP: f64 = 0.08;
const FOLD_AMP_MOD_BASE: f64 = 0.6;
const FOLD_AMP_MOD_SCALE: f64 = 0.4;
const FOLD_SECONDARY_AMP: f64 = 0.18;
const FOLD_ELEV_THRESHOLD: f64 = 0.5;
const FOLD_MAX_AMPLITUDE: f64 = 0.8;

// Rift valleys
const RIFT_AXIS_DEPTH: f64 = -0.18;
const RIFT_FLOOR_DEPTH: f64 = -0.12;
const RIFT_SHOULDER_UPLIFT: f64 = 0.05;
pub(super) const RIFT_HALF_WIDTH: u32 = 6;

// Back-arc basins
const BACK_ARC_START: f64 = 2.0;
const BACK_ARC_PEAK: f64 = 4.0;
const BACK_ARC_END: f64 = 7.0;
const BACK_ARC_DEPTH: f64 = 0.25;

// Plateau
const PLATEAU_START: f64 = 4.0;
const PLATEAU_BOOST: f64 = 0.6;
const PLATEAU_NOISE_SUPPRESS: f64 = 0.3;

// Coastal roughening
const COAST_ROUGHEN_REACH: f64 = 6.0;
const COAST_PASSIVE_AMP: f64 = 0.12;
const COAST_ACTIVE_AMP: f64 = 0.20;
const COAST_ISLAND_THRESHOLD: f64 = 0.55;
const COAST_ISLAND_AMP: f64 = 0.15;

/// Directional fold ridges perpendicular to the collision direction.
pub(super) fn apply_fold_ridges(
    n: usize,
    points: &[DVec3],
    dist_mountain: &[u32],
    is_continental: &[bool],
    stress: &[f32],
    stress_dir: &[DVec3],
    fbm_fold: &Fbm<Perlin>,
    elevation: &mut [f32],
) {
    for i in 0..n {
        if !is_continental[i] {
            continue;
        }
        let dm = dist_mountain[i];
        if dm == u32::MAX || dm > 12 {
            continue;
        }
        let stress_val = stress[i] as f64;
        if stress_val < 0.02 {
            continue;
        }
        let elev = elevation[i] as f64;
        if elev < FOLD_ELEV_THRESHOLD {
            continue;
        }

        let p = points[i];
        let dir = stress_dir[i];
        if dir.length_squared() < 0.01 {
            continue;
        }

        // Ridge strike = perpendicular to stress direction in tangent plane
        let normal = p.normalize();
        let strike = normal.cross(dir).normalize_or_zero();
        if strike.length_squared() < 0.01 {
            continue;
        }

        let u = p.dot(strike);

        // Primary fold system: 1 - |sin(phase)| gives sharp ridges
        let phase_warp =
            fbm_fold.get([p.x * 3.0 + 55.3, p.y * 3.0 + 33.7, p.z * 3.0 + 17.2])
                * FOLD_PHASE_WARP_AMP;
        let phase = (u + phase_warp) * FOLD_FREQ_PRIMARY * PI;
        let ridge = 1.0 - phase.sin().abs();
        let fold_centered = ridge - FOLD_MEAN_OFFSET;

        let amp_mod = FOLD_AMP_MOD_BASE
            + fbm_fold.get([p.x * 4.0 + 88.1, p.y * 4.0 + 62.3, p.z * 4.0 + 41.7])
                * FOLD_AMP_MOD_SCALE;

        let elev_boost = (elev - FOLD_ELEV_THRESHOLD).max(0.0) * 2.0;
        let fold_amp = (stress_val * elev_boost * FOLD_MAX_AMPLITUDE).min(FOLD_MAX_AMPLITUDE);

        let mut fold_contrib = fold_centered * fold_amp * amp_mod;

        // Secondary fold: higher frequency, slight cross-grain
        let u2 = 0.85 * u + 0.15 * p.dot(dir);
        let phase_warp2 =
            fbm_fold.get([p.x * 5.0 + 71.2, p.y * 5.0 + 19.8, p.z * 5.0 + 43.6])
                * FOLD_PHASE_WARP_AMP
                * 1.5;
        let phase2 = (u2 + phase_warp2) * FOLD_FREQ_SECONDARY * PI;
        let ridge2 = 1.0 - phase2.sin().abs();
        let fold2_centered = ridge2 - FOLD_MEAN_OFFSET;

        let amp_mod2 = 0.5
            + fbm_fold.get([p.x * 6.0 + 33.4, p.y * 6.0 + 77.1, p.z * 6.0 + 52.9]) * 0.5;
        fold_contrib += fold2_centered * fold_amp * amp_mod2 * FOLD_SECONDARY_AMP;

        elevation[i] += fold_contrib as f32;
    }
}

pub(super) fn apply_rift_valleys(
    n: usize,
    points: &[DVec3],
    dist_rift: &[u32],
    is_continental: &[bool],
    fbm_rift: &Fbm<Perlin>,
    elevation: &mut [f32],
) {
    let floor_end = (RIFT_HALF_WIDTH as f64 * 0.4).max(1.0);
    let shoulder_end = (RIFT_HALF_WIDTH as f64 * 0.7).max(2.0);

    for i in 0..n {
        let rd = dist_rift[i];
        if rd == u32::MAX || !is_continental[i] {
            continue;
        }
        let rd = rd as f64;
        let p = points[i];

        let rift_effect = if rd <= 0.5 {
            let volcanic = fbm_rift.get([p.x * 8.0, p.y * 8.0, p.z * 8.0]).abs();
            RIFT_AXIS_DEPTH + volcanic * 0.06
        } else if rd <= floor_end {
            let t = rd / floor_end;
            let volcanic = fbm_rift.get([p.x * 8.0, p.y * 8.0, p.z * 8.0]).abs();
            RIFT_FLOOR_DEPTH * (1.0 - t * 0.3) + volcanic * 0.03 * (1.0 - t)
        } else if rd <= shoulder_end {
            let t = (rd - floor_end) / (shoulder_end - floor_end);
            RIFT_SHOULDER_UPLIFT * (1.0 - t)
        } else {
            let t = ((rd - shoulder_end) / (RIFT_HALF_WIDTH as f64 - shoulder_end)).min(1.0);
            let fade = t * t * (3.0 - 2.0 * t);
            RIFT_SHOULDER_UPLIFT * (1.0 - fade) * 0.2
        };

        elevation[i] += rift_effect as f32;
    }
}

pub(super) fn apply_back_arc_basins(
    n: usize,
    dist_subduction: &[u32],
    dist_mountain: &[u32],
    is_continental: &[bool],
    stress: &[f32],
    elevation: &mut [f32],
) {
    for i in 0..n {
        if !is_continental[i] {
            continue;
        }
        let ds = dist_subduction[i];
        if ds == u32::MAX {
            continue;
        }
        let ds = ds as f64;
        if ds < BACK_ARC_START || ds > BACK_ARC_END {
            continue;
        }

        let dm = dist_mountain[i] as f64;
        let orogeny_factor = if dm < ds && dist_mountain[i] != u32::MAX {
            (dm / ds).max(0.0)
        } else {
            1.0
        };

        let stress_mod = (stress[i] as f64).max(0.1);
        let profile = if ds <= BACK_ARC_PEAK {
            let t = (ds - BACK_ARC_START) / (BACK_ARC_PEAK - BACK_ARC_START);
            t * t * (3.0 - 2.0 * t)
        } else {
            let t = (ds - BACK_ARC_PEAK) / (BACK_ARC_END - BACK_ARC_PEAK);
            let s = t * t * (3.0 - 2.0 * t);
            1.0 - s
        };

        elevation[i] -= (BACK_ARC_DEPTH * profile * stress_mod * orogeny_factor) as f32;
    }
}

pub(super) fn apply_plateaus(
    n: usize,
    dist_mountain: &[u32],
    dist_coast: &[u32],
    is_continental: &[bool],
    stress: &[f32],
    elevation: &mut [f32],
) {
    for i in 0..n {
        if !is_continental[i] {
            continue;
        }
        let dm = dist_mountain[i];
        if dm == u32::MAX {
            continue;
        }
        let dm = dm as f64;
        let dc = dist_coast[i] as f64;

        if dm > PLATEAU_START && dc > 5.0 {
            let stress_val = stress[i] as f64;
            if stress_val > 0.05 {
                let boost = PLATEAU_BOOST * stress_val;
                let suppress = 1.0 - PLATEAU_NOISE_SUPPRESS * stress_val;
                let current = elevation[i] as f64;
                let mean = current.max(0.5);
                elevation[i] = (mean * suppress + boost + current * (1.0 - suppress)) as f32;
            }
        }
    }
}

pub(super) fn apply_coastal_roughening(
    n: usize,
    points: &[DVec3],
    dist_coast: &[u32],
    is_continental: &[bool],
    stress: &[f32],
    fbm_coast: &Fbm<Perlin>,
    elevation: &mut [f32],
) {
    for i in 0..n {
        let dc = dist_coast[i];
        if dc == u32::MAX {
            continue;
        }
        let dc = dc as f64;
        if dc > COAST_ROUGHEN_REACH {
            continue;
        }
        let p = points[i];
        let t = dc / COAST_ROUGHEN_REACH;
        let falloff = (1.0 - t) * (1.0 - t);
        let stress_val = stress[i] as f64;

        let is_active = stress_val > 0.1;
        let amp = if is_active { COAST_ACTIVE_AMP } else { COAST_PASSIVE_AMP };
        let freq = if is_active { 9.0 } else { 6.0 };

        let coast_noise =
            fbm_coast.get([p.x * freq + 3.7, p.y * freq + 7.1, p.z * freq + 2.3]);
        elevation[i] += (coast_noise * amp * falloff * (1.0 + stress_val * 2.0)) as f32;

        // Coastal island scattering
        if !is_continental[i] && dc > 0.0 && dc <= 4.0 {
            let island_noise =
                fbm_coast.get([p.x * 8.0 + 5.1, p.y * 8.0 + 9.3, p.z * 8.0 + 2.7]);
            if island_noise > COAST_ISLAND_THRESHOLD {
                let excess =
                    (island_noise - COAST_ISLAND_THRESHOLD) / (1.0 - COAST_ISLAND_THRESHOLD);
                let dist_fade = 1.0 - dc / 4.0;
                let bump = excess * excess * COAST_ISLAND_AMP * dist_fade;
                if elevation[i] + bump as f32 > -0.1 {
                    elevation[i] += bump as f32;
                }
            }
        }
    }
}
