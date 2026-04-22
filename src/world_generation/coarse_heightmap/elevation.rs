use glam::DVec3;
use noise::{Fbm, NoiseFn, Perlin};

const CONTINENTAL_SHELF: f64 = 0.05;
const MOUNTAIN_BASE_PEAK: f64 = 3.5;
const MOUNTAIN_NOISE_RANGE: f64 = 3.5;
const MOUNTAIN_SIGMA: f64 = 5.0;
const FORELAND_DEPTH: f64 = 0.3;
const FORELAND_WIDTH: f64 = 10.0;
const COAST_RAMP_HOPS: f64 = 15.0;
const BASIN_THRESHOLD: f64 = -0.6;
const BASIN_DEPTH: f64 = 0.4;
const SHIELD_BASE: f64 = 0.5;
const BASIN_INTERIOR_BASE: f64 = 0.15;

const OCEAN_SHELF_DEPTH: f64 = -0.2;
const OCEAN_ABYSS_DEPTH: f64 = -4.2;
const RIDGE_CREST: f64 = -1.0;
const RIDGE_SIGMA: f64 = 4.0;
const ISLAND_ARC_PEAK: f64 = 0.8;
const ISLAND_ARC_SIGMA: f64 = 1.5;
const ISLAND_ARC_OFFSET: f64 = 2.0;
const ISLAND_ARC_THRESHOLD: f64 = 0.3;
const TRENCH_DEPTH: f64 = -6.0;
const TRENCH_WIDTH: f64 = 3.0;

pub(super) fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

pub(super) fn gaussian(x: f64, sigma: f64) -> f64 {
    (-0.5 * (x / sigma).powi(2)).exp()
}

pub(super) fn continental(
    dist_coast: f64,
    dist_mountain: f64,
    dist_subduction: f64,
    p: DVec3,
    noise_val: f64,
    stress_val: f64,
    terrain_class: f32,
    fbm_mountain: &Fbm<Perlin>,
    fbm_basin: &Fbm<Perlin>,
) -> f64 {
    let tc = terrain_class as f64;

    // Interior base: shields are higher and more rugged, basins are low and flat
    let interior_base = SHIELD_BASE * (1.0 - tc) + BASIN_INTERIOR_BASE * tc;
    let coast_profile = smoothstep(0.0, COAST_RAMP_HOPS, dist_coast);
    let noise_scale = 1.0 - tc * 0.5;
    let base =
        CONTINENTAL_SHELF + interior_base * coast_profile + noise_val * 0.3 * noise_scale;

    // Himalaya-style mountains driven by stress field
    let mountain_noise = fbm_mountain.get([p.x * 6.0, p.y * 6.0, p.z * 6.0]);
    let mountain_peak = MOUNTAIN_BASE_PEAK + MOUNTAIN_NOISE_RANGE * mountain_noise.abs();
    let effective_sigma = MOUNTAIN_SIGMA * (1.0 + stress_val * 0.5);
    let mountain_falloff = gaussian(dist_mountain, effective_sigma);
    let mountain_elev = mountain_peak * mountain_falloff * (0.3 + 0.7 * stress_val);

    // Andes-style subduction mountains — narrower, stress-modulated
    let sub_noise =
        fbm_mountain.get([p.x * 8.0 + 10.0, p.y * 8.0 + 10.0, p.z * 8.0 + 10.0]);
    let sub_peak = 2.0 + 2.0 * sub_noise.abs();
    let sub_falloff = gaussian(dist_subduction, 3.0) * (0.3 + 0.7 * stress_val);
    let sub_elev = sub_peak * sub_falloff;

    let mut elev = base.max(mountain_elev).max(sub_elev);

    // Foreland basin: depression ahead of mountain front
    if dist_mountain < FORELAND_WIDTH && dist_mountain > 2.0 {
        let t = dist_mountain / FORELAND_WIDTH;
        let peak_pos = 0.2;
        let profile = if t < peak_pos {
            let s = t / peak_pos;
            s * s * (3.0 - 2.0 * s)
        } else {
            let s = (t - peak_pos) / (1.0 - peak_pos);
            1.0 - s * s * (3.0 - 2.0 * s)
        };
        elev -= FORELAND_DEPTH * profile * (0.5 + 0.5 * stress_val);
    }

    // Inland seas: noise-driven basins in deep interior
    if dist_coast > 8.0 {
        let basin_noise =
            fbm_basin.get([p.x * 3.0 + 7.3, p.y * 3.0 + 3.1, p.z * 3.0 + 9.7]);
        if basin_noise < BASIN_THRESHOLD {
            let depression =
                BASIN_DEPTH * (basin_noise - BASIN_THRESHOLD) / (1.0 + BASIN_THRESHOLD.abs());
            elev = (elev + depression).max(-0.05);
        }
    }

    elev
}

pub(super) fn oceanic(
    dist_coast: f64,
    dist_ridge: f64,
    dist_arc: f64,
    dist_subduction: f64,
    p: DVec3,
    noise_val: f64,
    fbm_arc: &Fbm<Perlin>,
) -> f64 {
    let depth_profile = smoothstep(0.0, 20.0, dist_coast);
    let mut elev = OCEAN_SHELF_DEPTH
        + (OCEAN_ABYSS_DEPTH - OCEAN_SHELF_DEPTH) * depth_profile
        + noise_val * 0.2;

    // Mid-ocean ridge
    let ridge_falloff = gaussian(dist_ridge, RIDGE_SIGMA);
    elev = RIDGE_CREST * ridge_falloff + elev * (1.0 - ridge_falloff);

    // Island arcs at oceanic convergent boundaries
    let arc_noise = fbm_arc.get([p.x * 8.0, p.y * 8.0, p.z * 8.0]);
    if dist_arc < 6.0 && arc_noise > ISLAND_ARC_THRESHOLD {
        let arc_profile = gaussian(dist_arc - ISLAND_ARC_OFFSET, ISLAND_ARC_SIGMA);
        let island_elev =
            (0.1 + ISLAND_ARC_PEAK * (arc_noise - ISLAND_ARC_THRESHOLD)) * arc_profile;
        elev = elev.max(island_elev);
    }

    // Trench at subduction zones
    if dist_subduction < TRENCH_WIDTH {
        let trench_profile = 1.0 - dist_subduction / TRENCH_WIDTH;
        let trench_elev = TRENCH_DEPTH * trench_profile + elev * (1.0 - trench_profile);
        elev = elev.min(trench_elev);
    }

    elev
}
