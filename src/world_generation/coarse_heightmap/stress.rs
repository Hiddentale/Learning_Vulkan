use glam::DVec3;
use noise::{Fbm, NoiseFn, Perlin};

use crate::world_generation::sphere_geometry::plate_seed_placement::Adjacency;

const STRESS_DECAY: f64 = 0.75;
const STRESS_PROPAGATION_PASSES: usize = 12;
const STRESS_DIR_BLEND_PARENT: f64 = 0.8;
const STRESS_DIR_BLEND_TRAVEL: f64 = 0.2;
const STRESS_DIR_FACTOR_BASE: f64 = 0.3;
const STRESS_DIR_FACTOR_SCALE: f64 = 0.7;
const STRESS_DIR_FACTOR_MIN: f64 = 0.1;

const TERRAIN_CLASS_FREQ: f64 = 1.8;

/// Propagate stress inward from convergent boundaries.
/// Returns (stress, stress_dir): per-point stress in [0, 1] and the collision
/// direction vector (tangent to sphere surface).
pub(super) fn propagate(
    n: usize,
    points: &[DVec3],
    adjacency: &Adjacency,
    plate_ids: &[u32],
    mountain_seeds: &[u32],
    subduction_seeds: &[u32],
) -> (Vec<f32>, Vec<DVec3>) {
    let mut stress = vec![0.0f32; n];
    let mut stress_dir = vec![DVec3::ZERO; n];

    seed_boundaries(
        points, adjacency, plate_ids, mountain_seeds, subduction_seeds,
        &mut stress, &mut stress_dir,
    );

    let mut frontier: Vec<u32> = mountain_seeds
        .iter()
        .chain(subduction_seeds.iter())
        .copied()
        .collect();

    for _ in 0..STRESS_PROPAGATION_PASSES {
        let next = propagate_one_pass(
            &frontier, points, adjacency, plate_ids, &mut stress, &mut stress_dir,
        );
        if next.is_empty() {
            break;
        }
        frontier = next;
    }

    (stress, stress_dir)
}

fn seed_boundaries(
    points: &[DVec3],
    adjacency: &Adjacency,
    plate_ids: &[u32],
    mountain_seeds: &[u32],
    subduction_seeds: &[u32],
    stress: &mut [f32],
    stress_dir: &mut [DVec3],
) {
    for &s in mountain_seeds.iter().chain(subduction_seeds.iter()) {
        stress[s as usize] = 1.0;
        let p = points[s as usize];
        let my_plate = plate_ids[s as usize];
        let mut inward = DVec3::ZERO;
        for &nb in adjacency.neighbors_of(s) {
            if plate_ids[nb as usize] == my_plate {
                inward += points[nb as usize] - p;
            }
        }
        let tangent = inward - p * inward.dot(p);
        stress_dir[s as usize] = tangent.normalize_or_zero();
    }
}

fn propagate_one_pass(
    frontier: &[u32],
    points: &[DVec3],
    adjacency: &Adjacency,
    plate_ids: &[u32],
    stress: &mut [f32],
    stress_dir: &mut [DVec3],
) -> Vec<u32> {
    let mut next = Vec::new();

    for &r in frontier {
        let r_idx = r as usize;
        let my_plate = plate_ids[r_idx];
        let base = stress[r_idx] as f64 * STRESS_DECAY;
        if base < 0.005 {
            continue;
        }
        let dir = stress_dir[r_idx];
        let has_dir = dir.length_squared() > 0.01;

        for &nb in adjacency.neighbors_of(r) {
            let nb_idx = nb as usize;
            if plate_ids[nb_idx] != my_plate {
                continue;
            }

            let mut propagated = base;
            if has_dir {
                let travel = (points[nb_idx] - points[r_idx]).normalize_or_zero();
                let alignment = dir.dot(travel);
                let dir_factor =
                    (STRESS_DIR_FACTOR_BASE + STRESS_DIR_FACTOR_SCALE * alignment)
                        .max(STRESS_DIR_FACTOR_MIN);
                propagated *= dir_factor;
            }

            if propagated > stress[nb_idx] as f64 {
                stress[nb_idx] = propagated as f32;
                next.push(nb);

                if has_dir {
                    let travel = (points[nb_idx] - points[r_idx]).normalize_or_zero();
                    let blended =
                        dir * STRESS_DIR_BLEND_PARENT + travel * STRESS_DIR_BLEND_TRAVEL;
                    stress_dir[nb_idx] = blended.normalize_or_zero();
                }
            }
        }
    }

    next
}

/// Per-point terrain class in [0, 1]: 0 = shield/craton, 1 = sedimentary basin.
pub(super) fn classify_interior_terrain(n: usize, points: &[DVec3], seed: u64) -> Vec<f32> {
    let fbm: Fbm<Perlin> = Fbm::new(seed.wrapping_add(661) as u32);
    let mut class = vec![0.5f32; n];
    for i in 0..n {
        let p = points[i];
        let raw = fbm.get([
            p.x * TERRAIN_CLASS_FREQ + 7.3,
            p.y * TERRAIN_CLASS_FREQ + 3.1,
            p.z * TERRAIN_CLASS_FREQ + 9.7,
        ]);
        class[i] = (0.5 + raw * 0.6).clamp(0.0, 1.0) as f32;
    }
    class
}
