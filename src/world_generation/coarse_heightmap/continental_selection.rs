use std::collections::HashSet;

use glam::DVec3;

use crate::world_generation::sphere_geometry::plate_seed_placement::Adjacency;

const CONTINENTAL_FRACTION: f64 = 0.30;
const NUM_CONTINENTS: usize = 3;

pub(super) fn build_plate_adjacency(
    adjacency: &Adjacency,
    plate_ids: &[u32],
    plate_count: usize,
) -> Vec<HashSet<u32>> {
    let mut adj: Vec<HashSet<u32>> = vec![HashSet::new(); plate_count];
    for (i, &pid) in plate_ids.iter().enumerate() {
        for &nb in adjacency.neighbors_of(i as u32) {
            let nb_pid = plate_ids[nb as usize];
            if pid != nb_pid {
                adj[pid as usize].insert(nb_pid);
                adj[nb_pid as usize].insert(pid);
            }
        }
    }
    adj
}

pub(super) fn compute_plate_sizes(plate_ids: &[u32], plate_count: usize) -> Vec<usize> {
    let mut sizes = vec![0usize; plate_count];
    for &pid in plate_ids {
        sizes[pid as usize] += 1;
    }
    sizes
}

pub(super) fn compute_plate_centroids(
    points: &[DVec3],
    plate_ids: &[u32],
    plate_count: usize,
) -> Vec<DVec3> {
    let mut sums = vec![DVec3::ZERO; plate_count];
    for (i, &pid) in plate_ids.iter().enumerate() {
        sums[pid as usize] += points[i];
    }
    sums.iter().map(|s| s.normalize_or_zero()).collect()
}

/// Multi-seed continent selection: pick NUM_CONTINENTS well-separated nuclei,
/// then grow each by claiming adjacent plates round-robin until ~30% coverage.
pub(super) fn pick_continental_plates(
    plate_adj: &[HashSet<u32>],
    plate_sizes: &[usize],
    plate_centroids: &[DVec3],
    total_points: usize,
    seed: u64,
) -> HashSet<u32> {
    let plate_count = plate_adj.len();
    let target = (total_points as f64 * CONTINENTAL_FRACTION) as usize;
    let num_continents = NUM_CONTINENTS.min(plate_count);
    let mut rng = seed.wrapping_add(42);

    let continent_seeds = select_continent_seeds(
        plate_count, plate_centroids, num_continents, &mut rng,
    );

    let mut plate_continent = grow_continents(
        &continent_seeds, plate_adj, plate_sizes, plate_count, target, &mut rng,
    );

    let mut total_land: usize = plate_continent
        .iter()
        .enumerate()
        .filter(|(_, c)| c.is_some())
        .map(|(pid, _)| plate_sizes[pid])
        .sum();

    absorb_interior_seas(plate_adj, plate_sizes, &mut plate_continent, &mut total_land, target);

    plate_continent
        .iter()
        .enumerate()
        .filter_map(|(pid, c)| c.map(|_| pid as u32))
        .collect()
}

fn select_continent_seeds(
    plate_count: usize,
    plate_centroids: &[DVec3],
    num_continents: usize,
    rng: &mut u64,
) -> Vec<u32> {
    let first = (super::splitmix64(*rng) % plate_count as u64) as u32;
    *rng = super::splitmix64(*rng);
    let mut seeds: Vec<u32> = vec![first];
    let mut chosen: HashSet<u32> = HashSet::new();
    chosen.insert(first);

    for _ in 1..num_continents {
        let mut best_pid = u32::MAX;
        let mut best_score = f64::NEG_INFINITY;
        for pid in 0..plate_count as u32 {
            if chosen.contains(&pid) {
                continue;
            }
            let c = plate_centroids[pid as usize];
            let min_dist_sq = seeds
                .iter()
                .map(|&s| (c - plate_centroids[s as usize]).length_squared())
                .fold(f64::INFINITY, f64::min);
            *rng = super::splitmix64(*rng);
            let jitter = 1.0 + (*rng as f64 / u64::MAX as f64) * 0.1;
            let score = min_dist_sq * jitter;
            if score > best_score {
                best_score = score;
                best_pid = pid;
            }
        }
        if best_pid == u32::MAX {
            break;
        }
        seeds.push(best_pid);
        chosen.insert(best_pid);
    }

    seeds
}

fn grow_continents(
    continent_seeds: &[u32],
    plate_adj: &[HashSet<u32>],
    plate_sizes: &[usize],
    plate_count: usize,
    target: usize,
    rng: &mut u64,
) -> Vec<Option<usize>> {
    let num_c = continent_seeds.len();
    let per_continent_target = target / num_c;

    let mut plate_continent: Vec<Option<usize>> = vec![None; plate_count];
    let mut continent_area = vec![0usize; num_c];
    let mut total_land = 0usize;

    for (c, &pid) in continent_seeds.iter().enumerate() {
        plate_continent[pid as usize] = Some(c);
        continent_area[c] = plate_sizes[pid as usize];
        total_land += plate_sizes[pid as usize];
    }

    let mut progress = true;
    while progress && total_land < target {
        progress = false;
        for c in 0..num_c {
            if continent_area[c] >= per_continent_target || total_land >= target {
                continue;
            }
            let mut best_candidate = u32::MAX;
            let mut best_score = f64::NEG_INFINITY;
            for pid in 0..plate_count as u32 {
                if plate_continent[pid as usize].is_some() {
                    continue;
                }
                let mut touches_self = false;
                let mut touches_other = false;
                for &adj in &plate_adj[pid as usize] {
                    match plate_continent[adj as usize] {
                        Some(ac) if ac == c => touches_self = true,
                        Some(_) => {
                            touches_other = true;
                            break;
                        }
                        None => {}
                    }
                }
                if touches_self && !touches_other {
                    *rng = super::splitmix64(*rng);
                    let score =
                        plate_sizes[pid as usize] as f64 + (*rng as f64 / u64::MAX as f64) * 0.5;
                    if score > best_score {
                        best_score = score;
                        best_candidate = pid;
                    }
                }
            }
            if best_candidate != u32::MAX {
                plate_continent[best_candidate as usize] = Some(c);
                let area = plate_sizes[best_candidate as usize];
                continent_area[c] += area;
                total_land += area;
                progress = true;
            }
        }
    }

    plate_continent
}

fn absorb_interior_seas(
    plate_adj: &[HashSet<u32>],
    plate_sizes: &[usize],
    plate_continent: &mut [Option<usize>],
    total_land: &mut usize,
    target: usize,
) {
    let plate_count = plate_adj.len();
    let mut visited = vec![false; plate_count];
    let cap = (target as f64 * 1.15) as usize;

    for start in 0..plate_count {
        if plate_continent[start].is_some() || visited[start] {
            continue;
        }
        let mut component = vec![start];
        visited[start] = true;
        let mut qi = 0;
        while qi < component.len() {
            let pid = component[qi];
            qi += 1;
            for &adj in &plate_adj[pid] {
                let adj = adj as usize;
                if plate_continent[adj].is_none() && !visited[adj] {
                    visited[adj] = true;
                    component.push(adj);
                }
            }
        }

        let mut bordering = HashSet::new();
        for &pid in &component {
            for &adj in &plate_adj[pid] {
                if let Some(c) = plate_continent[adj as usize] {
                    bordering.insert(c);
                }
                if bordering.len() > 1 {
                    break;
                }
            }
            if bordering.len() > 1 {
                break;
            }
        }

        if bordering.len() == 1 {
            let comp_area: usize = component.iter().map(|&pid| plate_sizes[pid]).sum();
            if *total_land + comp_area <= cap {
                let c = *bordering.iter().next().unwrap();
                for &pid in &component {
                    plate_continent[pid] = Some(c);
                }
                *total_land += comp_area;
            }
        }
    }
}
