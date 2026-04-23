use std::collections::VecDeque;

use crate::world_generation::sphere_geometry::plate_seed_placement::{Adjacency, PlateAssignment};

pub(super) struct BoundarySeeds {
    pub coast_seeds: Vec<u32>,
    pub mountain_seeds: Vec<u32>,
    pub ridge_seeds: Vec<u32>,
    pub arc_seeds: Vec<u32>,
    pub subduction_seeds: Vec<u32>,
    pub rift_seeds: Vec<u32>,
}

pub(super) fn classify(
    adjacency: &Adjacency,
    assignment: &PlateAssignment,
    is_continental: &[bool],
    seed: u64,
) -> BoundarySeeds {
    let n = assignment.plate_ids.len();
    let mut seeds = BoundarySeeds {
        coast_seeds: Vec::new(),
        mountain_seeds: Vec::new(),
        ridge_seeds: Vec::new(),
        arc_seeds: Vec::new(),
        subduction_seeds: Vec::new(),
        rift_seeds: Vec::new(),
    };

    let is_convergent = |a: u32, b: u32| -> bool {
        let lo = a.min(b) as u64;
        let hi = a.max(b) as u64;
        let h = super::splitmix64(seed.wrapping_add(lo * 1000003 + hi));
        (h % 100) < 60
    };

    for i in 0..n {
        let my_plate = assignment.plate_ids[i];
        let my_cont = is_continental[i];

        for &nb in adjacency.neighbors_of(i as u32) {
            let nb_plate = assignment.plate_ids[nb as usize];
            if my_plate == nb_plate {
                continue;
            }
            let nb_cont = is_continental[nb as usize];

            if my_cont != nb_cont {
                seeds.coast_seeds.push(i as u32);
            }

            if my_cont && nb_cont {
                if is_convergent(my_plate, nb_plate) {
                    seeds.mountain_seeds.push(i as u32);
                } else {
                    seeds.rift_seeds.push(i as u32);
                }
            } else if my_cont && !nb_cont && is_convergent(my_plate, nb_plate) {
                seeds.subduction_seeds.push(i as u32);
            } else if !my_cont && !nb_cont {
                if is_convergent(my_plate, nb_plate) {
                    seeds.arc_seeds.push(i as u32);
                } else {
                    seeds.ridge_seeds.push(i as u32);
                }
            }

            break;
        }
    }

    seeds
}

pub(super) fn bfs_distance(n: usize, adjacency: &Adjacency, seeds: &[u32]) -> Vec<u32> {
    bfs_distance_capped(n, adjacency, seeds, u32::MAX)
}

pub(super) fn bfs_distance_capped(
    n: usize,
    adjacency: &Adjacency,
    seeds: &[u32],
    max_dist: u32,
) -> Vec<u32> {
    let mut dist = vec![u32::MAX; n];
    let mut queue = VecDeque::with_capacity(seeds.len());
    for &s in seeds {
        if dist[s as usize] == u32::MAX {
            dist[s as usize] = 0;
            queue.push_back(s);
        }
    }
    while let Some(current) = queue.pop_front() {
        let nd = dist[current as usize] + 1;
        if nd > max_dist {
            continue;
        }
        for &nb in adjacency.neighbors_of(current) {
            if dist[nb as usize] == u32::MAX {
                dist[nb as usize] = nd;
                queue.push_back(nb);
            }
        }
    }
    dist
}
