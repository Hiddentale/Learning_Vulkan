use std::collections::HashSet;

use glam::DVec3;

use super::plate_seed_placement::Adjacency;
use super::plates::{CrustType, Plate};

/// A boundary edge: a point where two plates meet.
pub(super) struct BoundaryEdge {
    pub(super) point: u32,
    pub(super) neighbor: u32,
    pub(super) plate_a: u32,
    pub(super) plate_b: u32,
    pub(super) crust_a: CrustType,
    pub(super) crust_b: CrustType,
    pub(super) age_a: f64,
    pub(super) age_b: f64,
}

/// Scan the adjacency graph for edges that cross plate boundaries.
pub(super) fn find_boundary_edges(
    plates: &[Plate],
    points: &[DVec3],
    adjacency: &Adjacency,
) -> Vec<BoundaryEdge> {
    let mut point_to_plate = vec![0u32; points.len()];
    let mut point_to_local = vec![0usize; points.len()];
    for (plate_idx, plate) in plates.iter().enumerate() {
        for (local, &global) in plate.point_indices.iter().enumerate() {
            point_to_plate[global as usize] = plate_idx as u32;
            point_to_local[global as usize] = local;
        }
    }

    let mut edges = Vec::new();
    let mut seen: HashSet<(u32, u32)> = HashSet::new();

    for point in 0..points.len() as u32 {
        let plate_a = point_to_plate[point as usize];
        for &neighbor in adjacency.neighbors_of(point) {
            let plate_b = point_to_plate[neighbor as usize];
            if plate_a == plate_b {
                continue;
            }
            let pair = (point.min(neighbor), point.max(neighbor));
            if !seen.insert(pair) {
                continue;
            }
            let local_a = point_to_local[point as usize];
            let local_b = point_to_local[neighbor as usize];
            edges.push(BoundaryEdge {
                point,
                neighbor,
                plate_a,
                plate_b,
                crust_a: plates[plate_a as usize].crust[local_a].crust_type,
                crust_b: plates[plate_b as usize].crust[local_b].crust_type,
                age_a: plates[plate_a as usize].crust[local_a].age,
                age_b: plates[plate_b as usize].crust[local_b].age,
            });
        }
    }

    edges
}
