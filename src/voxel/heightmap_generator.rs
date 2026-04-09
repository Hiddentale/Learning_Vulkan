use super::block::BlockType;
use super::erosion::ErosionMap;
use super::heightmap_quadtree::{QuadNode, HEIGHT_PAGE_SIZE};
use super::sphere::{self, CUBE_HALF_BLOCKS, PLANET_RADIUS_BLOCKS};
use super::terrain::{self, WorldNoises, SEA_LEVEL};
use crossbeam_channel::{Receiver, Sender};
use std::sync::Arc;
use std::thread;

const WORKER_COUNT: usize = 8;

/// Generate the height + material grid for one quadtree tile. Returns
/// `HEIGHT_PAGE_SIZE²` floats and bytes laid out row-major
/// (`data[gx + gz * stride]`). Height values are in blocks above (positive)
/// or below (negative) the planet radius, clamped to
/// `±MAX_TERRAIN_AMPLITUDE`.
pub fn generate_tile_heights(
    node: QuadNode,
    seed: u32,
    erosion_map: Option<&ErosionMap>,
) -> TileHeightsData {
    let noises = WorldNoises::new(seed);
    let stride = HEIGHT_PAGE_SIZE as usize;
    let mut heights = vec![0.0_f32; stride * stride];
    let mut materials = vec![0u8; stride * stride];
    let side = node.side_blocks();
    let u0 = node.ix as f64 * side - CUBE_HALF_BLOCKS;
    let v0 = node.iy as f64 * side - CUBE_HALF_BLOCKS;
    let denom = (stride - 1).max(1) as f64;
    let sea_level_radius = SEA_LEVEL as f64 + PLANET_RADIUS_BLOCKS as f64;
    for gz in 0..stride {
        for gx in 0..stride {
            let u = u0 + (gx as f64 / denom) * side;
            let v = v0 + (gz as f64 / denom) * side;
            let probe = sphere::face_local_to_world(node.face, u, v, 0.0);
            let (seabed_radius, block_type) = terrain::surface_radius_at_world(&noises, probe, erosion_map);
            let (visible, mat) = if seabed_radius < sea_level_radius {
                (sea_level_radius, BlockType::Water)
            } else {
                (seabed_radius, block_type)
            };
            let offset = (visible - PLANET_RADIUS_BLOCKS as f64) as f32;
            heights[gx + gz * stride] = offset.clamp(
                -super::heightmap_quadtree::MAX_TERRAIN_AMPLITUDE as f32,
                super::heightmap_quadtree::MAX_TERRAIN_AMPLITUDE as f32,
            );
            materials[gx + gz * stride] = mat.material_id();
        }
    }
    TileHeightsData { heights, materials }
}

/// Heights + materials for one quadtree tile page.
pub struct TileHeightsData {
    pub heights: Vec<f32>,
    pub materials: Vec<u8>,
}

#[derive(Copy, Clone, Debug)]
pub struct HeightsRequest {
    pub node: QuadNode,
    pub seed: u32,
}

pub struct HeightsResult {
    pub node: QuadNode,
    pub heights: Vec<f32>,
    pub materials: Vec<u8>,
}

/// Background thread pool for tile heights generation.
pub struct HeightsGenerator {
    request_tx: Sender<HeightsRequest>,
    result_rx: Receiver<HeightsResult>,
    _workers: Vec<thread::JoinHandle<()>>,
}

impl HeightsGenerator {
    pub fn new(erosion_map: Option<Arc<ErosionMap>>) -> Self {
        let (request_tx, request_rx) = crossbeam_channel::unbounded::<HeightsRequest>();
        let (result_tx, result_rx) = crossbeam_channel::unbounded::<HeightsResult>();
        let mut workers = Vec::with_capacity(WORKER_COUNT);
        for _ in 0..WORKER_COUNT {
            let rx = request_rx.clone();
            let tx = result_tx.clone();
            let emap = erosion_map.clone();
            workers.push(thread::spawn(move || {
                while let Ok(req) = rx.recv() {
                    let data = generate_tile_heights(req.node, req.seed, emap.as_deref());
                    if tx.send(HeightsResult { node: req.node, heights: data.heights, materials: data.materials }).is_err() {
                        break;
                    }
                }
            }));
        }
        Self { request_tx, result_rx, _workers: workers }
    }

    pub fn request(&self, req: HeightsRequest) {
        let _ = self.request_tx.send(req);
    }

    pub fn receive(&self) -> Vec<HeightsResult> {
        let mut out = Vec::new();
        while let Ok(r) = self.result_rx.try_recv() {
            out.push(r);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::heightmap_quadtree::{QuadNode, HEIGHT_PAGE_SIZE, MAX_TERRAIN_AMPLITUDE};
    use crate::voxel::sphere::Face;

    #[test]
    fn generate_tile_heights_returns_full_page_within_bounds() {
        let node = QuadNode::root(Face::PosY);
        let data = generate_tile_heights(node, 42, None);
        let expected = (HEIGHT_PAGE_SIZE as usize).pow(2);
        assert_eq!(data.heights.len(), expected);
        assert_eq!(data.materials.len(), expected);
        let amp = MAX_TERRAIN_AMPLITUDE as f32;
        for &h in &data.heights {
            assert!(h.is_finite(), "non-finite height: {}", h);
            assert!(h >= -amp && h <= amp, "height {} out of bounds", h);
        }
    }

    #[test]
    fn generate_tile_heights_matches_surface_radius_at_world() {
        use crate::voxel::sphere::{face_local_to_world, CUBE_HALF_BLOCKS};
        let node = QuadNode { face: Face::PosY, level: 3, ix: 2, iy: 5 };
        let data = generate_tile_heights(node, 99, None);
        let noises = WorldNoises::new(99);
        let stride = HEIGHT_PAGE_SIZE as usize;
        let side = node.side_blocks();
        let u0 = node.ix as f64 * side - CUBE_HALF_BLOCKS;
        let v0 = node.iy as f64 * side - CUBE_HALF_BLOCKS;
        let probe = face_local_to_world(node.face, u0, v0, 0.0);
        let (r, _) = terrain::surface_radius_at_world(&noises, probe, None);
        let sea = SEA_LEVEL as f64 + PLANET_RADIUS_BLOCKS as f64;
        let expected = (r.max(sea) - PLANET_RADIUS_BLOCKS as f64) as f32;
        let actual = data.heights[0 + 0 * stride];
        assert!(
            (actual - expected).abs() < 0.01,
            "corner height mismatch: actual={} expected={}",
            actual, expected
        );
    }
}
