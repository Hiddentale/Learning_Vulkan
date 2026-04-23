use super::block::BlockType;
use super::heightmap_quadtree::{QuadNode, HEIGHT_PAGE_SIZE};
use super::sphere::{self, CUBE_HALF_BLOCKS, PLANET_RADIUS_BLOCKS};
use super::terrain::{self, WorldNoises, SEA_LEVEL};
use crossbeam_channel::{Receiver, Sender};
use std::sync::Arc;
use std::thread;

const WORKER_COUNT: usize = 8;

/// Generate the height + material grid for one quadtree tile. Returns
/// `HEIGHT_PAGE_SIZE²` floats and bytes laid out row-major
/// (`data[gx + gz * stride]`). The page is `TILE_GRID_POSTS + 2` wide:
/// a 1-texel border on each side for bilinear filtering at tile edges.
/// Interior texels `[1..TILE_GRID_POSTS]` map to the tile's face-local
/// block range; border texels 0 and `TILE_GRID_POSTS + 1` oversample by
/// one post step into the neighboring tile's territory.
pub fn generate_tile_heights(node: QuadNode, seed: u32, terrain: Option<&super::terrain::TerrainData>) -> TileHeightsData {
    use super::heightmap_quadtree::TILE_GRID_POSTS;
    let noises = WorldNoises::new(seed);
    let stride = HEIGHT_PAGE_SIZE as usize;
    let mut heights = vec![0.0_f32; stride * stride];
    let mut materials = vec![0u8; stride * stride];
    let side = node.side_blocks();
    let u0 = node.ix as f64 * side - CUBE_HALF_BLOCKS;
    let v0 = node.iy as f64 * side - CUBE_HALF_BLOCKS;
    // Interior posts span [1, TILE_GRID_POSTS] in the page. The post step
    // covers the tile's full side in (TILE_GRID_POSTS - 1) intervals.
    let interior = TILE_GRID_POSTS as f64;
    let post_step = side / (interior - 1.0);
    let sea_level_radius = SEA_LEVEL as f64 + PLANET_RADIUS_BLOCKS as f64;
    for gz in 0..stride {
        for gx in 0..stride {
            // Map page texel to face-local (u, v). Texel 1 = first interior
            // post (u0), texel 0 = one step before u0 (border).
            let u = u0 + (gx as f64 - 1.0) * post_step;
            let v = v0 + (gz as f64 - 1.0) * post_step;
            let probe = sphere::face_local_to_world(node.face, u, v, 0.0);
            let (seabed_radius, block_type) = terrain::surface_radius_at_world(&noises, probe, terrain);
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
    pub fn new(terrain: Option<Arc<super::terrain::TerrainData>>) -> Self {
        let (request_tx, request_rx) = crossbeam_channel::unbounded::<HeightsRequest>();
        let (result_tx, result_rx) = crossbeam_channel::unbounded::<HeightsResult>();
        let mut workers = Vec::with_capacity(WORKER_COUNT);
        for _ in 0..WORKER_COUNT {
            let rx = request_rx.clone();
            let tx = result_tx.clone();
            let terrain_data = terrain.clone();
            workers.push(thread::spawn(move || {
                while let Ok(req) = rx.recv() {
                    let data = generate_tile_heights(req.node, req.seed, terrain_data.as_deref());
                    if tx
                        .send(HeightsResult {
                            node: req.node,
                            heights: data.heights,
                            materials: data.materials,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            }));
        }
        Self {
            request_tx,
            result_rx,
            _workers: workers,
        }
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
        let node = QuadNode {
            face: Face::PosY,
            level: 3,
            ix: 2,
            iy: 5,
        };
        let data = generate_tile_heights(node, 99, None);
        let noises = WorldNoises::new(99);
        let stride = HEIGHT_PAGE_SIZE as usize;
        let side = node.side_blocks();
        let u0 = node.ix as f64 * side - CUBE_HALF_BLOCKS;
        let v0 = node.iy as f64 * side - CUBE_HALF_BLOCKS;
        // Interior post (0,0) is at page texel (1,1) due to the 1-texel border.
        let probe = face_local_to_world(node.face, u0, v0, 0.0);
        let (r, _) = terrain::surface_radius_at_world(&noises, probe, None);
        let sea = SEA_LEVEL as f64 + PLANET_RADIUS_BLOCKS as f64;
        let expected = (r.max(sea) - PLANET_RADIUS_BLOCKS as f64) as f32;
        let actual = data.heights[1 + stride];
        assert!(
            (actual - expected).abs() < 0.01,
            "corner height mismatch: actual={} expected={}",
            actual,
            expected
        );
    }
}
