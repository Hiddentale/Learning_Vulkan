#![allow(dead_code)] // Wired up incrementally
use super::block::BlockType;
use super::erosion::ErosionMap;
use super::terrain::{self, WorldNoises, SEA_LEVEL};
use crossbeam_channel::{Receiver, Sender};
use std::sync::Arc;
use std::thread;

const WORKER_COUNT: usize = 2;

/// Vertex for heightmap tile mesh. 32 bytes, matches shader layout.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct HeightmapVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub material_id: u32,
    pub morph_delta_y: f32,
}

/// Completed heightmap tile mesh ready for GPU upload.
pub struct HeightmapTileMesh {
    pub pos: [i32; 2],
    pub lod: u8,
    pub vertices: Vec<HeightmapVertex>,
    pub indices: Vec<u16>,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
}

/// Request to generate a heightmap tile mesh on a background thread.
pub struct HeightmapRequest {
    pub pos: [i32; 2],
    pub lod: u8,
    pub tile_size_blocks: f64,
    pub grid_posts: usize,
    pub spacing: f64,
    pub coarse_spacing: f64,
    pub seed: u32,
}

/// Background thread pool for heightmap mesh generation.
pub struct HeightmapGenerator {
    request_tx: Sender<HeightmapRequest>,
    result_rx: Receiver<HeightmapTileMesh>,
    _workers: Vec<thread::JoinHandle<()>>,
}

impl HeightmapGenerator {
    pub fn new(erosion_map: Option<Arc<ErosionMap>>) -> Self {
        let (request_tx, request_rx) = crossbeam_channel::unbounded::<HeightmapRequest>();
        let (result_tx, result_rx) = crossbeam_channel::unbounded::<HeightmapTileMesh>();

        let mut workers = Vec::with_capacity(WORKER_COUNT);
        for _ in 0..WORKER_COUNT {
            let rx = request_rx.clone();
            let tx = result_tx.clone();
            let emap = erosion_map.clone();
            workers.push(thread::spawn(move || {
                while let Ok(req) = rx.recv() {
                    let mesh = generate_tile_mesh(&req, emap.as_deref());
                    if tx.send(mesh).is_err() {
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

    pub fn request(&self, req: HeightmapRequest) {
        let _ = self.request_tx.send(req);
    }

    /// Drain all completed tile meshes (non-blocking).
    pub fn receive(&self) -> Vec<HeightmapTileMesh> {
        let mut results = Vec::new();
        while let Ok(mesh) = self.result_rx.try_recv() {
            results.push(mesh);
        }
        results
    }
}

fn generate_tile_mesh(req: &HeightmapRequest, erosion_map: Option<&ErosionMap>) -> HeightmapTileMesh {
    let noises = WorldNoises::new(req.seed);
    let n = req.grid_posts;
    let chunk_size = 16.0_f64;
    let origin_x = req.pos[0] as f64 * chunk_size;
    let origin_z = req.pos[1] as f64 * chunk_size;

    // Sample height grid
    let mut heights = vec![0.0f32; n * n];
    let mut materials = vec![BlockType::Grass; n * n];
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;

    for gz in 0..n {
        for gx in 0..n {
            let wx = origin_x + gx as f64 * req.spacing;
            let wz = origin_z + gz as f64 * req.spacing;
            let (height, surface) = terrain::sample_surface(&noises, wx, wz, erosion_map);

            let y = if height < SEA_LEVEL { SEA_LEVEL as f32 } else { height as f32 };
            let mat = if height < SEA_LEVEL { BlockType::Water } else { surface };

            let idx = gx + gz * n;
            heights[idx] = y;
            materials[idx] = mat;
            min_y = min_y.min(y);
            max_y = max_y.max(y);
        }
    }

    // Sample coarser grid for geomorphing (morph_delta_y)
    let mut morph_deltas = vec![0.0f32; n * n];
    for gz in 0..n {
        for gx in 0..n {
            let wx = origin_x + gx as f64 * req.spacing;
            let wz = origin_z + gz as f64 * req.spacing;
            // Snap to coarser grid
            let snapped_x = (wx / req.coarse_spacing).floor() * req.coarse_spacing;
            let snapped_z = (wz / req.coarse_spacing).floor() * req.coarse_spacing;
            let frac_x = (wx - snapped_x) / req.coarse_spacing;
            let frac_z = (wz - snapped_z) / req.coarse_spacing;

            // Bilinear interpolation of coarser heights
            let (h00, _) = terrain::sample_surface(&noises, snapped_x, snapped_z, erosion_map);
            let (h10, _) = terrain::sample_surface(&noises, snapped_x + req.coarse_spacing, snapped_z, erosion_map);
            let (h01, _) = terrain::sample_surface(&noises, snapped_x, snapped_z + req.coarse_spacing, erosion_map);
            let (h11, _) = terrain::sample_surface(&noises, snapped_x + req.coarse_spacing, snapped_z + req.coarse_spacing, erosion_map);

            let coarse_y = (h00 as f64 * (1.0 - frac_x) * (1.0 - frac_z)
                + h10 as f64 * frac_x * (1.0 - frac_z)
                + h01 as f64 * (1.0 - frac_x) * frac_z
                + h11 as f64 * frac_x * frac_z) as f32;

            let coarse_y = coarse_y.max(SEA_LEVEL as f32);
            morph_deltas[gx + gz * n] = heights[gx + gz * n] - coarse_y;
        }
    }

    // Build vertices with normals via finite differences
    let spacing_f32 = req.spacing as f32;
    let mut vertices = Vec::with_capacity(n * n);
    for gz in 0..n {
        for gx in 0..n {
            let idx = gx + gz * n;
            let x = origin_x as f32 + gx as f32 * spacing_f32;
            let y = heights[idx];
            let z = origin_z as f32 + gz as f32 * spacing_f32;

            // Finite difference normals
            let h_left = if gx > 0 { heights[idx - 1] } else { y };
            let h_right = if gx < n - 1 { heights[idx + 1] } else { y };
            let h_back = if gz > 0 { heights[idx - n] } else { y };
            let h_front = if gz < n - 1 { heights[idx + n] } else { y };

            let nx = h_left - h_right;
            let ny = 2.0 * spacing_f32;
            let nz = h_back - h_front;
            let len = (nx * nx + ny * ny + nz * nz).sqrt();

            vertices.push(HeightmapVertex {
                position: [x, y, z],
                normal: [nx / len, ny / len, nz / len],
                material_id: materials[idx].material_id() as u32,
                morph_delta_y: morph_deltas[idx],
            });
        }
    }

    // Generate triangle indices (two triangles per quad)
    let quads = n - 1;
    let mut indices = Vec::with_capacity(quads * quads * 6);
    for gz in 0..quads {
        for gx in 0..quads {
            let tl = (gx + gz * n) as u16;
            let tr = (gx + 1 + gz * n) as u16;
            let bl = (gx + (gz + 1) * n) as u16;
            let br = (gx + 1 + (gz + 1) * n) as u16;
            // Triangle 1
            indices.push(tl);
            indices.push(tr);
            indices.push(bl);
            // Triangle 2
            indices.push(tr);
            indices.push(br);
            indices.push(bl);
        }
    }

    let tile_end_x = origin_x as f32 + (n - 1) as f32 * spacing_f32;
    let tile_end_z = origin_z as f32 + (n - 1) as f32 * spacing_f32;

    HeightmapTileMesh {
        pos: req.pos,
        lod: req.lod,
        vertices,
        indices,
        aabb_min: [origin_x as f32, min_y - 1.0, origin_z as f32],
        aabb_max: [tile_end_x, max_y + 1.0, tile_end_z],
    }
}
