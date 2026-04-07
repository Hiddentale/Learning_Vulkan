use super::block::BlockType;
use super::erosion::ErosionMap;
use super::sphere::{self, ChunkPos, Face};
use super::terrain::{self, WorldNoises, SEA_LEVEL};
use crossbeam_channel::{Receiver, Sender};
use glam::{DVec3, Vec3};
use std::sync::Arc;
use std::thread;

const WORKER_COUNT: usize = 2;

/// Vertex for heightmap tile mesh. 32 bytes, matches shader layout.
/// `position` is in world cartesian (relative to planet center) — the tile
/// mesh is already curved on the CPU so the vertex shader does no projection.
/// `morph_delta_r` is a scalar radial offset; geomorphing toward the coarser
/// LOD interpolates along the local outward radial direction.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct HeightmapVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub material_id: u32,
    pub morph_delta_r: f32,
}

/// Address of a heightmap tile on the cube-sphere. Two tiles with the same
/// `(cx, cz)` on different faces are distinct — this fixes the pre-sphere
/// pool key collision bug.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct TileKey {
    pub face: Face,
    pub cx: i32,
    pub cz: i32,
}

/// Completed heightmap tile mesh ready for GPU upload.
pub struct HeightmapTileMesh {
    pub key: TileKey,
    pub lod: u8,
    pub vertices: Vec<HeightmapVertex>,
    pub indices: Vec<u16>,
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
}

/// Request to generate a heightmap tile mesh on a background thread.
/// `tile_chunks_per_side` is the tile's footprint in face-local chunk units;
/// `grid_posts` is the resolution of the post grid (so each post represents
/// `tile_chunks_per_side / (grid_posts - 1)` chunks of terrain).
pub struct HeightmapRequest {
    pub key: TileKey,
    pub lod: u8,
    pub tile_chunks_per_side: i32,
    pub grid_posts: usize,
    pub coarse_grid_posts: usize,
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

/// Compute the world cartesian position of a heightmap grid post on the
/// surface. `grid_x` and `grid_z` are post indices in `[0, grid_posts)`.
/// The probe direction comes from the cube-sphere projection; the radial
/// distance comes from the same `surface_radius_at_world` call that drives
/// chunk-density terrain, keeping tiles consistent with chunks pixel-by-pixel.
fn surface_post_world(
    key: TileKey,
    tile_chunks_per_side: i32,
    grid_x: usize,
    grid_z: usize,
    grid_posts: usize,
    noises: &WorldNoises,
    erosion_map: Option<&ErosionMap>,
) -> (DVec3, BlockType) {
    // Map grid index to a face-local block-space offset inside the tile.
    let span_blocks = tile_chunks_per_side as f32 * super::chunk::CHUNK_SIZE as f32;
    let denom = (grid_posts - 1).max(1) as f32;
    let local_x = (grid_x as f32) / denom * span_blocks;
    let local_z = (grid_z as f32) / denom * span_blocks;
    // Probe direction from a point on the cube surface (radial depth = 0).
    let probe_cp = ChunkPos { face: key.face, cx: key.cx, cy: 0, cz: key.cz };
    let probe = sphere::chunk_to_world(probe_cp, Vec3::new(local_x, 0.0, local_z));
    let unit_dir = probe.normalize_or(DVec3::Y);
    let (radius, block) = terrain::surface_radius_at_world(noises, probe, erosion_map);
    let radius = radius.max(SEA_LEVEL as f64 + sphere::PLANET_RADIUS_BLOCKS as f64);
    (unit_dir * radius, block)
}

pub fn generate_tile_mesh(req: &HeightmapRequest, erosion_map: Option<&ErosionMap>) -> HeightmapTileMesh {
    let noises = WorldNoises::new(req.seed);
    let n = req.grid_posts;

    // First pass: world position + material per post.
    let mut positions = vec![DVec3::ZERO; n * n];
    let mut materials = vec![BlockType::Grass; n * n];
    for gz in 0..n {
        for gx in 0..n {
            let (pos, mat) = surface_post_world(req.key, req.tile_chunks_per_side, gx, gz, n, &noises, erosion_map);
            positions[gx + gz * n] = pos;
            materials[gx + gz * n] = mat;
        }
    }

    // Geomorph: each fine post computes a coarse-grid radial position by
    // re-sampling at a coarser stride. The vertex shader interpolates along
    // the local outward radial between fine and coarse as the player crosses
    // the LOD band. Coarse stride is `n / coarse_grid_posts`.
    let coarse_n = req.coarse_grid_posts.max(2);
    let mut coarse_radii = vec![0.0_f32; coarse_n * coarse_n];
    for gz in 0..coarse_n {
        for gx in 0..coarse_n {
            let (pos, _) = surface_post_world(req.key, req.tile_chunks_per_side, gx, gz, coarse_n, &noises, erosion_map);
            coarse_radii[gx + gz * coarse_n] = pos.length() as f32;
        }
    }

    // Build vertices.
    let mut vertices = Vec::with_capacity(n * n);
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for gz in 0..n {
        for gx in 0..n {
            let p = positions[gx + gz * n];
            let pf = [p.x as f32, p.y as f32, p.z as f32];

            // Finite-difference normal in world space using neighboring posts.
            let p_left = positions[gx.saturating_sub(1) + gz * n];
            let p_right = positions[(gx + 1).min(n - 1) + gz * n];
            let p_back = positions[gx + gz.saturating_sub(1) * n];
            let p_front = positions[gx + (gz + 1).min(n - 1) * n];
            let du = (p_right - p_left).as_vec3();
            let dv = (p_front - p_back).as_vec3();
            let normal = du.cross(dv).normalize_or(p.normalize_or(DVec3::Y).as_vec3());

            // Radial morph delta to coarse grid (bilinear interp on coarse_radii).
            let fx = (gx as f32 / (n - 1).max(1) as f32) * (coarse_n - 1) as f32;
            let fz = (gz as f32 / (n - 1).max(1) as f32) * (coarse_n - 1) as f32;
            let cx0 = (fx.floor() as usize).min(coarse_n - 1);
            let cz0 = (fz.floor() as usize).min(coarse_n - 1);
            let cx1 = (cx0 + 1).min(coarse_n - 1);
            let cz1 = (cz0 + 1).min(coarse_n - 1);
            let tx = fx - cx0 as f32;
            let tz = fz - cz0 as f32;
            let r00 = coarse_radii[cx0 + cz0 * coarse_n];
            let r10 = coarse_radii[cx1 + cz0 * coarse_n];
            let r01 = coarse_radii[cx0 + cz1 * coarse_n];
            let r11 = coarse_radii[cx1 + cz1 * coarse_n];
            let coarse_r = r00 * (1.0 - tx) * (1.0 - tz) + r10 * tx * (1.0 - tz) + r01 * (1.0 - tx) * tz + r11 * tx * tz;
            let fine_r = p.length() as f32;
            let morph_delta_r = fine_r - coarse_r;

            vertices.push(HeightmapVertex {
                position: pf,
                normal: normal.into(),
                material_id: materials[gx + gz * n].material_id() as u32,
                morph_delta_r,
            });
            for k in 0..3 {
                if pf[k] < min[k] {
                    min[k] = pf[k];
                }
                if pf[k] > max[k] {
                    max[k] = pf[k];
                }
            }
        }
    }

    // AABB pad: covers normal-direction lighting jitter and the outward
    // morph displacement (whichever is larger). 2 blocks is conservative.
    const AABB_PAD: f32 = 2.0;
    for k in 0..3 {
        min[k] -= AABB_PAD;
        max[k] += AABB_PAD;
    }

    // Triangle indices (two triangles per quad).
    let quads = n - 1;
    let mut indices = Vec::with_capacity(quads * quads * 6);
    for gz in 0..quads {
        for gx in 0..quads {
            let tl = (gx + gz * n) as u16;
            let tr = (gx + 1 + gz * n) as u16;
            let bl = (gx + (gz + 1) * n) as u16;
            let br = (gx + 1 + (gz + 1) * n) as u16;
            indices.push(tl);
            indices.push(tr);
            indices.push(bl);
            indices.push(tr);
            indices.push(br);
            indices.push(bl);
        }
    }

    HeightmapTileMesh {
        key: req.key,
        lod: req.lod,
        vertices,
        indices,
        aabb_min: min,
        aabb_max: max,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::CHUNK_SIZE;
    use crate::voxel::sphere::{FACE_SIDE_CHUNKS, PLANET_RADIUS_BLOCKS};

    const SEED: u32 = 42;
    const TILE_CHUNKS: i32 = 8;
    const POSTS: usize = 33;

    fn make_request(face: Face, cx: i32, cz: i32) -> HeightmapRequest {
        HeightmapRequest {
            key: TileKey { face, cx, cz },
            lod: 0,
            tile_chunks_per_side: TILE_CHUNKS,
            grid_posts: POSTS,
            coarse_grid_posts: 9,
            seed: SEED,
        }
    }

    /// Walk a chunk column from the top down to find the radius of the
    /// topmost solid (non-air, non-water) block. Mirrors the test ground
    /// truth in `terrain::surface_diagnostics` but uses only public API.
    fn topmost_solid_radius_via_generator(face: Face, cx: i32, cz: i32, x: usize, z: usize) -> Option<f64> {
        let chunks = terrain::generate_column(face, cx, cz, SEED, None);
        let cs = CHUNK_SIZE;
        for cy_rev in (0..chunks.len()).rev() {
            for ly_rev in (0..cs).rev() {
                let block = chunks[cy_rev].get(x, ly_rev, z);
                if block != BlockType::Air && block != BlockType::Water {
                    let cp = ChunkPos { face, cx, cy: cy_rev as i32, cz };
                    let world = sphere::chunk_to_world(cp, Vec3::new(x as f32 + 0.5, ly_rev as f32 + 0.5, z as f32 + 0.5));
                    return Some(world.length());
                }
            }
        }
        None
    }

    #[test]
    fn tile_key_distinguishes_same_chunk_coords_on_different_faces() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        let cx = 5;
        let cz = 7;
        for face in [Face::PosX, Face::NegX, Face::PosY, Face::NegY, Face::PosZ, Face::NegZ] {
            assert!(set.insert(TileKey { face, cx, cz }));
        }
        assert_eq!(set.len(), 6);
    }

    /// Bug 1: heightmap tiles must use the same 3D world-space surface as
    /// the chunked terrain, otherwise there's a visible discontinuity at
    /// every chunk/tile boundary. We sample tile vertices at multiple
    /// (face, cx, cz) — including face centers and tile-on-an-edge — and
    /// check the tile radius is within `tolerance` of the topmost solid
    /// block radius from `generate_column`. The tolerance covers
    /// overhangs/cave headers (the heightmap is a smooth surface; chunks
    /// have density-driven irregularities of up to ~OVERHANG_BAND blocks).
    #[test]
    fn tile_post_radius_matches_chunk_terrain_within_overhang_tolerance() {
        let tolerance = 8.0_f64;
        let cases = [
            (Face::PosY, FACE_SIDE_CHUNKS / 2, FACE_SIDE_CHUNKS / 2),
            (Face::PosX, FACE_SIDE_CHUNKS / 2, FACE_SIDE_CHUNKS / 2),
            (Face::NegZ, 4, 4),
            (Face::PosY, 0, FACE_SIDE_CHUNKS / 2), // tile spanning a face edge
        ];
        for (face, cx, cz) in cases {
            let req = make_request(face, cx, cz);
            let mesh = generate_tile_mesh(&req, None);
            // Check the corner posts (gx, gz) ∈ {0, last}².
            let corners: [(usize, usize); 4] = [(0, 0), (POSTS - 1, 0), (0, POSTS - 1), (POSTS - 1, POSTS - 1)];
            for (gx, gz) in corners {
                let v = mesh.vertices[gx + gz * POSTS];
                let radius = (v.position[0] as f64).hypot(v.position[1] as f64).hypot(v.position[2] as f64);
                // The corner post lives at the edge of the tile; the
                // matching chunk column is at (cx + dx_chunks, cz + dz_chunks)
                // where dx_chunks = tile_chunks * (gx / (posts-1)). For the
                // corners that lands on chunk boundaries, so we just look up
                // the chunk at that integer offset.
                let dx_chunks = (gx * TILE_CHUNKS as usize) / (POSTS - 1);
                let dz_chunks = (gz * TILE_CHUNKS as usize) / (POSTS - 1);
                let chunk_cx = cx + dx_chunks as i32;
                let chunk_cz = cz + dz_chunks as i32;
                if chunk_cx < 0 || chunk_cx >= FACE_SIDE_CHUNKS || chunk_cz < 0 || chunk_cz >= FACE_SIDE_CHUNKS {
                    continue; // tile spills past the face; cross-face lookup is a separate concern
                }
                let truth = topmost_solid_radius_via_generator(face, chunk_cx, chunk_cz, 0, 0)
                    .expect("expected solid ground at probe");
                let delta = (radius - truth).abs();
                assert!(
                    delta < tolerance,
                    "tile vs chunk mismatch at face={:?} chunk=({}, {}) corner=({}, {}): tile_r={:.3} chunk_r={:.3} delta={:.3} > {:.3}",
                    face, chunk_cx, chunk_cz, gx, gz, radius, truth, delta, tolerance
                );
            }
        }
    }

    /// Bug 4: the AABB used by frustum culling must enclose every projected
    /// vertex. The pre-sphere AABB was computed in flat-XZ space and would
    /// fail this on any tile that curves with the planet.
    #[test]
    fn tile_aabb_encloses_every_vertex() {
        for face in [Face::PosY, Face::PosX, Face::NegZ] {
            let req = make_request(face, FACE_SIDE_CHUNKS / 2, FACE_SIDE_CHUNKS / 2);
            let mesh = generate_tile_mesh(&req, None);
            for v in &mesh.vertices {
                for k in 0..3 {
                    assert!(
                        v.position[k] >= mesh.aabb_min[k] && v.position[k] <= mesh.aabb_max[k],
                        "vertex {:?} outside aabb {:?}..{:?}",
                        v.position,
                        mesh.aabb_min,
                        mesh.aabb_max
                    );
                }
            }
            // Sanity: every vertex sits on or above the planet radius.
            for v in &mesh.vertices {
                let r = (v.position[0] as f64).hypot(v.position[1] as f64).hypot(v.position[2] as f64);
                assert!(r >= PLANET_RADIUS_BLOCKS as f64 - 1.0, "vertex below planet: r={}", r);
            }
        }
    }

    /// Bug 3: tile selection must use a sphere-aware metric. With Chebyshev
    /// distance on `(cx, cz)`, a tile on an opposite face with low numerical
    /// chunk indices reads as "near" — completely wrong for picking the
    /// closest tile to the player.
    #[test]
    fn angular_distance_picks_geographically_closest_tile() {
        // Player straight above the +Y face center.
        let player = sphere::chunk_to_world(
            ChunkPos {
                face: Face::PosY,
                cx: FACE_SIDE_CHUNKS / 2,
                cy: 0,
                cz: FACE_SIDE_CHUNKS / 2,
            },
            Vec3::new(8.0, 8.0, 8.0),
        );

        // Three candidate tile centers: (a) +Y near center, (b) +X face,
        // (c) -Y antipode. Angular distance must order them a < b < c.
        let center_a = sphere::chunk_to_world(
            ChunkPos { face: Face::PosY, cx: FACE_SIDE_CHUNKS / 2, cy: 0, cz: FACE_SIDE_CHUNKS / 2 },
            Vec3::new(8.0, 0.0, 8.0),
        );
        let center_b = sphere::chunk_to_world(
            ChunkPos { face: Face::PosX, cx: FACE_SIDE_CHUNKS / 2, cy: 0, cz: FACE_SIDE_CHUNKS / 2 },
            Vec3::new(8.0, 0.0, 8.0),
        );
        let center_c = sphere::chunk_to_world(
            ChunkPos { face: Face::NegY, cx: FACE_SIDE_CHUNKS / 2, cy: 0, cz: FACE_SIDE_CHUNKS / 2 },
            Vec3::new(8.0, 0.0, 8.0),
        );

        let d_a = angular_distance(player, center_a);
        let d_b = angular_distance(player, center_b);
        let d_c = angular_distance(player, center_c);
        assert!(d_a < d_b, "+Y center should be nearer than +X face: {} vs {}", d_a, d_b);
        assert!(d_b < d_c, "+X face should be nearer than -Y antipode: {} vs {}", d_b, d_c);
        assert!((d_c - std::f32::consts::PI).abs() < 0.1, "-Y antipode should be ~π radians away, got {}", d_c);
    }
}

/// Angular distance (radians) between two cartesian world points, measured
/// from the planet center. Used by tile scheduling, eviction, and morph-band
/// assignment to replace the broken Chebyshev `(cx, cz)` distance metric.
pub fn angular_distance(a: DVec3, b: DVec3) -> f32 {
    let ua = a.normalize_or(DVec3::Y);
    let ub = b.normalize_or(DVec3::Y);
    ua.dot(ub).clamp(-1.0, 1.0).acos() as f32
}
