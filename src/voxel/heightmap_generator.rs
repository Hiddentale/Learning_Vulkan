use super::block::BlockType;
use super::erosion::ErosionMap;
use super::heightmap_quadtree::{QuadNode, HEIGHT_PAGE_SIZE};
use super::sphere::{self, ChunkPos, Face, CUBE_HALF_BLOCKS, PLANET_RADIUS_BLOCKS};
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
    let (seabed_radius, seabed_block) = terrain::surface_radius_at_world(noises, probe, erosion_map);
    let sea_level_radius = SEA_LEVEL as f64 + sphere::PLANET_RADIUS_BLOCKS as f64;
    // Where the seabed is below sea level, the visible top is the water
    // surface. Water is rendered as a solid block (own texture, no
    // transparency yet) so the heightmap material at these posts is Water.
    if seabed_radius < sea_level_radius {
        (unit_dir * sea_level_radius, BlockType::Water)
    } else {
        (unit_dir * seabed_radius, seabed_block)
    }
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

// ---------------------------------------------------------------------------
// Phase 2: heights-only generator for the SSE quadtree path.
//
// The new heightmap pipeline uploads a flat `Vec<f32>` of heights per tile
// (one per post on a `HEIGHT_PAGE_SIZE × HEIGHT_PAGE_SIZE` grid). The mesh
// shader does cube_to_sphere projection and normal computation itself, so
// this generator only sources the radial offset above the cube face plane:
// `offset = surface_radius_at_world(probe) − PLANET_RADIUS_BLOCKS`.
//
// Lives alongside `generate_tile_mesh` until Phase 5 deletes the legacy path.
// ---------------------------------------------------------------------------

/// Generate the height grid for one quadtree tile. Returns
/// `HEIGHT_PAGE_SIZE²` floats laid out row-major (`heights[gx + gz * stride]`).
/// Each value is in blocks above (positive) or below (negative) the planet
/// radius, clamped to `±MAX_TERRAIN_AMPLITUDE` so the GPU encoding stays in
/// a known range.
pub fn generate_tile_heights(
    node: QuadNode,
    seed: u32,
    erosion_map: Option<&ErosionMap>,
) -> Vec<f32> {
    let noises = WorldNoises::new(seed);
    let stride = HEIGHT_PAGE_SIZE as usize;
    let mut heights = vec![0.0_f32; stride * stride];
    let side = node.side_blocks();
    // Face-local (u, v) of the tile's lower-left corner, in blocks. Origin
    // is the face center (range [-CUBE_HALF, +CUBE_HALF]) so this matches
    // the convention used by `face_local_to_world`.
    let u0 = node.ix as f64 * side - CUBE_HALF_BLOCKS;
    let v0 = node.iy as f64 * side - CUBE_HALF_BLOCKS;
    let denom = (stride - 1).max(1) as f64;
    let sea_level_radius = SEA_LEVEL as f64 + PLANET_RADIUS_BLOCKS as f64;
    for gz in 0..stride {
        for gx in 0..stride {
            let u = u0 + (gx as f64 / denom) * side;
            let v = v0 + (gz as f64 / denom) * side;
            // Probe the surface at this post; sea floors clamp upward to
            // sea level so the heightmap shows water as the top surface.
            let probe = sphere::face_local_to_world(node.face, u, v, 0.0);
            let (seabed_radius, _) = terrain::surface_radius_at_world(&noises, probe, erosion_map);
            let visible = seabed_radius.max(sea_level_radius);
            let offset = (visible - PLANET_RADIUS_BLOCKS as f64) as f32;
            heights[gx + gz * stride] = offset.clamp(
                -super::heightmap_quadtree::MAX_TERRAIN_AMPLITUDE as f32,
                super::heightmap_quadtree::MAX_TERRAIN_AMPLITUDE as f32,
            );
        }
    }
    heights
}

/// Worker request for the new path. Distinct type from `HeightmapRequest`
/// so the legacy and new generators can coexist until Phase 5 deletes the
/// old code.
#[derive(Copy, Clone, Debug)]
pub struct HeightsRequest {
    pub node: QuadNode,
    pub seed: u32,
}

/// Background thread pool for heights-only generation. Mirrors
/// [`HeightmapGenerator`] but produces flat float pages.
pub struct HeightsGenerator {
    request_tx: Sender<HeightsRequest>,
    result_rx: Receiver<HeightsResult>,
    _workers: Vec<thread::JoinHandle<()>>,
}

pub struct HeightsResult {
    pub node: QuadNode,
    pub heights: Vec<f32>,
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
                    let heights = generate_tile_heights(req.node, req.seed, emap.as_deref());
                    if tx.send(HeightsResult { node: req.node, heights }).is_err() {
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
    use crate::voxel::chunk::CHUNK_SIZE;
    use crate::voxel::heightmap_quadtree::{QuadNode, HEIGHT_PAGE_SIZE, MAX_TERRAIN_AMPLITUDE};
    use crate::voxel::sphere::{FACE_SIDE_CHUNKS, PLANET_RADIUS_BLOCKS};

    #[test]
    fn generate_tile_heights_returns_full_page_within_bounds() {
        // Root tile of +Y face, deterministic seed.
        let node = QuadNode::root(Face::PosY);
        let heights = generate_tile_heights(node, 42, None);
        let expected = (HEIGHT_PAGE_SIZE as usize).pow(2);
        assert_eq!(heights.len(), expected);
        let amp = MAX_TERRAIN_AMPLITUDE as f32;
        for &h in &heights {
            assert!(h.is_finite(), "non-finite height: {}", h);
            assert!(h >= -amp && h <= amp, "height {} out of bounds", h);
        }
    }

    #[test]
    fn generate_tile_heights_matches_surface_radius_at_world() {
        // Sample the corner of a level-3 tile and confirm the offset
        // matches what surface_radius_at_world would return for the
        // same world point.
        use crate::voxel::sphere::{face_local_to_world, CUBE_HALF_BLOCKS};
        let node = QuadNode { face: Face::PosY, level: 3, ix: 2, iy: 5 };
        let heights = generate_tile_heights(node, 99, None);
        let noises = WorldNoises::new(99);
        let stride = HEIGHT_PAGE_SIZE as usize;
        let side = node.side_blocks();
        let u0 = node.ix as f64 * side - CUBE_HALF_BLOCKS;
        let v0 = node.iy as f64 * side - CUBE_HALF_BLOCKS;
        // Pick the (gx=0, gz=0) corner.
        let probe = face_local_to_world(node.face, u0, v0, 0.0);
        let (r, _) = terrain::surface_radius_at_world(&noises, probe, None);
        let sea = SEA_LEVEL as f64 + PLANET_RADIUS_BLOCKS as f64;
        let expected = (r.max(sea) - PLANET_RADIUS_BLOCKS as f64) as f32;
        let actual = heights[0 + 0 * stride];
        assert!(
            (actual - expected).abs() < 0.01,
            "corner height mismatch: actual={} expected={}",
            actual, expected
        );
    }

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
                // Water is now a solid block: the parity test must accept
                // it as the visible top surface (matches mesh chunk path).
                if block != BlockType::Air {
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

    /// **The parity contract**: the chunked terrain and the heightmap LOD
    /// must derive their visible surface from the same canonical function.
    /// After dropping overhang carving from `sample_density_block`, the
    /// chunked top-of-column block sits at `floor(surface_radius_at_world)`,
    /// and the heightmap tile post sits at the same `surface_radius_at_world`,
    /// so the difference is bounded by the rounding (≤ 1 block) plus a small
    /// FP slack. If this test fails, the two paths have drifted apart again
    /// and the LOD ladder is broken — fix the divergence, do not loosen the
    /// tolerance.
    #[test]
    fn heightmap_top_matches_chunked_top_within_one_block() {
        let tolerance = 1.5_f64;
        let cases = [
            (Face::PosY, FACE_SIDE_CHUNKS / 2, FACE_SIDE_CHUNKS / 2),
            (Face::PosX, FACE_SIDE_CHUNKS / 2, FACE_SIDE_CHUNKS / 2),
            (Face::NegZ, 4, 4),
            (Face::PosY, FACE_SIDE_CHUNKS / 4, FACE_SIDE_CHUNKS / 4 + 3),
        ];
        let noises = WorldNoises::new(SEED);
        for (face, cx, cz) in cases {
            // Sample several blocks within the chunk and compare per point.
            for &(lx, lz) in &[(0usize, 0usize), (4, 4), (8, 8), (15, 15), (3, 11)] {
                // Chunked: walk the column top-down, find the first non-air
                // block, take its world radius.
                let chunks = terrain::generate_column(face, cx, cz, SEED, None);
                let mut chunked_top = None;
                'find: for cy_rev in (0..chunks.len()).rev() {
                    for ly_rev in (0..CHUNK_SIZE).rev() {
                        if chunks[cy_rev].get(lx, ly_rev, lz) != BlockType::Air {
                            let cp = ChunkPos { face, cx, cy: cy_rev as i32, cz };
                            let world = sphere::chunk_to_world(
                                cp,
                                Vec3::new(lx as f32 + 0.5, ly_rev as f32 + 0.5, lz as f32 + 0.5),
                            );
                            chunked_top = Some(world.length());
                            break 'find;
                        }
                    }
                }
                let chunked_r = chunked_top.expect("expected something solid in column");

                // Analytical: sample the canonical surface function at the
                // matching probe direction.
                let probe = sphere::chunk_to_world(
                    ChunkPos { face, cx, cy: 0, cz },
                    Vec3::new(lx as f32 + 0.5, 0.5, lz as f32 + 0.5),
                );
                let (analytical_r, _) = terrain::surface_radius_at_world(&noises, probe, None);
                let analytical_r = analytical_r
                    .max(SEA_LEVEL as f64 + sphere::PLANET_RADIUS_BLOCKS as f64);

                let delta = (chunked_r - analytical_r).abs();
                assert!(
                    delta < tolerance,
                    "chunked vs heightmap top mismatch at face={:?} chunk=({},{}) local=({},{}): \
                     chunked={:.3} analytical={:.3} delta={:.3} > {:.3}",
                    face, cx, cz, lx, lz, chunked_r, analytical_r, delta, tolerance
                );
            }
        }
    }

    /// Older, weaker parity test. Kept because it covers tile-mesh corner
    /// vertices specifically (the tile mesh class, not just the underlying
    /// surface function). Tolerance kept loose because corner vertices may
    /// land slightly off the chunk grid.
    #[test]
    fn tile_post_radius_matches_chunk_terrain_within_overhang_tolerance() {
        // Tile post = analytical surface; chunk top = floor(surface) at the
        // chunk-grid resolution. The two can differ by ~1 block (analytical
        // floor) plus a fractional offset from the chunk's center-sample
        // probe (0.5). At PLANET_RADIUS=16001 the float-to-block rounding
        // can land exactly on the previous bound; bumped to 2.0.
        let tolerance = 2.0_f64;
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

    // Mirror of HEIGHTMAP_BANDS in vulkan_object.rs. The schedule tests are
    // pure (no GPU state), so we hard-code the bands here. If the renderer
    // bands change, update this constant — `schedule_completes_every_in_band_tile`
    // is the canary that catches drift.
    const TEST_BANDS: &[(i32, usize, usize, f32, f32)] = &[
        (8, 65, 17, -10.0, 0.50),
        (16, 65, 17, 0.50, 1.50),
    ];
    const TEST_CHUNKED_ARC: f32 = 0.416; // ~ WORLD_DISTANCE * CHUNK_SIZE / PLANET_RADIUS_BLOCKS at the current scale
    // Pinned to a small value so scheduler tests stay O(face²)-cheap regardless
    // of the real `FACE_SIDE_CHUNKS`. The schedule logic is independent of
    // planet size, so this only changes runtime, not coverage.
    const TEST_FACE_SIDE_CHUNKS: i32 = 128;
    #[allow(dead_code)]
    const _TEST_FACE_SIDE_FITS: () = assert!(TEST_FACE_SIDE_CHUNKS <= FACE_SIDE_CHUNKS);
    const PER_FRAME: usize = 32;

    /// Player at the +Y face center, surface altitude.
    fn player_at_posy_center() -> DVec3 {
        sphere::chunk_to_world(
            ChunkPos {
                face: Face::PosY,
                cx: TEST_FACE_SIDE_CHUNKS / 2,
                cy: 0,
                cz: TEST_FACE_SIDE_CHUNKS / 2,
            },
            Vec3::new(8.0, 8.0, 8.0),
        )
    }

    /// Enumerate every tile that should *eventually* be loaded for a given
    /// player position. Mirrors the body of `schedule_candidates` without
    /// the loaded-or-in-flight predicate or the cap.
    fn enumerate_in_band_tiles(player: DVec3) -> Vec<TileKey> {
        enumerate_in_band_tiles_with_shadow(player, 0.0)
    }

    fn enumerate_in_band_tiles_with_shadow(player: DVec3, mesh_shadow_arc: f32) -> Vec<TileKey> {
        let mut all = Vec::new();
        for &(tile_chunks, _grid, _coarse, extra_min, extra_max) in TEST_BANDS {
            let min_angle = (TEST_CHUNKED_ARC + extra_min).max(mesh_shadow_arc).max(0.0);
            let max_angle = TEST_CHUNKED_ARC + extra_max;
            for face in [Face::PosX, Face::NegX, Face::PosY, Face::NegY, Face::PosZ, Face::NegZ] {
                let mut cz = 0;
                while cz + tile_chunks <= TEST_FACE_SIDE_CHUNKS {
                    let mut cx = 0;
                    while cx + tile_chunks <= TEST_FACE_SIDE_CHUNKS {
                        let half_span = (tile_chunks as f32 * CHUNK_SIZE as f32) * 0.5;
                        let center = sphere::chunk_to_world(
                            ChunkPos { face, cx, cy: 0, cz },
                            Vec3::new(half_span, 0.0, half_span),
                        );
                        let dist = angular_distance(player, center);
                        if dist >= min_angle && dist <= max_angle {
                            all.push(TileKey { face, cx, cz });
                        }
                        cx += tile_chunks;
                    }
                    cz += tile_chunks;
                }
            }
        }
        all
    }

    /// **The schedule contract**: given a fixed player position, simulating
    /// frame-by-frame submission with `PER_FRAME` budget per frame and
    /// instant generation, every in-band tile is eventually queued. If this
    /// fails, the schedule has a starvation bug — some tile is structurally
    /// unreachable, the iteration order is dropping it, or the band math
    /// disagrees with the enumerator.
    #[test]
    fn schedule_completes_every_in_band_tile() {
        let player = player_at_posy_center();
        let expected: std::collections::HashSet<TileKey> =
            enumerate_in_band_tiles(player).into_iter().collect();
        assert!(!expected.is_empty(), "no in-band tiles for the test player position");

        let mut loaded: std::collections::HashSet<TileKey> = Default::default();
        for _frame in 0..1000 {
            let picks = schedule_candidates(
                player,
                TEST_BANDS,
                TEST_CHUNKED_ARC,
                0.0,
                TEST_FACE_SIDE_CHUNKS,
                &|k| loaded.contains(&k),
                PER_FRAME,
            );
            if picks.is_empty() {
                break;
            }
            for cand in picks {
                loaded.insert(cand.key);
            }
        }
        let missing: Vec<&TileKey> = expected.iter().filter(|k| !loaded.contains(k)).collect();
        assert!(
            missing.is_empty(),
            "schedule never queued {} of {} in-band tiles. examples: {:?}",
            missing.len(),
            expected.len(),
            missing.iter().take(5).collect::<Vec<_>>(),
        );
    }

    /// **Closest-first**: with `max_results = 1` and a fresh player, the
    /// returned tile is the one with smallest angular distance among all
    /// in-band candidates. Catches the previous `face × cz × cx` iteration
    /// order bug where the player's directly-below tiles were starved
    /// while distant +X-face tiles filled first.
    #[test]
    fn schedule_returns_closest_unloaded_tile_first() {
        let player = player_at_posy_center();
        let candidates = schedule_candidates(
            player,
            TEST_BANDS,
            TEST_CHUNKED_ARC,
            0.0,
            TEST_FACE_SIDE_CHUNKS,
            &|_| false,
            1,
        );
        assert_eq!(candidates.len(), 1);
        let picked = candidates[0];

        // Compare against the brute-force closest in-band tile.
        let all = enumerate_in_band_tiles(player);
        let mut best: Option<(TileKey, f32)> = None;
        for k in all {
            let half_span = {
                let tile_chunks =
                    if TEST_BANDS[0].0 == 8 && TEST_BANDS.iter().any(|b| b.0 == 8) {
                        // Use the band that contains this tile size.
                        TEST_BANDS
                            .iter()
                            .find(|b| {
                                let min = (TEST_CHUNKED_ARC + b.3).max(0.0);
                                let max = TEST_CHUNKED_ARC + b.4;
                                let cs = b.0 as f32 * CHUNK_SIZE as f32;
                                let center = sphere::chunk_to_world(
                                    ChunkPos { face: k.face, cx: k.cx, cy: 0, cz: k.cz },
                                    Vec3::new(cs * 0.5, 0.0, cs * 0.5),
                                );
                                let d = angular_distance(player, center);
                                d >= min && d <= max
                            })
                            .map(|b| b.0)
                            .unwrap_or(8)
                    } else {
                        8
                    };
                (tile_chunks as f32 * CHUNK_SIZE as f32) * 0.5
            };
            let center = sphere::chunk_to_world(
                ChunkPos { face: k.face, cx: k.cx, cy: 0, cz: k.cz },
                Vec3::new(half_span, 0.0, half_span),
            );
            let d = angular_distance(player, center);
            if best.map_or(true, |(_, bd)| d < bd) {
                best = Some((k, d));
            }
        }
        let (best_key, _) = best.unwrap();
        assert_eq!(
            picked.key, best_key,
            "schedule returned {:?}, brute-force closest is {:?}",
            picked.key, best_key
        );
    }

    /// **Mesh-shadow exclusion**: when the player is on the surface, the
    /// heightmap must NOT load tiles inside the mesh-chunk reach. Otherwise
    /// the smooth shell occludes the floored mesh blocks (the shell is at
    /// `surface_radius`, the mesh tops at `floor(surface_radius)`) and the
    /// player visually lands on the heightmap instead of the blocks.
    #[test]
    fn schedule_excludes_tiles_inside_mesh_shadow() {
        let player = player_at_posy_center();
        let mesh_shadow = TEST_CHUNKED_ARC;
        let picks = schedule_candidates(
            player,
            TEST_BANDS,
            TEST_CHUNKED_ARC,
            mesh_shadow,
            TEST_FACE_SIDE_CHUNKS,
            &|_| false,
            10_000,
        );
        // Every picked tile must lie at angular distance ≥ mesh_shadow.
        for cand in &picks {
            let cs = cand.tile_chunks as f32 * CHUNK_SIZE as f32;
            let center = sphere::chunk_to_world(
                ChunkPos { face: cand.key.face, cx: cand.key.cx, cy: 0, cz: cand.key.cz },
                Vec3::new(cs * 0.5, 0.0, cs * 0.5),
            );
            let dist = angular_distance(player, center);
            assert!(
                dist >= mesh_shadow - 1e-4,
                "tile {:?} at dist {:.3} is inside mesh shadow {:.3}",
                cand.key, dist, mesh_shadow,
            );
        }
        assert!(!picks.is_empty(), "schedule returned nothing — outer band should still fire");
    }

    /// **Altitude case**: with `mesh_shadow_arc = 0` (player above terrain
    /// band, mesh chunks empty), the heightmap covers from angular distance
    /// 0. The directly-below tile (angular distance ≈ 0) must be picked
    /// before any other.
    #[test]
    fn schedule_at_altitude_picks_directly_below_first() {
        let player = player_at_posy_center();
        let picks = schedule_candidates(
            player,
            TEST_BANDS,
            TEST_CHUNKED_ARC,
            0.0, // no mesh shadow — player at altitude
            TEST_FACE_SIDE_CHUNKS,
            &|_| false,
            1,
        );
        assert_eq!(picks.len(), 1);
        let pick = picks[0];
        let cs = pick.tile_chunks as f32 * CHUNK_SIZE as f32;
        let center = sphere::chunk_to_world(
            ChunkPos { face: pick.key.face, cx: pick.key.cx, cy: 0, cz: pick.key.cz },
            Vec3::new(cs * 0.5, 0.0, cs * 0.5),
        );
        let dist = angular_distance(player, center);
        assert!(
            dist < TEST_CHUNKED_ARC,
            "directly-below pick should be inside the chunked-arc reach, got {:.3} vs {:.3}",
            dist, TEST_CHUNKED_ARC,
        );
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

/// One picked tile candidate: the key, the band index, and that band's
/// tile size in chunks.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct ScheduledCandidate {
    pub key: TileKey,
    pub band_idx: u8,
    pub tile_chunks: i32,
}

/// **The pure scheduling decision**: given the player's world position, the
/// band layout, the angular reach of currently-active mesh chunks
/// (`mesh_shadow_arc`), and a predicate identifying tiles already loaded or
/// in flight, return the closest unloaded in-band tiles up to `max_results`.
///
/// Pure on purpose. The renderer's `schedule_heightmap_generation` is a thin
/// wrapper that calls this and submits the result. Keeping it pure means the
/// schedule contract — *every in-band tile is eventually queued, closest
/// first, none inside the mesh shadow* — is unit-testable without GPU state.
///
/// `bands` matches `HEIGHTMAP_BANDS`: `(tile_chunks, grid_posts,
/// coarse_grid_posts, extra_min, extra_max)`. The angular range of band `b`
/// is `[max(mesh_shadow_arc, chunked_arc + b.extra_min), chunked_arc +
/// b.extra_max]`. `mesh_shadow_arc` clamps the lower bound so the heightmap
/// never overlaps the mesh-chunk working set: when the player is on the
/// surface, mesh chunks cover `[0, chunked_arc]` and the heightmap stays
/// outside; when the player is at altitude (no mesh chunks), the caller
/// passes `mesh_shadow_arc = 0` and the heightmap covers from 0.
pub fn schedule_candidates(
    player_world: DVec3,
    bands: &[(i32, usize, usize, f32, f32)],
    chunked_arc: f32,
    mesh_shadow_arc: f32,
    face_side_chunks: i32,
    is_loaded_or_in_flight: &dyn Fn(TileKey) -> bool,
    max_results: usize,
) -> Vec<ScheduledCandidate> {
    use crate::voxel::chunk::CHUNK_SIZE;
    let faces = [Face::PosX, Face::NegX, Face::PosY, Face::NegY, Face::PosZ, Face::NegZ];
    let mut candidates: Vec<(ScheduledCandidate, f32)> = Vec::new();

    for (band_idx, &(tile_chunks, _grid, _coarse, extra_min, extra_max)) in bands.iter().enumerate() {
        let min_angle = (chunked_arc + extra_min).max(mesh_shadow_arc).max(0.0);
        let max_angle = chunked_arc + extra_max;
        if max_angle <= 0.0 || min_angle > std::f32::consts::PI {
            continue;
        }
        for &face in &faces {
            let mut cz = 0;
            while cz + tile_chunks <= face_side_chunks {
                let mut cx = 0;
                while cx + tile_chunks <= face_side_chunks {
                    let key = TileKey { face, cx, cz };
                    if !is_loaded_or_in_flight(key) {
                        let cs = CHUNK_SIZE as f32;
                        let half_span = (tile_chunks as f32 * cs) * 0.5;
                        let center_w = sphere::chunk_to_world(
                            ChunkPos { face, cx, cy: 0, cz },
                            Vec3::new(half_span, 0.0, half_span),
                        );
                        let dist = angular_distance(player_world, center_w);
                        if dist >= min_angle && dist <= max_angle {
                            candidates.push((
                                ScheduledCandidate {
                                    key,
                                    band_idx: band_idx as u8,
                                    tile_chunks,
                                },
                                dist,
                            ));
                        }
                    }
                    cx += tile_chunks;
                }
                cz += tile_chunks;
            }
        }
    }

    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    candidates.into_iter().take(max_results).map(|(c, _)| c).collect()
}
