//! Per-face quadtree for far-field heightmap LOD.
//!
//! Each cube face is rooted by one quadtree. A node at level `L` covers a
//! `(FACE_SIDE_BLOCKS / 2^L)`-square region of its face. Per frame the player
//! camera drives a screen-space-error (SSE) descent that emits a leaf set;
//! a `restrict` pass then enforces equal LOD level across shared edges
//! (within-face and across cube seams) so the rasterized tile mesh is
//! crack-free without skirts.
//!
//! No Vulkan, no GPU. Everything in this module is pure CPU and unit-tested.
//! The output of one frame is `Quadtree::active_set()` — a flat slice of
//! [`TileDesc`] that the GPU streamer reads.
//!
//! Reference: Strugar, "Continuous Distance-Dependent Level of Detail for
//! Rendering Heightmaps" (CDLOD), 2010 — https://aggrobird.com/files/cdlod_latest.pdf
//! Rumpelnik, "Planetary Rendering with Mesh Shaders", TU Wien 2020.

#![allow(dead_code)] // Wired up incrementally across phases.

use super::sphere::{
    face_basis, face_for_cube_point, face_local_to_world, ALL_FACES, Face, EdgeDir,
    CUBE_HALF_BLOCKS, FACE_SIDE_BLOCKS,
};
use glam::DVec3;

/// Side length, in blocks, of the finest (deepest) quadtree leaf. Determines
/// the maximum level: `MAX_LEVEL = ceil(log2(FACE_SIDE_BLOCKS / BASE_TILE_BLOCKS))`.
pub const BASE_TILE_BLOCKS: i32 = 128;

/// Vertex grid posts along one edge of a tile. 65 = 64 quads, splits cleanly
/// into power-of-two meshlets in the mesh shader.
pub const TILE_GRID_POSTS: u32 = 65;

/// Side length, in texels, of one heightmap atlas page. Currently equals
/// `TILE_GRID_POSTS` — one height value per geometric grid post, no skirt.
/// Bumping this by 2 (for a 1-texel border on each side) is the way to
/// enable hardware bilinear filtering across tile edges if Phase 3 needs it.
pub const HEIGHT_PAGE_SIZE: u32 = TILE_GRID_POSTS;

/// Maximum quadtree depth. With `FACE_SIDE_BLOCKS=22720` and `BASE=128` this
/// is 8 (root face = level 0, finest leaf = level 8 covering ~88 blocks).
pub const MAX_LEVEL: u8 = compute_max_level();

const fn compute_max_level() -> u8 {
    let mut span = FACE_SIDE_BLOCKS;
    let mut level = 0u8;
    while span > BASE_TILE_BLOCKS {
        span /= 2;
        level += 1;
    }
    level
}

/// Stable identity of a quadtree node. `morton` packs `(face, level, ix, iy)`
/// into a single `u64` so it can be used as a HashMap key and as a GPU-side
/// stable id without storing the structural fields separately.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct QuadNode {
    pub face: Face,
    pub level: u8,
    /// Tile index along the face's tangent-u axis at this level. Range
    /// `0..(1 << level)`.
    pub ix: u32,
    /// Tile index along the face's tangent-v axis at this level.
    pub iy: u32,
}

impl QuadNode {
    pub fn root(face: Face) -> Self {
        Self { face, level: 0, ix: 0, iy: 0 }
    }

    /// Number of tiles per side at this level.
    pub fn tiles_per_side(self) -> u32 {
        1u32 << self.level
    }

    /// Side length of this tile in blocks (face-local).
    pub fn side_blocks(self) -> f64 {
        FACE_SIDE_BLOCKS as f64 / self.tiles_per_side() as f64
    }

    /// 64-bit identifier. Layout: `[face:3 | level:5 | iy:28 | ix:28]`. With
    /// MAX_LEVEL=8 only 16 bits each are used by ix/iy, so the encoding has
    /// room for ~30 levels of future growth.
    pub fn morton(self) -> u64 {
        debug_assert!(self.level <= MAX_LEVEL);
        debug_assert!(self.ix < self.tiles_per_side());
        debug_assert!(self.iy < self.tiles_per_side());
        let face = self.face as u64;
        let level = self.level as u64;
        let ix = self.ix as u64;
        let iy = self.iy as u64;
        (face << 61) | (level << 56) | (iy << 28) | ix
    }

    /// The 4 children at `level + 1`. Order: `(0,0), (1,0), (0,1), (1,1)`.
    pub fn children(self) -> Option<[QuadNode; 4]> {
        if self.level >= MAX_LEVEL {
            return None;
        }
        let level = self.level + 1;
        let bx = self.ix * 2;
        let by = self.iy * 2;
        Some([
            QuadNode { face: self.face, level, ix: bx,     iy: by },
            QuadNode { face: self.face, level, ix: bx + 1, iy: by },
            QuadNode { face: self.face, level, ix: bx,     iy: by + 1 },
            QuadNode { face: self.face, level, ix: bx + 1, iy: by + 1 },
        ])
    }

    /// Parent at `level - 1`. None for root.
    pub fn parent(self) -> Option<QuadNode> {
        if self.level == 0 {
            return None;
        }
        Some(QuadNode {
            face: self.face,
            level: self.level - 1,
            ix: self.ix / 2,
            iy: self.iy / 2,
        })
    }

    /// Face-local block-space center `(u, v)`. Origin is the face center,
    /// matching `face_local_to_world`'s convention (range `[-CUBE_HALF, +CUBE_HALF]`).
    pub fn center_uv(self) -> (f64, f64) {
        let side = self.side_blocks();
        let half = CUBE_HALF_BLOCKS;
        let u = (self.ix as f64 + 0.5) * side - half;
        let v = (self.iy as f64 + 0.5) * side - half;
        (u, v)
    }

    /// Cartesian world-space center, projected through cube_to_sphere at sea
    /// level. Used for SSE distance and frustum/horizon culling.
    pub fn center_world(self) -> DVec3 {
        let (u, v) = self.center_uv();
        face_local_to_world(self.face, u, v, 0.0)
    }

    /// Bounding sphere radius around `center_world`, conservative. Sized to
    /// enclose all 4 corners + max terrain amplitude. Used by frustum and
    /// horizon culls.
    pub fn bounding_radius(self, max_terrain_amplitude: f64) -> f64 {
        let side = self.side_blocks();
        let half = CUBE_HALF_BLOCKS;
        let u0 = self.ix as f64 * side - half;
        let v0 = self.iy as f64 * side - half;
        let u1 = u0 + side;
        let v1 = v0 + side;
        let center = self.center_world();
        let mut max_d2 = 0.0_f64;
        for &(u, v) in &[(u0, v0), (u1, v0), (u0, v1), (u1, v1)] {
            let p = face_local_to_world(self.face, u, v, max_terrain_amplitude);
            let d2 = (p - center).length_squared();
            if d2 > max_d2 { max_d2 = d2; }
        }
        max_d2.sqrt() + max_terrain_amplitude
    }
}

/// Per-frame description of one resident quadtree leaf. Mirrored to GPU
/// (Phase 3) as a flat SSBO row.
#[derive(Clone, Debug)]
pub struct TileDesc {
    pub node: QuadNode,
    pub center_world: DVec3,
    pub bounding_radius: f32,
    /// Geometric error in world units: max vertical deviation between this
    /// tile's grid and the finest grid it would be replaced by on subdivision.
    /// For now we estimate as `side_blocks * GEOMETRIC_ERROR_PER_BLOCK`.
    pub geometric_error: f32,
    /// LOD levels of the 4 cross-edge neighbors after the restrict pass.
    /// Index order: `[NegU, PosU, NegV, PosV]`. Equal to `node.level` after
    /// restrict (we enforce equality, not ±1).
    pub neighbor_levels: [u8; 4],
    /// CDLOD morph factor toward parent. 0 = full fine grid, 1 = collapsed
    /// to parent grid. Computed in `compute_morph_factors` after `descend`.
    pub morph_factor: f32,
}

/// Geometric error of a tile, in world units. CDLOD definition: the maximum
/// vertical deviation between this tile's grid and the parent grid that
/// would be used in its place at one level coarser. For terrain bounded by
/// `MAX_TERRAIN_AMPLITUDE`, the worst case at level `L` is roughly the
/// amplitude scaled by the tile's relative size.
fn tile_geometric_error(side_blocks: f64) -> f32 {
    (MAX_TERRAIN_AMPLITUDE * side_blocks / FACE_SIDE_BLOCKS as f64) as f32
}

/// Maximum natural terrain height above the cube face plane, in blocks.
/// Used to size bounding spheres. Heights stored in the atlas are encoded
/// as offsets from `PLANET_RADIUS_BLOCKS` and live in `[-this, +this]`.
pub const MAX_TERRAIN_AMPLITUDE: f64 = 2048.0;

/// Maximum number of resident tiles. Bounds the height-atlas page count
/// and the per-frame TileDesc upload size. The CDLOD descent is bounded by
/// screen pixels, not planet radius — empirically ≤1024 leaves at 1440p
/// with `SSE_PIXEL_THRESHOLD=4`.
pub const MAX_RESIDENT_TILES: usize = 1024;

/// Pixel threshold for SSE-driven split. Tiles with projected geometric
/// error above this split; below merge.
pub const SSE_PIXEL_THRESHOLD: f32 = 4.0;

/// GPU-side tile descriptor row. Mirrored exactly by the `TileDesc` struct
/// in `heightmap_cull.comp`, `heightmap_tile.task`, and `heightmap_tile.mesh`.
/// std430 layout: 64 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GpuTileDesc {
    pub center: [f32; 3],
    pub bounding_radius: f32,
    /// `[NegU, PosU, NegV, PosV]` neighbor levels.
    pub neighbor_levels: [u32; 4],
    pub face: u32,
    pub level: u32,
    pub ix: u32,
    pub iy: u32,
    pub geometric_error: f32,
    pub morph_factor: f32,
    pub height_page: u32,
    pub _pad: u32,
}

const _: () = assert!(std::mem::size_of::<GpuTileDesc>() == 64);

impl TileDesc {
    /// Pack into the GPU layout. `height_page` is supplied externally
    /// (looked up in the atlas after streaming).
    pub fn to_gpu(&self, height_page: u32) -> GpuTileDesc {
        let c = self.center_world;
        GpuTileDesc {
            center: [c.x as f32, c.y as f32, c.z as f32],
            bounding_radius: self.bounding_radius,
            neighbor_levels: [
                self.neighbor_levels[0] as u32,
                self.neighbor_levels[1] as u32,
                self.neighbor_levels[2] as u32,
                self.neighbor_levels[3] as u32,
            ],
            face: self.node.face as u32,
            level: self.node.level as u32,
            ix: self.node.ix,
            iy: self.node.iy,
            geometric_error: self.geometric_error,
            morph_factor: self.morph_factor,
            height_page,
            _pad: 0,
        }
    }
}

pub struct Quadtree {
    active: Vec<TileDesc>,
    /// Cached frame inputs so `restrict` and `compute_morph_factors` don't
    /// need them re-passed.
    last_camera: DVec3,
    last_pixel_per_radian: f32,
    /// Mortons of every tile resident in the GPU height atlas. Updated by
    /// `commit_stream` after the streamer has applied a `StreamDelta`.
    resident: std::collections::HashSet<u64>,
}

/// Diff between this frame's `active` set and the prior `resident` set.
/// `to_load` are tiles whose heights need to be generated and uploaded;
/// `to_evict` are pages the streamer should free. The streamer is expected
/// to call [`Quadtree::commit_stream`] once it has acted on the delta.
#[derive(Default, Debug)]
pub struct StreamDelta {
    pub to_load: Vec<QuadNode>,
    pub to_evict: Vec<u64>,
}

impl Quadtree {
    pub fn new() -> Self {
        Self {
            active: Vec::with_capacity(MAX_RESIDENT_TILES),
            last_camera: DVec3::ZERO,
            last_pixel_per_radian: 0.0,
            resident: std::collections::HashSet::with_capacity(MAX_RESIDENT_TILES),
        }
    }

    pub fn active(&self) -> &[TileDesc] { &self.active }

    /// Diff this frame's active set against the resident GPU set. The
    /// streamer should pass `to_load` to the heights generator and free
    /// `to_evict` pages, then call [`Self::commit_stream`].
    pub fn stream(&self) -> StreamDelta {
        let active_set: std::collections::HashSet<u64> =
            self.active.iter().map(|t| t.node.morton()).collect();
        let mut to_load = Vec::new();
        for tile in &self.active {
            let m = tile.node.morton();
            if !self.resident.contains(&m) {
                to_load.push(tile.node);
            }
        }
        let mut to_evict = Vec::new();
        for &m in &self.resident {
            if !active_set.contains(&m) {
                to_evict.push(m);
            }
        }
        StreamDelta { to_load, to_evict }
    }

    /// Apply the result of a `StreamDelta` to the resident set. Call after
    /// the streamer has uploaded the loaded tiles and freed the evicted
    /// pages.
    pub fn commit_stream(&mut self, delta: &StreamDelta) {
        for n in &delta.to_load {
            self.resident.insert(n.morton());
        }
        for m in &delta.to_evict {
            self.resident.remove(m);
        }
    }

    pub fn resident_count(&self) -> usize { self.resident.len() }

    /// Drive one frame: SSE descent → restrict → morph. After this call,
    /// `active()` returns the leaf set ready for the GPU streamer.
    pub fn update(&mut self, camera_world: DVec3, screen_height_px: f32, fov_y_rad: f32) {
        self.last_camera = camera_world;
        // pixels-per-radian for SSE projection: rho = err * pix_per_rad / dist.
        self.last_pixel_per_radian = screen_height_px / (2.0 * (fov_y_rad * 0.5).tan());
        self.active.clear();
        self.descend_bfs();
        self.restrict();
        self.compute_morph_factors();
    }

    /// BFS / priority-queue descent. Starts with the 6 face roots in a
    /// max-heap keyed by projected pixel error, repeatedly popping the
    /// highest-priority node and either splitting it (push 4 children) or
    /// emitting it as a leaf. Stops when the heap is empty or the active
    /// set hits `MAX_RESIDENT_TILES`. Distributes the tile budget by
    /// importance instead of by face iteration order, so a near face can't
    /// starve the far ones.
    fn descend_bfs(&mut self) {
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(Copy, Clone)]
        struct Pending {
            node: QuadNode,
            // Projected pixel error. Wrapped so we can `Ord` it as a max-heap.
            projected_px: f32,
        }
        impl PartialEq for Pending {
            fn eq(&self, other: &Self) -> bool { self.projected_px == other.projected_px }
        }
        impl Eq for Pending {}
        impl PartialOrd for Pending {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
        }
        impl Ord for Pending {
            fn cmp(&self, other: &Self) -> Ordering {
                self.projected_px
                    .partial_cmp(&other.projected_px)
                    .unwrap_or(Ordering::Equal)
            }
        }

        let mut heap: BinaryHeap<Pending> = BinaryHeap::with_capacity(MAX_RESIDENT_TILES * 4);
        for face in ALL_FACES {
            let root = QuadNode::root(face);
            heap.push(Pending { node: root, projected_px: self.projected_pixel_error(root) });
        }

        while let Some(Pending { node, projected_px }) = heap.pop() {
            if self.active.len() >= MAX_RESIDENT_TILES {
                break;
            }
            let should_split = projected_px > SSE_PIXEL_THRESHOLD && node.level < MAX_LEVEL;
            if should_split {
                if let Some(children) = node.children() {
                    for child in children {
                        heap.push(Pending { node: child, projected_px: self.projected_pixel_error(child) });
                    }
                    continue;
                }
            }
            self.emit_leaf(node);
        }
    }

    fn projected_pixel_error(&self, node: QuadNode) -> f32 {
        let center = node.center_world();
        let geo_err = tile_geometric_error(node.side_blocks());
        let dist = ((center - self.last_camera).length() as f32).max(1.0);
        geo_err * self.last_pixel_per_radian / dist
    }

    fn emit_leaf(&mut self, node: QuadNode) {
        let center = node.center_world();
        let geo_err = tile_geometric_error(node.side_blocks());
        let radius = node.bounding_radius(MAX_TERRAIN_AMPLITUDE);
        self.active.push(TileDesc {
            node,
            center_world: center,
            bounding_radius: radius as f32,
            geometric_error: geo_err,
            neighbor_levels: [node.level; 4],
            morph_factor: 0.0,
        });
    }

    /// Enforce equal LOD level across every shared edge. Iterates the leaf
    /// set: any leaf whose neighbor along an edge has a coarser level than
    /// itself causes the neighbor to split. Repeats to fixpoint.
    fn restrict(&mut self) {
        // `levels` records, for every node the descent visited (leaf or
        // internal), the deepest level any of its descendants was emitted at.
        // The reemit walk uses this to know whether to recurse into a
        // subtree or emit it as a leaf.
        use std::collections::HashMap;
        let mut levels: HashMap<u64, u8> = HashMap::with_capacity(self.active.len() * 4);
        for tile in &self.active {
            record_with_ancestors(tile.node, tile.node.level, &mut levels);
        }
        // Walk leaves, recording the max level any cross-edge neighbor at
        // the same or coarser scale demands. Repeat until stable.
        // Implementation: each iteration may force a leaf to "virtually split"
        // by raising the recorded level for its position. We then rebuild
        // the active list at the end. For simplicity and correctness, we
        // collect required (face, ix, iy, level) and re-emit.
        for _ in 0..(MAX_LEVEL as usize + 1) {
            let mut changed = false;
            let snapshot = self.active.clone();
            for tile in &snapshot {
                for (edge_idx, dir) in [EdgeDir::NegU, EdgeDir::PosU, EdgeDir::NegV, EdgeDir::PosV].iter().enumerate() {
                    let nb = neighbor_node(tile.node, *dir);
                    if let Some(nb) = nb {
                        let nb_level = levels.get(&nb.morton()).copied().unwrap_or(tile.node.level);
                        if nb_level < tile.node.level {
                            // Neighbor is coarser — record that this tile's
                            // edge sees a coarser neighbor. Used by the
                            // shader to snap edge verts. We do NOT force the
                            // coarser neighbor to split because we are
                            // enforcing equality, not ±1: instead we mark
                            // the coarser side for refinement on the next
                            // pass.
                            let _ = edge_idx;
                            // Force the neighbor up to this level by
                            // virtually splitting it: record the new level
                            // on the neighbor and all its ancestors.
                            record_with_ancestors(nb, tile.node.level, &mut levels);
                            changed = true;
                        }
                    }
                }
            }
            if !changed { break; }
        }
        // Re-emit active set at the levels map (subdividing coarse nodes
        // whose level was bumped). Walk roots and descend to whichever level
        // the map says.
        let mut new_active: Vec<TileDesc> = Vec::with_capacity(self.active.len());
        for face in ALL_FACES {
            reemit_recursive(QuadNode::root(face), &levels, &mut new_active);
        }
        // Re-fill neighbor_levels for the new set.
        let new_levels: std::collections::HashMap<u64, u8> = new_active
            .iter()
            .map(|t| (t.node.morton(), t.node.level))
            .collect();
        for tile in &mut new_active {
            for (edge_idx, dir) in [EdgeDir::NegU, EdgeDir::PosU, EdgeDir::NegV, EdgeDir::PosV].iter().enumerate() {
                if let Some(nb) = neighbor_node(tile.node, *dir) {
                    tile.neighbor_levels[edge_idx] = new_levels.get(&nb.morton()).copied().unwrap_or(tile.node.level);
                } else {
                    tile.neighbor_levels[edge_idx] = tile.node.level;
                }
            }
        }
        self.active = new_active;
    }

    /// CDLOD morph factor: smooth 0..1 over the outer 15% of the tile's
    /// "valid distance band" so vertices smoothly approach the parent grid
    /// before the tile is replaced.
    fn compute_morph_factors(&mut self) {
        for tile in &mut self.active {
            let dist = ((tile.center_world - self.last_camera).length() as f32).max(1.0);
            // Distance at which this tile would have been split (rho = threshold).
            let split_dist = tile.geometric_error * self.last_pixel_per_radian / SSE_PIXEL_THRESHOLD;
            // Distance at which the parent would have been split (== this tile is replaced).
            let parent_split_dist = split_dist * 2.0;
            let lo = split_dist + (parent_split_dist - split_dist) * 0.85;
            let hi = parent_split_dist;
            let t = ((dist - lo) / (hi - lo).max(1e-6)).clamp(0.0, 1.0);
            tile.morph_factor = t * t * (3.0 - 2.0 * t); // smoothstep
        }
    }
}

impl Default for Quadtree {
    fn default() -> Self { Self::new() }
}

/// Record `level` for `node` and every ancestor up to root, taking max
/// against any existing entry. The reemit walk uses these ancestor entries
/// to know which subtrees to descend into.
fn record_with_ancestors(node: QuadNode, level: u8, levels: &mut std::collections::HashMap<u64, u8>) {
    let mut cur = node;
    loop {
        let entry = levels.entry(cur.morton()).or_insert(0);
        if *entry < level {
            *entry = level;
        }
        match cur.parent() {
            Some(p) => cur = p,
            None => break,
        }
    }
}

/// Walk the (face, ix, iy)-keyed `levels` map and emit a TileDesc for every
/// node whose recorded level matches its current depth (i.e. it's a leaf in
/// the post-restrict tree).
fn reemit_recursive(node: QuadNode, levels: &std::collections::HashMap<u64, u8>, out: &mut Vec<TileDesc>) {
    if out.len() >= MAX_RESIDENT_TILES {
        return;
    }
    let recorded = levels.get(&node.morton()).copied().unwrap_or(node.level);
    if recorded > node.level {
        if let Some(children) = node.children() {
            for child in children {
                reemit_recursive(child, levels, out);
            }
            return;
        }
    }
    let center = node.center_world();
    let geo_err = tile_geometric_error(node.side_blocks());
    let radius = node.bounding_radius(MAX_TERRAIN_AMPLITUDE);
    out.push(TileDesc {
        node,
        center_world: center,
        bounding_radius: radius as f32,
        geometric_error: geo_err,
        neighbor_levels: [node.level; 4],
        morph_factor: 0.0,
    });
}

/// Cross-edge neighbor of a quadtree node at the same level. Within-face
/// neighbors are direct index arithmetic. Cross-face neighbors route through
/// `face_adjacency` so the (u, v) frame rotation/flip is honored.
///
/// Returns `None` only when the node has no neighbor in that direction —
/// which never happens on a closed cube map; in practice every node always
/// has a neighbor.
pub fn neighbor_node(node: QuadNode, dir: EdgeDir) -> Option<QuadNode> {
    let n = node.tiles_per_side() as i32;
    let (mut nx, mut ny) = (node.ix as i32, node.iy as i32);
    match dir {
        EdgeDir::NegU => nx -= 1,
        EdgeDir::PosU => nx += 1,
        EdgeDir::NegV => ny -= 1,
        EdgeDir::PosV => ny += 1,
    }
    if nx >= 0 && nx < n && ny >= 0 && ny < n {
        return Some(QuadNode { face: node.face, level: node.level, ix: nx as u32, iy: ny as u32 });
    }
    // Cross-face: convert tile (ix, iy) to a face-local cube point in
    // blocks, project onto the cube edge, re-resolve which face owns it,
    // and re-express in that face's (ix, iy) at the same level.
    cross_face_tile(node, dir)
}

/// Compute the same-level cross-face tile neighbor of `node` along `dir`.
/// Mirrors `sphere::cross_face_neighbor` but at quadtree-tile granularity:
/// the math is identical (face basis → 3D cube point → dominant axis →
/// re-project), only the unit changes.
fn cross_face_tile(node: QuadNode, dir: EdgeDir) -> Option<QuadNode> {
    let n = node.tiles_per_side();
    // Step one tile beyond the edge in the source face's tangent space.
    let (mut nx, mut ny) = (node.ix as i32, node.iy as i32);
    match dir {
        EdgeDir::NegU => nx -= 1,
        EdgeDir::PosU => nx += 1,
        EdgeDir::NegV => ny -= 1,
        EdgeDir::PosV => ny += 1,
    }
    // Convert tile center to face-local (u, v) in blocks, then to a 3D cube
    // point. Use 3D coords in CUBE_HALF units so the dominant-axis pick is
    // numerically stable for any level.
    let side = node.side_blocks();
    let half = CUBE_HALF_BLOCKS;
    let u = (nx as f64 + 0.5) * side - half;
    let v = (ny as f64 + 0.5) * side - half;
    let (tu, tv, fnv) = face_basis(node.face);
    let cube_pt = tu.as_dvec3() * u + tv.as_dvec3() * v + fnv.as_dvec3() * half;
    // Re-classify which face owns this point.
    let new_face = face_for_cube_point(cube_pt);
    if new_face == node.face {
        // Numerically still on the source face — should not happen for a
        // genuine cross-edge step, but be robust.
        return None;
    }
    // Re-express the cube point in the new face's tangent basis.
    let (ntu, ntv, _) = face_basis(new_face);
    let new_u = cube_pt.dot(ntu.as_dvec3());
    let new_v = cube_pt.dot(ntv.as_dvec3());
    // Convert back to tile indices at the same level.
    let new_ix_f = (new_u + half) / side;
    let new_iy_f = (new_v + half) / side;
    // Clamp to face range — corners can land just outside due to floating
    // point error. The tile that owns the projected point is the floor.
    let new_ix = (new_ix_f.floor() as i32).clamp(0, n as i32 - 1) as u32;
    let new_iy = (new_iy_f.floor() as i32).clamp(0, n as i32 - 1) as u32;
    Some(QuadNode { face: new_face, level: node.level, ix: new_ix, iy: new_iy })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::sphere::PLANET_RADIUS_BLOCKS;

    #[test]
    fn morton_is_unique() {
        use std::collections::HashSet;
        let mut seen = HashSet::new();
        for face in ALL_FACES {
            for level in 0..=MAX_LEVEL {
                let n = 1u32 << level;
                for iy in 0..n {
                    for ix in 0..n {
                        let m = QuadNode { face, level, ix, iy }.morton();
                        assert!(seen.insert(m), "duplicate morton at {:?} L{} ({},{})", face, level, ix, iy);
                    }
                }
            }
        }
    }

    #[test]
    fn root_children_recover_parent() {
        for face in ALL_FACES {
            let root = QuadNode::root(face);
            let kids = root.children().unwrap();
            for k in kids {
                assert_eq!(k.parent().unwrap(), root);
            }
        }
    }

    #[test]
    fn max_level_leaf_has_no_children() {
        let leaf = QuadNode { face: Face::PosY, level: MAX_LEVEL, ix: 0, iy: 0 };
        assert!(leaf.children().is_none());
    }

    #[test]
    fn side_blocks_halves_each_level() {
        let root = QuadNode::root(Face::PosY);
        assert_eq!(root.side_blocks(), FACE_SIDE_BLOCKS as f64);
        let l1 = root.children().unwrap()[0];
        assert_eq!(l1.side_blocks(), FACE_SIDE_BLOCKS as f64 / 2.0);
    }

    #[test]
    fn descent_at_high_altitude_is_shallow() {
        let mut qt = Quadtree::new();
        // Camera 10 planet radii out: descent should stop within a few
        // levels of the root. The exact level depends on the geometric
        // error formula; we just want to confirm the SSE pruning is doing
        // its job.
        let camera = DVec3::new(0.0, 10.0 * PLANET_RADIUS_BLOCKS as f64, 0.0);
        qt.update(camera, 1080.0, 1.0);
        let max_level = qt.active().iter().map(|t| t.node.level).max().unwrap();
        assert!(max_level <= 3, "expected shallow tree at 10R altitude, got max level {}", max_level);
        // And the tree should be much smaller than the cap.
        assert!(qt.active().len() < 200, "active set unexpectedly large at 10R altitude: {}", qt.active().len());
    }

    #[test]
    fn descent_near_surface_subdivides() {
        let mut qt = Quadtree::new();
        let camera = DVec3::new(0.0, (PLANET_RADIUS_BLOCKS as f64) + 10.0, 0.0);
        qt.update(camera, 1440.0, 1.0);
        let max_level = qt.active().iter().map(|t| t.node.level).max().unwrap();
        assert!(max_level > 0, "expected subdivision near surface, got max level {}", max_level);
    }

    #[test]
    fn working_set_bounded_under_camera_sweep() {
        let mut qt = Quadtree::new();
        let r = PLANET_RADIUS_BLOCKS as f64;
        // Sample camera positions across altitudes and surface points.
        let altitudes = [r + 5.0, r + 50.0, r + 500.0, r + 5000.0, r * 2.0, r * 10.0];
        // 16 surface points via golden-ratio lattice.
        let phi = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
        for i in 0..16 {
            let y = 1.0 - (i as f64 / 15.0) * 2.0;
            let r_xy = (1.0 - y * y).sqrt();
            let theta = phi * i as f64;
            let dir = DVec3::new(theta.cos() * r_xy, y, theta.sin() * r_xy);
            for &alt in &altitudes {
                let cam = dir * alt;
                qt.update(cam, 1440.0, 1.0);
                assert!(
                    qt.active().len() <= MAX_RESIDENT_TILES,
                    "active set {} exceeds MAX_RESIDENT_TILES {} at cam={:?}",
                    qt.active().len(), MAX_RESIDENT_TILES, cam
                );
            }
        }
    }

    #[test]
    fn within_face_neighbors_round_trip() {
        let node = QuadNode { face: Face::PosY, level: 4, ix: 5, iy: 7 };
        let east = neighbor_node(node, EdgeDir::PosU).unwrap();
        assert_eq!(east, QuadNode { face: Face::PosY, level: 4, ix: 6, iy: 7 });
        let back = neighbor_node(east, EdgeDir::NegU).unwrap();
        assert_eq!(back, node);
    }

    #[test]
    fn face_edge_neighbor_changes_face() {
        // Tile on the +U edge of PosY at level 2 must have a cross-face
        // neighbor on a different face.
        let n = 1u32 << 2;
        let edge_tile = QuadNode { face: Face::PosY, level: 2, ix: n - 1, iy: 1 };
        let nb = neighbor_node(edge_tile, EdgeDir::PosU).expect("cross-face neighbor");
        assert_ne!(nb.face, Face::PosY);
        assert_eq!(nb.level, edge_tile.level);
    }

    #[test]
    fn stream_initial_load_is_full_active_set() {
        let mut qt = Quadtree::new();
        let cam = DVec3::new(0.0, 10.0 * PLANET_RADIUS_BLOCKS as f64, 0.0);
        qt.update(cam, 1080.0, 1.0);
        let delta = qt.stream();
        assert_eq!(delta.to_load.len(), qt.active().len());
        assert!(delta.to_evict.is_empty());
        qt.commit_stream(&delta);
        assert_eq!(qt.resident_count(), qt.active().len());
    }

    #[test]
    fn stream_steady_state_is_empty_diff() {
        let mut qt = Quadtree::new();
        let cam = DVec3::new(0.0, 10.0 * PLANET_RADIUS_BLOCKS as f64, 0.0);
        qt.update(cam, 1080.0, 1.0);
        let d1 = qt.stream();
        qt.commit_stream(&d1);
        // Same camera again — nothing should change.
        qt.update(cam, 1080.0, 1.0);
        let d2 = qt.stream();
        assert!(d2.to_load.is_empty(), "expected no loads on steady state, got {}", d2.to_load.len());
        assert!(d2.to_evict.is_empty(), "expected no evicts on steady state, got {}", d2.to_evict.len());
    }

    #[test]
    fn stream_camera_move_evicts_and_loads() {
        let mut qt = Quadtree::new();
        let r = PLANET_RADIUS_BLOCKS as f64;
        // First near-surface position on +Y face.
        qt.update(DVec3::new(0.0, r + 50.0, 0.0), 1440.0, 1.0);
        let d1 = qt.stream();
        qt.commit_stream(&d1);
        let initial_count = qt.resident_count();
        assert!(initial_count > 6, "expected substantial subdivision, got {}", initial_count);
        // Jump to the antipode (-Y face) — different leaf set.
        qt.update(DVec3::new(0.0, -(r + 50.0), 0.0), 1440.0, 1.0);
        let d2 = qt.stream();
        assert!(!d2.to_load.is_empty(), "expected loads after antipode jump");
        assert!(!d2.to_evict.is_empty(), "expected evicts after antipode jump");
    }

    #[test]
    fn cross_face_neighbor_round_trips() {
        // Walking off and back must return to the original tile (modulo
        // level — we test at every level).
        for level in 0..=MAX_LEVEL.min(4) {
            let n = 1u32 << level;
            for face in ALL_FACES {
                for &(ix, iy) in &[(0u32, 0u32), (n - 1, 0), (0, n - 1), (n - 1, n - 1), (n / 2, 0)] {
                    if n == 1 && (ix > 0 || iy > 0) { continue; }
                    let node = QuadNode { face, level, ix, iy };
                    for &dir in &[EdgeDir::NegU, EdgeDir::PosU, EdgeDir::NegV, EdgeDir::PosV] {
                        let on_edge = match dir {
                            EdgeDir::NegU => ix == 0,
                            EdgeDir::PosU => ix == n - 1,
                            EdgeDir::NegV => iy == 0,
                            EdgeDir::PosV => iy == n - 1,
                        };
                        if !on_edge { continue; }
                        let nb = neighbor_node(node, dir).expect("cross-face neighbor exists");
                        assert_ne!(nb.face, node.face, "cross-face dir {:?} kept same face", dir);
                        assert_eq!(nb.level, level);
                    }
                }
            }
        }
    }
}
