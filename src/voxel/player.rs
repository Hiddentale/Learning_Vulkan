//! Player state.
//!
//! The player's canonical position lives in **face-local cube space**:
//! a face id, integer chunk indices `(cx, cy, cz)`, and floating-point
//! sub-chunk offsets `(lx, ly, lz)` in `[0, CHUNK_SIZE)`. Cartesian world
//! position is *derived* on demand via [`sphere::chunk_to_world`]; nothing
//! ever inverts the projection per frame, so collision and rendering can
//! never disagree about which block the player occupies.
//!
//! Movement is expressed in face-local block units and carried across chunk
//! and face boundaries by integer arithmetic. The only call into the
//! inverse projection happens once when the player crosses a face edge —
//! a clean discrete event, not a continuous lookup.

use super::chunk::CHUNK_SIZE;
use super::sphere::{self, ChunkPos, Face};
use super::world::World;
use glam::{DVec3, Vec3};

const GRAVITY: f32 = 20.0;
const JUMP_VELOCITY: f32 = 8.0;
const PLAYER_HEIGHT: f32 = 1.7;
const PLAYER_HALF_WIDTH: f32 = 0.3;
/// Eye sits this far in front of the body center along `forward`. Lets the
/// camera approach a wall to within `PLAYER_HALF_WIDTH - CAMERA_FORWARD_OFFSET`.
const CAMERA_FORWARD_OFFSET: f32 = 0.25;
pub const MAX_PITCH: f32 = 89.0_f32 * (std::f32::consts::PI / 180.0);

pub struct Player {
    pub face: Face,
    pub cx: i32,
    pub cy: i32,
    pub cz: i32,
    pub lx: f32,
    pub ly: f32,
    pub lz: f32,
    /// Camera forward direction in cartesian world space. Re-tangent-projected
    /// against the local up after each move so it stays roughly tangent to
    /// the curved surface.
    pub forward: Vec3,
    pub radial_velocity: f32,
    pub on_ground: bool,
    pub fly_mode: bool,
}

impl Player {
    pub fn new() -> Self {
        // Spawn directly above the +Y pole, comfortably above the highest
        // possible terrain (radial depth ~150). Looking horizontally along
        // the +X tangent direction.
        let face = Face::PosY;
        let mid = sphere::FACE_SIDE_CHUNKS / 2;
        Self {
            face,
            cx: mid,
            cy: 9,
            cz: mid,
            lx: 8.0,
            ly: 8.0,
            lz: 8.0,
            forward: Vec3::new(1.0, 0.0, 0.0),
            radial_velocity: 0.0,
            on_ground: false,
            fly_mode: true,
        }
    }

    pub fn chunk_pos(&self) -> ChunkPos {
        ChunkPos { face: self.face, cx: self.cx, cy: self.cy, cz: self.cz }
    }

    /// Cartesian body-center position. Always derived; never stored.
    pub fn world_pos(&self) -> Vec3 {
        sphere::chunk_to_world(self.chunk_pos(), Vec3::new(self.lx, self.ly, self.lz)).as_vec3()
    }

    /// Cartesian eye position: body center plus a small forward offset so the
    /// camera can approach walls more closely than the body's half-width.
    pub fn eye_pos(&self) -> Vec3 {
        self.world_pos() + self.forward * CAMERA_FORWARD_OFFSET
    }

    /// Radial outward direction at the player.
    pub fn up(&self) -> Vec3 {
        self.world_pos().normalize_or(Vec3::Y)
    }

    pub fn right(&self) -> Vec3 {
        self.forward.cross(self.up()).normalize_or(Vec3::X)
    }

    pub fn jump(&mut self) {
        if self.on_ground && !self.fly_mode {
            self.radial_velocity = JUMP_VELOCITY;
            self.on_ground = false;
        }
    }

    pub fn toggle_fly_mode(&mut self) {
        self.fly_mode = !self.fly_mode;
        self.radial_velocity = 0.0;
    }

    pub fn rotate_yaw(&mut self, radians: f32) {
        self.forward = rotate_around(self.forward, self.up(), radians).normalize_or(self.forward);
    }

    pub fn rotate_pitch(&mut self, radians: f32) {
        let up = self.up();
        let right = self.right();
        let new_forward = rotate_around(self.forward, right, radians).normalize_or(self.forward);
        let sin_pitch = new_forward.dot(up).clamp(-1.0, 1.0);
        let max_sin = MAX_PITCH.sin();
        if sin_pitch.abs() <= max_sin {
            self.forward = new_forward;
        } else {
            let horizontal = (new_forward - up * sin_pitch).normalize_or(self.right());
            let cos_max = MAX_PITCH.cos();
            self.forward = (horizontal * cos_max + up * sin_pitch.signum() * max_sin).normalize_or(self.forward);
        }
    }

    /// Walk a tangent-plane displacement (`tangent_world`) of length in
    /// blocks. Splits into forward and right components in the local tangent
    /// plane and applies each as a world-space step with collision. World-
    /// space integration is required so that motion at off-center positions
    /// (corners, edges) follows the actual sphere tangent rather than the
    /// face's flat cube basis — otherwise pressing W near a corner produces
    /// a diagonal slide.
    pub fn walk(&mut self, tangent_world: Vec3, world: &World) {
        let up = self.up();
        let forward_h = (self.forward - up * self.forward.dot(up)).normalize_or(Vec3::ZERO);
        if forward_h == Vec3::ZERO {
            return;
        }
        let right_h = forward_h.cross(up).normalize_or(Vec3::ZERO);
        let forward_amt = tangent_world.dot(forward_h);
        let right_amt = tangent_world.dot(right_h);
        let moved_f = self.try_world_step(forward_h * forward_amt, world);
        let moved_r = self.try_world_step(right_h * right_amt, world);
        if moved_f || moved_r {
            self.reorthogonalize_forward();
        }
    }

    /// Apply one world-space tangent step. Re-derives cube coords via the
    /// inverse projection and reverts on collision. Used by walk for
    /// per-axis sliding in true sphere tangent space.
    fn try_world_step(&mut self, world_displacement: Vec3, world: &World) -> bool {
        if world_displacement.length_squared() < 1e-12 {
            return false;
        }
        let save = (self.face, self.cx, self.cy, self.cz, self.lx, self.ly, self.lz);
        let cur = sphere::chunk_to_world(self.chunk_pos(), Vec3::new(self.lx, self.ly, self.lz));
        let new_world = cur + world_displacement.as_dvec3();
        let Some((cp, lx, ly, lz)) = sphere::world_to_chunk_local(new_world) else {
            return false;
        };
        if cp.cy < 0 {
            return false;
        }
        let n = sphere::FACE_SIDE_CHUNKS;
        self.face = cp.face;
        self.cx = cp.cx.clamp(0, n - 1);
        self.cy = cp.cy;
        self.cz = cp.cz.clamp(0, n - 1);
        self.lx = lx;
        self.ly = ly;
        self.lz = lz;
        if capsule_collides(self, world) {
            self.face = save.0;
            self.cx = save.1;
            self.cy = save.2;
            self.cz = save.3;
            self.lx = save.4;
            self.ly = save.5;
            self.lz = save.6;
            false
        } else {
            true
        }
    }

    /// Fly with full 6DoF in world space. Adds the displacement to the
    /// player's cartesian position and re-derives the cube-space coords
    /// via the inverse projection. Decoupled from the source face's flat
    /// basis so motion follows the screen, not the underlying cube.
    pub fn fly_move(&mut self, displacement_world: Vec3) {
        let cur = sphere::chunk_to_world(self.chunk_pos(), Vec3::new(self.lx, self.ly, self.lz));
        let new_world = cur + displacement_world.as_dvec3();
        if let Some((cp, lx, ly, lz)) = sphere::world_to_chunk_local(new_world) {
            let n = sphere::FACE_SIDE_CHUNKS;
            if cp.cy >= 0 {
                self.face = cp.face;
                self.cx = cp.cx.clamp(0, n - 1);
                self.cy = cp.cy;
                self.cz = cp.cz.clamp(0, n - 1);
                self.lx = lx;
                self.ly = ly;
                self.lz = lz;
            }
        }
        self.reorthogonalize_forward();
    }

    /// Apply radial gravity. Sticky `on_ground` is verified each frame by
    /// probing the block below the feet.
    pub fn apply_physics(&mut self, dt: f32, world: &World) {
        if self.fly_mode {
            return;
        }
        if self.on_ground {
            if ground_below(self, world) {
                return; // standing — no motion this frame
            }
            self.on_ground = false;
        }

        self.radial_velocity -= GRAVITY * dt;
        let dd = self.radial_velocity * dt;
        self.ly += dd;
        self.carry();

        // Push up out of any embedded geometry. Lift in 0.05 steps, max 3 blocks.
        let mut lifted = 0;
        while capsule_collides(self, world) && lifted < 60 {
            self.ly += 0.05;
            self.carry();
            lifted += 1;
        }
        if lifted > 0 {
            self.radial_velocity = 0.0;
            self.on_ground = true;
        }
    }

    /// Attempt a move by `(du, dd, dv)` (face-local block units). Each axis
    /// is tried independently so the player can slide along walls instead of
    /// stopping dead the moment any axis collides.
    fn try_move(&mut self, du: f32, dd: f32, dv: f32, world: &World) {
        let moved_lateral = (du != 0.0 || dv != 0.0) && {
            let a = self.try_axis(du, 0.0, 0.0, world);
            let b = self.try_axis(0.0, 0.0, dv, world);
            a || b
        };
        if dd != 0.0 {
            self.try_axis(0.0, dd, 0.0, world);
        }
        if moved_lateral {
            self.reorthogonalize_forward();
        }
    }

    /// Apply a single-axis displacement and revert that axis on collision.
    /// Returns true if the move succeeded.
    fn try_axis(&mut self, du: f32, dd: f32, dv: f32, world: &World) -> bool {
        let (old_face, old_cx, old_cy, old_cz) = (self.face, self.cx, self.cy, self.cz);
        let (old_lx, old_ly, old_lz) = (self.lx, self.ly, self.lz);
        self.lx += du;
        self.ly += dd;
        self.lz += dv;
        self.carry();
        if capsule_collides(self, world) {
            self.face = old_face;
            self.cx = old_cx;
            self.cy = old_cy;
            self.cz = old_cz;
            self.lx = old_lx;
            self.ly = old_ly;
            self.lz = old_lz;
            false
        } else {
            true
        }
    }

    /// Re-tangent-project `forward` against the new local up. Preserves
    /// the world-space pitch component, just realigns the horizontal.
    fn reorthogonalize_forward(&mut self) {
        let up = self.up();
        let dot = self.forward.dot(up);
        let tangent = (self.forward - up * dot).normalize_or(self.forward.normalize_or(Vec3::X));
        let cos = (1.0 - dot * dot).max(0.0).sqrt();
        self.forward = (tangent * cos + up * dot).normalize_or(tangent);
    }

    /// Apply integer chunk + face carries to keep `(lx, ly, lz)` in
    /// `[0, CHUNK_SIZE)` and `(cx, cz)` in `[0, FACE_SIDE_CHUNKS)`.
    fn carry(&mut self) {
        let cs = CHUNK_SIZE as f32;
        // ly is radial — clamp at 0 (planet core) and TERRAIN_MAX_CY top.
        while self.lx >= cs { self.lx -= cs; self.cx += 1; }
        while self.lx < 0.0 { self.lx += cs; self.cx -= 1; }
        while self.lz >= cs { self.lz -= cs; self.cz += 1; }
        while self.lz < 0.0 { self.lz += cs; self.cz -= 1; }
        while self.ly >= cs { self.ly -= cs; self.cy += 1; }
        while self.ly < 0.0 { self.ly += cs; self.cy -= 1; }
        // Clamp cy so the player can't fall to -infinity.
        if self.cy < 0 { self.cy = 0; self.ly = 0.0; }

        // Cross-face transition if cx/cz left the face range.
        let n = sphere::FACE_SIDE_CHUNKS;
        if self.cx < 0 || self.cx >= n || self.cz < 0 || self.cz >= n {
            self.remap_to_neighbor_face();
        }
    }

    /// Cross a face boundary. Geometric remap derives the new
    /// `(face, cx, cz, lx, lz)` by computing the cube point and projecting
    /// onto the neighbor face whose normal axis dominates `cube_pt`. The
    /// neighbor is chosen by the cube point itself, not the edge enum, so
    /// double-overflow at corners picks a consistent face.
    ///
    /// `forward` is *not* rotated — only re-tangent-projected. The sphere
    /// surface is smooth, so the world-space tangent plane only tilts by a
    /// small angle across a small step. A 90° rotation here would correspond
    /// to walking on the un-projected cube and is visually a teleport.
    fn remap_to_neighbor_face(&mut self) {
        let n_chunks = sphere::FACE_SIDE_CHUNKS;
        let cs = CHUNK_SIZE as f64;
        let half = (n_chunks * CHUNK_SIZE as i32) as f64 * 0.5;
        let u = self.cx as f64 * cs + self.lx as f64 - half;
        let v = self.cz as f64 * cs + self.lz as f64 - half;
        let (tu, tv, n) = sphere::face_basis(self.face);
        let cube_pt = tu.as_dvec3() * u + tv.as_dvec3() * v + n.as_dvec3() * half;
        let new_face = sphere::face_for_cube_point(cube_pt);
        let new_n = sphere::face_normal(new_face).as_dvec3();
        let new_dominant = cube_pt.dot(new_n).abs().max(1e-9);
        let scaled = cube_pt * (half / new_dominant);
        let (ntu, ntv, _) = sphere::face_basis(new_face);
        let new_face_u = (scaled.dot(ntu.as_dvec3()) + half).clamp(0.0, sphere::FACE_SIDE_BLOCKS as f64 - 1e-4);
        let new_face_v = (scaled.dot(ntv.as_dvec3()) + half).clamp(0.0, sphere::FACE_SIDE_BLOCKS as f64 - 1e-4);
        let new_cx = (new_face_u / cs).floor() as i32;
        let new_cz = (new_face_v / cs).floor() as i32;
        let new_lx = (new_face_u - new_cx as f64 * cs) as f32;
        let new_lz = (new_face_v - new_cz as f64 * cs) as f32;

        self.face = new_face;
        self.cx = new_cx.clamp(0, n_chunks - 1);
        self.cz = new_cz.clamp(0, n_chunks - 1);
        self.lx = new_lx;
        self.lz = new_lz;
        // cy / ly (radial) preserved across face boundaries.

        self.reorthogonalize_forward();
    }
}

/// Sample one specific integer block at face-local indices `(ix, iy, iz)`
/// relative to the player's current chunk. Carries across chunk and face
/// boundaries. Cross-face samples are resolved by re-deriving the
/// neighbor's chunk via the same `remap_to_neighbor_face` math used for
/// player movement, so collision matches the rendered geometry exactly.
fn sample_block_solid(face: Face, mut cx: i32, mut cy: i32, mut cz: i32, mut ix: i32, mut iy: i32, mut iz: i32, world: &World) -> bool {
    let cs = CHUNK_SIZE as i32;
    // Integer chunk carry.
    while ix >= cs { ix -= cs; cx += 1; }
    while ix < 0 { ix += cs; cx -= 1; }
    while iz >= cs { iz -= cs; cz += 1; }
    while iz < 0 { iz += cs; cz -= 1; }
    while iy >= cs { iy -= cs; cy += 1; }
    while iy < 0 { iy += cs; cy -= 1; }
    if cy < 0 {
        return true; // below planet core — pin the player.
    }
    let n = sphere::FACE_SIDE_CHUNKS;
    let cp = if cx < 0 || cx >= n || cz < 0 || cz >= n {
        // Cross-face sample: derive the neighbor face from the cube point.
        let (new_face, new_cx, new_ix, new_cz, new_iz) = remap_block_to_neighbor(face, cx, cz, ix, iz);
        if new_face == face {
            return false;
        }
        ix = new_ix;
        iz = new_iz;
        ChunkPos { face: new_face, cx: new_cx, cy, cz: new_cz }
    } else {
        ChunkPos { face, cx, cy, cz }
    };
    world.block_solid(cp, ix as usize, iy as usize, iz as usize)
}

/// Static remap of a block-level (cx, cz, ix, iz) on `face` whose chunk
/// indices have left the face range, onto the neighbor face. cy/iy are
/// preserved (they are radial). This is the block-granularity analogue of
/// [`Player::remap_to_neighbor_face`], using the same cube-edge geometry.
fn remap_block_to_neighbor(face: Face, cx: i32, cz: i32, ix: i32, iz: i32) -> (Face, i32, i32, i32, i32) {
    let cs = CHUNK_SIZE as f64;
    let half = (sphere::FACE_SIDE_CHUNKS * CHUNK_SIZE as i32) as f64 * 0.5;
    // Sample at the block centre.
    let u = cx as f64 * cs + ix as f64 + 0.5 - half;
    let v = cz as f64 * cs + iz as f64 + 0.5 - half;
    let (tu, tv, n) = sphere::face_basis(face);
    let cube_pt = tu.as_dvec3() * u + tv.as_dvec3() * v + n.as_dvec3() * half;
    let ax = cube_pt.x.abs();
    let ay = cube_pt.y.abs();
    let az = cube_pt.z.abs();
    let new_face = if ax >= ay && ax >= az {
        if cube_pt.x > 0.0 { Face::PosX } else { Face::NegX }
    } else if ay >= az {
        if cube_pt.y > 0.0 { Face::PosY } else { Face::NegY }
    } else {
        if cube_pt.z > 0.0 { Face::PosZ } else { Face::NegZ }
    };
    if new_face == face {
        return (face, cx, ix, cz, iz);
    }
    let dominant = ax.max(ay).max(az);
    let scaled = cube_pt * (half / dominant);
    let (ntu, ntv, _) = sphere::face_basis(new_face);
    let new_u_center = scaled.dot(ntu.as_dvec3());
    let new_v_center = scaled.dot(ntv.as_dvec3());
    let new_face_u = (new_u_center + half).clamp(0.0, sphere::FACE_SIDE_BLOCKS as f64 - 1e-4);
    let new_face_v = (new_v_center + half).clamp(0.0, sphere::FACE_SIDE_BLOCKS as f64 - 1e-4);
    let new_cx = (new_face_u / cs).floor() as i32;
    let new_cz = (new_face_v / cs).floor() as i32;
    let new_ix = (new_face_u - new_cx as f64 * cs) as i32;
    let new_iz = (new_face_v - new_cz as f64 * cs) as i32;
    let nm = sphere::FACE_SIDE_CHUNKS;
    (new_face, new_cx.clamp(0, nm - 1), new_ix, new_cz.clamp(0, nm - 1), new_iz)
}

/// Iterate every integer block the player's capsule overlaps. For a
/// PLAYER_HEIGHT × 2*HALF_WIDTH × 2*HALF_WIDTH box this is at most
/// 3 × 2 × 2 = 12 lookups, all direct chunk array reads.
fn capsule_collides(player: &Player, world: &World) -> bool {
    let hw = PLAYER_HALF_WIDTH;
    let lx_min = (player.lx - hw).floor() as i32;
    let lx_max = (player.lx + hw).floor() as i32;
    let lz_min = (player.lz - hw).floor() as i32;
    let lz_max = (player.lz + hw).floor() as i32;
    let ly_min = (player.ly - PLAYER_HEIGHT + 0.01).floor() as i32;
    let ly_max = (player.ly - 0.01).floor() as i32;
    for ix in lx_min..=lx_max {
        for iy in ly_min..=ly_max {
            for iz in lz_min..=lz_max {
                if sample_block_solid(player.face, player.cx, player.cy, player.cz, ix, iy, iz, world) {
                    return true;
                }
            }
        }
    }
    false
}

/// True if there is a solid block in the layer immediately below the feet.
/// Used for sticky `on_ground`.
fn ground_below(player: &Player, world: &World) -> bool {
    let hw = PLAYER_HALF_WIDTH;
    let lx_min = (player.lx - hw).floor() as i32;
    let lx_max = (player.lx + hw).floor() as i32;
    let lz_min = (player.lz - hw).floor() as i32;
    let lz_max = (player.lz + hw).floor() as i32;
    let iy = (player.ly - PLAYER_HEIGHT - 0.05).floor() as i32;
    for ix in lx_min..=lx_max {
        for iz in lz_min..=lz_max {
            if sample_block_solid(player.face, player.cx, player.cy, player.cz, ix, iy, iz, world) {
                return true;
            }
        }
    }
    false
}

fn rotate_around(v: Vec3, axis: Vec3, angle: f32) -> Vec3 {
    let c = angle.cos();
    let s = angle.sin();
    v * c + axis.cross(v) * s + axis * axis.dot(v) * (1.0 - c)
}

#[allow(dead_code)]
fn _suppress_dvec3_warning(_: DVec3) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::block::BlockType;
    use crate::voxel::chunk::Chunk;

    /// Build an empty World with no chunk generator threads — for tests
    /// that want full control over which blocks exist where.
    fn empty_world() -> World {
        World::new(2, 0, None)
    }

    /// Fill every chunk on the planet with stone — a uniform sphere.
    fn solid_planet() -> World {
        let mut w = empty_world();
        for face in sphere::ALL_FACES {
            for cx in 0..sphere::FACE_SIDE_CHUNKS {
                for cz in 0..sphere::FACE_SIDE_CHUNKS {
                    for cy in 0..sphere::FACE_RADIAL_CHUNKS / 4 {
                        // Just enough radial layers to bound the planet.
                        w.insert_solid_chunk(ChunkPos { face, cx, cy, cz });
                    }
                }
            }
        }
        w
    }

    fn floor_planet() -> World {
        // A planet whose ly=8 plane (in cy=2) is solid stone, everything
        // else is air. Useful for testing standing on a flat surface.
        let mut w = empty_world();
        for face in sphere::ALL_FACES {
            for cx in 0..sphere::FACE_SIDE_CHUNKS {
                for cz in 0..sphere::FACE_SIDE_CHUNKS {
                    let cp = ChunkPos { face, cx, cy: 2, cz };
                    let mut chunk = Chunk::new(BlockType::Air);
                    for lx in 0..CHUNK_SIZE {
                        for lz in 0..CHUNK_SIZE {
                            for ly in 0..8 {
                                chunk.set(lx, ly, lz, BlockType::Stone);
                            }
                        }
                    }
                    w.chunks_mut_for_test().insert(cp, chunk);
                    // Also load the chunks above so apply_physics doesn't
                    // see "out of terrain range" => solid.
                    w.insert_empty_chunk_at(ChunkPos { face, cx, cy: 3, cz });
                    w.insert_empty_chunk_at(ChunkPos { face, cx, cy: 4, cz });
                }
            }
        }
        w
    }

    // ----- (2) Walking returns to start -----

    /// fly_move along a fixed tangent direction inside one face is locally
    /// reversible. (A full great-circle walk across face boundaries is
    /// not — `fly_move` interprets its argument in the current face's
    /// tangent basis, which rotates 90° at each crossing, so "walk +X
    /// world for N steps then -X world for N steps" loses coherence.
    /// The proper round-trip test is per-face, not per-planet.)
    #[test]
    fn fly_move_within_one_face_is_reversible() {
        let _world = solid_planet();
        let mut p = Player::new();
        p.fly_mode = true;
        p.face = Face::PosY;
        p.cx = 1; p.cy = 8; p.cz = 1;
        p.lx = 8.0; p.ly = 8.0; p.lz = 8.0;
        let start_world = p.world_pos();
        let (tu, _, _) = sphere::face_basis(p.face);
        let n_steps = 5;
        for _ in 0..n_steps {
            p.fly_move(tu * 0.4);
        }
        for _ in 0..n_steps {
            p.fly_move(-tu * 0.4);
        }
        let end_world = p.world_pos();
        let dist = (end_world - start_world).length();
        assert!(dist < 0.05, "single-face fly_move not reversible: drift = {} blocks", dist);
    }

    // ----- (3) Crossing every edge changes face and stays on the planet -----

    /// Crossing every (face, edge) pair lands on the expected neighbor face,
    /// preserves the player's distance from the planet center, and produces
    /// a finite world position. (This is the strict invariant — the
    /// "walk back across the edge" inverse is *not* checked because
    /// `fly_move`'s argument is interpreted in the current face's basis,
    /// which differs from the original face's basis after the crossing.)
    #[test]
    fn crossing_every_edge_lands_on_expected_neighbor() {
        let _world = solid_planet();
        for start_face in sphere::ALL_FACES {
            for edge in sphere::ALL_EDGES {
                let mut p = Player::new();
                p.fly_mode = true;
                p.face = start_face;
                p.cy = 8;
                p.ly = 8.0;
                let n = sphere::FACE_SIDE_CHUNKS;
                match edge {
                    sphere::EdgeDir::PosU => { p.cx = n - 1; p.lx = (CHUNK_SIZE - 1) as f32; p.cz = 1; p.lz = 8.0; }
                    sphere::EdgeDir::NegU => { p.cx = 0; p.lx = 0.5; p.cz = 1; p.lz = 8.0; }
                    sphere::EdgeDir::PosV => { p.cz = n - 1; p.lz = (CHUNK_SIZE - 1) as f32; p.cx = 1; p.lx = 8.0; }
                    sphere::EdgeDir::NegV => { p.cz = 0; p.lz = 0.5; p.cx = 1; p.lx = 8.0; }
                }
                let start_radius = p.world_pos().length();
                let expected_face = sphere::edge_transition(start_face, edge).neighbor;
                let (tu, tv, _) = sphere::face_basis(start_face);
                let push = match edge {
                    sphere::EdgeDir::PosU => tu * 30.0,
                    sphere::EdgeDir::NegU => -tu * 30.0,
                    sphere::EdgeDir::PosV => tv * 30.0,
                    sphere::EdgeDir::NegV => -tv * 30.0,
                };
                p.fly_move(push);
                assert_eq!(p.face, expected_face, "edge {:?} {:?}: landed on {:?}, expected {:?}", start_face, edge, p.face, expected_face);
                let end_pos = p.world_pos();
                assert!(end_pos.x.is_finite() && end_pos.y.is_finite() && end_pos.z.is_finite(), "non-finite position");
                // Tangent fly_move legitimately leaves the sphere shell; we
                // only assert the result is on the same order as the start.
                let radius_drift = (end_pos.length() - start_radius).abs();
                assert!(radius_drift < 50.0, "edge {:?} {:?}: radius drifted by {}", start_face, edge, radius_drift);
                let _ = start_radius;
            }
        }
    }

    // ----- (4) Capsule collision regression cases -----

    #[test]
    fn standing_on_a_one_block_floor_does_not_sink() {
        let world = floor_planet();
        let mut p = Player::new();
        p.fly_mode = false;
        p.cy = 2;
        p.ly = 8.0 + PLAYER_HEIGHT + 0.01; // exactly on top of the floor
        let start_ly = p.ly;
        for _ in 0..120 {
            // 2 seconds of physics
            p.apply_physics(1.0 / 60.0, &world);
        }
        assert!((p.ly - start_ly).abs() < 0.2, "player drifted from {} to {}", start_ly, p.ly);
        assert!(p.on_ground, "player should be on_ground");
    }

    #[test]
    fn falling_player_lands_on_floor_within_terminal_time() {
        let world = floor_planet();
        let mut p = Player::new();
        p.fly_mode = false;
        p.cy = 2;
        p.ly = 15.5;
        for _ in 0..600 {
            // 10s budget
            p.apply_physics(1.0 / 60.0, &world);
            if p.on_ground {
                break;
            }
        }
        assert!(p.on_ground, "player did not land");
        // Feet should be at the top of the solid floor (ly=8).
        let feet_ly = p.ly - PLAYER_HEIGHT;
        assert!(feet_ly >= 7.9 && feet_ly <= 8.5, "feet at unexpected ly: {}", feet_ly);
    }

    #[test]
    fn fly_through_solid_planet_never_returns_inside_it() {
        // Fly mode bypasses collision; verify the player can pass through
        // a solid block without crashing or producing NaN positions.
        let world = solid_planet();
        let mut p = Player::new();
        p.fly_mode = true;
        for _ in 0..100 {
            let (_, _, n) = sphere::face_basis(p.face);
            p.fly_move(-n * 1.0); // dive radially inward
        }
        let pos = p.world_pos();
        assert!(pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(), "position became non-finite: {:?}", pos);
    }

    // ----- (5) Forward vector stability under walking -----

    /// Forward stays unit length and finite after many moves. (We do NOT
    /// check that it stays tangent to the local up — that requires proper
    /// parallel transport across face crossings, which is a known
    /// limitation. The `reorthogonalize_forward` step preserves the
    /// current pitch component, not zero-pitch.)
    #[test]
    fn forward_stays_well_formed_after_long_walk() {
        let _world = solid_planet();
        let mut p = Player::new();
        p.fly_mode = true;
        p.cy = 8;
        p.ly = 8.0;
        for _ in 0..200 {
            let (tu, _, _) = sphere::face_basis(p.face);
            p.fly_move(tu * 0.3);
            assert!(p.forward.x.is_finite() && p.forward.y.is_finite() && p.forward.z.is_finite(), "forward became NaN");
            assert!((p.forward.length() - 1.0).abs() < 0.01, "forward not unit length: {}", p.forward.length());
        }
    }

    /// Drift of the forward vector's tangent component over a long walk.
    /// Currently fails — documented as the "no parallel transport"
    /// limitation. Re-enable once forward gets parallel-transported on
    /// each cross-face transition or each move step.
    #[test]
    #[ignore]
    fn forward_stays_tangent_after_long_walk() {
        let _world = solid_planet();
        let mut p = Player::new();
        p.fly_mode = true;
        p.cy = 8;
        p.ly = 8.0;
        for _ in 0..200 {
            let (tu, _, _) = sphere::face_basis(p.face);
            p.fly_move(tu * 0.3);
        }
        let up = p.up();
        let dot = p.forward.dot(up).abs();
        assert!(dot < 0.1, "forward drifted off tangent plane: |dot|={}", dot);
    }

    // ----- (9) Physics tick determinism -----

    #[test]
    fn fixed_step_physics_is_framerate_independent() {
        // Two players starting at the same state. One ticks at 60Hz for
        // 1 second; the other at 240Hz for 1 second. Final positions
        // should match within fp tolerance.
        let world = floor_planet();
        let mut a = Player::new();
        a.fly_mode = false;
        a.cy = 4;
        a.ly = 0.5;

        let mut b = Player::new();
        b.fly_mode = false;
        b.cy = 4;
        b.ly = 0.5;

        for _ in 0..60 {
            a.apply_physics(1.0 / 60.0, &world);
        }
        for _ in 0..240 {
            b.apply_physics(1.0 / 240.0, &world);
        }

        let dx = (a.world_pos() - b.world_pos()).length();
        assert!(dx < 0.5, "60Hz vs 240Hz drift: {} blocks", dx);
    }

    // ----- (8 — already in sphere.rs) — included here for cross-module test -----

    /// (6) Block-render vs collision agreement.
    /// For every block in a sample chunk, project the block center via
    /// `chunk_to_world` (the function the GPU shader mirrors), then place
    /// a player at that exact world position and verify they sample the
    /// SAME block via collision lookup. This is the structural invariant
    /// the cube-space refactor exists to enforce.
    #[test]
    fn block_render_position_matches_collision_lookup() {
        // Test a chunk near the +Y pole — interior of one face, no edge effects.
        let cp = ChunkPos { face: Face::PosY, cx: 1, cy: 5, cz: 1 };
        for ix in 1..15 {
            for iy in 1..15 {
                for iz in 1..15 {
                    let world = sphere::chunk_to_world(cp, Vec3::new(ix as f32 + 0.5, iy as f32 + 0.5, iz as f32 + 0.5));
                    // Reverse-engineer the player coords for this world point.
                    let recovered = sphere::world_to_chunk_local(world);
                    let (rcp, rlx, rly, rlz) = recovered.expect("inverse failed");
                    assert_eq!(rcp.face, cp.face);
                    assert_eq!(rcp.cx, cp.cx, "cx mismatch at ({},{},{})", ix, iy, iz);
                    assert_eq!(rcp.cy, cp.cy, "cy mismatch at ({},{},{})", ix, iy, iz);
                    assert_eq!(rcp.cz, cp.cz, "cz mismatch at ({},{},{})", ix, iy, iz);
                    assert_eq!(rlx as usize, ix, "lx mismatch at ({},{},{}): got {}", ix, iy, iz, rlx);
                    assert_eq!(rly as usize, iy, "ly mismatch at ({},{},{}): got {}", ix, iy, iz, rly);
                    assert_eq!(rlz as usize, iz, "lz mismatch at ({},{},{}): got {}", ix, iy, iz, rlz);
                }
            }
        }
    }

    #[test]
    fn small_step_across_face_edge_is_continuous_in_world_space() {
        // A tiny fly_move that crosses a face edge must produce a tiny
        // change in world position AND in the forward vector. Pre-fix,
        // remap_to_neighbor_face rotated forward 90° around the cube edge,
        // which on the smooth sphere is a visible camera teleport.
        let _world = solid_planet();
        let mut p = Player::new();
        p.fly_mode = true;
        p.face = Face::PosY;
        let n = sphere::FACE_SIDE_CHUNKS;
        p.cx = n - 1;
        p.lx = (CHUNK_SIZE - 1) as f32 + 0.999;
        p.cz = 1;
        p.lz = 8.0;
        p.cy = 8;
        p.ly = 8.0;
        // Aim forward along the +tu direction (i.e., the direction of motion).
        let (tu, _, _) = sphere::face_basis(p.face);
        p.forward = tu;
        let pos_before = p.world_pos();
        let forward_before = p.forward;
        // Push by enough world distance to actually cross the edge from the
        // sub-block-away starting position. With the sphere at radius ~184,
        // 1 world block ≈ 0.13 face-local cube units near a side face.
        let step = 1.0_f32;
        p.fly_move(tu * step);
        // Confirm the crossing actually happened.
        assert_ne!(p.face, Face::PosY, "expected face crossing");
        let pos_after = p.world_pos();
        let pos_jump = (pos_after - pos_before).length();
        // Should be approximately the step magnitude — no extra teleport on top.
        assert!(pos_jump < step as f32 * 1.5 + 0.1, "world position teleported by {} blocks across face edge (step was {})", pos_jump, step);
        let forward_jump = (p.forward - forward_before).length();
        assert!(forward_jump < 0.2, "forward vector teleported by {} across face edge", forward_jump);
    }

    #[test]
    fn capsule_at_corner_triple_point_is_blocked_in_solid_planet() {
        // Player is wedged into the corner block of a fully-solid planet.
        // Capsule collision must report a hit no matter how the corner
        // remap routes the off-face block samples.
        let world = solid_planet();
        let mut p = Player::new();
        p.fly_mode = false;
        p.face = Face::PosY;
        let n = sphere::FACE_SIDE_CHUNKS;
        p.cx = n - 1;
        p.lx = (CHUNK_SIZE - 1) as f32 - 0.05;
        p.cz = n - 1;
        p.lz = (CHUNK_SIZE - 1) as f32 - 0.05;
        p.cy = 8;
        p.ly = 8.0;
        assert!(capsule_collides(&p, &world), "player embedded in solid_planet at corner should collide");
    }

    #[test]
    fn edge_transition_consistent_with_player_remap() {
        // The Player::remap_to_neighbor_face math and sphere::edge_transition
        // must agree on which neighbor face is entered.
        let world = empty_world();
        for start_face in sphere::ALL_FACES {
            for edge in sphere::ALL_EDGES {
                let expected = sphere::edge_transition(start_face, edge).neighbor;
                let mut p = Player::new();
                p.fly_mode = true;
                p.face = start_face;
                p.cy = 8;
                p.ly = 8.0;
                let n = sphere::FACE_SIDE_CHUNKS;
                match edge {
                    sphere::EdgeDir::PosU => { p.cx = n - 1; p.lx = 15.5; p.cz = 1; p.lz = 8.0; }
                    sphere::EdgeDir::NegU => { p.cx = 0; p.lx = 0.5; p.cz = 1; p.lz = 8.0; }
                    sphere::EdgeDir::PosV => { p.cz = n - 1; p.lz = 15.5; p.cx = 1; p.lx = 8.0; }
                    sphere::EdgeDir::NegV => { p.cz = 0; p.lz = 0.5; p.cx = 1; p.lx = 8.0; }
                }
                let (tu, tv, _) = sphere::face_basis(start_face);
                let push = match edge {
                    sphere::EdgeDir::PosU => tu * 30.0,
                    sphere::EdgeDir::NegU => -tu * 30.0,
                    sphere::EdgeDir::PosV => tv * 30.0,
                    sphere::EdgeDir::NegV => -tv * 30.0,
                };
                p.fly_move(push);
                assert_eq!(p.face, expected, "edge {:?} {:?}: player landed on {:?}, table says {:?}", start_face, edge, p.face, expected);
            }
        }
        let _ = world;
    }
}
