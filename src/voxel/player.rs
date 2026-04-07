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

    /// Cartesian world position. Always derived; never stored.
    pub fn world_pos(&self) -> Vec3 {
        sphere::chunk_to_world(self.chunk_pos(), Vec3::new(self.lx, self.ly, self.lz)).as_vec3()
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
    /// blocks. Projects onto the player's current face basis to get
    /// face-local `(du, dv)` and applies them with collision.
    pub fn walk(&mut self, tangent_world: Vec3, world: &World) {
        let (tu, tv, _) = sphere::face_basis(self.face);
        let du = tangent_world.dot(tu);
        let dv = tangent_world.dot(tv);
        self.try_move(du, 0.0, dv, world);
    }

    /// Fly with full 6DoF in the player's local frame: tangent (du, dv) +
    /// radial (dd). Used by fly mode; no collision response.
    pub fn fly_move(&mut self, displacement_world: Vec3) {
        let (tu, tv, n) = sphere::face_basis(self.face);
        let du = displacement_world.dot(tu);
        let dv = displacement_world.dot(tv);
        let dd = displacement_world.dot(n);
        self.lx += du;
        self.ly += dd;
        self.lz += dv;
        self.carry();
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

    /// Attempt a horizontal move by `(du, dv)` (face-local block units).
    /// On collision, revert. No vertical motion — gravity is handled in
    /// `apply_physics`.
    fn try_move(&mut self, du: f32, dd: f32, dv: f32, world: &World) {
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
        } else if du != 0.0 || dv != 0.0 {
            self.reorthogonalize_forward();
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

    /// Cross a face boundary. The player's current `(cx, lx, cz, lz)` are
    /// out of face range; reproject onto the neighbor face by computing the
    /// cube-surface point and dotting against the neighbor's basis.
    ///
    /// This is **not** the inverse of [`sphere::cube_to_sphere_unit`] —
    /// it works directly on the cube point with no normalization, no
    /// fixed-point iteration, and no rounding ambiguity. The geometry it
    /// preserves is the cube edge, which is the actual boundary between
    /// the two faces.
    fn remap_to_neighbor_face(&mut self) {
        let cs = CHUNK_SIZE as f64;
        let half = (sphere::FACE_SIDE_CHUNKS * CHUNK_SIZE as i32) as f64 * 0.5;
        // Face-local cube tangent coords (centered at cube center).
        let u = self.cx as f64 * cs + self.lx as f64 - half;
        let v = self.cz as f64 * cs + self.lz as f64 - half;
        let (tu, tv, n) = sphere::face_basis(self.face);
        // Cube point on the original face's surface plane, slightly past
        // the edge in the tangent direction.
        let cube_pt = tu.as_dvec3() * u + tv.as_dvec3() * v + n.as_dvec3() * half;

        // The neighbor face is the one whose normal axis dominates.
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
        if new_face == self.face {
            return;
        }
        // Scale the cube point so its dominant axis sits on the new face plane.
        let dominant = ax.max(ay).max(az);
        let scaled = cube_pt * (half / dominant);

        // Re-express in the new face's tangent basis.
        let (ntu, ntv, _) = sphere::face_basis(new_face);
        let new_u_center = scaled.dot(ntu.as_dvec3());
        let new_v_center = scaled.dot(ntv.as_dvec3());
        let new_face_u = new_u_center + half; // [0, FACE_SIDE_BLOCKS]
        let new_face_v = new_v_center + half;

        let n_chunks = sphere::FACE_SIDE_CHUNKS;
        let max_face = n_chunks as f64 * cs - 1e-4;
        let new_face_u = new_face_u.clamp(0.0, max_face);
        let new_face_v = new_face_v.clamp(0.0, max_face);
        let new_cx = (new_face_u / cs).floor() as i32;
        let new_cz = (new_face_v / cs).floor() as i32;
        let new_lx = (new_face_u - new_cx as f64 * cs) as f32;
        let new_lz = (new_face_v - new_cz as f64 * cs) as f32;

        self.face = new_face;
        self.cx = new_cx.clamp(0, n_chunks - 1);
        self.cz = new_cz.clamp(0, n_chunks - 1);
        self.lx = new_lx;
        self.lz = new_lz;
        // cy / ly (radial) preserved.
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
