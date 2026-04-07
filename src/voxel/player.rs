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

    /// Apply radial gravity. Sticky `on_ground` to avoid bobbing.
    pub fn apply_physics(&mut self, dt: f32, world: &World) {
        if self.fly_mode {
            return;
        }
        if self.on_ground {
            // Walked off the edge? Probe one step down.
            let probe_ly = self.ly - 0.2;
            if !sample_solid_at(self.face, self.cx, self.cy, self.cz, self.lx, probe_ly - PLAYER_HEIGHT, self.lz, world) {
                self.on_ground = false;
            } else {
                return;
            }
        }
        self.radial_velocity -= GRAVITY * dt;
        let dd = self.radial_velocity * dt;
        self.try_move(0.0, dd, 0.0, world);
    }

    /// Attempt to move by `(du, dd, dv)` in face-local block units. The
    /// move is applied as one integer carry, then collision-tested as a
    /// capsule footprint. On collision the move is reverted and, if it was
    /// a downward move, `on_ground` becomes true.
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
            if dd < 0.0 {
                self.radial_velocity = 0.0;
                self.on_ground = true;
            }
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

    /// Convert the player's current cube-space position to a world point,
    /// then re-derive `(face, cx, cy, cz, lx, ly, lz)` on whichever face
    /// owns that point. Used only at chunk-boundary face crossings (a
    /// discrete event), not for collision.
    fn remap_to_neighbor_face(&mut self) {
        let world = sphere::chunk_to_world(self.chunk_pos(), Vec3::new(self.lx, self.ly, self.lz));
        if let Some((cp, lx, ly, lz)) = sphere::world_to_chunk_local(world) {
            self.face = cp.face;
            self.cx = cp.cx;
            self.cy = cp.cy;
            self.cz = cp.cz;
            self.lx = lx;
            self.ly = ly;
            self.lz = lz;
        }
        // Re-orthogonalize forward against the new face's local up.
        self.reorthogonalize_forward();
    }
}

/// Sample whether the chunk at `(face, cx, cy, cz)` has a solid block at
/// the given sub-chunk position. Sub-chunk coordinates outside `[0, 16)`
/// are first carried into the appropriate neighbor (same face only — for
/// capsule sampling we always stay within one chunk's worth of the player).
fn sample_solid_at(face: Face, mut cx: i32, mut cy: i32, mut cz: i32, mut lx: f32, mut ly: f32, mut lz: f32, world: &World) -> bool {
    let cs = CHUNK_SIZE as f32;
    while lx >= cs { lx -= cs; cx += 1; }
    while lx < 0.0 { lx += cs; cx -= 1; }
    while lz >= cs { lz -= cs; cz += 1; }
    while lz < 0.0 { lz += cs; cz -= 1; }
    while ly >= cs { ly -= cs; cy += 1; }
    while ly < 0.0 { ly += cs; cy -= 1; }
    if cy < 0 {
        return true; // anything below the planet core is "solid" — pin the player.
    }
    let n = sphere::FACE_SIDE_CHUNKS;
    let cp = if cx < 0 || cx >= n || cz < 0 || cz >= n {
        // Cross-face sample: re-derive via world position. One-shot, fine.
        let world_pt = sphere::chunk_to_world(ChunkPos { face, cx, cy, cz }, Vec3::new(lx, ly, lz));
        match sphere::world_to_chunk_local(world_pt) {
            Some((cp, nlx, nly, nlz)) => {
                lx = nlx; ly = nly; lz = nlz;
                cp
            }
            None => return false,
        }
    } else {
        ChunkPos { face, cx, cy, cz }
    };
    world.block_solid(cp, lx as usize, ly as usize, lz as usize)
}

fn capsule_collides(player: &Player, world: &World) -> bool {
    // Sample 4 footprint corners at 3 heights (head, mid, feet).
    let hw = PLAYER_HALF_WIDTH;
    let height = PLAYER_HEIGHT;
    let offsets = [
        ( hw,  hw),
        ( hw, -hw),
        (-hw,  hw),
        (-hw, -hw),
    ];
    let heights = [0.0_f32, -0.5 * height, -(height - 0.05)];
    for h in heights {
        for (dx, dz) in offsets {
            if sample_solid_at(player.face, player.cx, player.cy, player.cz, player.lx + dx, player.ly + h, player.lz + dz, world) {
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
