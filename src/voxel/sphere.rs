//! Spherical (cube-map) world projection.
//!
//! This module is the *only* place in the codebase that knows the world is a
//! sphere. Every other module addresses chunks via [`ChunkPos`] (which carries
//! a [`Face`] identity) and asks this module to convert to/from cartesian
//! world space. Phase A: projection is still flat — only the [`ChunkPos`]
//! identity exists, and [`Face`] is hardcoded to [`Face::PosY`] everywhere.
//! Phase B replaces the flat conversions with the inflated-cube projection.

#![allow(dead_code)] // Phase A pre-stages constants and Face variants used by later phases.

use super::chunk::CHUNK_SIZE;
use glam::{DVec3, Vec3};

/// Tiny planet for testing — circumference ≈ 2π·48 ≈ 300 blocks. At a walking
/// speed of ~5 blocks/s a player circles the equator in roughly one minute.
pub const PLANET_RADIUS_BLOCKS: i32 = 48;

/// Each cube face spans this many chunks along its u and v axes. With
/// CHUNK_SIZE=16 this gives a 48×48 block face — matching the planet radius
/// so the inflated cube is roughly a unit-aspect sphere.
pub const FACE_SIDE_CHUNKS: i32 = 3;

/// Vertical (radial) chunk extent above the cube surface. Currently mirrors
/// the legacy flat terrain range; Phase D will remap this to radial depth.
pub const FACE_RADIAL_CHUNKS: i32 = 48;

/// Identity of one of the six cube faces. Phase A: every chunk uses
/// [`Face::PosY`], so behavior is byte-identical to the flat-grid world.
#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Face {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

/// World identity of a chunk: a face plus integer coordinates within that
/// face's local grid. `cx`/`cz` are the surface (u, v) chunk indices and `cy`
/// is the radial depth above the cube surface.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct ChunkPos {
    pub face: Face,
    pub cx: i32,
    pub cy: i32,
    pub cz: i32,
}

impl ChunkPos {
    /// Construct a chunk position on the +Y face. Phase A uses this everywhere.
    pub const fn posy(cx: i32, cy: i32, cz: i32) -> Self {
        Self { face: Face::PosY, cx, cy, cz }
    }

    /// Raw integer grid coordinates `[cx, cy, cz]`, discarding the face.
    /// Used at boundaries with code that still indexes into a flat grid
    /// (GPU pools, region storage, SVDAG super-chunk groups).
    pub const fn coords(self) -> [i32; 3] {
        [self.cx, self.cy, self.cz]
    }
}

/// Convert a block-space (wx, wy, wz) integer coordinate to the chunk that
/// owns it. Phase A: face is always PosY. Phase B will route through
/// [`world_to_chunk`] proper.
pub fn block_to_chunk(wx: i32, wy: i32, wz: i32) -> ChunkPos {
    let size = CHUNK_SIZE as i32;
    ChunkPos::posy(wx.div_euclid(size), wy.div_euclid(size), wz.div_euclid(size))
}

// ---------------------------------------------------------------------------
// Phase B1: cube ↔ sphere projection math.
//
// The mapping is the Catmull/Everitt closed-form "equal-area-ish" projection,
// not raw normalize(). Raw normalization stretches cube corners by ~30% and
// makes erosion kernels anisotropic; this form keeps area distortion small
// across the face while remaining a clean analytic inverse.
// ---------------------------------------------------------------------------

/// Half side length of the cube in blocks. The cube is inscribed so that its
/// corner inflates exactly to the sphere — the face plane sits at this
/// distance from the planet center.
pub const CUBE_HALF_BLOCKS: f64 = (FACE_SIDE_CHUNKS * CHUNK_SIZE as i32 / 2) as f64;

/// Map a point on the unit cube `[-1, 1]³` (any face) to the unit sphere.
pub fn cube_to_sphere_unit(p: DVec3) -> DVec3 {
    let (x, y, z) = (p.x, p.y, p.z);
    let xs = x * (1.0 - y * y * 0.5 - z * z * 0.5 + y * y * z * z / 3.0).sqrt();
    let ys = y * (1.0 - z * z * 0.5 - x * x * 0.5 + z * z * x * x / 3.0).sqrt();
    let zs = z * (1.0 - x * x * 0.5 - y * y * 0.5 + x * x * y * y / 3.0).sqrt();
    DVec3::new(xs, ys, zs)
}

/// Inverse of [`cube_to_sphere_unit`]. Pins the dominant-axis face and
/// fixed-point iterates the two tangent components against the closed-form
/// reduction
///
/// ```text
///   ys² = y² · (0.5 − z²/6),     zs² = z² · (0.5 − y²/6)   (when |x| = 1)
/// ```
///
/// which derives directly from the Catmull/Everitt forward map. Converges
/// to <1e-12 in ~15 iterations across the entire face.
pub fn sphere_to_cube_unit(s: DVec3) -> DVec3 {
    let ax = s.x.abs();
    let ay = s.y.abs();
    let az = s.z.abs();
    if ax.max(ay).max(az) < 1e-12 {
        return DVec3::ZERO;
    }
    if ax >= ay && ax >= az {
        let (u, v) = solve_tangents(s.y, s.z);
        DVec3::new(s.x.signum(), u, v)
    } else if ay >= az {
        let (u, v) = solve_tangents(s.x, s.z);
        DVec3::new(u, s.y.signum(), v)
    } else {
        let (u, v) = solve_tangents(s.x, s.y);
        DVec3::new(u, v, s.z.signum())
    }
}

/// Fixed-point inverse of `(a, b) ↦ (a·√(0.5 − b²/6), b·√(0.5 − a²/6))`,
/// the reduction of [`cube_to_sphere_unit`] when one axis is pinned to ±1.
fn solve_tangents(target_a: f64, target_b: f64) -> (f64, f64) {
    // Initial guess: scale by 1/√0.5 = √2.
    let mut a = target_a * std::f64::consts::SQRT_2;
    let mut b = target_b * std::f64::consts::SQRT_2;
    for _ in 0..32 {
        let new_a = target_a / (0.5 - b * b / 6.0).max(1e-12).sqrt();
        let new_b = target_b / (0.5 - a * a / 6.0).max(1e-12).sqrt();
        a = new_a.clamp(-1.0, 1.0);
        b = new_b.clamp(-1.0, 1.0);
    }
    (a, b)
}

/// Outward unit normal of a face.
pub fn face_normal(face: Face) -> Vec3 {
    match face {
        Face::PosX => Vec3::X,
        Face::NegX => -Vec3::X,
        Face::PosY => Vec3::Y,
        Face::NegY => -Vec3::Y,
        Face::PosZ => Vec3::Z,
        Face::NegZ => -Vec3::Z,
    }
}

/// `(tangent_u, tangent_v, normal)` basis for a face. The basis is
/// right-handed: `tangent_u × tangent_v = normal`. **This table is the
/// authoritative source for every per-face axis convention** — the edge
/// transition table in Phase C is derived from it programmatically, never
/// hand-written.
pub fn face_basis(face: Face) -> (Vec3, Vec3, Vec3) {
    // Each row is (tangent_u, tangent_v, normal) and is verified
    // right-handed by `face_basis_is_right_handed`.
    match face {
        Face::PosX => (Vec3::Y, Vec3::Z, Vec3::X),
        Face::NegX => (Vec3::Z, Vec3::Y, -Vec3::X),
        Face::PosY => (Vec3::Z, Vec3::X, Vec3::Y),
        Face::NegY => (Vec3::X, Vec3::Z, -Vec3::Y),
        Face::PosZ => (Vec3::X, Vec3::Y, Vec3::Z),
        Face::NegZ => (Vec3::Y, Vec3::X, -Vec3::Z),
    }
}

/// Map a face-local block-space position `(u, v)` (in blocks, may extend past
/// the face edges) and radial depth `d` (blocks above the cube surface) to a
/// cartesian position relative to the planet center, in blocks. Inflated-cube
/// projection: the cube point at `(u, v, +half)` is normalized through
/// [`cube_to_sphere_unit`] then offset radially by `radius + d`.
pub fn face_local_to_world(face: Face, u: f64, v: f64, d: f64) -> DVec3 {
    let (tu, tv, n) = face_basis(face);
    let cube_pt = tu.as_dvec3() * u + tv.as_dvec3() * v + n.as_dvec3() * CUBE_HALF_BLOCKS;
    let unit_cube = cube_pt / CUBE_HALF_BLOCKS;
    let unit_sphere = cube_to_sphere_unit(unit_cube);
    unit_sphere * (PLANET_RADIUS_BLOCKS as f64 + d)
}

/// Convert a chunk position plus a local block-space offset `(lx, ly, lz)`
/// inside that chunk to a cartesian world position relative to the planet
/// center. `ly` becomes radial depth above the cube surface.
pub fn chunk_to_world(cp: ChunkPos, local: Vec3) -> DVec3 {
    let cs = CHUNK_SIZE as f64;
    let half = (FACE_SIDE_CHUNKS * CHUNK_SIZE as i32) as f64 * 0.5;
    let u = cp.cx as f64 * cs + local.x as f64 - half;
    let v = cp.cz as f64 * cs + local.z as f64 - half;
    let d = cp.cy as f64 * cs + local.y as f64;
    face_local_to_world(cp.face, u, v, d)
}

#[cfg(test)]
mod projection_tests {
    use super::*;

    fn approx_eq(a: DVec3, b: DVec3, eps: f64) -> bool {
        (a - b).length() < eps
    }

    #[test]
    fn cube_corners_inflate_onto_unit_sphere() {
        for &s in &[-1.0_f64, 1.0] {
            for &t in &[-1.0_f64, 1.0] {
                for &u in &[-1.0_f64, 1.0] {
                    let p = cube_to_sphere_unit(DVec3::new(s, t, u));
                    assert!((p.length() - 1.0).abs() < 1e-9, "corner {:?} not on unit sphere: |p|={}", (s, t, u), p.length());
                }
            }
        }
    }

    #[test]
    fn cube_face_centers_inflate_onto_axes() {
        let p = cube_to_sphere_unit(DVec3::new(0.0, 1.0, 0.0));
        assert!(approx_eq(p, DVec3::Y, 1e-12));
        let p = cube_to_sphere_unit(DVec3::new(1.0, 0.0, 0.0));
        assert!(approx_eq(p, DVec3::X, 1e-12));
    }

    #[test]
    fn sphere_to_cube_round_trip_on_posy_face() {
        // Sample a grid on the +Y cube face and round-trip through the
        // sphere. Round-trip error must be < 1e-6 to be useful for noise.
        let n = 16;
        let mut max_err: f64 = 0.0;
        for i in 0..=n {
            for j in 0..=n {
                let u = -1.0 + 2.0 * i as f64 / n as f64;
                let v = -1.0 + 2.0 * j as f64 / n as f64;
                let cube = DVec3::new(u, 1.0, v);
                let sphere = cube_to_sphere_unit(cube);
                let back = sphere_to_cube_unit(sphere);
                let err = (back - cube).length();
                if err > max_err {
                    max_err = err;
                }
            }
        }
        assert!(max_err < 1e-6, "max round-trip error {} exceeds 1e-6", max_err);
    }

    #[test]
    fn face_basis_is_right_handed() {
        for face in [Face::PosX, Face::NegX, Face::PosY, Face::NegY, Face::PosZ, Face::NegZ] {
            let (tu, tv, n) = face_basis(face);
            let cross = tu.cross(tv);
            assert!((cross - n).length() < 1e-6, "face {:?} basis not right-handed", face);
            assert!((n - face_normal(face)).length() < 1e-6, "face {:?} normal mismatch", face);
        }
    }

    #[test]
    fn face_center_chunk_maps_to_top_of_planet() {
        // The chunk at the geometric centre of the +Y face, sampled at its
        // local centre, should sit directly above the +Y pole.
        let mid_chunk = FACE_SIDE_CHUNKS / 2;
        let cp = ChunkPos::posy(mid_chunk, 0, mid_chunk);
        let half_block = CHUNK_SIZE as f32 * 0.5;
        let world = chunk_to_world(cp, Vec3::new(half_block, 0.0, half_block));
        // The +Y face center inflates to (0, R, 0).
        assert!((world.x).abs() < 1e-6, "expected x≈0, got {}", world.x);
        assert!((world.z).abs() < 1e-6, "expected z≈0, got {}", world.z);
        assert!((world.y - PLANET_RADIUS_BLOCKS as f64).abs() < 1e-6);
    }

    #[test]
    fn radial_offset_increases_world_radius() {
        let cp = ChunkPos::posy(FACE_SIDE_CHUNKS / 2, 0, FACE_SIDE_CHUNKS / 2);
        let half = CHUNK_SIZE as f32 * 0.5;
        let surface = chunk_to_world(cp, Vec3::new(half, 0.0, half));
        let above = chunk_to_world(cp, Vec3::new(half, 32.0, half));
        assert!((above.length() - surface.length() - 32.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunkpos_round_trips_through_coords() {
        let cp = ChunkPos::posy(3, -2, 7);
        assert_eq!(cp.coords(), [3, -2, 7]);
        assert_eq!(cp.face, Face::PosY);
    }

    #[test]
    fn block_to_chunk_matches_div_euclid() {
        assert_eq!(block_to_chunk(0, 0, 0), ChunkPos::posy(0, 0, 0));
        assert_eq!(block_to_chunk(15, 15, 15), ChunkPos::posy(0, 0, 0));
        assert_eq!(block_to_chunk(16, 0, 0), ChunkPos::posy(1, 0, 0));
        assert_eq!(block_to_chunk(-1, -1, -1), ChunkPos::posy(-1, -1, -1));
    }

    #[test]
    fn tiny_planet_circumference_is_walkable() {
        // At ~5 blocks/sec walking speed, circumference should be ~300 blocks
        // so a full lap takes about a minute.
        let circumference = (2.0 * std::f32::consts::PI * PLANET_RADIUS_BLOCKS as f32) as i32;
        assert!(circumference < 400, "planet too large for one-minute lap: {}", circumference);
    }
}
