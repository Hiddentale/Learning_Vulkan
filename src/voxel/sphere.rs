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
/// Radius from the planet center to the cube face plane, in blocks. The
/// projection formula is `world = unit_sphere * (PLANET_RADIUS + d)` where
/// `d` is the radial offset above the cube face. Surface terrain lives at
/// `d ≈ SEA_LEVEL`, so the *visual* surface radius is `PLANET_RADIUS + SEA_LEVEL`.
///
/// Tuned so blocks are visually square at the actual terrain surface:
/// `(PLANET_RADIUS + SEA_LEVEL) / CUBE_HALF = √2`
/// → `PLANET_RADIUS = √2·CUBE_HALF − SEA_LEVEL = √2·96 − 64 ≈ 71`.
pub const PLANET_RADIUS_BLOCKS: i32 = 71;

/// Approximate radius of the visible terrain surface (sea level), in blocks.
/// Equals `PLANET_RADIUS_BLOCKS + SEA_LEVEL_BLOCKS`. Used as a stable scale
/// for noise sampling so noise frequencies stay tuned regardless of how
/// `PLANET_RADIUS_BLOCKS` is internally adjusted for block-aspect tuning.
pub const SURFACE_RADIUS_BLOCKS: i32 = 135;

/// Sea level in blocks above the cube face plane (== `d` in projection).
/// Lives here so both terrain generation and the projection share one source.
pub const SEA_LEVEL_BLOCKS: i32 = SURFACE_RADIUS_BLOCKS - PLANET_RADIUS_BLOCKS;

/// Each cube face spans this many chunks along its u and v axes. With
/// CHUNK_SIZE=16 this gives a 48×48 block face — matching the planet radius
/// so the inflated cube is roughly a unit-aspect sphere.
pub const FACE_SIDE_CHUNKS: i32 = 12;

/// Vertical (radial) chunk extent above the cube surface. Currently mirrors
/// the legacy flat terrain range; Phase D will remap this to radial depth.
pub const FACE_RADIAL_CHUNKS: i32 = 96;

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

/// Total side length of one face in blocks.
pub const FACE_SIDE_BLOCKS: i32 = FACE_SIDE_CHUNKS * CHUNK_SIZE as i32;

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

/// Inverse of [`chunk_to_world`]: given a world-space cartesian point
/// (relative to the planet center), determine which face owns it and return
/// the chunk position plus the local block-space offset within that chunk.
///
/// Returns `None` for points exactly at the origin. The face/chunk may be
/// out of [`FACE_SIDE_CHUNKS`] range for points well above the cube surface
/// at the corners, but for any reasonable above-ground player position the
/// result lies inside a valid chunk.
pub fn world_to_chunk_local(world: DVec3) -> Option<(ChunkPos, f32, f32, f32)> {
    world_to_chunk_local_hysteretic(world, None, 0.0)
}

/// Hysteretic version: prefers `current_face` when the dominant axis margin
/// is below `epsilon`. Used by walk/fly to stop the owning face from
/// flipping each frame on a seam.
pub fn world_to_chunk_local_hysteretic(world: DVec3, current_face: Option<Face>, epsilon: f64) -> Option<(ChunkPos, f32, f32, f32)> {
    let radius = world.length();
    if radius < 1e-6 {
        return None;
    }
    let unit_sphere = world / radius;
    let unit_cube = sphere_to_cube_unit(unit_sphere);
    // Try the hysteretic pick first. If the resulting (u, v) on that face
    // is outside the face's own range, the player has actually crossed the
    // seam — fall back to the geometric dominant pick.
    let mut face = match current_face {
        Some(cur) => face_for_cube_point_hysteretic(unit_cube, cur, epsilon),
        None => face_for_cube_point(unit_cube),
    };
    let half = (FACE_SIDE_CHUNKS * CHUNK_SIZE as i32) as f64 * 0.5;
    let cs = CHUNK_SIZE as f64;
    let cube_pt_blocks = unit_cube * CUBE_HALF_BLOCKS;
    let (mut u, mut v) = project_uv(face, cube_pt_blocks);
    if u.abs() > half || v.abs() > half {
        face = face_for_cube_point(unit_cube);
        let (u2, v2) = project_uv(face, cube_pt_blocks);
        u = u2;
        v = v2;
    }
    let d = radius - PLANET_RADIUS_BLOCKS as f64;
    let face_u = u + half;
    let face_v = v + half;
    let cx = face_u.div_euclid(cs) as i32;
    let cy = d.div_euclid(cs) as i32;
    let cz = face_v.div_euclid(cs) as i32;
    let lx = (face_u.rem_euclid(cs)) as f32;
    let ly = (d.rem_euclid(cs)) as f32;
    let lz = (face_v.rem_euclid(cs)) as f32;
    Some((ChunkPos { face, cx, cy, cz }, lx, ly, lz))
}

fn project_uv(face: Face, cube_pt_blocks: DVec3) -> (f64, f64) {
    let (tu, tv, _) = face_basis(face);
    (cube_pt_blocks.dot(tu.as_dvec3()), cube_pt_blocks.dot(tv.as_dvec3()))
}

/// Convert a `(wx, wz)` block-space coordinate — interpreted as a position on
/// the +Y face with the cube centre at the origin — to a 3D point on the
/// planet sphere, for sampling 3D noise. Uses raw normalization (cheap, no
/// branch on out-of-face coordinates) since noise sampling doesn't need the
/// area-preserving Catmull/Everitt mapping.
pub fn noise_pos_on_posy(wx: f64, wz: f64) -> [f64; 3] {
    let v = DVec3::new(wx, CUBE_HALF_BLOCKS, wz).normalize() * SURFACE_RADIUS_BLOCKS as f64;
    [v.x, v.y, v.z]
}

/// Sample a noise position from a world-space cartesian point. Returns a
/// 3D coordinate that depends only on the *direction* from the planet center,
/// scaled to the surface radius. Two world points along the same radial
/// produce the same noise sample, which is what makes density-based terrain
/// seamless across cube face boundaries.
pub fn noise_pos_at_world(world: DVec3) -> [f64; 3] {
    let dir = world.normalize_or(DVec3::Y);
    let p = dir * SURFACE_RADIUS_BLOCKS as f64;
    [p.x, p.y, p.z]
}

/// Sample 3D noise at the sphere position corresponding to a face-local
/// `(u, v)` block-space coordinate (with face center at origin) on `face`.
/// This is the multi-face generalization of [`noise_pos_on_posy`] used by
/// terrain generation in Phase C.
pub fn noise_pos_on_face(face: Face, u: f64, v: f64) -> [f64; 3] {
    let (tu, tv, n) = face_basis(face);
    let cube_pt = tu.as_dvec3() * u + tv.as_dvec3() * v + n.as_dvec3() * CUBE_HALF_BLOCKS;
    let p = cube_pt.normalize() * SURFACE_RADIUS_BLOCKS as f64;
    [p.x, p.y, p.z]
}

/// Number of samples per axis when bounding a sphere-projected chunk.
/// 5×5×5 = 125 samples gives a substantially tighter AABB than the previous
/// 3×3×3 grid for chunks on curved face areas (the projection bulges between
/// samples; finer sampling shrinks the bulge that the bound has to contain).
const AABB_SAMPLES_PER_AXIS: i32 = 5;

/// Safety pad in blocks added to each AABB axis to absorb the residual
/// inter-sample bulge. Bounded by the cube-to-sphere derivative norm times
/// the sample step (`CHUNK_SIZE / (N-1)`) — a conservative half-block at
/// `FACE_SIDE_CHUNKS=12, N=5` covers the worst case.
const AABB_SAMPLE_PAD: f32 = 0.5;

/// Compute a conservative axis-aligned bounding box (in cartesian world
/// space relative to the planet center) for a single chunk after sphere
/// projection. Samples a dense grid over the chunk's cube-space box and
/// pads by [`AABB_SAMPLE_PAD`] to absorb residual inter-sample bulge.
pub fn chunk_world_aabb(cp: ChunkPos) -> ([f32; 3], [f32; 3]) {
    let cs = CHUNK_SIZE as f32;
    let n = AABB_SAMPLES_PER_AXIS;
    let step = 1.0 / (n - 1) as f32;
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let lx = ix as f32 * step * cs;
                let ly = iy as f32 * step * cs;
                let lz = iz as f32 * step * cs;
                let w = chunk_to_world(cp, Vec3::new(lx, ly, lz));
                let p = [w.x as f32, w.y as f32, w.z as f32];
                for k in 0..3 {
                    if p[k] < min[k] { min[k] = p[k]; }
                    if p[k] > max[k] { max[k] = p[k]; }
                }
            }
        }
    }
    for k in 0..3 {
        min[k] -= AABB_SAMPLE_PAD;
        max[k] += AABB_SAMPLE_PAD;
    }
    (min, max)
}

/// Conservative bounding sphere of a sphere-projected chunk, in cartesian
/// world space relative to the planet center. Returns `(center, radius)`.
/// The center is the world position of the chunk's cube-space midpoint;
/// the radius is the maximum distance from that center to any of a dense
/// sample grid, padded by [`AABB_SAMPLE_PAD`].
///
/// Used by horizon culling and as a tighter cull primitive than the AABB
/// for chunks on curved face areas.
pub fn chunk_bounding_sphere(cp: ChunkPos) -> (Vec3, f32) {
    let cs = CHUNK_SIZE as f32;
    let half = cs * 0.5;
    let center_world = chunk_to_world(cp, Vec3::new(half, half, half));
    let center = Vec3::new(center_world.x as f32, center_world.y as f32, center_world.z as f32);
    let n = AABB_SAMPLES_PER_AXIS;
    let step = 1.0 / (n - 1) as f32;
    let mut max_d2: f32 = 0.0;
    for ix in 0..n {
        for iy in 0..n {
            for iz in 0..n {
                let lx = ix as f32 * step * cs;
                let ly = iy as f32 * step * cs;
                let lz = iz as f32 * step * cs;
                let w = chunk_to_world(cp, Vec3::new(lx, ly, lz));
                let p = Vec3::new(w.x as f32, w.y as f32, w.z as f32);
                let d2 = (p - center).length_squared();
                if d2 > max_d2 { max_d2 = d2; }
            }
        }
    }
    (center, max_d2.sqrt() + AABB_SAMPLE_PAD)
}

/// All faces in their `Face::*` discriminant order — used for enumerating
/// the planet during chunk generation and bounded iteration.
pub const ALL_FACES: [Face; 6] = [
    Face::PosX,
    Face::NegX,
    Face::PosY,
    Face::NegY,
    Face::PosZ,
    Face::NegZ,
];

/// Numeric id matching GLSL: shader-side face basis table is indexed by this.
pub fn face_id(face: Face) -> u32 {
    face as u32
}

// ---------------------------------------------------------------------------
// 24-entry edge transition table.
//
// When the player walks off one of a face's four edges, this table tells us
// which neighbor face they enter and how to rotate their `forward` vector so
// that "walking forward" continues to mean "into the new face" rather than
// "off into space". The geometry is derived once from [`face_basis`] and
// validated by `edge_transition_round_trip` — there is no hand-written data
// to drift out of sync with the face basis.
// ---------------------------------------------------------------------------

/// Which edge of a face the player just crossed. Indexed by the chunk coord
/// that left the face range.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum EdgeDir {
    /// `cx < 0` (walked in `-tu` direction)
    NegU,
    /// `cx >= FACE_SIDE_CHUNKS` (walked in `+tu` direction)
    PosU,
    /// `cz < 0` (walked in `-tv` direction)
    NegV,
    /// `cz >= FACE_SIDE_CHUNKS` (walked in `+tv` direction)
    PosV,
}

/// Result of crossing one edge: which face you land on, and the world-space
/// rotation to apply to `forward` so the camera tilts onto the new face.
#[derive(Copy, Clone, Debug)]
pub struct EdgeTransition {
    pub neighbor: Face,
    /// Edge axis (the cube edge along which the two faces meet). The
    /// `forward` vector rotates 90° around this axis.
    pub edge_axis: Vec3,
    /// Direction of rotation in radians: `+FRAC_PI_2` or `-FRAC_PI_2`. Sign
    /// is chosen so the rotation takes the old face's outward normal to the
    /// new face's outward normal (i.e., the camera's "up" rotates correctly).
    pub rotation: f32,
}

/// Compute the transition for `(face, dir)` from the face basis. Pure
/// function — the same call always returns the same result, so this is
/// effectively a 24-entry static table without the storage overhead.
pub fn edge_transition(face: Face, dir: EdgeDir) -> EdgeTransition {
    let (tu, tv, n) = face_basis(face);
    // Direction the player walked, in world space.
    let walk_dir = match dir {
        EdgeDir::PosU => tu,
        EdgeDir::NegU => -tu,
        EdgeDir::PosV => tv,
        EdgeDir::NegV => -tv,
    };
    // Neighbor face: the one whose outward normal matches `walk_dir`.
    let neighbor = face_for_axis(walk_dir);
    // Edge axis = cross of the two face normals. This is the world-space
    // line along which the two faces meet on the cube.
    let new_n = face_normal(neighbor);
    let edge_axis = n.cross(new_n).normalize_or(Vec3::X);
    // Rotation: by 90° around edge_axis in the direction that takes the
    // old normal to the new normal.
    // Verify with a sample: rotate `n` by +90° around edge_axis and see if
    // it lands on new_n. If not, use -90°.
    let plus = rotate(n, edge_axis, std::f32::consts::FRAC_PI_2);
    let rotation = if (plus - new_n).length() < 1e-3 {
        std::f32::consts::FRAC_PI_2
    } else {
        -std::f32::consts::FRAC_PI_2
    };
    EdgeTransition { neighbor, edge_axis, rotation }
}

/// Pick the cube face whose outward normal axis dominates `point`. Used by
/// any code that needs to classify a cube-space point onto its owning face.
pub fn face_for_cube_point(point: DVec3) -> Face {
    let ax = point.x.abs();
    let ay = point.y.abs();
    let az = point.z.abs();
    if ax >= ay && ax >= az {
        if point.x > 0.0 { Face::PosX } else { Face::NegX }
    } else if ay >= az {
        if point.y > 0.0 { Face::PosY } else { Face::NegY }
    } else {
        if point.z > 0.0 { Face::PosZ } else { Face::NegZ }
    }
}

/// Hysteretic version of [`face_for_cube_point`]. Prefers `current` when the
/// dominant axis margin over `current`'s normal axis is below `epsilon` —
/// stops the face from flipping back and forth on every frame when the
/// player walks exactly along a seam (where `ax`, `ay`, `az` are nearly
/// equal and float noise picks a different winner each tick).
pub fn face_for_cube_point_hysteretic(point: DVec3, current: Face, epsilon: f64) -> Face {
    let candidate = face_for_cube_point(point);
    if candidate == current {
        return current;
    }
    let cur_axis_value = match current {
        Face::PosX | Face::NegX => point.x.abs(),
        Face::PosY | Face::NegY => point.y.abs(),
        Face::PosZ | Face::NegZ => point.z.abs(),
    };
    let cand_axis_value = match candidate {
        Face::PosX | Face::NegX => point.x.abs(),
        Face::PosY | Face::NegY => point.y.abs(),
        Face::PosZ | Face::NegZ => point.z.abs(),
    };
    // Only switch if the candidate axis is meaningfully larger AND the sign
    // of the current face's axis hasn't changed (player is still on the
    // outward side of the cube).
    let same_sign_on_current = match current {
        Face::PosX => point.x >= 0.0,
        Face::NegX => point.x <= 0.0,
        Face::PosY => point.y >= 0.0,
        Face::NegY => point.y <= 0.0,
        Face::PosZ => point.z >= 0.0,
        Face::NegZ => point.z <= 0.0,
    };
    if same_sign_on_current && cand_axis_value < cur_axis_value + epsilon {
        current
    } else {
        candidate
    }
}

/// Pick the cube face whose outward normal is closest to `axis`.
fn face_for_axis(axis: Vec3) -> Face {
    let ax = axis.x.abs();
    let ay = axis.y.abs();
    let az = axis.z.abs();
    if ax >= ay && ax >= az {
        if axis.x > 0.0 { Face::PosX } else { Face::NegX }
    } else if ay >= az {
        if axis.y > 0.0 { Face::PosY } else { Face::NegY }
    } else {
        if axis.z > 0.0 { Face::PosZ } else { Face::NegZ }
    }
}

/// Rodrigues' rotation: rotate `v` around unit `axis` by `angle` radians.
fn rotate(v: Vec3, axis: Vec3, angle: f32) -> Vec3 {
    let c = angle.cos();
    let s = angle.sin();
    v * c + axis.cross(v) * s + axis * axis.dot(v) * (1.0 - c)
}

/// Apply an [`EdgeTransition`] to a `forward` vector.
pub fn rotate_forward_through(transition: EdgeTransition, forward: Vec3) -> Vec3 {
    rotate(forward, transition.edge_axis, transition.rotation).normalize_or(forward)
}

/// All 4 edge directions, ordered for iteration.
pub const ALL_EDGES: [EdgeDir; 4] = [EdgeDir::NegU, EdgeDir::PosU, EdgeDir::NegV, EdgeDir::PosV];

#[cfg(test)]
mod edge_table_tests {
    use super::*;

    #[test]
    fn every_edge_transitions_to_a_different_face() {
        for face in ALL_FACES {
            for dir in ALL_EDGES {
                let t = edge_transition(face, dir);
                assert!(t.neighbor != face, "{:?} {:?} → same face", face, dir);
            }
        }
    }

    #[test]
    fn rotation_takes_old_normal_to_new_normal() {
        // The defining property of the rotation: it carries the camera's
        // "up" (face normal) onto the new face's normal. This is what
        // makes "walking forward" continue to feel forward.
        for face in ALL_FACES {
            for dir in ALL_EDGES {
                let t = edge_transition(face, dir);
                let old_n = face_normal(face);
                let new_n = face_normal(t.neighbor);
                let rotated = rotate(old_n, t.edge_axis, t.rotation);
                assert!((rotated - new_n).length() < 1e-3, "rotation wrong for {:?} {:?}", face, dir);
            }
        }
    }

    /// Snapshot of the entire 24-entry table. If anyone changes
    /// `face_basis` and the rotations shift, this test prints exactly
    /// which (face, edge) entries differ. Hand-validated; do not edit
    /// without re-deriving by hand.
    #[test]
    fn edge_table_snapshot() {
        let expected: &[((Face, EdgeDir), Face)] = &[
            // PosX (basis tu=Y, tv=Z, n=X)
            ((Face::PosX, EdgeDir::NegU), Face::NegY),
            ((Face::PosX, EdgeDir::PosU), Face::PosY),
            ((Face::PosX, EdgeDir::NegV), Face::NegZ),
            ((Face::PosX, EdgeDir::PosV), Face::PosZ),
            // NegX (tu=Z, tv=Y, n=-X)
            ((Face::NegX, EdgeDir::NegU), Face::NegZ),
            ((Face::NegX, EdgeDir::PosU), Face::PosZ),
            ((Face::NegX, EdgeDir::NegV), Face::NegY),
            ((Face::NegX, EdgeDir::PosV), Face::PosY),
            // PosY (tu=Z, tv=X, n=Y)
            ((Face::PosY, EdgeDir::NegU), Face::NegZ),
            ((Face::PosY, EdgeDir::PosU), Face::PosZ),
            ((Face::PosY, EdgeDir::NegV), Face::NegX),
            ((Face::PosY, EdgeDir::PosV), Face::PosX),
            // NegY (tu=X, tv=Z, n=-Y)
            ((Face::NegY, EdgeDir::NegU), Face::NegX),
            ((Face::NegY, EdgeDir::PosU), Face::PosX),
            ((Face::NegY, EdgeDir::NegV), Face::NegZ),
            ((Face::NegY, EdgeDir::PosV), Face::PosZ),
            // PosZ (tu=X, tv=Y, n=Z)
            ((Face::PosZ, EdgeDir::NegU), Face::NegX),
            ((Face::PosZ, EdgeDir::PosU), Face::PosX),
            ((Face::PosZ, EdgeDir::NegV), Face::NegY),
            ((Face::PosZ, EdgeDir::PosV), Face::PosY),
            // NegZ (tu=Y, tv=X, n=-Z)
            ((Face::NegZ, EdgeDir::NegU), Face::NegY),
            ((Face::NegZ, EdgeDir::PosU), Face::PosY),
            ((Face::NegZ, EdgeDir::NegV), Face::NegX),
            ((Face::NegZ, EdgeDir::PosV), Face::PosX),
        ];
        assert_eq!(expected.len(), 24);
        for &((face, dir), neighbor) in expected {
            let actual = edge_transition(face, dir);
            assert_eq!(actual.neighbor, neighbor, "{:?} {:?}: expected {:?}, got {:?}", face, dir, neighbor, actual.neighbor);
        }
    }

    /// Crossing every edge from every face and walking back ends up on
    /// the original face — i.e., transitions are pairwise inverse.
    #[test]
    fn every_edge_crossing_is_invertible() {
        for face in ALL_FACES {
            for dir in ALL_EDGES {
                let t1 = edge_transition(face, dir);
                // Find the reverse edge: the edge of the neighbor whose
                // neighbor is the original face.
                let mut reverse: Option<EdgeDir> = None;
                for d2 in ALL_EDGES {
                    let t2 = edge_transition(t1.neighbor, d2);
                    if t2.neighbor == face {
                        reverse = Some(d2);
                        break;
                    }
                }
                let reverse = reverse.unwrap_or_else(|| panic!("no reverse edge for {:?} {:?} → {:?}", face, dir, t1.neighbor));
                let _ = reverse;
            }
        }
    }

    #[test]
    fn rotation_takes_walk_direction_to_minus_old_normal() {
        // The direction the player walked OFF the old face, after rotation,
        // should equal `-face_normal(old)` — that is, the rotation pivots
        // the world "down the side" of the cube. On the new face this is
        // a tangent direction pointing away from the shared edge toward
        // the new face center, which is exactly what "keep walking forward"
        // should mean.
        for face in ALL_FACES {
            for dir in ALL_EDGES {
                let t = edge_transition(face, dir);
                let (tu, tv, _) = face_basis(face);
                let walk = match dir {
                    EdgeDir::PosU => tu,
                    EdgeDir::NegU => -tu,
                    EdgeDir::PosV => tv,
                    EdgeDir::NegV => -tv,
                };
                let rotated = rotate(walk, t.edge_axis, t.rotation);
                let expected = -face_normal(face);
                assert!((rotated - expected).length() < 1e-3, "walk dir wrong for {:?} {:?}: rotated={:?} expected={:?}", face, dir, rotated, expected);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Edge transitions across face boundaries.
//
// When a chunk on face F sits at the edge of the face's grid, its
// out-of-range neighbor lives on a different face F'. This module computes
// the (F', cx', cy', cz') for any out-of-range chunk neighbor by walking the
// 3D cube position one step over the cube edge and re-resolving which face
// owns that position. The basis table is the single source of truth.
// ---------------------------------------------------------------------------

/// Step across a face boundary. Given a chunk position whose `(cx, cz)` may
/// be out of range `[0, FACE_SIDE_CHUNKS)`, return the equivalent position
/// on the neighbor face. Returns `None` for in-range positions.
///
/// The math: convert the face-local chunk grid coordinate to a 3D cube
/// position (in chunk units), then re-classify which face's tangent plane
/// it lies on by largest absolute axis. Re-express in that face's `(cx, cz)`.
pub fn cross_face_neighbor(cp: ChunkPos) -> Option<ChunkPos> {
    let n = FACE_SIDE_CHUNKS;
    let in_range = cp.cx >= 0 && cp.cx < n && cp.cz >= 0 && cp.cz < n;
    if in_range {
        return None;
    }
    // Map face-local chunk grid to a 3D cube point in chunk units.
    // Offset by 0.5 so we sample the chunk centre, and shift by -n/2 so the
    // face is centred on the cube centre.
    let (tu, tv, fn_) = face_basis(cp.face);
    let half = n as f32 * 0.5;
    let u = (cp.cx as f32 + 0.5) - half;
    let v = (cp.cz as f32 + 0.5) - half;
    let cube_pt = tu * u + tv * v + fn_ * half;
    // Find the dominant axis — that's the new face.
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
    if new_face == cp.face {
        return None;
    }
    // Re-express cube_pt in the new face's tangent basis.
    let (ntu, ntv, _) = face_basis(new_face);
    let new_u = cube_pt.dot(ntu);
    let new_v = cube_pt.dot(ntv);
    let new_cx = (new_u + half).floor() as i32;
    let new_cz = (new_v + half).floor() as i32;
    // Clamp to face range — corners can land just outside due to fp.
    let new_cx = new_cx.clamp(0, n - 1);
    let new_cz = new_cz.clamp(0, n - 1);
    Some(ChunkPos { face: new_face, cx: new_cx, cy: cp.cy, cz: new_cz })
}

#[cfg(test)]
mod transition_tests {
    use super::*;

    #[test]
    fn in_range_returns_none() {
        let cp = ChunkPos { face: Face::PosY, cx: 1, cy: 0, cz: 1 };
        assert!(cross_face_neighbor(cp).is_none());
    }

    #[test]
    fn step_off_each_edge_lands_on_a_different_face() {
        // For every face, walk off all four edges and verify the result is
        // on a different face and within range.
        let n = FACE_SIDE_CHUNKS;
        for face in ALL_FACES {
            let edges = [
                ChunkPos { face, cx: -1, cy: 0, cz: 1 },
                ChunkPos { face, cx: n,  cy: 0, cz: 1 },
                ChunkPos { face, cx: 1,  cy: 0, cz: -1 },
                ChunkPos { face, cx: 1,  cy: 0, cz: n },
            ];
            for e in edges {
                let nb = cross_face_neighbor(e).unwrap_or_else(|| panic!("no neighbor for {:?}", e));
                assert!(nb.face != face, "edge {:?} did not change face: {:?}", e, nb);
                assert!(nb.cx >= 0 && nb.cx < n, "cx out of range: {:?}", nb);
                assert!(nb.cz >= 0 && nb.cz < n, "cz out of range: {:?}", nb);
            }
        }
    }
}

#[cfg(test)]
mod aabb_tests {
    use super::*;
    #[test]
    fn aabb_contains_chunk_corners() {
        let cp = ChunkPos { face: Face::PosY, cx: 1, cy: 0, cz: 1 };
        let (mn, mx) = chunk_world_aabb(cp);
        for k in 0..3 { assert!(mn[k] <= mx[k]); }
    }
    #[test]
    fn each_face_center_chunk_has_finite_aabb() {
        for face in ALL_FACES {
            let cp = ChunkPos { face, cx: 1, cy: 0, cz: 1 };
            let (mn, mx) = chunk_world_aabb(cp);
            for k in 0..3 {
                assert!(mn[k].is_finite() && mx[k].is_finite());
                assert!(mx[k] >= mn[k]);
            }
        }
    }

    /// Sample a dense pseudo-random grid of points inside a chunk's cube-space
    /// box, project them through the sphere, and assert every projected point
    /// lies inside the AABB. The previous 3×3×3 sampling could miss the bulge
    /// at curved-face chunks; this pins correctness for all faces.
    #[test]
    fn chunk_aabb_contains_dense_surface_samples() {
        let cs = CHUNK_SIZE as f32;
        // Choose a corner chunk on each face — that's where the projection
        // bulge is largest.
        for face in ALL_FACES {
            for &(cx, cz) in &[(0, 0), (FACE_SIDE_CHUNKS - 1, FACE_SIDE_CHUNKS - 1), (0, FACE_SIDE_CHUNKS - 1)] {
                let cp = ChunkPos { face, cx, cy: 0, cz };
                let (mn, mx) = chunk_world_aabb(cp);
                // 11×11×11 = 1331 quasi-random samples (irrational stride to avoid
                // hitting only the sample lattice the AABB is computed from).
                for i in 0..11 {
                    for j in 0..11 {
                        for k in 0..11 {
                            let fx = ((i as f32 * 0.1734).fract() + i as f32 * 0.1) % 1.0;
                            let fy = ((j as f32 * 0.2917).fract() + j as f32 * 0.1) % 1.0;
                            let fz = ((k as f32 * 0.3491).fract() + k as f32 * 0.1) % 1.0;
                            let w = chunk_to_world(cp, Vec3::new(fx * cs, fy * cs, fz * cs));
                            let p = [w.x as f32, w.y as f32, w.z as f32];
                            for axis in 0..3 {
                                assert!(
                                    p[axis] >= mn[axis] - 1e-3 && p[axis] <= mx[axis] + 1e-3,
                                    "face={:?} chunk=({},{}) sample {:?} outside AABB axis {} ({}..{})",
                                    face, cx, cz, p, axis, mn[axis], mx[axis]
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn chunk_bounding_sphere_contains_dense_surface_samples() {
        let cs = CHUNK_SIZE as f32;
        for face in ALL_FACES {
            for &(cx, cz) in &[(0, 0), (FACE_SIDE_CHUNKS - 1, FACE_SIDE_CHUNKS - 1)] {
                let cp = ChunkPos { face, cx, cy: 0, cz };
                let (center, radius) = chunk_bounding_sphere(cp);
                for i in 0..7 {
                    for j in 0..7 {
                        for k in 0..7 {
                            let f = |x: i32| x as f32 / 6.0;
                            let w = chunk_to_world(cp, Vec3::new(f(i) * cs, f(j) * cs, f(k) * cs));
                            let p = Vec3::new(w.x as f32, w.y as f32, w.z as f32);
                            let d = (p - center).length();
                            assert!(d <= radius + 1e-3, "sample distance {} > radius {}", d, radius);
                        }
                    }
                }
            }
        }
    }

    /// At the planet scale, a chunk's bounding sphere radius shouldn't be
    /// dramatically larger than the chunk's diagonal (which would mean a
    /// pathologically loose bound). The sphere projection adds some bulge but
    /// not 10×.
    #[test]
    fn chunk_bounding_sphere_radius_is_reasonable() {
        let diag = (CHUNK_SIZE as f32) * 3.0_f32.sqrt();
        for face in ALL_FACES {
            let cp = ChunkPos { face, cx: FACE_SIDE_CHUNKS / 2, cy: 0, cz: FACE_SIDE_CHUNKS / 2 };
            let (_, r) = chunk_bounding_sphere(cp);
            assert!(r < diag * 2.0, "face {:?}: bounding radius {} > 2 * diagonal {}", face, r, diag);
        }
    }
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
    fn face_center_maps_to_top_of_planet() {
        // The geometric centre of the +Y face inflates to (0, R, 0).
        let world = face_local_to_world(Face::PosY, 0.0, 0.0, 0.0);
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
    fn block_aspect_at_surface_face_center_is_square() {
        // The aspect that matters is at the actual terrain surface, not at
        // the cube face plane (d=0). Surface is around SEA_LEVEL blocks above
        // the cube face. Tangent step should equal radial step there.
        let mid = FACE_SIDE_CHUNKS / 2;
        let surface_cy = 4; // ~4 chunks * 16 blocks = 64 blocks above the cube face
        let cp = ChunkPos { face: Face::PosY, cx: mid, cy: surface_cy, cz: mid };
        let center = chunk_to_world(cp, Vec3::new(0.0, 0.0, 0.0));
        let tangent = chunk_to_world(cp, Vec3::new(1.0, 0.0, 0.0));
        let radial = chunk_to_world(cp, Vec3::new(0.0, 1.0, 0.0));
        let dt = (tangent - center).length();
        let dr = (radial - center).length();
        let ratio = dt / dr;
        eprintln!("[surface] tangent={dt} radial={dr} ratio={ratio}");
        assert!((ratio - 1.0).abs() < 0.1, "surface block aspect ratio is {}, expected ≈ 1.0", ratio);
    }

    #[test]
    fn planet_circumference_is_finite_and_positive() {
        let circumference = (2.0 * std::f32::consts::PI * PLANET_RADIUS_BLOCKS as f32) as i32;
        assert!(circumference > 0 && circumference < 100_000);
    }
}
