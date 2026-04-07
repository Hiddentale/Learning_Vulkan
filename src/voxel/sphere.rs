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
