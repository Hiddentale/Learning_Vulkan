//! Cube-space raycast for block selection (break / place).
//!
//! The flat-grid DDA was retired in Phase D'. The cube-space replacement
//! steps a small fixed distance in world space and asks the world about
//! the block under each sample. Skips duplicate samples so each integer
//! block is tested at most once. Reach is a Euclidean cap because the
//! sphere surface is locally Euclidean for the player's reach distance.

use super::block::BlockType;
use super::sphere::{self, ChunkPos};
use super::world::World;
use glam::Vec3;

const MAX_REACH: f32 = 8.0;
const STEP: f32 = 0.1;

/// Result of a successful raycast: the solid block hit and the empty
/// adjacent block (used for placement).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct RaycastHit {
    pub hit: BlockRef,
    pub adjacent: BlockRef,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct BlockRef {
    pub chunk: ChunkPos,
    pub lx: usize,
    pub ly: usize,
    pub lz: usize,
}

/// Cast a ray from `origin` along `direction` and return the first solid
/// block hit within `MAX_REACH`, plus the previous (empty) sample as the
/// placement target.
pub fn raycast(origin: Vec3, direction: Vec3, world: &World) -> Option<RaycastHit> {
    let dir = direction.normalize_or_zero();
    if dir == Vec3::ZERO {
        return None;
    }
    let max_steps = (MAX_REACH / STEP) as usize + 1;
    let mut prev: Option<BlockRef> = None;
    for i in 0..max_steps {
        let t = i as f32 * STEP;
        let p = (origin + dir * t).as_dvec3();
        let Some((cp, lx, ly, lz)) = sphere::world_to_chunk_local(p) else {
            continue;
        };
        if cp.cy < 0 || !chunk_in_face_range(cp) {
            continue;
        }
        let here = BlockRef {
            chunk: cp,
            lx: lx.floor() as usize,
            ly: ly.floor() as usize,
            lz: lz.floor() as usize,
        };
        if Some(here) == prev {
            continue;
        }
        let block = world.block_at(cp, here.lx, here.ly, here.lz);
        if block != BlockType::Air {
            return Some(RaycastHit {
                hit: here,
                adjacent: prev.unwrap_or(here),
            });
        }
        prev = Some(here);
    }
    None
}

fn chunk_in_face_range(cp: ChunkPos) -> bool {
    let n = sphere::FACE_SIDE_CHUNKS;
    cp.cx >= 0 && cp.cx < n && cp.cz >= 0 && cp.cz < n && cp.cy >= 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::chunk::{Chunk, CHUNK_SIZE};
    use crate::voxel::sphere::Face;

    fn empty_world() -> World {
        World::new(2, 0, None)
    }

    #[test]
    fn raycast_misses_into_empty_space() {
        let world = empty_world();
        let origin = Vec3::new(0.0, 1000.0, 0.0);
        let dir = Vec3::new(0.0, 1.0, 0.0);
        assert!(raycast(origin, dir, &world).is_none());
    }

    #[test]
    fn raycast_hits_a_solid_block_directly_below_pole() {
        // Place a solid stone block at the +Y pole and shoot a ray down at it.
        let mut world = empty_world();
        let cp = ChunkPos { face: Face::PosY, cx: sphere::FACE_SIDE_CHUNKS / 2, cy: 5, cz: sphere::FACE_SIDE_CHUNKS / 2 };
        let mut chunk = Chunk::new(BlockType::Air);
        chunk.set(8, 8, 8, BlockType::Stone);
        world.chunks_mut_for_test().insert(cp, chunk);
        let block_world = sphere::chunk_to_world(cp, Vec3::new(8.5, 8.5, 8.5)).as_vec3();
        let origin = block_world + Vec3::new(0.0, 5.0, 0.0);
        let dir = -Vec3::Y;
        let hit = raycast(origin, dir, &world).expect("ray should hit");
        assert_eq!(hit.hit.chunk, cp);
        assert_eq!((hit.hit.lx, hit.hit.ly, hit.hit.lz), (8, 8, 8));
    }
}
