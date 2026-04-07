use super::block::BlockType;
use super::chunk::{Chunk, CHUNK_SIZE};
use super::chunk_generator::ChunkGenerator;
use super::erosion::ErosionMap;
use super::metric::MetricField;
use super::sphere::{self, ChunkPos};
use std::collections::HashMap;
use std::sync::Arc;

pub const TERRAIN_MIN_CY: i32 = 0;
pub const TERRAIN_MAX_CY: i32 = 47; // 768 blocks tall (48 × 16)

pub struct World {
    chunks: HashMap<ChunkPos, Chunk>,
    render_distance: i32,
    generator: ChunkGenerator,
    pub metric: MetricField,
}

/// Result of a world update: which chunks were added/removed.
pub struct WorldDelta {
    pub loaded: Vec<ChunkPos>,
    pub unloaded: Vec<ChunkPos>,
}

impl World {
    pub fn new(render_distance: i32, seed: u32, erosion_map: Option<Arc<ErosionMap>>) -> Self {
        Self {
            chunks: HashMap::new(),
            render_distance,
            generator: ChunkGenerator::new(seed, erosion_map),
            metric: MetricField::new(),
        }
    }

    /// Non-blocking world update. Unloads out-of-range chunks immediately,
    /// requests missing columns from background threads, and receives any
    /// completed columns. Never blocks the calling thread.
    pub fn update(&mut self, player_cx: i32, _player_cy: i32, player_cz: i32) -> WorldDelta {
        let mut loaded = Vec::new();
        let mut unloaded = Vec::new();

        // Unload chunks outside render distance.
        // Chunks within XZ range always stay (full column needed for physics/gravity).
        // Beyond that, use 3D distance to cull vertically.
        let keys: Vec<ChunkPos> = self.chunks.keys().copied().collect();
        for pos in keys {
            // Phase C: never unload chunks based on flat XZ distance —
            // every chunk on the planet is always loaded for the tiny
            // world. Distance-based eviction returns in Phase E.
            let _ = (pos, player_cx, player_cz);
            let _ = &mut unloaded;
            break;
        }

        // Phase C: enumerate all 6 faces × full face grid. The planet is
        // small enough that every column is always resident.
        let _ = player_cx;
        let _ = player_cz;
        for &face in &sphere::ALL_FACES {
            for cz in 0..sphere::FACE_SIDE_CHUNKS {
                for cx in 0..sphere::FACE_SIDE_CHUNKS {
                    let has_any = (TERRAIN_MIN_CY..=TERRAIN_MAX_CY).any(|cy| self.chunks.contains_key(&ChunkPos { face, cx, cy, cz }));
                    if !has_any && !self.generator.is_pending(face, cx, cz) {
                        self.generator.request(face, cx, cz);
                    }
                }
            }
        }

        // Receive completed columns — keep all Y layers (full columns for physics)
        for col in self.generator.receive() {
            for (i, chunk) in col.chunks.into_iter().enumerate() {
                let cy = TERRAIN_MIN_CY + i as i32;
                let key = ChunkPos { face: col.face, cx: col.cx, cy, cz: col.cz };
                loaded.push(key);
                self.chunks.insert(key, chunk);
            }
        }

        WorldDelta { loaded, unloaded }
    }

    pub fn get_block(&self, wx: f32, wy: f32, wz: f32) -> BlockType {
        let world = glam::DVec3::new(wx as f64, wy as f64, wz as f64);
        let Some((cp, lx, ly, lz)) = sphere::world_to_chunk_local(world) else {
            return BlockType::Air;
        };
        if !valid_chunk(cp) {
            return BlockType::Air;
        }
        let lxi = lx as usize;
        let lyi = ly as usize;
        let lzi = lz as usize;
        match self.chunks.get(&cp) {
            Some(chunk) => chunk.get(lxi.min(CHUNK_SIZE - 1), lyi.min(CHUNK_SIZE - 1), lzi.min(CHUNK_SIZE - 1)),
            None => BlockType::Air,
        }
    }

    /// Returns true if the block at this world-space cartesian position is
    /// solid. Below the planet surface (radial depth 0..TERRAIN_MAX_CY*16)
    /// unloaded chunks are treated as solid so the player cannot fall
    /// through ungenerated terrain.
    pub fn is_solid(&self, wx: f32, wy: f32, wz: f32) -> bool {
        let world = glam::DVec3::new(wx as f64, wy as f64, wz as f64);
        let Some((cp, lx, ly, lz)) = sphere::world_to_chunk_local(world) else {
            return false;
        };
        if !valid_chunk(cp) {
            return false;
        }
        let lxi = (lx as usize).min(CHUNK_SIZE - 1);
        let lyi = (ly as usize).min(CHUNK_SIZE - 1);
        let lzi = (lz as usize).min(CHUNK_SIZE - 1);
        match self.chunks.get(&cp) {
            Some(chunk) => chunk.get(lxi, lyi, lzi).is_opaque(),
            None => (TERRAIN_MIN_CY..=TERRAIN_MAX_CY).contains(&cp.cy),
        }
    }

    pub fn set_block(&mut self, wx: i32, wy: i32, wz: i32, block: BlockType) -> bool {
        // Phase D: block edits go through the sphere inverse projection.
        let world = glam::DVec3::new(wx as f64 + 0.5, wy as f64 + 0.5, wz as f64 + 0.5);
        let Some((cp, lx, ly, lz)) = sphere::world_to_chunk_local(world) else {
            return false;
        };
        if !valid_chunk(cp) {
            return false;
        }
        let lxi = (lx as usize).min(CHUNK_SIZE - 1);
        let lyi = (ly as usize).min(CHUNK_SIZE - 1);
        let lzi = (lz as usize).min(CHUNK_SIZE - 1);
        match self.chunks.get_mut(&cp) {
            Some(chunk) => {
                chunk.set(lxi, lyi, lzi, block);
                true
            }
            None => false,
        }
    }

    pub fn block_to_chunk(wx: i32, wy: i32, wz: i32) -> [i32; 3] {
        sphere::block_to_chunk(wx, wy, wz).coords()
    }

    pub fn get_chunk(&self, cx: i32, cy: i32, cz: i32) -> Option<&Chunk> {
        self.chunks.get(&ChunkPos::posy(cx, cy, cz))
    }

    pub fn get_chunk_at(&self, cp: ChunkPos) -> Option<&Chunk> {
        self.chunks.get(&cp)
    }

    pub fn chunk_positions(&self) -> impl Iterator<Item = ChunkPos> + '_ {
        self.chunks.keys().copied()
    }

    #[cfg(test)]
    pub fn insert_empty_chunk(&mut self, cx: i32, cy: i32, cz: i32) {
        self.chunks.insert(ChunkPos::posy(cx, cy, cz), Chunk::new(BlockType::Air));
    }
}

fn valid_chunk(cp: ChunkPos) -> bool {
    let n = sphere::FACE_SIDE_CHUNKS;
    cp.cx >= 0 && cp.cx < n && cp.cz >= 0 && cp.cz < n && cp.cy >= TERRAIN_MIN_CY && cp.cy <= TERRAIN_MAX_CY
}

/// 3D Chebyshev distance from a chunk to the player position.
pub fn chunk_distance(cx: i32, cy: i32, cz: i32, px: i32, py: i32, pz: i32) -> i32 {
    (cx - px).abs().max((cy - py).abs()).max((cz - pz).abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PLAYER_CY: i32 = 5;
    const TERRAIN_LAYERS: usize = (TERRAIN_MAX_CY - TERRAIN_MIN_CY + 1) as usize;

    fn drain_world(world: &mut World, player_cx: i32, player_cz: i32) {
        let rd = world.render_distance;
        let expected_columns = ((2 * rd + 1) * (2 * rd + 1)) as usize;
        let expected_chunks = expected_columns * TERRAIN_LAYERS;
        for _ in 0..500 {
            world.update(player_cx, TEST_PLAYER_CY, player_cz);
            if world.chunks.len() >= expected_chunks {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("drain_world timed out: have {}, expected {}", world.chunks.len(), expected_chunks);
    }

    #[test]
    fn chunks_load_within_render_distance() {
        let mut world = World::new(4, 42, None);
        drain_world(&mut world, 0, 0);
        let rd = 4;
        let expected = ((2 * rd + 1) * (2 * rd + 1)) as usize * TERRAIN_LAYERS;
        assert_eq!(world.chunks.len(), expected);
    }

    #[test]
    fn chunks_unload_on_move() {
        let mut world = World::new(4, 42, None);
        drain_world(&mut world, 0, 0);
        let delta = world.update(10, 5, 0);
        assert!(!delta.unloaded.is_empty());
    }

    #[test]
    fn chunk_distance_3d_chebyshev() {
        assert_eq!(chunk_distance(0, 0, 0, 0, 0, 0), 0);
        assert_eq!(chunk_distance(5, 3, 2, 0, 0, 0), 5);
        assert_eq!(chunk_distance(3, 10, 7, 0, 0, 0), 10);
        // Y dominates when player is high above terrain
        assert_eq!(chunk_distance(1, 2, 1, 1, 20, 1), 18);
        // XZ only (same as old 2D when Y is equal)
        assert_eq!(chunk_distance(5, 0, 3, 0, 0, 0), 5);
        assert_eq!(chunk_distance(3, 0, 7, 0, 0, 0), 7);
    }
}
