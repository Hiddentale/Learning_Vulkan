use super::block::BlockType;
use super::chunk::{Chunk, CHUNK_SIZE};
use super::chunk_generator::ChunkGenerator;
use super::metric::MetricField;
use std::collections::HashMap;

pub const TERRAIN_MIN_CY: i32 = 0;
pub const TERRAIN_MAX_CY: i32 = 15;

pub struct World {
    chunks: HashMap<[i32; 3], Chunk>,
    render_distance: i32,
    generator: ChunkGenerator,
    pub metric: MetricField,
}

/// Result of a world update: which chunks were added/removed.
pub struct WorldDelta {
    pub loaded: Vec<[i32; 3]>,
    pub unloaded: Vec<[i32; 3]>,
}

impl World {
    pub fn new(render_distance: i32, seed: u32) -> Self {
        Self {
            chunks: HashMap::new(),
            render_distance,
            generator: ChunkGenerator::new(seed),
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
        let keys: Vec<[i32; 3]> = self.chunks.keys().copied().collect();
        for pos in keys {
            let xz_dist = (pos[0] - player_cx).abs().max((pos[2] - player_cz).abs());
            if xz_dist > self.render_distance {
                self.chunks.remove(&pos);
                unloaded.push(pos);
            }
        }

        // Request generation for missing columns, closest first (spiral outward).
        for ring in 0..=self.render_distance {
            for cz in (player_cz - ring)..=(player_cz + ring) {
                for cx in (player_cx - ring)..=(player_cx + ring) {
                    if (cx - player_cx).abs() != ring && (cz - player_cz).abs() != ring {
                        continue;
                    }
                    let xz_dist = (cx - player_cx).abs().max((cz - player_cz).abs());
                    if xz_dist > self.render_distance {
                        continue;
                    }
                    let has_any = (TERRAIN_MIN_CY..=TERRAIN_MAX_CY).any(|cy| self.chunks.contains_key(&[cx, cy, cz]));
                    if !has_any && !self.generator.is_pending(cx, cz) {
                        self.generator.request(cx, cz);
                    }
                }
            }
        }

        // Receive completed columns — keep all Y layers (full columns for physics)
        for col in self.generator.receive() {
            for (i, chunk) in col.chunks.into_iter().enumerate() {
                let cy = TERRAIN_MIN_CY + i as i32;
                let key = [col.cx, cy, col.cz];
                loaded.push(key);
                self.chunks.insert(key, chunk);
            }
        }

        WorldDelta { loaded, unloaded }
    }

    pub fn get_block(&self, wx: f32, wy: f32, wz: f32) -> BlockType {
        let (cx, cy, cz, lx, ly, lz) = world_to_chunk(wx, wy, wz);
        match self.chunks.get(&[cx, cy, cz]) {
            Some(chunk) => chunk.get(lx, ly, lz),
            None => BlockType::Air,
        }
    }

    /// Returns true if the block is solid, or if the chunk is not loaded but
    /// within terrain range. Unloaded terrain chunks act as solid for physics —
    /// the player must not fall through space that hasn't been generated yet.
    /// Above the terrain range, unloaded means genuinely empty sky.
    pub fn is_solid(&self, wx: f32, wy: f32, wz: f32) -> bool {
        let (cx, cy, cz, lx, ly, lz) = world_to_chunk(wx, wy, wz);
        match self.chunks.get(&[cx, cy, cz]) {
            Some(chunk) => chunk.get(lx, ly, lz).is_opaque(),
            None => (TERRAIN_MIN_CY..=TERRAIN_MAX_CY).contains(&cy),
        }
    }

    pub fn set_block(&mut self, wx: i32, wy: i32, wz: i32, block: BlockType) -> bool {
        let size = CHUNK_SIZE as i32;
        let cx = wx.div_euclid(size);
        let cy = wy.div_euclid(size);
        let cz = wz.div_euclid(size);
        let lx = wx.rem_euclid(size) as usize;
        let ly = wy.rem_euclid(size) as usize;
        let lz = wz.rem_euclid(size) as usize;
        match self.chunks.get_mut(&[cx, cy, cz]) {
            Some(chunk) => {
                chunk.set(lx, ly, lz, block);
                true
            }
            None => false,
        }
    }

    pub fn block_to_chunk(wx: i32, wy: i32, wz: i32) -> [i32; 3] {
        let size = CHUNK_SIZE as i32;
        [wx.div_euclid(size), wy.div_euclid(size), wz.div_euclid(size)]
    }

    pub fn get_chunk(&self, cx: i32, cy: i32, cz: i32) -> Option<&Chunk> {
        self.chunks.get(&[cx, cy, cz])
    }

    pub fn chunk_positions(&self) -> impl Iterator<Item = [i32; 3]> + '_ {
        self.chunks.keys().copied()
    }

    #[cfg(test)]
    pub fn insert_empty_chunk(&mut self, cx: i32, cy: i32, cz: i32) {
        self.chunks.insert([cx, cy, cz], Chunk::new(BlockType::Air));
    }
}

fn world_to_chunk(wx: f32, wy: f32, wz: f32) -> (i32, i32, i32, usize, usize, usize) {
    let size = CHUNK_SIZE as f32;
    let cx = wx.div_euclid(size) as i32;
    let cy = wy.div_euclid(size) as i32;
    let cz = wz.div_euclid(size) as i32;
    let lx = wx.rem_euclid(size) as usize;
    let ly = wy.rem_euclid(size) as usize;
    let lz = wz.rem_euclid(size) as usize;
    (cx, cy, cz, lx, ly, lz)
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
        let mut world = World::new(4, 42);
        drain_world(&mut world, 0, 0);
        let rd = 4;
        let expected = ((2 * rd + 1) * (2 * rd + 1)) as usize * TERRAIN_LAYERS;
        assert_eq!(world.chunks.len(), expected);
    }

    #[test]
    fn chunks_unload_on_move() {
        let mut world = World::new(4, 42);
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
