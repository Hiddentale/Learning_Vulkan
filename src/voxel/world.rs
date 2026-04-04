use super::block::BlockType;
use super::chunk::{Chunk, CHUNK_SIZE};
use super::chunk_generator::ChunkGenerator;
use super::metric::MetricField;
use std::collections::HashMap;

pub const MIN_CHUNK_Y: i32 = 0;
pub const MAX_CHUNK_Y: i32 = 15;

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
    pub fn new(render_distance: i32) -> Self {
        Self {
            chunks: HashMap::new(),
            render_distance,
            generator: ChunkGenerator::new(),
            metric: MetricField::new(),
        }
    }

    /// Non-blocking world update. Unloads out-of-range chunks immediately,
    /// requests missing columns from background threads, and receives any
    /// completed columns. Never blocks the calling thread.
    pub fn update(&mut self, player_cx: i32, player_cz: i32) -> WorldDelta {
        let mut loaded = Vec::new();
        let mut unloaded = Vec::new();

        // Unload columns outside render distance
        let keys: Vec<[i32; 3]> = self.chunks.keys().copied().collect();
        for pos in keys {
            if !in_range(pos[0], pos[2], player_cx, player_cz, self.render_distance) {
                self.chunks.remove(&pos);
                unloaded.push(pos);
                self.generator.cancel(pos[0], pos[2]);
            }
        }

        // Request generation for missing columns, closest first (spiral outward)
        for ring in 0..=self.render_distance {
            for cz in (player_cz - ring)..=(player_cz + ring) {
                for cx in (player_cx - ring)..=(player_cx + ring) {
                    // Only process the border of this ring
                    if (cx - player_cx).abs() != ring && (cz - player_cz).abs() != ring {
                        continue;
                    }
                    if !self.chunks.contains_key(&[cx, MIN_CHUNK_Y, cz]) && !self.generator.is_pending(cx, cz) {
                        self.generator.request(cx, cz);
                    }
                }
            }
        }

        // Receive completed columns from workers
        for col in self.generator.receive() {
            if !in_range(col.cx, col.cz, player_cx, player_cz, self.render_distance) {
                continue;
            }
            for (i, chunk) in col.chunks.into_iter().enumerate() {
                let cy = MIN_CHUNK_Y + i as i32;
                let key = [col.cx, cy, col.cz];
                loaded.push(key);
                self.chunks.insert(key, chunk);
            }
        }

        WorldDelta { loaded, unloaded }
    }

    pub fn get_block(&self, wx: f32, wy: f32, wz: f32) -> BlockType {
        let (cx, cy, cz, lx, ly, lz) = world_to_chunk(wx, wy, wz);
        if !(MIN_CHUNK_Y..=MAX_CHUNK_Y).contains(&cy) {
            return BlockType::Air;
        }
        match self.chunks.get(&[cx, cy, cz]) {
            Some(chunk) => chunk.get(lx, ly, lz),
            None => BlockType::Air,
        }
    }

    pub fn is_solid(&self, wx: f32, wy: f32, wz: f32) -> bool {
        self.get_block(wx, wy, wz).is_opaque()
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

fn in_range(cx: i32, cz: i32, player_cx: i32, player_cz: i32, radius: i32) -> bool {
    (cx - player_cx).abs() <= radius && (cz - player_cz).abs() <= radius
}

/// Chebyshev distance from a chunk column to the player column.
pub fn chunk_distance(cx: i32, cz: i32, player_cx: i32, player_cz: i32) -> i32 {
    (cx - player_cx).abs().max((cz - player_cz).abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drain_world(world: &mut World, player_cx: i32, player_cz: i32) {
        let rd = world.render_distance;
        let expected_columns = ((2 * rd + 1) * (2 * rd + 1)) as usize;
        let expected_chunks = expected_columns * (MAX_CHUNK_Y - MIN_CHUNK_Y + 1) as usize;
        for _ in 0..500 {
            world.update(player_cx, player_cz);
            if world.chunks.len() >= expected_chunks {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("drain_world timed out");
    }

    #[test]
    fn chunks_load_within_render_distance() {
        let mut world = World::new(4);
        drain_world(&mut world, 0, 0);
        let expected = ((2 * 4 + 1) * (2 * 4 + 1)) as usize * 16;
        assert_eq!(world.chunks.len(), expected);
    }

    #[test]
    fn chunks_unload_on_move() {
        let mut world = World::new(4);
        drain_world(&mut world, 0, 0);
        let delta = world.update(10, 0);
        assert!(!delta.unloaded.is_empty());
    }

    #[test]
    fn chunk_distance_chebyshev() {
        assert_eq!(chunk_distance(0, 0, 0, 0), 0);
        assert_eq!(chunk_distance(5, 3, 0, 0), 5);
        assert_eq!(chunk_distance(3, 7, 0, 0), 7);
    }
}
