use super::block::BlockType;
use super::chunk::{Chunk, CHUNK_SIZE};
use super::chunk_generator::ChunkGenerator;
use std::collections::HashMap;

pub const MIN_CHUNK_Y: i32 = 0;
pub const MAX_CHUNK_Y: i32 = 15;

pub struct World {
    chunks: HashMap<[i32; 3], Chunk>,
    render_distance: i32,
    generator: ChunkGenerator,
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
        }
    }

    /// Non-blocking world update. Unloads out-of-range chunks immediately,
    /// requests missing columns from background threads, and receives any
    /// completed columns. Never blocks the calling thread.
    pub fn update(&mut self, player_cx: i32, player_cz: i32) -> WorldDelta {
        let mut loaded = Vec::new();
        let mut unloaded = Vec::new();

        // Unload columns outside render distance (always immediate)
        let keys: Vec<[i32; 3]> = self.chunks.keys().copied().collect();
        for pos in keys {
            if !in_range(pos[0], pos[2], player_cx, player_cz, self.render_distance) {
                self.chunks.remove(&pos);
                unloaded.push(pos);
                // Cancel if still pending (best-effort)
                self.generator.cancel(pos[0], pos[2]);
            }
        }

        // Request generation for missing columns (non-blocking, just enqueues)
        for cz in (player_cz - self.render_distance)..=(player_cz + self.render_distance) {
            for cx in (player_cx - self.render_distance)..=(player_cx + self.render_distance) {
                if !self.chunks.contains_key(&[cx, MIN_CHUNK_Y, cz]) && !self.generator.is_pending(cx, cz) {
                    self.generator.request(cx, cz);
                }
            }
        }

        // Receive completed columns from workers (non-blocking drain)
        for col in self.generator.receive() {
            // Skip if column was unloaded while generating (player moved away)
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

    /// Returns the block at a world-space position, or Air if the chunk isn't loaded.
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

    /// Returns true if the block at a world-space position is solid.
    pub fn is_solid(&self, wx: f32, wy: f32, wz: f32) -> bool {
        self.get_block(wx, wy, wz).is_opaque()
    }

    pub fn get_chunk(&self, cx: i32, cy: i32, cz: i32) -> Option<&Chunk> {
        self.chunks.get(&[cx, cy, cz])
    }

    pub fn chunk_positions(&self) -> impl Iterator<Item = [i32; 3]> + '_ {
        self.chunks.keys().copied()
    }
}

/// Converts world-space float coordinates to (chunk_x, chunk_y, chunk_z, local_x, local_y, local_z).
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
