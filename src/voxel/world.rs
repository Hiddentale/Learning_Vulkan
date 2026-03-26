use super::block::BlockType;
use super::chunk::{Chunk, CHUNK_SIZE};
use super::terrain;
use std::collections::HashMap;

pub struct World {
    chunks: HashMap<[i32; 2], Chunk>,
    render_distance: i32,
}

/// Result of a world update: which chunks were added/removed.
pub struct WorldDelta {
    pub loaded: Vec<[i32; 2]>,
    pub unloaded: Vec<[i32; 2]>,
}

impl World {
    pub fn new(render_distance: i32) -> Self {
        Self {
            chunks: HashMap::new(),
            render_distance,
        }
    }

    /// Updates loaded chunks based on the player's chunk coordinate.
    /// Returns which chunks were added or removed.
    pub fn update(&mut self, player_cx: i32, player_cz: i32) -> WorldDelta {
        let mut loaded = Vec::new();
        let mut unloaded = Vec::new();

        // Unload chunks outside render distance
        let keys: Vec<[i32; 2]> = self.chunks.keys().copied().collect();
        for pos in keys {
            if !in_range(pos[0], pos[1], player_cx, player_cz, self.render_distance) {
                self.chunks.remove(&pos);
                unloaded.push(pos);
            }
        }

        // Load missing chunks within render distance
        for cz in (player_cz - self.render_distance)..=(player_cz + self.render_distance) {
            for cx in (player_cx - self.render_distance)..=(player_cx + self.render_distance) {
                if let std::collections::hash_map::Entry::Vacant(e) = self.chunks.entry([cx, cz]) {
                    let chunk = terrain::generate_chunk(cx, cz);
                    e.insert(chunk);
                    loaded.push([cx, cz]);
                }
            }
        }

        WorldDelta { loaded, unloaded }
    }

    /// Returns the block at a world-space position, or Air if the chunk isn't loaded.
    pub fn get_block(&self, wx: f32, wy: f32, wz: f32) -> BlockType {
        // Out-of-bounds vertically is always air
        if wy < 0.0 || wy >= CHUNK_SIZE as f32 {
            return BlockType::Air;
        }

        let (cx, cz, lx, ly, lz) = world_to_chunk(wx, wy, wz);
        match self.chunks.get(&[cx, cz]) {
            Some(chunk) => chunk.get(lx, ly, lz),
            None => BlockType::Air,
        }
    }

    /// Returns true if the block at a world-space position is solid.
    pub fn is_solid(&self, wx: f32, wy: f32, wz: f32) -> bool {
        self.get_block(wx, wy, wz).is_opaque()
    }

    pub fn get_chunk(&self, cx: i32, cz: i32) -> Option<&Chunk> {
        self.chunks.get(&[cx, cz])
    }

    pub fn chunk_positions(&self) -> impl Iterator<Item = [i32; 2]> + '_ {
        self.chunks.keys().copied()
    }
}

/// Converts world-space float coordinates to (chunk_x, chunk_z, local_x, local_y, local_z).
fn world_to_chunk(wx: f32, wy: f32, wz: f32) -> (i32, i32, usize, usize, usize) {
    let size = CHUNK_SIZE as f32;
    let cx = wx.div_euclid(size) as i32;
    let cz = wz.div_euclid(size) as i32;
    let lx = wx.rem_euclid(size) as usize;
    let ly = wy as usize;
    let lz = wz.rem_euclid(size) as usize;
    (cx, cz, lx, ly, lz)
}

fn in_range(cx: i32, cz: i32, player_cx: i32, player_cz: i32, radius: i32) -> bool {
    (cx - player_cx).abs() <= radius && (cz - player_cz).abs() <= radius
}
