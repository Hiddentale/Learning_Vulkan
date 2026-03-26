use super::chunk::Chunk;
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

    pub fn get_chunk(&self, cx: i32, cz: i32) -> Option<&Chunk> {
        self.chunks.get(&[cx, cz])
    }

    pub fn chunk_positions(&self) -> impl Iterator<Item = [i32; 2]> + '_ {
        self.chunks.keys().copied()
    }
}

fn in_range(cx: i32, cz: i32, player_cx: i32, player_cz: i32, radius: i32) -> bool {
    (cx - player_cx).abs() <= radius && (cz - player_cz).abs() <= radius
}
