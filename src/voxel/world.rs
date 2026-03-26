use super::chunk::Chunk;
use super::terrain;
use std::collections::HashMap;

pub struct World {
    chunks: HashMap<[i32; 2], Chunk>,
}

impl World {
    /// Creates a world with a grid of chunks centered around the origin.
    pub fn generate(radius: i32) -> Self {
        let mut chunks = HashMap::new();
        for cz in -radius..=radius {
            for cx in -radius..=radius {
                let chunk = terrain::generate_chunk(cx, cz);
                chunks.insert([cx, cz], chunk);
            }
        }
        Self { chunks }
    }

    pub fn get_chunk(&self, cx: i32, cz: i32) -> Option<&Chunk> {
        self.chunks.get(&[cx, cz])
    }

    pub fn chunk_positions(&self) -> impl Iterator<Item = [i32; 2]> + '_ {
        self.chunks.keys().copied()
    }
}
