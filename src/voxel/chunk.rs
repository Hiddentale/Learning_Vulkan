use super::block::BlockType;

pub const CHUNK_SIZE: usize = 16;
const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

pub struct Chunk {
    blocks: [BlockType; CHUNK_VOLUME],
}

impl Chunk {
    pub fn new(fill: BlockType) -> Self {
        Self {
            blocks: [fill; CHUNK_VOLUME],
        }
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> BlockType {
        self.blocks[index(x, y, z)]
    }

    pub fn set(&mut self, x: usize, y: usize, z: usize, block: BlockType) {
        self.blocks[index(x, y, z)] = block;
    }
}

/// Converts (x, y, z) to a flat index. Y is the vertical axis (outermost).
/// Layout: x + z * CHUNK_SIZE + y * CHUNK_SIZE * CHUNK_SIZE
fn index(x: usize, y: usize, z: usize) -> usize {
    debug_assert!(x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE);
    x + z * CHUNK_SIZE + y * CHUNK_SIZE * CHUNK_SIZE
}
