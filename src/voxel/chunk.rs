use super::block::BlockType;

pub const CHUNK_SIZE: usize = 16;
const CHUNK_VOLUME: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;

#[derive(Clone)]
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

    /// Raw block data as bytes for GPU upload. Safe because BlockType is #[repr(u8)].
    #[allow(dead_code)] // Used in Phase 1 (voxel pool upload)
    pub fn as_bytes(&self) -> &[u8; CHUNK_VOLUME] {
        // Safety: BlockType is #[repr(u8)] so [BlockType; N] has the same layout as [u8; N]
        unsafe { &*(self.blocks.as_ptr() as *const [u8; CHUNK_VOLUME]) }
    }

    /// True iff every block in this chunk is `BlockType::Air`. Used by the
    /// upload pipeline to skip chunks that emit zero geometry.
    pub fn is_uniform_air(&self) -> bool {
        let bytes = self.as_bytes();
        let first = bytes[0];
        first == BlockType::Air as u8 && bytes.iter().all(|&b| b == first)
    }

    /// True iff every block in this chunk is the same opaque type — i.e. the
    /// chunk is one solid material with no internal air. Combined with a
    /// neighbor check this identifies "buried" chunks that emit no geometry.
    pub fn is_uniform_opaque(&self) -> bool {
        let bytes = self.as_bytes();
        let first = bytes[0];
        // SAFETY: BlockType is #[repr(u8)] with discriminants 0..=7
        let kind: BlockType = unsafe { std::mem::transmute(first) };
        kind.is_opaque() && bytes.iter().all(|&b| b == first)
    }
}

/// Converts (x, y, z) to a flat index. Y is the vertical axis (outermost).
/// Layout: x + z * CHUNK_SIZE + y * CHUNK_SIZE * CHUNK_SIZE
fn index(x: usize, y: usize, z: usize) -> usize {
    debug_assert!(x < CHUNK_SIZE && y < CHUNK_SIZE && z < CHUNK_SIZE);
    x + z * CHUNK_SIZE + y * CHUNK_SIZE * CHUNK_SIZE
}
