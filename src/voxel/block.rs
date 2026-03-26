#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockType {
    Air,
    Solid,
}

impl BlockType {
    pub fn is_opaque(self) -> bool {
        self != BlockType::Air
    }
}
