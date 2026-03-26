#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockType {
    Air,
    Grass,
    Dirt,
    Stone,
}

impl BlockType {
    pub fn is_opaque(self) -> bool {
        self != BlockType::Air
    }

    pub fn color(self) -> [f32; 3] {
        match self {
            BlockType::Air => [0.0, 0.0, 0.0],
            BlockType::Grass => [0.3, 0.7, 0.2],
            BlockType::Dirt => [0.55, 0.35, 0.15],
            BlockType::Stone => [0.5, 0.5, 0.5],
        }
    }
}
