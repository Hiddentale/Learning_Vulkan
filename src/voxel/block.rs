#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockType {
    Air,
    Grass,
    Dirt,
    Stone,
    Water,
    Sand,
    Snow,
    Gravel,
}

impl BlockType {
    pub fn is_opaque(self) -> bool {
        !matches!(self, BlockType::Air | BlockType::Water)
    }

    pub fn is_transparent(self) -> bool {
        self == BlockType::Water
    }

    pub fn material_id(self) -> u8 {
        match self {
            BlockType::Air => 0,
            BlockType::Grass => 1,
            BlockType::Dirt => 2,
            BlockType::Stone => 3,
            BlockType::Water => 4,
            BlockType::Sand => 5,
            BlockType::Snow => 6,
            BlockType::Gravel => 7,
        }
    }
}
