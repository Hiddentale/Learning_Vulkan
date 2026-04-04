use super::block::BlockType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Biome {
    Ocean,
    Plains,
    Desert,
    Mountains,
    Tundra,
}

/// Selects a biome from temperature and humidity values in [-1, 1].
pub fn determine_biome(temperature: f64, humidity: f64, height: usize, sea_level: usize) -> Biome {
    if height < sea_level.saturating_sub(5) {
        return Biome::Ocean;
    }
    if temperature < -0.3 {
        return Biome::Tundra;
    }
    if temperature > 0.3 && humidity < -0.1 {
        return Biome::Desert;
    }
    if humidity > 0.3 || temperature > 0.1 {
        return Biome::Plains;
    }
    Biome::Mountains
}

/// The block placed on the terrain surface.
pub fn surface_block(biome: Biome) -> BlockType {
    match biome {
        Biome::Ocean => BlockType::Sand,
        Biome::Plains => BlockType::Grass,
        Biome::Desert => BlockType::Sand,
        Biome::Mountains => BlockType::Stone,
        Biome::Tundra => BlockType::Snow,
    }
}

/// The block placed below the surface (dirt layer).
pub fn subsurface_block(biome: Biome) -> BlockType {
    match biome {
        Biome::Ocean => BlockType::Sand,
        Biome::Plains => BlockType::Dirt,
        Biome::Desert => BlockType::Sand,
        Biome::Mountains => BlockType::Gravel,
        Biome::Tundra => BlockType::Dirt,
    }
}
