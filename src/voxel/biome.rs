use super::block::BlockType;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Biome {
    DeepOcean,
    Ocean,
    Beach,
    Plains,
    Forest,
    Swamp,
    Desert,
    Savanna,
    Badlands,
    Mountains,
    SnowyMountains,
    Tundra,
    IceSpikes,
}

/// Selects a biome from the 6-parameter noise router.
/// `continentalness`, `temperature`, `humidity`, `erosion`, `weirdness` are in [-1, 1].
/// `height` is the terrain surface Y, `sea_level` is the water line.
pub fn determine_biome(
    continentalness: f64,
    temperature: f64,
    humidity: f64,
    erosion: f64,
    weirdness: f64,
    height: usize,
    sea_level: usize,
) -> Biome {
    // Ocean biomes by continentalness
    if continentalness < -0.4 {
        return Biome::DeepOcean;
    }
    if continentalness < -0.2 || height + 3 < sea_level {
        return Biome::Ocean;
    }

    // Beach at coastlines
    let near_sea = height <= sea_level + 3;
    if near_sea && continentalness < 0.05 {
        return if temperature < -0.3 { Biome::Tundra } else { Biome::Beach };
    }

    // Extreme cold
    if temperature < -0.5 {
        return if weirdness > 0.3 { Biome::IceSpikes } else { Biome::Tundra };
    }

    // Cold + high erosion = snowy mountains
    if temperature < -0.1 && erosion > 0.2 {
        return Biome::SnowyMountains;
    }
    if temperature < -0.1 {
        return Biome::Tundra;
    }

    // Hot + dry + high erosion = badlands
    if temperature > 0.3 && humidity < -0.3 && erosion > 0.2 {
        return Biome::Badlands;
    }

    // Hot + dry = desert
    if temperature > 0.3 && humidity < -0.1 {
        return Biome::Desert;
    }

    // Hot + moderate humidity = savanna
    if temperature > 0.2 && humidity < 0.2 {
        return Biome::Savanna;
    }

    // Wet + low-lying = swamp
    if humidity > 0.4 && erosion < -0.1 && height < sea_level + 10 {
        return Biome::Swamp;
    }

    // Humid = forest
    if humidity > 0.1 {
        return Biome::Forest;
    }

    // High erosion = mountains
    if erosion > 0.3 {
        return Biome::Mountains;
    }

    Biome::Plains
}

/// The block placed on the terrain surface.
pub fn surface_block(biome: Biome) -> BlockType {
    match biome {
        Biome::DeepOcean => BlockType::Sand,
        Biome::Ocean => BlockType::Sand,
        Biome::Beach => BlockType::Sand,
        Biome::Plains => BlockType::Grass,
        Biome::Forest => BlockType::Grass,
        Biome::Swamp => BlockType::Dirt,
        Biome::Desert => BlockType::Sand,
        Biome::Savanna => BlockType::Sand,
        Biome::Badlands => BlockType::Gravel,
        Biome::Mountains => BlockType::Stone,
        Biome::SnowyMountains => BlockType::Snow,
        Biome::Tundra => BlockType::Snow,
        Biome::IceSpikes => BlockType::Snow,
    }
}

/// The block placed below the surface (dirt layer).
pub fn subsurface_block(biome: Biome) -> BlockType {
    match biome {
        Biome::DeepOcean => BlockType::Sand,
        Biome::Ocean => BlockType::Sand,
        Biome::Beach => BlockType::Sand,
        Biome::Plains => BlockType::Dirt,
        Biome::Forest => BlockType::Dirt,
        Biome::Swamp => BlockType::Dirt,
        Biome::Desert => BlockType::Sand,
        Biome::Savanna => BlockType::Dirt,
        Biome::Badlands => BlockType::Stone,
        Biome::Mountains => BlockType::Gravel,
        Biome::SnowyMountains => BlockType::Stone,
        Biome::Tundra => BlockType::Dirt,
        Biome::IceSpikes => BlockType::Snow,
    }
}
