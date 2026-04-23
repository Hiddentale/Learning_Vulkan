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

/// Selects a biome from climate and terrain parameters.
/// - `temperature`: degrees Celsius (real climate or noise-mapped)
/// - `precipitation`: mm/year (real climate or noise-mapped)
/// - `continentality`: [0, 1] coastal-to-inland (real climate or noise-mapped)
/// - `height`: terrain surface Y
/// - `sea_level`: water line
/// - `river_order`: Strahler stream order (0 = no river, 1+ = river)
/// - `weirdness`: noise [-1, 1], gates fantasy features
pub fn determine_biome(
    temperature: f64,
    precipitation: f64,
    continentality: f64,
    height: usize,
    sea_level: usize,
    river_order: u8,
    weirdness: f64,
) -> Biome {
    // Oceanic biomes
    if height + 3 < sea_level {
        return Biome::Ocean;
    }
    if height < sea_level {
        return Biome::DeepOcean;
    }

    // River/stream biomes
    if river_order > 0 {
        return if temperature < 0.0 { Biome::Tundra } else { Biome::Forest };
    }

    // Beach/coastal
    let near_sea = height <= sea_level + 3;
    if near_sea && continentality < 0.2 {
        return if temperature < -3.0 { Biome::Tundra } else { Biome::Beach };
    }

    // Temperature-based biomes
    if temperature < -10.0 {
        return if weirdness > 0.3 { Biome::IceSpikes } else { Biome::Tundra };
    }
    if temperature < 0.0 {
        return if height > sea_level + 200 { Biome::SnowyMountains } else { Biome::Tundra };
    }

    // Precipitation/humidity based
    if precipitation < 250.0 {
        if temperature > 20.0 {
            return Biome::Desert;
        } else if height > sea_level + 300 {
            return Biome::Badlands;
        } else {
            return Biome::Savanna;
        }
    }

    // Elevation-based
    if height > sea_level + 300 {
        return Biome::Mountains;
    }

    // Default: moderate temperature and precipitation
    if precipitation > 1000.0 {
        Biome::Swamp
    } else if precipitation > 600.0 {
        Biome::Forest
    } else {
        Biome::Plains
    }
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
