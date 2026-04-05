use super::biome::{self, Biome};
use super::block::BlockType;
use super::chunk::{Chunk, CHUNK_SIZE};
use super::world::{MAX_CHUNK_Y, MIN_CHUNK_Y};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti};

const SEA_LEVEL: usize = 64;
const DIRT_DEPTH: usize = 4;
const MIN_HEIGHT: usize = 4;
const MAX_HEIGHT: usize = 200;
const CAVE_THRESHOLD: f64 = 0.55;
const CHUNK_LAYERS: usize = (MAX_CHUNK_Y - MIN_CHUNK_Y + 1) as usize;

// Noise scales
const CONTINENT_SCALE: f64 = 0.002;
const DETAIL_SCALE: f64 = 0.02;
const MOUNTAIN_SCALE: f64 = 0.005;
const CAVE_SCALE: f64 = 0.05;
const TEMPERATURE_SCALE: f64 = 0.001;
const HUMIDITY_SCALE: f64 = 0.001;
const WARP_SCALE: f64 = 0.003;
const WARP_STRENGTH: f64 = 80.0;
const OVERHANG_SCALE: f64 = 0.04;
const OVERHANG_STRENGTH: f64 = 8.0;
const OVERHANG_BAND: usize = 20;

// Height contributions
const CONTINENT_AMPLITUDE: f64 = 40.0;
const DETAIL_AMPLITUDE: f64 = 8.0;
const MOUNTAIN_AMPLITUDE: f64 = 50.0;

struct WorldNoises {
    continent: Fbm<Perlin>,
    detail: Fbm<Perlin>,
    mountain: RidgedMulti<Perlin>,
    cave: Perlin,
    temperature: Fbm<Perlin>,
    humidity: Fbm<Perlin>,
    warp_x: Fbm<Perlin>,
    warp_z: Fbm<Perlin>,
    overhang: Perlin,
}

impl WorldNoises {
    fn new(seed: u32) -> Self {
        Self {
            continent: Fbm::<Perlin>::new(seed)
                .set_frequency(CONTINENT_SCALE)
                .set_octaves(4)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            detail: Fbm::<Perlin>::new(seed + 1)
                .set_frequency(DETAIL_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            mountain: RidgedMulti::<Perlin>::new(seed + 2).set_frequency(MOUNTAIN_SCALE).set_octaves(4),
            cave: Perlin::new(seed + 3),
            temperature: Fbm::<Perlin>::new(seed + 4)
                .set_frequency(TEMPERATURE_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            humidity: Fbm::<Perlin>::new(seed + 5)
                .set_frequency(HUMIDITY_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            warp_x: Fbm::<Perlin>::new(seed + 6)
                .set_frequency(WARP_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            warp_z: Fbm::<Perlin>::new(seed + 7)
                .set_frequency(WARP_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            overhang: Perlin::new(seed + 8),
        }
    }
}

/// Generates a full column of chunks at the given (chunk_x, chunk_z) coordinates.
pub fn generate_column(chunk_x: i32, chunk_z: i32, seed: u32) -> Vec<Chunk> {
    let noises = WorldNoises::new(seed);
    let mut chunks: Vec<Chunk> = (0..CHUNK_LAYERS).map(|_| Chunk::new(BlockType::Air)).collect();

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let wx = chunk_x as f64 * CHUNK_SIZE as f64 + x as f64;
            let wz = chunk_z as f64 * CHUNK_SIZE as f64 + z as f64;

            let warped_x = wx + noises.warp_x.get([wx, wz]) * WARP_STRENGTH;
            let warped_z = wz + noises.warp_z.get([wx, wz]) * WARP_STRENGTH;

            let temperature = noises.temperature.get([warped_x, warped_z]);
            let humidity = noises.humidity.get([warped_x, warped_z]);
            let height = sample_height(&noises, warped_x, warped_z, temperature);
            let biome = biome::determine_biome(temperature, humidity, height, SEA_LEVEL);

            fill_surface(&mut chunks, x, z, wx, wz, height, biome, &noises);
            carve_caves(&mut chunks, x, z, wx, wz, height, &noises);
            fill_water(&mut chunks, x, z, height);
        }
    }

    chunks
}

fn sample_height(noises: &WorldNoises, wx: f64, wz: f64, temperature: f64) -> usize {
    let base = SEA_LEVEL as f64 + noises.continent.get([wx, wz]) * CONTINENT_AMPLITUDE;
    let detail = noises.detail.get([wx, wz]) * DETAIL_AMPLITUDE;

    // Mountains only where temperature is moderate (not desert, not tundra)
    let mountain_factor = (1.0 - (temperature.abs() * 2.0)).max(0.0);
    let mountain = noises.mountain.get([wx, wz]) * MOUNTAIN_AMPLITUDE * mountain_factor;

    let height = (base + detail + mountain).clamp(MIN_HEIGHT as f64, MAX_HEIGHT as f64);
    height as usize
}

fn fill_surface(chunks: &mut [Chunk], x: usize, z: usize, wx: f64, wz: f64, surface_y: usize, biome: Biome, noises: &WorldNoises) {
    let surface = biome::surface_block(biome);
    let subsurface = biome::subsurface_block(biome);
    let band_bottom = surface_y.saturating_sub(OVERHANG_BAND);
    let band_top = (surface_y + OVERHANG_BAND).min(MAX_HEIGHT);

    // Below the overhang band: always solid
    for y in 0..band_bottom {
        let block = if y + DIRT_DEPTH > surface_y { subsurface } else { BlockType::Stone };
        set_block(chunks, x, y, z, block);
    }

    // Within the overhang band: use 3D density to decide solid vs air
    for y in band_bottom..=band_top {
        // Base density: positive below surface, negative above
        let base_density = (surface_y as f64 - y as f64) / OVERHANG_BAND as f64;
        let noise_val = noises.overhang.get([wx * OVERHANG_SCALE, y as f64 * OVERHANG_SCALE, wz * OVERHANG_SCALE]);
        let density = base_density + noise_val * (OVERHANG_STRENGTH / OVERHANG_BAND as f64);

        if density > 0.0 {
            let block = if y >= surface_y {
                surface
            } else if y + DIRT_DEPTH > surface_y {
                subsurface
            } else {
                BlockType::Stone
            };
            set_block(chunks, x, y, z, block);
        }
    }
}

fn carve_caves(chunks: &mut [Chunk], x: usize, z: usize, wx: f64, wz: f64, surface_y: usize, noises: &WorldNoises) {
    // Don't carve near the surface or below bedrock
    let cave_top = surface_y.saturating_sub(5);
    for y in 1..=cave_top {
        let value = noises.cave.get([wx * CAVE_SCALE, y as f64 * CAVE_SCALE, wz * CAVE_SCALE]);
        if value > CAVE_THRESHOLD {
            set_block(chunks, x, y, z, BlockType::Air);
        }
    }
}

fn fill_water(chunks: &mut [Chunk], x: usize, z: usize, surface_y: usize) {
    if surface_y >= SEA_LEVEL {
        return;
    }
    for y in (surface_y + 1)..=SEA_LEVEL {
        set_block(chunks, x, y, z, BlockType::Water);
    }
}

fn set_block(chunks: &mut [Chunk], x: usize, y: usize, z: usize, block: BlockType) {
    let chunk_index = y / CHUNK_SIZE;
    let local_y = y % CHUNK_SIZE;
    if chunk_index < chunks.len() {
        chunks[chunk_index].set(x, local_y, z, block);
    }
}
