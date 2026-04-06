use super::biome::{self, Biome};
use super::block::BlockType;
use super::chunk::{Chunk, CHUNK_SIZE};
use super::world::{TERRAIN_MAX_CY, TERRAIN_MIN_CY};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti};

const SEA_LEVEL: usize = 64;
const DIRT_DEPTH: usize = 4;
const MIN_HEIGHT: usize = 4;
const MAX_HEIGHT: usize = 200;
const CAVE_THRESHOLD: f64 = 0.55;
const CHUNK_LAYERS: usize = (TERRAIN_MAX_CY - TERRAIN_MIN_CY + 1) as usize;

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

/// Generate a 64³ LOD super-chunk by sampling terrain noise at `voxel_size` spacing.
/// `origin` is the world-space block coordinate of the super-chunk corner.
/// Each voxel represents `voxel_size³` world blocks.
pub fn generate_lod_super_chunk(origin: [i32; 3], voxel_size: u32, seed: u32) -> LodVoxelGrid {
    let noises = WorldNoises::new(seed);
    let vs = voxel_size as f64;
    let grid_size = CHUNK_SIZE * 4; // 64
    let mut blocks = vec![BlockType::Air; grid_size * grid_size * grid_size];

    for gz in 0..grid_size {
        for gx in 0..grid_size {
            let wx = origin[0] as f64 + gx as f64 * vs;
            let wz = origin[2] as f64 + gz as f64 * vs;

            let warped_x = wx + noises.warp_x.get([wx, wz]) * WARP_STRENGTH;
            let warped_z = wz + noises.warp_z.get([wx, wz]) * WARP_STRENGTH;

            let temperature = noises.temperature.get([warped_x, warped_z]);
            let humidity = noises.humidity.get([warped_x, warped_z]);
            let height = sample_height(&noises, warped_x, warped_z, temperature);
            let biome = biome::determine_biome(temperature, humidity, height, SEA_LEVEL);
            let surface = biome::surface_block(biome);
            let subsurface = biome::subsurface_block(biome);

            for gy in 0..grid_size {
                let wy = origin[1] as f64 + gy as f64 * vs;
                // Sample height at the TOP of the voxel. A coarse voxel is only solid
                // if terrain reaches its top. This makes the LOD surface always at or
                // below the true height. At LOD boundaries the step goes DOWN (exposing
                // a top face, fully lit) instead of UP (exposing a side face, dark seam).
                let y_top = (wy + vs - 1.0) as usize;
                let block = sample_block(y_top, height, surface, subsurface, &noises, wx, wy + vs * 0.5, wz);
                blocks[gx + gz * grid_size + gy * grid_size * grid_size] = block;
            }
        }
    }

    // Strip underground: keep only the top SURFACE_DEPTH solid voxels per column.
    // LOD chunks are only seen from above — removing underground reduces SVDAG size.
    const SURFACE_DEPTH: usize = 2;
    for gz in 0..grid_size {
        for gx in 0..grid_size {
            let col = gx + gz * grid_size;
            let mut top = 0;
            for gy in (0..grid_size).rev() {
                if blocks[col + gy * grid_size * grid_size] != BlockType::Air {
                    top = gy;
                    break;
                }
            }
            if top >= SURFACE_DEPTH {
                for gy in 0..top - SURFACE_DEPTH {
                    blocks[col + gy * grid_size * grid_size] = BlockType::Air;
                }
            }
        }
    }

    LodVoxelGrid { blocks, size: grid_size }
}

/// Sample a single block at world position, using the same logic as full-res terrain.
fn sample_block(y: usize, height: usize, surface: BlockType, subsurface: BlockType, noises: &WorldNoises, wx: f64, wy: f64, wz: f64) -> BlockType {
    // Water above surface but below sea level
    if y > height && y <= SEA_LEVEL {
        return BlockType::Water;
    }
    if y > height + OVERHANG_BAND {
        return BlockType::Air;
    }

    // Overhang band: 3D density check
    let band_bottom = height.saturating_sub(OVERHANG_BAND);
    let band_top = height + OVERHANG_BAND;
    if y >= band_bottom && y <= band_top {
        let base_density = (height as f64 - y as f64) / OVERHANG_BAND as f64;
        let noise_val = noises.overhang.get([wx * OVERHANG_SCALE, wy * OVERHANG_SCALE, wz * OVERHANG_SCALE]);
        let density = base_density + noise_val * (OVERHANG_STRENGTH / OVERHANG_BAND as f64);
        if density <= 0.0 {
            return BlockType::Air;
        }
    }

    // Solid terrain
    let block = if y >= height {
        surface
    } else if y + DIRT_DEPTH > height {
        subsurface
    } else {
        BlockType::Stone
    };

    // Cave carving (skip near surface)
    if y >= 1 && y + 5 <= height {
        let cave_val = noises.cave.get([wx * CAVE_SCALE, wy * CAVE_SCALE, wz * CAVE_SCALE]);
        if cave_val > CAVE_THRESHOLD {
            return BlockType::Air;
        }
    }

    block
}

/// A flat 64³ voxel grid for LOD super-chunk generation.
pub struct LodVoxelGrid {
    blocks: Vec<BlockType>,
    size: usize,
}

impl super::svdag::VoxelSource for LodVoxelGrid {
    fn get(&self, x: usize, y: usize, z: usize) -> BlockType {
        self.blocks[x + z * self.size + y * self.size * self.size]
    }
}
