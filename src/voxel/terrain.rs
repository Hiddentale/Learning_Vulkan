use super::biome::{self, Biome};
use super::block::BlockType;
use super::chunk::{Chunk, CHUNK_SIZE};
use super::sphere;
use super::world::{TERRAIN_MAX_CY, TERRAIN_MIN_CY};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti};

pub(crate) const SEA_LEVEL: usize = 64;
const DIRT_DEPTH: usize = 4;
const MIN_HEIGHT: usize = 4;
const MAX_HEIGHT: usize = 700;
const CAVE_THRESHOLD: f64 = 0.55;
const CHUNK_LAYERS: usize = (TERRAIN_MAX_CY - TERRAIN_MIN_CY + 1) as usize;

// Noise scales — each parameter operates at a different spatial frequency
const CONTINENTALNESS_SCALE: f64 = 0.0008;
const EROSION_SCALE: f64 = 0.002;
const WEIRDNESS_SCALE: f64 = 0.004;
const DETAIL_SCALE: f64 = 0.02;
const MOUNTAIN_SCALE: f64 = 0.005;
const CAVE_SCALE: f64 = 0.05;
const TEMPERATURE_SCALE: f64 = 0.001;
const HUMIDITY_SCALE: f64 = 0.001;
const WARP_SCALE: f64 = 0.003;
pub(crate) const WARP_STRENGTH: f64 = 80.0;
const OVERHANG_SCALE: f64 = 0.04;
const OVERHANG_STRENGTH: f64 = 8.0;
const OVERHANG_BAND: usize = 20;

// Height contributions
const MOUNTAIN_AMPLITUDE: f64 = 50.0;
const DETAIL_AMPLITUDE: f64 = 8.0;
const WEIRDNESS_AMPLITUDE: f64 = 15.0;

pub(crate) struct WorldNoises {
    pub(crate) continentalness: Fbm<Perlin>,
    pub(crate) erosion_noise: Fbm<Perlin>,
    pub(crate) weirdness: Fbm<Perlin>,
    detail: Fbm<Perlin>,
    mountain: RidgedMulti<Perlin>,
    cave: Perlin,
    temperature: Fbm<Perlin>,
    humidity: Fbm<Perlin>,
    pub(crate) warp_x: Fbm<Perlin>,
    pub(crate) warp_z: Fbm<Perlin>,
    overhang: Perlin,
}

impl WorldNoises {
    pub(crate) fn new(seed: u32) -> Self {
        Self {
            continentalness: Fbm::<Perlin>::new(seed)
                .set_frequency(CONTINENTALNESS_SCALE)
                .set_octaves(5)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            erosion_noise: Fbm::<Perlin>::new(seed + 9)
                .set_frequency(EROSION_SCALE)
                .set_octaves(4)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            weirdness: Fbm::<Perlin>::new(seed + 10)
                .set_frequency(WEIRDNESS_SCALE)
                .set_octaves(3)
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

/// All noise router parameters for a single (x, z) position.
#[allow(dead_code)]
struct TerrainParams {
    continentalness: f64,
    erosion: f64,
    weirdness: f64,
    temperature: f64,
    humidity: f64,
    height: usize,
    biome: Biome,
}

/// Sample all terrain parameters at world coordinates, applying domain warping.
/// Phase B2a: 2D `(wx, wz)` is reinterpreted as a position on the +Y cube
/// face and projected onto the planet sphere; all noise is then sampled in
/// 3D at that sphere point. This makes terrain seamless across face edges
/// for free (3D noise has no seams) but means the world is finite — anything
/// generated outside the face range gets pulled toward the same sphere
/// surface point as its on-face counterpart.
fn sample_params(noises: &WorldNoises, wx: f64, wz: f64, erosion_map: Option<&super::erosion::ErosionMap>) -> TerrainParams {
    let warp_p = sphere::noise_pos_on_posy(wx, wz);
    let warped_x = wx + noises.warp_x.get(warp_p) * WARP_STRENGTH;
    let warped_z = wz + noises.warp_z.get(warp_p) * WARP_STRENGTH;
    let p = sphere::noise_pos_on_posy(warped_x, warped_z);

    let continentalness = noises.continentalness.get(p);
    let erosion = noises.erosion_noise.get(p);
    let weirdness = noises.weirdness.get(p);
    let temperature = noises.temperature.get(p);
    let humidity = noises.humidity.get(p);
    let mut height = compute_height_from_params(noises, warped_x, warped_z, continentalness, erosion, weirdness);

    // Apply hydraulic erosion delta
    if let Some(emap) = erosion_map {
        let delta = emap.sample(wx, wz);
        height = (height as f64 + delta).clamp(MIN_HEIGHT as f64, MAX_HEIGHT as f64) as usize;
    }

    let biome = biome::determine_biome(continentalness, temperature, humidity, erosion, weirdness, height, SEA_LEVEL);

    TerrainParams {
        continentalness,
        erosion,
        weirdness,
        temperature,
        humidity,
        height,
        biome,
    }
}

/// Maps continentalness [-1, 1] to a base height offset from sea level.
/// Piecewise linear: deep ocean → shelf → coast → lowland → highland.
fn continental_curve(c: f64) -> f64 {
    if c < -0.4 {
        // Deep ocean: -40 at c=-1.0 to -10 at c=-0.4
        lerp(-40.0, -10.0, (c + 1.0) / 0.6)
    } else if c < -0.2 {
        // Ocean shelf: -10 to 0
        lerp(-10.0, 0.0, (c + 0.4) / 0.2)
    } else if c < 0.0 {
        // Coast: 0 to +5
        lerp(0.0, 5.0, (c + 0.2) / 0.2)
    } else if c < 0.5 {
        // Lowland: +5 to +30
        lerp(5.0, 30.0, c / 0.5)
    } else {
        // Highland: +30 to +80
        lerp(30.0, 80.0, (c - 0.5) / 0.5)
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

pub(crate) fn compute_height_from_params(noises: &WorldNoises, wx: f64, wz: f64, continentalness: f64, erosion: f64, weirdness: f64) -> usize {
    let base = continental_curve(continentalness);
    let p = sphere::noise_pos_on_posy(wx, wz);

    // Erosion controls terrain roughness: high erosion = full mountains, low = flat
    let erosion_factor = (0.3 + erosion * 0.7).clamp(0.3, 1.0);
    let mountain = noises.mountain.get(p) * MOUNTAIN_AMPLITUDE * erosion_factor;
    let detail = noises.detail.get(p) * DETAIL_AMPLITUDE * erosion_factor;
    let weirdness_offset = weirdness * WEIRDNESS_AMPLITUDE;

    let height = SEA_LEVEL as f64 + base + mountain + detail + weirdness_offset;
    height.clamp(MIN_HEIGHT as f64, MAX_HEIGHT as f64) as usize
}

/// Sample the surface block type at world coordinates, applying domain warping.
/// Returns (height, surface_block_type). Used by heightmap generator.
pub(crate) fn sample_surface(noises: &WorldNoises, wx: f64, wz: f64, erosion_map: Option<&super::erosion::ErosionMap>) -> (usize, BlockType) {
    let params = sample_params(noises, wx, wz, erosion_map);
    let surface = biome::surface_block(params.biome);
    (params.height, surface)
}

/// Generates a full column of chunks at the given (chunk_x, chunk_z) coordinates.
pub fn generate_column(chunk_x: i32, chunk_z: i32, seed: u32, erosion_map: Option<&super::erosion::ErosionMap>) -> Vec<Chunk> {
    let noises = WorldNoises::new(seed);
    let mut chunks: Vec<Chunk> = (0..CHUNK_LAYERS).map(|_| Chunk::new(BlockType::Air)).collect();

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let wx = chunk_x as f64 * CHUNK_SIZE as f64 + x as f64;
            let wz = chunk_z as f64 * CHUNK_SIZE as f64 + z as f64;
            let params = sample_params(&noises, wx, wz, erosion_map);

            fill_surface(&mut chunks, x, z, wx, wz, params.height, params.biome, &noises);
            carve_caves(&mut chunks, x, z, wx, wz, params.height, &noises);
            fill_water(&mut chunks, x, z, params.height);
        }
    }

    chunks
}

fn fill_surface(chunks: &mut [Chunk], x: usize, z: usize, wx: f64, wz: f64, surface_y: usize, biome: Biome, noises: &WorldNoises) {
    let surface = biome::surface_block(biome);
    let subsurface = biome::subsurface_block(biome);
    let band_bottom = surface_y.saturating_sub(OVERHANG_BAND);
    let band_top = (surface_y + OVERHANG_BAND).min(MAX_HEIGHT);

    for y in 0..band_bottom {
        let block = if y + DIRT_DEPTH > surface_y { subsurface } else { BlockType::Stone };
        set_block(chunks, x, y, z, block);
    }

    for y in band_bottom..=band_top {
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
pub fn generate_lod_super_chunk(origin: [i32; 3], voxel_size: u32, seed: u32, erosion_map: Option<&super::erosion::ErosionMap>) -> LodVoxelGrid {
    let noises = WorldNoises::new(seed);
    let vs = voxel_size as f64;
    let grid_size = CHUNK_SIZE * 4; // 64
    let mut blocks = vec![BlockType::Air; grid_size * grid_size * grid_size];

    for gz in 0..grid_size {
        for gx in 0..grid_size {
            let wx = origin[0] as f64 + gx as f64 * vs;
            let wz = origin[2] as f64 + gz as f64 * vs;
            let params = sample_params(&noises, wx, wz, erosion_map);
            let surface = biome::surface_block(params.biome);
            let subsurface = biome::subsurface_block(params.biome);

            for gy in 0..grid_size {
                let wy = origin[1] as f64 + gy as f64 * vs;
                let y_top = (wy + vs - 1.0) as usize;
                let block = sample_block(y_top, params.height, surface, subsurface, &noises, wx, wy + vs * 0.5, wz);
                blocks[gx + gz * grid_size + gy * grid_size * grid_size] = block;
            }
        }
    }

    // Strip underground: keep only top SURFACE_DEPTH solid voxels per column.
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

fn sample_block(y: usize, height: usize, surface: BlockType, subsurface: BlockType, noises: &WorldNoises, wx: f64, wy: f64, wz: f64) -> BlockType {
    if y > height && y <= SEA_LEVEL {
        return BlockType::Water;
    }
    if y > height + OVERHANG_BAND {
        return BlockType::Air;
    }

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

    let block = if y >= height {
        surface
    } else if y + DIRT_DEPTH > height {
        subsurface
    } else {
        BlockType::Stone
    };

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
