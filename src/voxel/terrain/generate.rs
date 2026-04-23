use super::height::{sample_params_at_world, sample_density_block};
use super::noises::{WorldNoises, SEA_LEVEL, MOUNTAIN_AMPLITUDE, WEIRDNESS_AMPLITUDE, CAVE_MIN_DEPTH, OVERHANG_SCALE_FACTOR, OVERHANG_STRENGTH_FACTOR, OVERHANG_BAND_SIZE, CAVE_THRESHOLD, CAVE_SCALE_FACTOR, DIRT_DEPTH};
use super::super::biome;
use super::super::block::BlockType;
use super::super::chunk::{Chunk, CHUNK_SIZE};
use super::super::sphere::{self, Face};
use super::super::world::{TERRAIN_MAX_CY, TERRAIN_MIN_CY};
use noise::NoiseFn;

pub const CHUNK_LAYERS: usize = (TERRAIN_MAX_CY - TERRAIN_MIN_CY + 1) as usize;

/// Generates a full column of chunks via 3D density sampling. For each block
/// in the column, we compute the world cartesian position via the cube-to-
/// sphere projection and evaluate a density function at that point. Density
/// > 0 is solid; density <= 0 with `|world| < surface_radius_at_sea_level`
/// is water; otherwise air. Because density is purely a function of world
/// position (and noise on direction), terrain is seamless across face edges.
///
/// If `detail_col` is provided, uses pre-computed detail noise offsets instead
/// of generating them on-the-fly. This avoids frame stutter during chunk loads.
pub fn generate_column(
    face: Face,
    chunk_x: i32,
    chunk_z: i32,
    seed: u32,
    erosion_map: Option<&super::super::erosion::ErosionMap>,
    detail_col: Option<&crate::world_generation::detail_noise::DetailColumn>,
) -> Vec<Chunk> {
    let noises = WorldNoises::new(seed);
    let mut chunks: Vec<Chunk> = (0..CHUNK_LAYERS).map(|_| Chunk::new(BlockType::Air)).collect();

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            fill_density_column(&mut chunks, face, chunk_x, chunk_z, x, z, &noises, erosion_map, detail_col);
        }
    }

    chunks
}

/// Per-(x, z) column fill: walks every radial layer, evaluates density at the
/// block center, and writes the resulting block type. The per-direction
/// noise (continentalness, mountain, biome, …) is sampled ONCE for the
/// whole column — it depends only on the direction, which is constant as
/// `ly` varies. Only the 3D overhang and cave noise sample per block.
///
/// If `detail_col` is provided, uses pre-computed detail noise offsets.
/// Otherwise, detail is computed on-the-fly.
fn fill_density_column(
    chunks: &mut [Chunk],
    face: Face,
    chunk_x: i32,
    chunk_z: i32,
    x: usize,
    z: usize,
    noises: &WorldNoises,
    erosion_map: Option<&super::super::erosion::ErosionMap>,
    detail_col: Option<&crate::world_generation::detail_noise::DetailColumn>,
) {
    // Sample one representative point in the column to fix the direction.
    let probe = sphere::chunk_to_world(
        sphere::ChunkPos {
            face,
            cx: chunk_x,
            cy: 0,
            cz: chunk_z,
        },
        glam::Vec3::new(x as f32 + 0.5, 0.5, z as f32 + 0.5),
    );

    // Get pre-computed detail offset if available
    let detail_offset = detail_col.map(|col| col[z * 16 + x]).unwrap_or(0.0);

    let mut params = sample_params_at_world(noises, probe, erosion_map);

    // Apply pre-computed detail offset instead of computing it on-the-fly
    if let Some(_) = detail_col {
        // If we have pre-computed detail, we've already incorporated domain warp
        // and slope-gated amplitude, so we just apply the offset
        params.height = (params.height as f64 + detail_offset as f64)
            .clamp(super::noises::MIN_HEIGHT as f64, super::noises::MAX_HEIGHT as f64) as usize;
    }

    let surface_radius = sphere::PLANET_RADIUS_BLOCKS as f64 + params.height as f64;
    let sea_radius = sphere::SURFACE_RADIUS_BLOCKS as f64;
    let max_radius_seen = sea_radius + MOUNTAIN_AMPLITUDE + WEIRDNESS_AMPLITUDE + 50.0;
    let surface_block = biome::surface_block(params.biome);
    let subsurface_block = biome::subsurface_block(params.biome);

    for (cy, chunk) in chunks.iter_mut().enumerate().take(CHUNK_LAYERS) {
        for ly in 0..CHUNK_SIZE {
            let cp = sphere::ChunkPos {
                face,
                cx: chunk_x,
                cy: cy as i32,
                cz: chunk_z,
            };
            let local = glam::Vec3::new(x as f32 + 0.5, ly as f32 + 0.5, z as f32 + 0.5);
            let world = sphere::chunk_to_world(cp, local);
            let r = world.length();
            if r > max_radius_seen + 1.0 {
                continue;
            }
            let block = sample_density_block(world, r, surface_radius, sea_radius, surface_block, subsurface_block, noises);
            if block != BlockType::Air {
                chunk.set(x, ly, z, block);
            }
        }
    }
}

fn sample_block(y: usize, height: usize, surface: BlockType, subsurface: BlockType, noises: &WorldNoises, wx: f64, wy: f64, wz: f64) -> BlockType {
    if y > height && y <= SEA_LEVEL {
        return BlockType::Water;
    }
    if y > height + OVERHANG_BAND_SIZE {
        return BlockType::Air;
    }

    let band_bottom = height.saturating_sub(OVERHANG_BAND_SIZE);
    let band_top = height + OVERHANG_BAND_SIZE;
    if y >= band_bottom && y <= band_top {
        let base_density = (height as f64 - y as f64) / OVERHANG_BAND_SIZE as f64;
        let noise_val = noises.overhang.get([wx * OVERHANG_SCALE_FACTOR, wy * OVERHANG_SCALE_FACTOR, wz * OVERHANG_SCALE_FACTOR]);
        let density = base_density + noise_val * (OVERHANG_STRENGTH_FACTOR / OVERHANG_BAND_SIZE as f64);
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
        let cave_val = noises.cave.get([wx * CAVE_SCALE_FACTOR, wy * CAVE_SCALE_FACTOR, wz * CAVE_SCALE_FACTOR]);
        if cave_val > CAVE_THRESHOLD {
            return BlockType::Air;
        }
    }

    block
}

/// Generate a 64³ LOD super-chunk by sampling terrain noise at `voxel_size` spacing.
pub fn generate_lod_super_chunk(origin: [i32; 3], voxel_size: u32, seed: u32, erosion_map: Option<&super::super::erosion::ErosionMap>) -> LodVoxelGrid {
    let noises = WorldNoises::new(seed);
    let vs = voxel_size as f64;
    let grid_size = CHUNK_SIZE * 4; // 64
    let mut blocks = vec![BlockType::Air; grid_size * grid_size * grid_size];

    for gz in 0..grid_size {
        for gx in 0..grid_size {
            let wx = origin[0] as f64 + gx as f64 * vs;
            let wz = origin[2] as f64 + gz as f64 * vs;
            // Phase C: LOD super-chunk path is disabled. Hardcode +Y face.
            let params = super::height::sample_params(&noises, Face::PosY, wx, wz, erosion_map);
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

/// A flat 64³ voxel grid for LOD super-chunk generation.
pub struct LodVoxelGrid {
    blocks: Vec<BlockType>,
    size: usize,
}

impl super::super::svdag::VoxelSource for LodVoxelGrid {
    fn get(&self, x: usize, y: usize, z: usize) -> BlockType {
        self.blocks[x + z * self.size + y * self.size * self.size]
    }
}
