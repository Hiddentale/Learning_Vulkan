use super::block::BlockType;
use super::chunk::{Chunk, CHUNK_SIZE};
use noise::{NoiseFn, Perlin};

const SEED: u32 = 42;
const NOISE_SCALE: f64 = 0.08;
const MIN_HEIGHT: usize = 4;
const DIRT_DEPTH: usize = 3;

/// Generates a chunk at the given chunk coordinates using Perlin noise.
/// `chunk_x` and `chunk_z` are chunk-space coordinates (multiply by CHUNK_SIZE for world-space).
pub fn generate_chunk(chunk_x: i32, chunk_z: i32) -> Chunk {
    let perlin = Perlin::new(SEED);
    let mut chunk = Chunk::new(BlockType::Air);

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let world_x = chunk_x as f64 * CHUNK_SIZE as f64 + x as f64;
            let world_z = chunk_z as f64 * CHUNK_SIZE as f64 + z as f64;

            let noise_value = perlin.get([world_x * NOISE_SCALE, world_z * NOISE_SCALE]);
            let height = height_from_noise(noise_value);

            fill_column(&mut chunk, x, z, height);
        }
    }

    chunk
}

/// Maps noise [-1, 1] to a block height within the chunk.
fn height_from_noise(noise_value: f64) -> usize {
    let normalized = (noise_value + 1.0) * 0.5; // [0, 1]
    let max_height = CHUNK_SIZE - 1;
    let height = MIN_HEIGHT + (normalized * (max_height - MIN_HEIGHT) as f64) as usize;
    height.min(max_height)
}

/// Fills a single column: stone at the bottom, dirt above, grass on top.
fn fill_column(chunk: &mut Chunk, x: usize, z: usize, surface_y: usize) {
    for y in 0..=surface_y {
        let block = if y == surface_y {
            BlockType::Grass
        } else if y + DIRT_DEPTH > surface_y {
            BlockType::Dirt
        } else {
            BlockType::Stone
        };
        chunk.set(x, y, z, block);
    }
}
