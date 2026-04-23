pub(crate) mod noises;
pub(crate) mod height;
pub(crate) mod generate;
mod surface_diagnostics;

// Re-export public API
pub(crate) use noises::{WorldNoises, SEA_LEVEL, WARP_STRENGTH};
pub use generate::{generate_column, generate_lod_super_chunk, LodVoxelGrid};
pub use height::{surface_radius_at_world, compute_height_from_params};

pub(crate) use height::{sample_params_at_world, TerrainParams};
pub(crate) use generate::CHUNK_LAYERS;
