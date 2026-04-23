use std::sync::{Arc, Mutex};
use crate::world_generation::{
    terrain_amplification::AmplifiedTerrain,
    climate::ClimateMap,
    river_networking::FlowData,
    detail_noise::DetailCache,
};

/// Bundle of external terrain data sources for voxel generation.
/// Provides seamless access to amplified heightmap, climate, river data,
/// and pre-computed detail noise during chunk generation.
pub(crate) struct TerrainData {
    pub(crate) amplified: Arc<AmplifiedTerrain>,
    pub(crate) climate: Arc<ClimateMap>,
    pub(crate) flow: Arc<FlowData>,
    pub(crate) detail_cache: Arc<Mutex<DetailCache>>,
}
