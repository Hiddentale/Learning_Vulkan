use crate::storage::world_meta::WorldMeta;
use crate::voxel::erosion::ErosionMap;
use crate::voxel::erosion_worker::ErosionWorker;
use std::path::PathBuf;
use std::sync::Arc;

pub enum GameState {
    TitleScreen,
    WorldSelect {
        worlds: Vec<(PathBuf, WorldMeta)>,
    },
    CreateWorld {
        name: String,
        seed_text: String,
    },
    PreGenerating {
        world_dir: PathBuf,
        seed: u32,
        loaded: usize,
        total: usize,
        erosion_worker: Option<ErosionWorker>,
        erosion_map: Option<Arc<ErosionMap>>,
    },
    EnteringWorld {
        world_dir: PathBuf,
        seed: u32,
    },
    Playing,
}

impl GameState {
    pub fn is_menu(&self) -> bool {
        !matches!(self, GameState::Playing)
    }
}
