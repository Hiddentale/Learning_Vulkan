use crate::storage::world_meta::WorldMeta;
use std::path::PathBuf;

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
        /// Columns loaded so far.
        loaded: usize,
        /// Total columns to generate.
        total: usize,
    },
    Playing,
}

impl GameState {
    pub fn is_menu(&self) -> bool {
        !matches!(self, GameState::Playing)
    }
}
