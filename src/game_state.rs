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
        loaded: usize,
        total: usize,
    },
    /// Enter an existing world without pre-generation.
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
