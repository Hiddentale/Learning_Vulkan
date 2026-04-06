use super::erosion::{self, ErosionMap};
use crossbeam_channel::Receiver;
use std::path::PathBuf;
use std::thread;

const EROSION_GRID_SIZE: usize = 2048;
const EROSION_CELL_SIZE: f32 = 32.0;
const EROSION_ITERATIONS: usize = 100;

/// Background thread that generates the coarse erosion map.
pub struct ErosionWorker {
    result_rx: Receiver<ErosionMap>,
    _worker: thread::JoinHandle<()>,
}

impl ErosionWorker {
    /// Start erosion generation on a background thread.
    /// If a cached erosion map exists on disk, loads it instead.
    pub fn start(seed: u32, save_path: PathBuf) -> Self {
        let (tx, rx) = crossbeam_channel::bounded(1);
        let worker = thread::spawn(move || {
            let map = if save_path.exists() {
                match ErosionMap::load(&save_path) {
                    Ok(m) => m,
                    Err(e) => {
                        log::warn!("Failed to load erosion map: {e} — regenerating");
                        let m = erosion::generate_erosion_map(EROSION_GRID_SIZE, EROSION_CELL_SIZE, seed, EROSION_ITERATIONS);
                        let _ = m.save(&save_path);
                        m
                    }
                }
            } else {
                let m = erosion::generate_erosion_map(EROSION_GRID_SIZE, EROSION_CELL_SIZE, seed, EROSION_ITERATIONS);
                let _ = m.save(&save_path);
                m
            };
            let _ = tx.send(map);
        });
        Self {
            result_rx: rx,
            _worker: worker,
        }
    }

    /// Non-blocking check for completion. Returns the map if ready.
    pub fn try_receive(&self) -> Option<ErosionMap> {
        self.result_rx.try_recv().ok()
    }
}
