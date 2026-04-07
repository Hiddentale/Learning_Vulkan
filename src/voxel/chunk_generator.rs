use super::chunk::Chunk;
use super::erosion::ErosionMap;
use super::sphere::Face;
use super::terrain;
use crossbeam_channel::{Receiver, Sender};
use std::collections::HashSet;
use std::sync::Arc;
use std::thread;

const WORKER_COUNT: usize = 4;

pub struct GeneratedColumn {
    pub face: Face,
    pub cx: i32,
    pub cz: i32,
    pub chunks: Vec<Chunk>,
}

#[derive(Copy, Clone, Hash, PartialEq, Eq)]
struct ColumnKey {
    face: Face,
    cx: i32,
    cz: i32,
}

/// Manages background threads that generate terrain columns.
/// Main thread sends requests, workers generate, main thread receives results.
pub struct ChunkGenerator {
    request_tx: Sender<ColumnKey>,
    result_rx: Receiver<GeneratedColumn>,
    pending: HashSet<ColumnKey>,
    _workers: Vec<thread::JoinHandle<()>>,
}

impl ChunkGenerator {
    pub fn new(seed: u32, erosion_map: Option<Arc<ErosionMap>>) -> Self {
        let (request_tx, request_rx) = crossbeam_channel::unbounded::<ColumnKey>();
        let (result_tx, result_rx) = crossbeam_channel::unbounded::<GeneratedColumn>();

        let mut workers = Vec::with_capacity(WORKER_COUNT);
        for _ in 0..WORKER_COUNT {
            let rx = request_rx.clone();
            let tx = result_tx.clone();
            let emap = erosion_map.clone();
            workers.push(thread::spawn(move || {
                while let Ok(ColumnKey { face, cx, cz }) = rx.recv() {
                    let chunks = terrain::generate_column(face, cx, cz, seed, emap.as_deref());
                    if tx.send(GeneratedColumn { face, cx, cz, chunks }).is_err() {
                        break;
                    }
                }
            }));
        }

        Self {
            request_tx,
            result_rx,
            pending: HashSet::new(),
            _workers: workers,
        }
    }

    /// Requests generation of a column if not already pending.
    pub fn request(&mut self, face: Face, cx: i32, cz: i32) {
        let key = ColumnKey { face, cx, cz };
        if self.pending.insert(key) {
            let _ = self.request_tx.send(key);
        }
    }

    /// Drains all completed columns (non-blocking). Returns them for the caller to insert.
    pub fn receive(&mut self) -> Vec<GeneratedColumn> {
        let mut results = Vec::new();
        while let Ok(col) = self.result_rx.try_recv() {
            self.pending.remove(&ColumnKey { face: col.face, cx: col.cx, cz: col.cz });
            results.push(col);
        }
        results
    }

    /// Returns true if a column is currently queued or being generated.
    pub fn is_pending(&self, face: Face, cx: i32, cz: i32) -> bool {
        self.pending.contains(&ColumnKey { face, cx, cz })
    }
}

impl Drop for ChunkGenerator {
    fn drop(&mut self) {
        // Drop the sender to signal workers to exit
        // (request_tx is dropped when self is dropped, closing the channel)
    }
}
