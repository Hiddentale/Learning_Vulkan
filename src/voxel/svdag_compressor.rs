#![allow(dead_code)] // Wired up in Phase 3

use super::chunk::Chunk;
use super::svdag::{rle_encode, svdag_from_chunk, svdag_materials};
use crossbeam_channel::{Receiver, Sender};
use std::thread;

const WORKER_COUNT: usize = 2;

/// Request to compress a chunk into SVDAG format.
pub struct CompressionRequest {
    pub pos: [i32; 3],
    pub chunk: Chunk,
}

/// Result of SVDAG compression.
pub struct CompressionResult {
    pub pos: [i32; 3],
    pub dag_data: Vec<u8>,
    pub material_data: Vec<u8>,
}

/// Background thread pool that compresses dense chunks into SVDAGs.
pub struct SvdagCompressor {
    request_tx: Sender<CompressionRequest>,
    result_rx: Receiver<CompressionResult>,
    _workers: Vec<thread::JoinHandle<()>>,
}

impl SvdagCompressor {
    pub fn new() -> Self {
        let (request_tx, request_rx) = crossbeam_channel::unbounded::<CompressionRequest>();
        let (result_tx, result_rx) = crossbeam_channel::unbounded::<CompressionResult>();

        let mut workers = Vec::with_capacity(WORKER_COUNT);
        for _ in 0..WORKER_COUNT {
            let rx = request_rx.clone();
            let tx = result_tx.clone();
            workers.push(thread::spawn(move || {
                while let Ok(req) = rx.recv() {
                    let dag_data = svdag_from_chunk(&req.chunk);
                    let materials = svdag_materials(&req.chunk);
                    let material_data = rle_encode(&materials);
                    if tx
                        .send(CompressionResult {
                            pos: req.pos,
                            dag_data,
                            material_data,
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            }));
        }

        Self {
            request_tx,
            result_rx,
            _workers: workers,
        }
    }

    /// Queue a chunk for background SVDAG compression.
    pub fn request(&self, pos: [i32; 3], chunk: Chunk) {
        let _ = self.request_tx.send(CompressionRequest { pos, chunk });
    }

    /// Drain all completed compressions (non-blocking).
    pub fn receive(&self) -> Vec<CompressionResult> {
        let mut results = Vec::new();
        while let Ok(result) = self.result_rx.try_recv() {
            results.push(result);
        }
        results
    }
}
