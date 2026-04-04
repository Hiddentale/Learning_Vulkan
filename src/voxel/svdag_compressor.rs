#![allow(dead_code)] // Wired up in Phase 3

use super::chunk::Chunk;
use super::svdag::{rle_encode, svdag_from_chunk, svdag_lod_merge, svdag_materials};
use crossbeam_channel::{Receiver, Sender};
use std::thread;

const WORKER_COUNT: usize = 2;

/// Request to compress a chunk (or group of chunks) into SVDAG format.
pub enum CompressionRequest {
    /// Single chunk at LOD-0.
    Single { pos: [i32; 3], chunk: Box<Chunk> },
    /// 8 chunks merged into LOD-1 at half resolution.
    LodMerge {
        pos: [i32; 3],
        children: Box<[Chunk; 8]>,
        lod_level: u32,
    },
}

/// Result of SVDAG compression.
pub struct CompressionResult {
    pub pos: [i32; 3],
    pub dag_data: Vec<u8>,
    pub material_data: Vec<u8>,
    pub lod_level: u32,
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
                    let result = match req {
                        CompressionRequest::Single { pos, chunk } => {
                            let dag_data = svdag_from_chunk(&chunk);
                            let materials = svdag_materials(&chunk);
                            let material_data = rle_encode(&materials);
                            CompressionResult {
                                pos,
                                dag_data,
                                material_data,
                                lod_level: 0,
                            }
                        }
                        CompressionRequest::LodMerge { pos, children, lod_level } => {
                            let refs: [&Chunk; 8] = [
                                &children[0],
                                &children[1],
                                &children[2],
                                &children[3],
                                &children[4],
                                &children[5],
                                &children[6],
                                &children[7],
                            ];
                            let dag_data = svdag_lod_merge(refs);
                            // Materials from merged chunk not supported yet — empty
                            CompressionResult {
                                pos,
                                dag_data,
                                material_data: Vec::new(),
                                lod_level,
                            }
                        }
                    };
                    if tx.send(result).is_err() {
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

    /// Queue a single chunk for background SVDAG compression at LOD-0.
    pub fn request(&self, pos: [i32; 3], chunk: Chunk) {
        let _ = self.request_tx.send(CompressionRequest::Single { pos, chunk: Box::new(chunk) });
    }

    /// Queue a LOD merge of 8 chunks.
    pub fn request_lod_merge(&self, pos: [i32; 3], children: Box<[Chunk; 8]>, lod_level: u32) {
        let _ = self.request_tx.send(CompressionRequest::LodMerge { pos, children, lod_level });
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
