#![allow(dead_code)] // Wired up in Phase 3

use super::chunk::Chunk;
use super::svdag::{svdag_compress, svdag_from_chunk, svdag_from_super_chunk, svdag_lod_merge, svdag_lod_merge_super, SuperChunkGrid};
use super::terrain;
use crossbeam_channel::{Receiver, Sender};
use std::thread;

const WORKER_COUNT: usize = 4;

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
    /// 4×4×4 chunks grouped into a single 64³ SVDAG at full resolution.
    SuperChunk { pos: [i32; 3], chunks: Box<[Option<Chunk>; 64]> },
    /// Merge 2×2×2 super-chunk SVDAGs into one LOD super-chunk at half resolution.
    SuperChunkLodMerge {
        pos: [i32; 3],
        children: Box<[Vec<u8>; 8]>,
        lod_level: u32,
    },
    /// Generate terrain directly at LOD resolution (no raw chunks needed).
    LodGenerate {
        pos: [i32; 3],
        origin: [i32; 3],
        voxel_size: u32,
        lod_level: u32,
        seed: u32,
    },
}

/// Result of SVDAG compression.
pub struct CompressionResult {
    pub pos: [i32; 3],
    pub dag_data: Vec<u8>,
    pub lod_level: u32,
    /// True if all source chunks were present (safe to cache to disk).
    pub complete: bool,
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
                            CompressionResult {
                                pos,
                                dag_data,
                                lod_level: 0,
                                complete: true,
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
                            CompressionResult {
                                pos,
                                dag_data,
                                lod_level,
                                complete: true,
                            }
                        }
                        CompressionRequest::SuperChunk { pos, chunks } => {
                            let complete = chunks.iter().all(|c| c.is_some());
                            let grid = SuperChunkGrid { chunks };
                            let dag_data = svdag_from_super_chunk(&grid);
                            CompressionResult {
                                pos,
                                dag_data,
                                lod_level: 2,
                                complete,
                            }
                        }
                        CompressionRequest::SuperChunkLodMerge { pos, children, lod_level } => {
                            let refs: [&[u8]; 8] = [
                                &children[0],
                                &children[1],
                                &children[2],
                                &children[3],
                                &children[4],
                                &children[5],
                                &children[6],
                                &children[7],
                            ];
                            let dag_data = svdag_lod_merge_super(refs);
                            CompressionResult {
                                pos,
                                dag_data,
                                lod_level,
                                complete: true,
                            }
                        }
                        CompressionRequest::LodGenerate {
                            pos,
                            origin,
                            voxel_size,
                            lod_level,
                            seed,
                        } => {
                            let grid = terrain::generate_lod_super_chunk(origin, voxel_size, seed);
                            let dag_data = svdag_compress(&grid, 64);
                            CompressionResult {
                                pos,
                                dag_data,
                                lod_level,
                                complete: true,
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

    /// Queue a LOD merge of 2×2×2 super-chunk SVDAGs.
    pub fn request_super_chunk_lod_merge(&self, pos: [i32; 3], children: Box<[Vec<u8>; 8]>, lod_level: u32) {
        let _ = self.request_tx.send(CompressionRequest::SuperChunkLodMerge { pos, children, lod_level });
    }

    /// Generate terrain directly at LOD resolution on a background thread.
    pub fn request_lod_generate(&self, pos: [i32; 3], origin: [i32; 3], voxel_size: u32, lod_level: u32, seed: u32) {
        let _ = self.request_tx.send(CompressionRequest::LodGenerate {
            pos,
            origin,
            voxel_size,
            lod_level,
            seed,
        });
    }

    /// Queue a 4×4×4 super-chunk for compression into a single 64³ SVDAG.
    pub fn request_super_chunk(&self, pos: [i32; 3], chunks: Box<[Option<Chunk>; 64]>) {
        let _ = self.request_tx.send(CompressionRequest::SuperChunk { pos, chunks });
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
