# SVDAG Implementation Report

## Summary

This branch (`hbsp_and_svdags`) implements a hybrid rendering architecture: mesh shaders for near-field terrain (0-8 chunks) and Sparse Voxel DAG ray marching for far-field terrain (8-24 chunks). The SVDAG pipeline follows the Aokana framework's multi-pass compute approach. 2,306 lines added across 21 files.

## Architecture

### Two disjoint rendering systems

| System | Distance | Chunks | Method |
|--------|----------|--------|--------|
| Mesh shader | 0-8 (128 blocks) | ~4,600 in VoxelPool | Task/mesh shader rasterization, textured, editable |
| SVDAG ray march | 8-24 (128-384 blocks) | Up to 16,384 in SvdagPool | 3-pass compute pipeline, flat palette colors |

The systems never compete for the same chunks. Each frame, chunks within MESH_DISTANCE are guaranteed to be in VoxelPool. Chunks beyond MESH_DISTANCE are routed to the SVDAG pipeline. When the player moves, chunks seamlessly transfer between systems.

### SVDAG data structure

Each 16x16x16 chunk is compressed into a 3-level SVDAG:
- Level 0 (root): 16^3 -> 8 children of 8^3
- Level 1 (mid): 8^3 -> 8 children of 4^3
- Level 2 (leaf): 4^3 = 64 bytes, one u8 material ID per voxel

Materials are embedded directly in leaf nodes (64 bytes per leaf, where 0 = air). This eliminates a separate material SSBO entirely. All-air octants produce no leaf nodes, so sparse chunks stay small. A typical terrain chunk compresses to ~1-2KB.

The 4-byte header at offset 0 stores the root node's offset (children are written before parents during DFS construction). Subtree deduplication via `HashMap<Vec<u8>, u32>` merges identical subtrees.

### Aokana-style 3-pass compute pipeline

**Pass 1 — Chunk selection (`svdag_cull.comp`):**
One thread per SVDAG chunk. Tests AABB against 6 frustum planes. Surviving chunks written to compact `visible_indices` buffer via atomic counter. Reduces ~thousands of chunks to ~hundreds of visible ones.

**Pass 2 — Tile assignment (`svdag_tile_assign.comp`):**
Screen divided into 8x8 pixel tiles. One thread per tile. Projects each visible chunk's AABB to screen space, tests overlap with the tile. Writes per-tile chunk lists (max 32 per tile). This is the key performance structure — each pixel only ray marches 1-5 chunks instead of all visible chunks.

**Pass 3 — Ray march (`svdag_raymarch.comp`):**
One thread per pixel. Reads its tile's chunk list. For each chunk: ray-AABB intersection, then DDA voxel traversal through the 3-level SVDAG. On hit: reads material from leaf node, computes lighting. Writes color (rgba8) and depth (r32f) to storage images.

**Composite — Fragment pass (`svdag_composite.frag`):**
Fullscreen triangle samples the compute output. Alpha-blended over mesh shader output with depth testing (gl_FragDepth). Discards transparent pixels (no SVDAG hit).

### Streaming and budget

Chunks are not all compressed immediately. A `svdag_pending` set tracks far chunks awaiting compression. Each frame, the 16 closest pending chunks are sent to 2 background compression threads. Completed SVDAGs are uploaded to GPU when results arrive.

VRAM budget: 64MB geometry SSBO. When approaching 90% capacity, the farthest chunks from the player are evicted. Chunk info buffer supports up to 16,384 entries.

World terrain generation uses distance-ordered requests (spiral outward from player) so near-field mesh shader terrain appears immediately at startup.

## Design decisions and why

### Embedded materials in leaf nodes (not separate SSBO)
**Why:** A flat 4KB material array per chunk (our first approach) exceeded the 16MB material budget with ~33K far chunks. DFS-ordered materials (the paper's approach) require complex GPU-side index counting. Embedding materials in 64-byte leaves eliminates the material SSBO entirely — zero separate allocation, zero budget, the material IS the leaf.

### Fragment shader ray march -> Compute pipeline (Aokana-style)
**Why:** The initial fragment shader approach drew a fullscreen triangle and looped over ALL SVDAG chunks for every pixel — O(pixels x chunks). At 1080p with 500 chunks, that's 1 billion AABB tests per frame. The 3-pass compute approach reduces per-pixel work to ~3 chunks via tile-based culling.

### Clean mesh/SVDAG separation (not tier-based handoff)
**Why:** The original plan used tiers (Near/Transition/Far) with chunks moving between VoxelPool and SvdagPool. This caused: visual overlap (both systems rendering the same chunk), budget crashes (pool handoff timing), and complexity. The clean separation — mesh owns 0-8, SVDAG owns 8-24, no overlap — eliminated all handoff bugs.

### Distance-ordered terrain generation
**Why:** With WORLD_DISTANCE=24, the chunk generator receives ~9000 column requests. The original grid-scan loop started from corners (-24,-24), so close terrain took minutes to appear. Spiral outward from player center ensures mesh shader terrain loads first.

### Depth-tested compositing
**Why:** Without depth testing, SVDAG fragments paint over mesh shader geometry. Writing `gl_FragDepth` from the ray march hit position and enabling depth test on the composite pipeline naturally rejects SVDAG pixels behind near-field rasterized geometry.

## Problems encountered and how they were resolved

### 1. Swapchain image as compute storage (validation errors)
**Problem:** Compute shader can't write to a swapchain image — wrong layout (PRESENT_SRC vs GENERAL) and missing STORAGE usage flag.
**Resolution:** Created a dedicated RGBA8 storage image. Later replaced the entire approach with the Aokana compute pipeline + composite fragment pass, which eliminated the issue structurally.

### 2. Material budget exceeded (crash)
**Problem:** Storing 4KB flat material arrays per chunk * 33,000 chunks = 132MB, far exceeding any budget.
**Resolution:** Embedded materials in SVDAG leaf nodes. Zero separate material storage.

### 3. Terrain overlap (both systems rendering same chunks)
**Problem:** Tier-based pool handoff was non-atomic. Chunks appeared in both VoxelPool and SvdagPool simultaneously.
**Resolution:** Removed tiers entirely. Clean separation: mesh distance 0-8, SVDAG distance 8-24. No shared chunks.

### 4. Chunks not appearing at startup
**Problem:** With WORLD_DISTANCE=48, terrain generation requested 9,409 columns starting from far corners. Close terrain took minutes.
**Resolution:** Changed request loop to spiral outward from player position. Close chunks generate first.

### 5. VoxelPool overflow on player movement
**Problem:** World unloads at WORLD_DISTANCE but VoxelPool should only hold MESH_DISTANCE chunks. Moving chunks stayed in VoxelPool forever.
**Resolution:** Each frame, sweep VoxelPool and evict chunks beyond MESH_DISTANCE. Also ensure all chunks within MESH_DISTANCE are promoted to VoxelPool from SVDAG.

### 6. Fullscreen ray march performance (5 FPS)
**Problem:** Fragment shader looped over ALL SVDAG chunks per pixel.
**Resolution:** Aokana-style 3-pass compute: frustum cull, tile assignment, then per-tile ray march. Each pixel tests ~3 chunks instead of hundreds.

### 7. SVDAG eviction thrashing
**Problem:** Budget fills -> evict arbitrary chunks -> they get re-queued -> re-uploaded -> evict again.
**Resolution:** Evict farthest chunks from player (not arbitrary). Cap SVDAG streaming distance to match budget capacity.

## Open bugs

### SVDAG flickering at render edges
**Status:** Unresolved. Chunks at the SVDAG boundary flicker in and out. Persists even when standing still with no camera movement. Confirmed NOT a double-buffer race (setting MAX_FRAMES_IN_FLIGHT=1 doesn't fix it).

**Possible causes to investigate:**
- Tile assignment edge case: tiles at the boundary of visible chunk screen projections may non-deterministically include/exclude chunks
- Floating-point precision in AABB projection causing some frames to miss edge tiles
- The `visible_count` atomic counter or `cmd_fill_buffer` interaction with host-coherent memory
- The image clear (GENERAL layout) + compute write timing

### SVDAG colors don't match mesh shader
**Status:** Expected for now. The mesh shader uses texture sampling (sampler2DArray with stochastic tiling, edge detection). The SVDAG uses flat palette colors. Lighting formula matches, but textures make the near-field look much richer.

## Future work

### Textures on SVDAG (planned, user requested)
Bind the texture array to the ray march compute shader. Compute UVs from the voxel position and face normal (same as mesh shader). Would require adding the texture sampler to the march descriptor set and computing atlas coordinates in the shader.

### LOD hierarchy (infrastructure exists, not wired)
- `svdag_lod_merge()` merges 8 chunks into one at half resolution
- `lod_for_distance()` returns LOD level 0-3 by distance
- `SvdagCompressor::request_lod_merge()` queues LOD merge on background threads
- Wire up: distance 8-16 = LOD-0, 16-32 = LOD-1, 32-48 = LOD-2

### Persistence (save/load SVDAG data)
Currently all terrain is regenerated from scratch every launch. Saving compressed SVDAGs to disk would:
- Eliminate startup compression delay
- Allow much larger SVDAG render distance (stream from NVMe instead of CPU-generate)
- Match the paper's approach: "2-17% of scene data in VRAM, streaming on-demand"

### 64^3 super-chunks
Aokana uses 64^3 chunks. We use 16^3, meaning 64x more chunks for the same volume. Grouping 4x4x4 blocks of 16-chunks into 64-chunk super-SVDAGs would reduce chunk management overhead, improve ray march coherence, and better match the paper's architecture.

### Hi-Z occlusion in tile assignment
The tile assignment shader currently only does screen-space AABB overlap. Adding Hi-Z depth testing (reading the mesh shader's depth pyramid) would skip tiles where mesh geometry fully occludes the SVDAG chunk — the paper's "tile selection with Hi-Z culling."

## File inventory

| File | Purpose |
|------|---------|
| `src/voxel/svdag.rs` | SVDAG node format, compression, lookup, LOD merge (417 lines) |
| `src/voxel/svdag_compressor.rs` | Background compression thread pool (99 lines) |
| `src/graphical_core/svdag_pool.rs` | GPU buffer management for SVDAG data (182 lines) |
| `src/graphical_core/svdag_pipeline.rs` | 3-pass compute + composite pipeline (518 lines) |
| `src/shaders/svdag_cull.comp` | Pass 1: frustum cull |
| `src/shaders/svdag_tile_assign.comp` | Pass 2: tile-based chunk assignment |
| `src/shaders/svdag_raymarch.comp` | Pass 3: per-pixel ray march |
| `src/shaders/svdag_composite.frag` | Composite: alpha-blend over mesh output |
| `docs/hybrid_hash_svdag_plan.md` | Original 7-phase implementation plan |
| `docs/svdag_known_issues.md` | Issues discovered from paper analysis |
