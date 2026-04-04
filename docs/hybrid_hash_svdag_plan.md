# Hybrid Hash + SVDAG Rendering Architecture

## Overview

Near-field (0-100m / ~6 chunks): Hash-based spatial partitioning with the current chunk system, fully editable, mesh shader rendered.

Far-field (100m+): Sparse Voxel DAGs based on the Aokana framework (May 2025, ACM I3D), GPU ray marched, multi-level LOD, streamed.

The existing mesh shader pipeline is never removed — only supplemented. Every phase is independently committable and testable.

---

## Phase 0: Preparatory Refactoring

No visual changes. Separate near-field and far-field chunk sets in the world management layer.

### Step 0.1 — Distance tiers in World

Modify `World::update()` to classify chunks into two tiers based on Chebyshev distance from the player:
- Near-field: distance <= `NEAR_RADIUS` (6 chunks, ~96 blocks)
- Far-field: distance > `NEAR_RADIUS` && <= `render_distance`

Extend `WorldDelta` to return `promoted` (far→near) and `demoted` (near→far) vectors in addition to `loaded`/`unloaded`.

**Test:** Unit test that `WorldDelta` correctly classifies chunks. Visually nothing changes.

### Step 0.2 — Tier-aware VoxelPool

Add a `ChunkTier` field to the internal slot tracking in `VoxelPool`. The `chunk_slots` map becomes `HashMap<[i32;3], (u32, ChunkTier)>`. Bookkeeping only — both tiers still render via mesh shaders.

**Test:** Existing rendering works identically. `VoxelPool::chunk_count()` can report near vs far counts separately.

---

## Phase 1: SVDAG Data Structure (CPU-Only)

Pure Rust, no GPU involvement. Exhaustively testable.

### Step 1.1 — SVDAG node format and compression

Create `src/voxel/svdag.rs`. For 16x16x16 chunks, the DAG is 3 levels deep:

```
Level 0 (root):  16x16x16 → 8 children (8x8x8 octants)
Level 1 (mid):   8x8x8   → 8 children (4x4x4 octants)
Level 2 (leaf):  4x4x4   → 64-bit occupancy bitmap
```

Node format:
```rust
struct SvdagNode {
    header: u32,      // low 8 bits = child mask, upper bits = metadata
    leaf_count: u32,  // total leaves in subtree (for DFS material indexing)
    // Followed by 0-8 child pointers (u32 offsets into node pool)
}
```

Implement:
- `svdag_from_chunk(chunk: &Chunk) -> Vec<u8>` — compress dense to DAG with subtree deduplication via `HashMap<&[u8], u32>`
- `svdag_lookup(dag: &[u8], x: u8, y: u8, z: u8) -> BlockType` — random access for validation

**Test:** Round-trip every voxel in a chunk. Compression ratio tests on typical terrain (expect 5-20x for terrain with large air/solid regions).

### Step 1.2 — LOD aggregation

Implement `svdag_lod_merge(children: [&[u8]; 8]) -> Vec<u8>` that takes 8 LOD-0 SVDAGs and produces one LOD-1 at half resolution. Rule: parent voxel is solid if majority (>=4 of 8) child voxels are solid. Material by majority vote.

**Test:** Create 8 known chunks, merge to LOD-1, verify dimensions and contents.

### Step 1.3 — Material encoding

DFS-ordered flat material array alongside the SVDAG. During ray-march hit, the voxel's DFS position indexes into this array. `BlockType` is `repr(u8)` with 8 variants — one byte per material entry. Compressed via run-length encoding.

**Test:** Verify DFS ordering matches traversal order. Verify RLE compression on typical terrain.

---

## Phase 2: GPU Buffer and Background Compression

### Step 2.1 — SvdagPool

Create `src/graphical_core/svdag_pool.rs` mirroring VoxelPool's slot-based architecture:

- **SVDAG geometry SSBO:** Variable-size per-chunk DAG data. Bump allocator with free list. ~64MB budget.
- **SVDAG material SSBO:** RLE-compressed material arrays, co-allocated with geometry.
- **SVDAG chunk info SSBO:** Per-chunk metadata for the ray marcher:

```rust
#[repr(C)]
struct GpuSvdagChunkInfo {
    aabb_min: [f32; 3],
    dag_offset: u32,       // byte offset into geometry SSBO
    aabb_max: [f32; 3],
    material_offset: u32,  // byte offset into material SSBO
    dag_size: u32,
    lod_level: u32,
    _pad: [u32; 2],
}
```

All buffers use host-visible coherent memory (matching VoxelPool).

**Test:** Upload test SVDAGs, read back via CPU pointer, verify data integrity. No rendering.

### Step 2.2 — Background compression thread

Extend the `ChunkGenerator` threading model with a separate SVDAG compression thread pool. When a chunk is demoted (near→far):

1. Dense chunk data sent to compression worker via `crossbeam_channel`
2. Worker calls `svdag_from_chunk()`
3. Compressed result returned to main thread
4. Main thread uploads to `SvdagPool` and removes from `VoxelPool`

The chunk stays visible in mesh shader pipeline until SVDAG is ready. Track state with:
```rust
enum DemotionState {
    Dense,
    Compressing,
    Compressed(SvdagHandle),
}
```

**Test:** Trigger demotion by moving the player. Chunks compress in the background without frame drops.

---

## Phase 3: Ray March Compute Pipeline

The big one. Build the GPU ray marching pipeline for far-field SVDAG chunks.

### Step 3.1 — Ray march compute shader

Create `src/shaders/svdag_raymarch.comp`:

1. Each thread handles one pixel (dispatched as 8x8 tile workgroups)
2. Reconstruct world-space ray from inverse VP matrix (already in `CameraUBO`)
3. Iterate over assigned SVDAG chunks
4. Ray-AABB intersection to enter the chunk
5. Traverse 3-level SVDAG: compute child octant, check child mask, descend
6. At leaf level: DDA through 4x4x4 bitmap
7. On hit: write depth + material to output

Bindings:
- SVDAG geometry SSBO (readonly)
- SVDAG material SSBO (readonly)
- SVDAG chunk info SSBO (readonly)
- CameraUBO (shared with mesh pipeline, binding 1)
- Depth buffer (for early-out against near-field geometry)

### Step 3.2 — SvdagPipeline struct

Create `src/graphical_core/svdag_pipeline.rs` following `mesh_pipeline.rs` patterns:
- Descriptor set layout with SVDAG-specific bindings
- Compute pipeline with the ray march shader
- Push constants for camera, chunk count, screen dimensions

### Step 3.3 — Integrate into command buffer recording

Modify `record_mesh_shader_command_buffer` to add the SVDAG pass:

```
1. Phase 1 mesh shader (near-field, previously visible)
2. Build depth pyramid
3. Phase 2 mesh shader (near-field, previously invisible)
4. Pipeline barrier: depth ATTACHMENT → SHADER_READ
5. SVDAG ray march compute dispatch (far-field)
```

The ray march reads the existing depth buffer for early-out and writes new depth for far-field pixels. Both pipelines write to the same color attachment.

**Test:** Move far enough that some chunks become far-field. They render via ray marching. Compare visually with mesh shader rendering of the same chunks.

---

## Phase 4: Boundary Transition

Eliminate visual popping at the near/far boundary.

### Step 4.1 — Overlap zone

Define a 2-chunk-wide transition band around `NEAR_RADIUS`. Chunks in this band exist in BOTH systems:
- Dense data in `VoxelPool` (mesh shader)
- Compressed SVDAG in `SvdagPool` (ray march)

The switch is instantaneous — no compression delay at the boundary.

### Step 4.2 — Depth-based compositing

Both renderers write the same depth buffer. Near-field geometry (closer) naturally wins via depth test. Identical voxel data = identical silhouettes = seamless transition.

### Step 4.3 — Alpha fade (if needed)

If LOD differences cause visible pop (LOD-0 mesh vs LOD-1 SVDAG), add distance-based alpha fade. Chunks within 1 chunk of the boundary render in both systems with cross-fade. Only implement if the depth-based compositing produces artifacts.

**Test:** Fly back and forth across the boundary rapidly. No popping, no gaps, no z-fighting.

---

## Phase 5: LOD Hierarchy and Streaming

### Step 5.1 — LOD level selection

Select LOD level by distance from player:

| Distance (chunks) | LOD | Resolution |
|--------------------|-----|------------|
| 0-6                | Mesh shaders | Full 16³ |
| 6-12               | LOD-0 SVDAG | Full 16³ |
| 12-24              | LOD-1 | 8³ (8 chunks merged) |
| 24-48              | LOD-2 | 4³ equivalent |
| 48+                | LOD-3 | 2³ equivalent |

LOD levels are pre-computed on CPU via `svdag_lod_merge()`.

### Step 5.2 — VRAM budget management

Track total SVDAG VRAM usage. When approaching the 64MB budget:
1. Evict least-recently-viewed LOD-0 chunks first
2. Keep their LOD-1+ parents (which cover the same area at lower resolution)
3. Target: ~5% of total scene data in VRAM at any time

### Step 5.3 — Async streaming

When new far-field chunks become visible:
1. Check `SvdagPool` for existing SVDAG
2. Check CPU-side LRU cache
3. Queue for generation (terrain gen → SVDAG compression)
4. Meanwhile render at parent LOD level (coarser but immediate)

**Test:** Render distance 48+ chunks. VRAM stays under budget. Distant terrain pops in at coarse LOD first, then refines.

---

## Phase 6: Edit Propagation

### Step 6.1 — Dirty chunk tracking

When `VulkanApplication::set_block()` modifies a chunk, mark the corresponding SVDAG chunk and its LOD ancestors as dirty.

### Step 6.2 — Lazy SVDAG rebuild

Dirty SVDAGs are recompressed on background threads. The stale SVDAG stays visible until the rebuild completes. Single-block edits are invisible at far-field distance — this latency is acceptable.

For large edits (explosions), trigger immediate recompression of affected LOD-1+ chunks.

**Test:** Place/break blocks near the far-field boundary. Far-field representation updates within a few frames.

---

## Key Architectural Decisions

1. **Per-chunk shallow SVDAGs (3 levels deep)** rather than one global deep SVDAG. Matches Aokana and the existing chunk-centric architecture. Each SVDAG is self-contained and independently streamable.

2. **Compute ray march for far-field**, mesh shaders for near-field. Ray marching avoids generating sub-pixel triangles. Cost is bounded by screen resolution, not scene complexity.

3. **Shared depth buffer** between both pipelines. Mesh shader writes depth during rasterization; ray marcher reads for early-out and writes for far-field pixels. No compositing pass needed.

4. **Host-visible SSBOs** for SVDAG data, matching VoxelPool's existing mapped-memory pattern. Device-local + staging can be optimized later.

5. **Background compression on CPU threads**, matching ChunkGenerator's existing crossbeam channel worker model.

---

## File Map

| File | Purpose |
|------|---------|
| `src/voxel/svdag.rs` | SVDAG node format, compression, lookup, LOD merge |
| `src/graphical_core/svdag_pool.rs` | GPU buffer management for SVDAG data |
| `src/graphical_core/svdag_pipeline.rs` | Compute pipeline for ray marching |
| `src/shaders/svdag_raymarch.comp` | Ray march compute shader |
| `src/voxel/world.rs` | Distance tier classification, demotion/promotion |
| `src/graphical_core/voxel_pool.rs` | Tier-aware slot management |
| `src/graphical_core/commands.rs` | Command buffer integration |
| `src/graphical_core/vulkan_object.rs` | SvdagPool + SvdagPipeline ownership, render loop |
