# LOD Session Report

## What Was Built

### Title Screen & World Management
- **Game state machine:** TitleScreen, WorldSelect, CreateWorld, PreGenerating, EnteringWorld, Playing
- **World metadata:** `worlds/{name}/world.toml` with name, seed, timestamp
- **Per-world storage:** each world gets its own `svdag/`, `svdag_lod1/`, etc.
- **Pre-generation:** waits for raw terrain + LOD to settle before entering world
- **Existing world re-entry:** skips pre-gen, streams progressively
- **UI rendering:** bitmap font atlas (procedural 5x7 glyphs), quad batching pipeline

### LOD Hierarchy
- **Direct noise sampling:** `generate_lod_super_chunk(origin, voxel_size, seed)` samples terrain noise at LOD resolution. No dependency on lower LOD levels.
- **4 LOD bands** beyond LOD-0 (full-res 64m super-chunks):
  - LOD-1: 128m coverage, 2m voxels, distance 24-48 chunks
  - LOD-2: 256m coverage, 4m voxels, distance 48-96 chunks
  - LOD-3: 512m coverage, 8m voxels, distance 96-192 chunks
  - LOD-4: 1024m coverage, 16m voxels, distance 192-384 chunks
- **Shader:** DDA scales by `voxel_size = aabb_extent / dag_size`, handles variable voxel sizes
- **Background generation:** 4 worker threads, MAX_LOD_IN_FLIGHT=32, 8 submissions/frame
- **Disk cache:** per-LOD-level RegionStores (`svdag/`, `svdag_lod1/`, etc.)
- **Far plane:** increased from 500m to 10,000m

### Infrastructure Changes
- **Parameterized seed:** threaded through terrain → chunk_generator → world
- **Split VulkanApplication:** core Vulkan (always present) vs WorldResources (created on world entry)
- **Depth pyramid descriptor rebinding** after swapchain recreation

## Current Constants

| Constant | Value | Location |
|----------|-------|----------|
| MESH_DISTANCE | 8 chunks (128m) | vulkan_object.rs |
| LOD0_DISTANCE | 24 chunks (384m) | vulkan_object.rs |
| LOD1_DISTANCE | 48 chunks (768m) | vulkan_object.rs |
| LOD2_DISTANCE | 96 chunks (1536m) | vulkan_object.rs |
| LOD3_DISTANCE | 192 chunks (3072m) | vulkan_object.rs |
| LOD4_DISTANCE | 384 chunks (6144m) | vulkan_object.rs |
| WORLD_DISTANCE | 24 (=LOD0_DISTANCE) | vulkan_object.rs |
| SVDAG_DISTANCE | 384 (=LOD4_DISTANCE) | vulkan_object.rs |
| MAX_SVDAG_CHUNKS | 32,768 | vulkan_object.rs |
| SVDAG_GEOMETRY_BUDGET | 128 MB | svdag_pool.rs |
| MAX_CHUNKS_PER_TILE | 64 | svdag_pipeline.rs + both shaders |
| SUPER_CHUNK_VOXELS | 64 | vulkan_object.rs |
| FAR_PLANE | 10,000m | camera.rs |
| WORKER_COUNT | 4 | svdag_compressor.rs |

## Known Bugs

### Bug 1: Black/Gray Walls at LOD Boundaries
**Symptom:** Visible stone walls forming concentric squares at each LOD band boundary (384m, 768m, 1536m, 3072m). The walls are the exposed cross-section of underground terrain.

**Root cause:** At LOD band boundaries, adjacent chunks are at different resolutions. The terrain surface height differs slightly between LOD levels (due to noise sampling at different intervals). This height mismatch exposes the underground stone of the closer chunk as a visible wall.

**What was tried:**
1. **Shader edge-skip (direction-based):** Skip hits on edge voxels when ray enters from side. FAILED — created directional artifacts (missing strips facing camera direction).
2. **Shader edge-skip (underground check):** Skip edge voxels where voxel-above is solid. FAILED — made walls worse, extra SVDAG lookups per hit.
3. **Pool key coexistence `([i32;3], u32)`:** Allow LOD-0 and LOD-1 at same position to coexist. REVERTED — didn't fix walls, worsened tile overflow.
4. **EdgeStrippedSource:** Strip underground from LOD-0 edge columns during SVDAG compression. CURRENTLY ACTIVE — strips edge columns where 4 consecutive voxels above are solid. Effect unclear.
5. **Surface stripping in generate_lod_super_chunk:** Strip underground from all LOD-1+ columns, keeping only top 4 voxels. CURRENTLY ACTIVE — removes deep underground from LOD chunks.

**What hasn't been tried:**
- Generating LOD chunks with 1-voxel overlap beyond AABB (Aokana approach)
- Not generating LOD-0 super-chunks at positions that overlap with LOD-1 groups
- Fading/blending at LOD transitions

### Bug 2: Missing Chunks at LOD Boundaries
**Symptom:** Horizontal strips of missing terrain at LOD band boundaries. Missing strips are always the width of the LOD square. They face the same direction (toward the player's initial view). They flicker subtly.

**Root cause (probable):** Tile overflow. With multiple LOD levels, center-screen tiles have many overlapping chunks. The distance-sorted overflow (MAX_CHUNKS_PER_TILE=64) drops far chunks. May also involve the pool position key collision — LOD-0 at position [24,0,0] blocks LOD-1 at the same position.

**What was tried:**
1. Increased MAX_CHUNKS_PER_TILE from 32 to 64. Effect untested with clean data.
2. Pool key coexistence (reverted).

**Observation:** Missing chunks load permanently when player gets within ~1 LOD band + mesh distance. This suggests the chunks DO exist but aren't rendered at distance (tile overflow or frustum issue).

### Bug 3: Player Falls Through Blocks Outside Initial Spawn
**Symptom:** Only the mesh shader region around the initial spawn (0,0) has solid terrain. Walking/flying outside that area, the player falls through all blocks.

**Root cause (probable):** When the player moves, `world.update(player_cx, player_cz)` requests terrain generation at the new position. But chunk generation is async — the player reaches the new position before chunks are ready. With SPRINT_MULTIPLIER=100 (300 m/s), the player outruns generation entirely.

**What was tried:** Nothing yet.

**Contributing factors:**
- SPRINT_MULTIPLIER was increased to 100 (was 10) for traversing 5km+ worlds
- No chunk-loading barrier on player movement
- No ground detection failsafe (player just falls if no chunks exist)

### Bug 4: LOD-0/LOD-1 Position Key Collision
**Symptom:** LOD-0 super-chunk at position [24,0,0] (AABB 384-448m) and LOD-1 at [24,0,0] (AABB 384-512m) share the same pool key. LOD-0 arrives first, `has_chunk` returns true, LOD-1 is never uploaded. The area 448-512m has no coverage.

**Root cause:** `SvdagPool` keys chunks by `[i32; 3]` position only, not by LOD level. Positions that are valid for both LOD-0 (align=4) and LOD-1 (align=8) collide.

**What was tried:**
1. Pool key coexistence `([i32;3], u32)` — worked mechanically but reverted because it worsened tile overflow without fixing the visual seams.

**Status:** UNRESOLVED. This is likely a contributing factor to Bug 2 (missing chunks at boundaries).

## Architecture Diagram

```
                    Distance from player (chunks)
    0       8        24        48       96      192      384
    |-------|---------|---------|--------|--------|--------|
    | Mesh  |  LOD-0  | LOD-1   | LOD-2  | LOD-3  | LOD-4 |
    |shader |  64m    | 128m    | 256m   | 512m   | 1024m  |
    |VoxelP.| full-res| 2m vox  | 4m vox | 8m vox | 16m vox|
    |-------|---------|---------|--------|--------|--------|
    
    Raw chunks exist only within WORLD_DISTANCE (24 chunks = 384m).
    LOD-1+ generated directly from noise (no raw chunks needed).
    All LOD levels are 64x64x64 SVDAGs with variable voxel size.
```

## File Inventory (modified this session)

| File | Lines | Changes |
|------|-------|---------|
| src/main.rs | ~500 | Game state machine, UI rendering, pre-gen |
| src/game_state.rs | ~30 | GameState enum |
| src/graphical_core/vulkan_object.rs | ~850 | WorldResources, enter/exit_world, LOD scheduling |
| src/graphical_core/svdag_pool.rs | ~340 | dag_size parameter, budget increase |
| src/graphical_core/svdag_pipeline.rs | ~400 | Tile budget increase |
| src/graphical_core/ui_pipeline.rs | ~430 | NEW: bitmap font rendering pipeline |
| src/graphical_core/camera.rs | ~170 | Far plane increase |
| src/graphical_core/commands.rs | ~400 | draw_sky made public |
| src/graphical_core/mesh_pipeline.rs | ~300 | update_depth_pyramid method |
| src/voxel/terrain.rs | ~280 | generate_lod_super_chunk, surface stripping |
| src/voxel/svdag.rs | ~640 | EdgeStrippedSource, LOD merge functions |
| src/voxel/svdag_compressor.rs | ~180 | LodGenerate request, more workers |
| src/voxel/world.rs | ~200 | Seed parameter |
| src/storage/world_meta.rs | ~120 | NEW: world metadata, per-LOD cache dirs |
| src/shaders/svdag_raymarch.comp | ~300 | Variable voxel size DDA |
| src/shaders/svdag_tile_assign.comp | ~180 | Tile budget increase |
| src/shaders/ui.vert | ~20 | NEW: UI vertex shader |
| src/shaders/ui.frag | ~20 | NEW: UI fragment shader |
