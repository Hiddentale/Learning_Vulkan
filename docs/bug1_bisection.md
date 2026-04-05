# Bug 1 Bisection: Gray Gaps at LOD Boundaries

## Symptom
- Gray (sky-colored) strips at every LOD boundary (LOD-0/1, LOD-1/2, LOD-2/3, LOD-3/4)
- Exactly 1 chunk wide
- Always on the **inner** (closer to player) LOD side
- Directional: face the same direction (toward initial view)
- Visible as concentric squares from above

## Bisection Tests

| # | Test | Change | Result | Ruled Out |
|---|------|--------|--------|-----------|
| 1 | Surface stripping on LOD-0 super-chunks | Replace EdgeStrippedSource with SurfaceStrippedSource | No change | Underground cross-section visibility |
| 2 | LOD debug coloring in ray march shader | Tint pixels by AABB extent (red/blue/green/yellow/magenta) | Gaps are sky color (gray), not terrain. Confirms no chunk renders there | — |
| 3 | Disable Hi-Z occlusion in tile assign | Comment out hiz_visible check | Gaps got BIGGER (half border missing). Likely coincidental (different angle/load state) | Inconclusive |
| 4 | Increase MAX_CHUNKS_PER_TILE to 256 | Changed in both shaders + svdag_pipeline.rs | No change | Tile overflow |
| 5 | Disable frustum culling | Make all chunks pass svdag_cull.comp | No change (after regen) | Frustum culling |
| 6 | Composite pool key (pos, lod_level) | Allow LOD-0 and LOD-1 at same position | No change | Pool key collision |
| 7 | Red=no tile chunks, Cyan=chunks but no hit | Debug color in ray march | Gaps are cyan (chunks assigned, ray misses) | Tile assignment (chunks ARE there) |
| 8 | Only LOD-0 + LOD-1 | Disabled LOD-2/3/4 | Gap persists. LOD-0 not centered on player. | Multi-LOD interaction |
| 9 | Camera rotation test | Rotate camera, observe gap | Gap stays on same world side | View-dependent rendering |

| 10 | Log failed super-chunk groups | Log groups where all_present fails | Groups at x=24 and z=-24 permanently fail (children at distance 25+ never generated) | **ROOT CAUSE 1: WORLD_DISTANCE too small** |
| 11 | WORLD_DISTANCE = LOD0_DISTANCE + 4 | Extend raw chunk generation by 4 | LOD-0 boundary gap eliminated | — |
| 12 | Composite pool key ([i32;3], u32) | Allow coexistence of LOD-N and LOD-(N+1) at same position | LOD-1+ boundary gaps eliminated | **ROOT CAUSE 2: Pool key collision** |

## Root Causes Found

### Root Cause 1: WORLD_DISTANCE == LOD0_DISTANCE
`WORLD_DISTANCE` was set to `LOD0_DISTANCE` (24 chunks). LOD-0 super-chunk groups at the boundary (e.g. [24,0,0]) need children at positions 24-27. Chunks at distance 25+ were never generated. The `all_present` check failed permanently, leaving a 1-chunk-wide gap on the sides where the alignment ran past the world boundary.

**Fix:** `WORLD_DISTANCE = LOD0_DISTANCE + 4`

### Root Cause 2: Pool key collision
`SvdagPool` keyed chunks by `[i32;3]` position only. At LOD boundaries, LOD-N and LOD-(N+1) share a position (e.g. [24,0,0]). LOD-N arrives first and claims the slot. LOD-(N+1) is blocked by `has_chunk()`. But LOD-N's AABB (span=2^N) is smaller than LOD-(N+1)'s (span=2^(N+1)), leaving the difference uncovered.

**Fix:** Composite key `([i32;3], u32)` including lod_level. LOD scheduler uses `has_chunk_at_lod()`.

| 13 | Composite lod_in_flight key | HashSet<([i32;3], u32)> so LOD levels don't block each other in scheduling | No change — gaps persist after 5+ minutes | lod_in_flight collision |
| 14 | Force DDA hit on first voxel | trace_chunk always returns true | Gaps filled — chunks ARE there, DDA enters them | DDA not reaching chunks |
| 15 | Log empty chunk drops | Log when dag_data.len() <= 68 | **359 drops** — all LOD-0 y=8,12 (above terrain). Infinite retry cycle. Added lod_empty tracking. | — (partial fix, thrashing stopped but gaps remain) |
| 16 | Increase DDA steps 192→512 | More iterations in trace_chunk | No change | DDA step exhaustion |
| 17 | Disable surface stripping in generate_lod_super_chunk | Remove SURFACE_DEPTH stripping for LOD-1+ | No change | Surface stripping emptying chunks |
| 18 | Increase DDA to 512 steps | trace_chunk loop limit 192→512 | No change | DDA step exhaustion |
| 19 | Log all uploads | Log pos/lod/dag_bytes for every uploaded chunk | All LOD levels present (342 LOD-0, 126 each LOD-1/2/3/4), 40-58KB data, zero drops | Chunks missing from pool |
| 20 | Disable Hi-Z (clean retest) | Comment out hiz_visible with all fixes active | No change | Hi-Z occlusion |

| 21 | Disable LOD-0 entirely | Clear svdag_pending each frame | Gaps persist between LOD-1/2, LOD-2/3 etc. | LOD-0 interference |
| 22 | Probe center column of gap chunks | svdag_lookup at (32,y,32) from top down | **GREEN** — solid terrain found in SVDAG data | SVDAG data empty on GPU |

| 23 | DDA entry diagnostics | Output entry_y, steps, exit reason as RGB | Orange: entry_y≈50/64 (top of chunk), ~200 steps, exited via t_exit (left AABB sideways) | **ROOT CAUSE: AABB too tall** |

| 24 | Tight Y AABB | Compute max_solid_y in terrain gen, cap aabb_max.y to terrain height | No change | AABB too tall / ray entering from side |
| 25 | Revert composite key to simple [i32;3] | Remove LOD coexistence at same position | Gaps stayed. Composite key is irrelevant | Composite key causing gaps |
| 26 | Disable t_exit check in DDA | Only exit via bounds check | Gaps stayed (purple). DDA walks out the side of the chunk | t_exit miscalculation |

| 27 | LOD band overlap (+align to max_dist) | Each inner LOD extends past its boundary into the outer LOD's range | **GAPS ELIMINATED** | — |

## Root Causes — All Found and Fixed

### Root Cause 1: WORLD_DISTANCE == LOD0_DISTANCE (LOD-0 gap)
Super-chunk groups at the boundary lacked child chunks beyond WORLD_DISTANCE.
**Fix:** `WORLD_DISTANCE = LOD0_DISTANCE + 8 + 4`

### Root Cause 2: Shallow-angle ray miss at LOD transitions (LOD-1+ gaps)
At LOD boundaries, the outer LOD's first chunks are viewed at a shallow angle. The DDA ray enters the AABB from the side above terrain and traverses horizontally, exiting without descending to terrain height. This is correct DDA behavior — the ray genuinely misses. The inner LOD renders the same terrain fine because it's closer (steeper angle).
**Fix:** Overlap LOD bands. Each band's max_dist extends by +align so inner LOD chunks cover the transition zone. Standard technique in LOD ray marching systems.

## LOD-1+ Gaps — Still Unresolved
Despite the DDA diagnostics showing entry at y≈50/64 with horizontal traversal, tightening the AABB Y to actual terrain height did NOT fix the gaps. The hypothesis that rays enter from the side above terrain was wrong — or the tight AABB doesn't change the ray entry enough.

### What we know for certain:
- Chunks exist in pool with valid 40-58KB SVDAG data
- svdag_lookup at center column finds solid terrain (GREEN probe test)
- DDA enters the chunk, takes ~200 steps, exits via t_exit
- DDA entry Y is near top of chunk (~50/64)
- Forced first-voxel hit fills the gaps completely
- Not: Hi-Z, frustum cull, tile overflow, surface stripping, pool key collision, step count, LOD-0 interference, AABB height

## Status
**LOD-0 boundary gap: FIXED** (WORLD_DISTANCE + 4)
**LOD-1+ boundary gaps: UNRESOLVED** — Chunks exist in pool with valid terrain data (40-58KB). DDA enters them. Forced-hit fills gaps. But normal DDA finds no solid voxels. Not Hi-Z, not step count, not surface stripping, not tile overflow, not LOD-0 interference. Something causes the DDA to miss terrain that is present in the SVDAG data.

## Key Insight
**The gap is world-direction-dependent, not view-dependent.** The entire rendering pipeline is ruled out. The problem is in chunk generation or scheduling.

## What We Know
- Chunks exist in the pool (pool dump confirmed positions and AABBs at boundaries)
- Gaps are rendering gaps, not generation gaps — the data is there but not reaching pixels
- The problem is view-dependent (directional), ruling out generation/upload issues
- Not tile overflow (256 didn't help)
- The pipeline path is: pool → frustum cull → visible list → tile assign (project + Hi-Z + distance sort) → ray march

## Ruled Out
- Underground cross-section visibility (surface stripping)
- Tile overflow (256 per tile didn't help)
- Frustum culling (disabled, no change)
- Pool key collision (composite key, no change)

## Remaining Suspects
- Chunks not generated at boundary positions (generation gap)
- AABB projection in tile assign producing wrong screen-space bounds
- Ray march DDA bug for specific chunk geometries
- Stale cache data (regen fixed 2-sided → 1-sided gap)

## Debug Infrastructure Added
- Shader LOD coloring: `svdag_raymarch.comp` tints by AABB extent
- Pool state dump: `SvdagPool::dump_state()` fires when LOD settles (lod_idle_frames == 3)
