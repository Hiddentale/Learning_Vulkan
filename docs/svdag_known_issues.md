# SVDAG Known Issues and Required Fixes

Based on the Aokana framework (ACM SIGGRAPH 2025) and "Beyond Fixed Chunks" research survey.

## Current Problems

### 1. Material storage is uncompressed (causes budget crash)
We store a flat 4KB material array per SVDAG chunk — the same size as the uncompressed dense chunk. This defeats the purpose of SVDAG compression. With render_distance=24 and ~33,000 far chunks, that's 132MB of material data alone.

**Paper approach:** "Color data remains decoupled from geometry using depth-first search ordering for efficient indexing, achieving 4-6x compression through block-based encoding with tolerance thresholds."

**Fix:** Embed materials directly in SVDAG leaf nodes. Instead of a 64-bit occupancy bitmap (8 bytes per 4x4x4 leaf), store 64 bytes — one u8 per voxel where 0=air and nonzero=material ID. This eliminates the material SSBO entirely. All-air octants are still pruned by the SVDAG tree, so sparse chunks stay small.

### 2. No streaming budget (loads everything into VRAM)
We try to SVDAG-compress and upload ALL far chunks simultaneously. The paper maintains only 2-17% of total scene data in VRAM at any time, streaming on-demand based on visibility and distance.

**Fix (future):** Implement visibility-driven streaming. Only compress and upload chunks that are in the view frustum and within a VRAM budget. Evict least-recently-viewed chunks when approaching the budget.

### 3. No frustum culling on SVDAG chunks
The ray march fragment shader iterates over ALL SVDAG chunks for every pixel. No frustum or Hi-Z culling. Aokana does "chunk selection (frustum culling), tile selection with Hi-Z culling" in compute passes before ray marching.

**Fix:** Add a simple frustum test per SVDAG chunk in the fragment shader — skip chunks whose AABB is entirely behind the camera. This is a quick win before implementing proper compute-based culling.

### 4. Chunk size mismatch with paper
Aokana uses 64x64x64 chunks. We use 16x16x16. This means 64x more chunks to manage for the same volume — more SVDAG overhead, more chunk info entries, more per-chunk ray-AABB tests.

**Fix (future):** Consider grouping 4x4x4 blocks of 16-chunks into 64-chunk super-chunks for the SVDAG representation, matching the paper's granularity.
