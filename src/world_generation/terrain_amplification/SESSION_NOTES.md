# Terrain Amplification — Session Notes

## What's Built

The full terrain-diffusion ONNX pipeline is implemented and runs end-to-end in ~10 seconds on GPU:

```
src/world_generation/terrain_amplification/
  mod.rs              — orchestrator: amplify() → 6 cube-face heightmaps
  rasterize.rs        — coarse heightmap (Fibonacci sphere) → 6 cube-face grids
  session.rs          — ONNX model loading, CUDA/CPU fallback, auto PATH setup
  scheduler.rs        — EDM Karras sigmas, DPM-Solver++, trigonometric flow-matching
  tiling.rs           — BlendGrid, linear weight window, tile iteration
  coarse_stage.rs     — stage 1: 20-step DPM-Solver++ (64×64 tiles, stride 48)
  latent_stage.rs     — stage 2: 2-step flow-matching (64×64 tiles, stride 32)
  decoder_stage.rs    — stage 3: 1-step flow-matching (512×512 tiles, stride 384)
```

### Infrastructure working:
- ONNX Runtime loads via `data/onnxruntime/` (CUDA + cuDNN DLLs bundled)
- CUDA GPU acceleration confirmed (~10s vs ~3min CPU)
- Models load once per stage, run all 6 faces, then drop (3 loads total)
- Logging to `terrain_amplification.log` with per-stage timing
- HTML globe export for visual debugging (256×256 per face)

### ONNX models location:
- `data/models/terrain-diffusion-30m/coarse_model.onnx` (22 MB)
- `data/models/terrain-diffusion-30m/base_model.onnx` (2 GB)
- `data/models/terrain-diffusion-30m/decoder_model.onnx` (224 MB)

### GPU setup:
- `data/onnxruntime/` contains onnxruntime.dll (CUDA build) + CUDA runtime + cuDNN DLLs
- `session.rs` auto-adds this folder to PATH at startup via `ensure_ort_path()`
- `.cargo/config.toml` sets `ORT_DYLIB_PATH` for `load-dynamic` feature

## What's Broken — Fix These First

### 1. Cube face seam blending (BROKEN)
The `CUBE_EDGES` adjacency table in `mod.rs` has wrong edge mappings. The table
was guessed instead of derived from `voxel::sphere::face_basis()`. Need to:
- Read `face_basis()` for all 6 faces to get (tu, tv, normal) per face
- Programmatically determine which border of face A maps to which border of face B
- Check if the border traversal is reversed
- The `sphere_export.py` file in `src/tectonic_simulation/js/` has a correct
  `_CUBE_EDGES` table (lines 85-98) that can be used as reference
- Also consider: blend-weighted sampling approach from sphere_export.py instead
  of post-hoc averaging

### 2. Conditioning not working (NO LAND)
The diffusion model ignores our coarse heightmap conditioning — output is ~90% ocean
with no resemblance to the input continents. Likely causes:
- **COND_SNR values wrong**: We use `[0.3, 0.1, 1.0, 0.1, 1.0]` from Python defaults,
  but these control how strongly the model follows conditioning. May need higher values.
- **Conditioning normalization**: The coarse heightmap is in km, the model expects
  specific normalized ranges. The sqrt encoding (`sign(x) * sqrt(|x|)`) may not
  match what the model was trained on.
- **Coarse resolution too small**: Our 16×16 conditioning padded to 64×64 may be
  too sparse. The model was trained on real DEM data at specific resolutions.
- Compare against the Java `WorldPipeline.coarseTile()` (lines 130-195) and the
  Python `SyntheticMapFactory` to see exactly how conditioning is prepared.

### 3. Elevation range skewed
Output: -9600m to +1335m (heavily ocean-biased). Earth-like would be -11000m to +8800m.
The low-frequency + residual combination may need the Laplacian denoise step that
the Java uses (`LaplacianUtils.laplacianDenoise` + `laplacianDecode`) which we skipped.

## Bugs Fixed This Session

1. ~~Flow-matching used Euler step instead of trigonometric formula~~ → Fixed in scheduler.rs
2. ~~Coarse output missing p5 transformation (ch1 = ch0 - ch1)~~ → Fixed
3. ~~Latent conditioning mask value was raw 1.0 instead of normalized 0.723~~ → Fixed
4. ~~Latent reinitialization multiplied σ_data on sample term~~ → Fixed
5. ~~Decoder upsample mixed tile offset with pixel index~~ → Fixed
6. ~~sqrt reversal used sign(x)*x*|x| instead of sign(x)*x²~~ → Fixed

## Audit Notes

A full audit against the Java reference found 10 bugs total. 6 were fixed.
Remaining concerns:
- Seed offsets use XOR instead of Java's additive offsets (cosmetic, affects reproducibility)
- Decoder tile size is 512 (our ONNX model) vs Java's 256 (different model config)
- Missing Laplacian denoise/decode in final elevation computation
- Weight channel handling differs (our BlendGrid vs Java's explicit 7th channel)

## Key Reference Files

- Java pipeline: `o:/Programming/Rust/terrain-diffusion-mc/src/main/java/.../WorldPipeline.java`
- Java scheduler: `o:/Programming/Rust/terrain-diffusion-mc/src/main/java/.../EDMScheduler.java`
- Python pipeline: `o:/Programming/Rust/terrain-diffusion/terrain_diffusion/inference/world_pipeline.py`
- Python sphere export: `o:/Programming/Rust/manifold/src/tectonic_simulation/js/sphere_export.py`
- Engine projection: `o:/Programming/Rust/manifold/src/voxel/sphere.rs` (face_basis, cube_to_sphere_unit)

## How to Run

```bash
# Run tests (no GPU needed for unit tests)
cargo test --release -- world_generation

# Run amplification smoke test (needs ONNX models + GPU)
cargo test --release -- amplify_smoke --ignored --nocapture

# Generate visual globe (HTML)
cargo test --release -- amplify_globe_export --ignored --nocapture
# → amplified_globe.html in terrain_amplification/

# Check log
cat terrain_amplification.log
```

## Prompt for Next Session

```
Context: We're building a terrain-diffusion pipeline in Rust for procedural planet
generation. The ONNX inference pipeline runs end-to-end (10s on GPU) but has two
critical issues to fix:

1. CUBE FACE SEAM BLENDING: The CUBE_EDGES adjacency table in
   src/world_generation/terrain_amplification/mod.rs is wrong — seams between
   cube faces are visible. Need to derive correct edge mappings from
   voxel::sphere::face_basis(). Reference: sphere_export.py in
   src/tectonic_simulation/js/ has the correct table.

2. CONDITIONING NOT WORKING: The diffusion model ignores our coarse heightmap —
   output is ~90% ocean with no land resembling the input. The conditioning
   normalization, SNR values, or preparation is wrong. Compare against
   WorldPipeline.java coarseTile() method and SyntheticMapFactory in the Python
   terrain-diffusion repo at o:/Programming/Rust/terrain-diffusion/.

Read src/world_generation/terrain_amplification/SESSION_NOTES.md for full context.
The plan doc is at docs/ideas/world_generation_pipeline_most_recent_plan.md.
```
