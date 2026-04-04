# Project Roadmap

## Current State

Chunk-based voxel world with multi-octave terrain generation (Fbm + RidgedMulti noise), domain warping, biome system (Plains, Desert, Mountains, Tundra, Ocean), 3D cave carving, overhang density fields, and water placement below sea level. World is 256 blocks tall (16 vertically stacked 16³ chunks per column) with 8 block types (Air, Grass, Dirt, Stone, Water, Sand, Snow, Gravel). Async chunk generation on 4 background threads with non-blocking main thread — terrain streams in progressively from center outward.

Dual rendering pipeline: legacy CPU-meshed indirect draw path, and a mesh shader path (`VK_EXT_mesh_shader`) where raw voxel data (~4KB/chunk) is uploaded to SSBOs and geometry is generated on-the-fly by task + mesh shaders. Task shader performs two-phase frustum + Hi-Z occlusion culling. Mesh shader subdivides chunks into 4×4×4 sub-chunks (64 meshlets/chunk) and procedurally generates cube-face vertices with hidden-face removal using boundary slice data. Incremental slot-based voxel pool — no `device_wait_idle`, no VBO rebuild on chunk boundary crossing. Toggled via `MANIFOLD_MESH_SHADERS=1`.

Palette-based materials, directional + ambient lighting, stochastic texture sampling, edge detection shading. Procedural sky with atmospheric gradient and sun disc. Vulkan 1.2 with multiview always enabled. Unified rendering pipeline where VR is the baseline and desktop is the fallback — same shaders, same UBO layout (`gl_ViewIndex`-indexed VP arrays), same pipeline serves both paths. OpenXR session with stereo swapchains (`arraySize=2`), multiview render pass, pose-to-matrix conversion, frame loop with composition layer submission, session state machine, and combined stereo frustum culling. Two-phase Hi-Z occlusion culling with depth pyramid mip-chain generation.

---

## Next Steps

- [x] Remove legacy pipeline (mesh_pool, meshing.rs, shader.vert, cull.comp, transform SSBO, indirect buffer)
- [x] Make mesh shader path the default
- [x] Block placement / removal (interactive editing)
- [ ] Greedy face merging in mesh shader (reduce triangle count)

---

## Phase 7 — Advanced Voxel Techniques (long-term)

These are researched and documented in `docs/ideas/` but should only be tackled when earlier phases create a concrete need:

- [ ] Hash-based spatial structures for editable near-field (0-100m)
- [ ] Sparse Voxel DAGs with streaming for distant terrain (100m+)
- [x] Mesh shaders (`VK_EXT_mesh_shader`) for compressed chunk rendering
- [ ] Visibility buffer rendering
- [ ] Hardware ray tracing for shadows / global illumination
- [ ] Hybrid rasterize-near / ray-march-far pipeline (Aokana-style)

---

## Design Principles (from idea docs)

1. **Start coupled, separate when needed.** Don't build abstraction layers for problems that don't exist yet.
2. **VR-first is easier than VR-retrofit.** Keep camera source pluggable, UI world-space-ready, and target 90fps from the start.
3. **Hybrid architecture emerges, not planned.** Hash near-field + SVDAG far-field is the end state, not the starting point.
4. **Discrete blocks are an advantage.** Simpler destruction physics, better compression, artistic control over every block.
