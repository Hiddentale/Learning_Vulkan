# Manifold

A from-scratch voxel engine built with **Rust** and **Vulkan**, mainly built for VR.

![Terrain preview](docs/images/preview.png)
![Terrain preview](docs/images/preview2.png)
![Terrain preview](docs/images/preview3.png)

## Features

**Rendering**
- Hybrid pipeline: mesh shaders (near-field, 0-128m) + SVDAG ray marching (mid-field, 128m-1.5km) + rasterized heightmap (far-field, 1.5-6km)
- Hi-Z occlusion culling via depth pyramid
- Hierarchical DDA with empty octant caching for SVDAG traversal
- Stochastic texture tiling, palette-based materials (256-entry GPU palette)
- VR-ready: OpenXR + multiview rendering, stereo frustum culling

**Terrain generation**
- 6-parameter noise router (continentalness, erosion, weirdness, temperature, humidity, depth)
- 13 biomes with continental terrain shaping
- Tectonic plate simulation with convergent/divergent boundary stress
- Hydraulic erosion on a 2048x2048 coarse grid
- Domain warping, 3D cave carving, overhang density fields
- 768-block world height (48 vertical chunk layers)

**World**
- Async chunk generation on background threads
- SVDAG disk cache with LZ4 compression and mmap region files
- LOD hierarchy with direct noise sampling for distant terrain
- Block placement and removal
- Player physics with gravity, collision, fly/walk toggle

## Architecture

```
src/
  graphical_core/   Vulkan renderer, pipelines, GPU pools, command recording
  voxel/            Chunks, terrain, biomes, SVDAG compression, erosion, heightmap
  storage/          Region files, world metadata, disk caching
  vr/               OpenXR session, stereo swapchains, multiview
  shaders/          GLSL mesh/task/vertex/fragment/compute shaders (SPIR-V)
```

Voxel data and rendering are decoupled. Terrain generation runs on background thread pools (crossbeam channels). Three independent GPU pools manage near-field mesh chunks, SVDAG ray march data, and heightmap tile geometry.

## Build

Requires the [Vulkan SDK](https://vulkan.lunarg.com/) and `glslc` on PATH.

```
cargo run --release
```

## Controls

| Key | Action |
|-----|--------|
| WASD | Move |
| Mouse | Look |
| Space | Jump (walk mode) |
| Shift | Fast fly |
| E / Q | Fly up / down |
| F | Toggle fly/walk |
| Left click | Break block |
| Right click | Place block |
| Esc | Release cursor |
