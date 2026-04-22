# Tectonic Simulation v2 — Session Notes

## What we did

### Architecture refactor (per-plate sub-mesh model)

Rewrote the tectonic simulation from a "global point cloud + rebuild Delaunay every step" model to a "per-plate reference mesh + accumulated rotation" model, following Cortial et al. 2019.

**plates.rs** — `Plate` now owns: `reference_points` (birth-frame vertices), `triangles`, `adjacency` (per-triangle halfedge neighbors, `None` at plate boundaries), `bounding_cap`, `rotation: DQuat` (accumulated R_k), plus the existing `rotation_axis`/`angular_speed`. Added `walk_to_point` (triangle walk with CW/CCW handling + epsilon tolerance), `point_in_triangle`, `to_reference`/`to_world`, `insert_boundary_vertex`, `nearest_boundary_edge`, `nearest_start_triangle`. `SampleCache` holds plate + triangle + barycentric coords per sample.

**simulate.rs** — `Simulation` stores `sample_points` (fixed Fibonacci grid), `sample_cache`, `sample_adjacency`. `advance_rotations` updates quaternions instead of mutating points. `assign_samples` does warm-start walk per sample (steps 1-8 from paper). `detect_boundaries` scans sample adjacency for cross-plate edges. Physics phases wired up (see below).

**resample.rs** — New Fibonacci grid, walk old plate meshes to transfer crust via barycentric interpolation, partition new Delaunay into per-plate sub-meshes, build adjacency + cache. Fallback for walk failures: nearest-vertex assignment with K=6 majority vote on crust type.

**plate_initializer.rs** — Returns `(Vec<Plate>, Vec<SampleCache>)`. Builds per-plate sub-meshes from initial Delaunay partition with `rotation = DQuat::IDENTITY`.

### Physics implemented

- **Subduction** (Section 4.1) — Uplift via nearest-source-only (not stacked), fold direction update, slab pull on rotation axes, subducted_distance advancement. Andean orogeny flip when oceanic crust uplifts above sea level.
- **Continental collision** (Section 4.2) — Terrane detection via plate mesh vertex adjacency BFS. Elevation surge with radial falloff. Himalayan orogeny. Terrane transfer (sloppy mesh surgery: vertices move to target plate, triangles dropped, resample rebuilds).
- **Oceanic crust generation** (Section 4.3) — Per-step: inserts new vertices at divergent boundary edges via `insert_boundary_vertex`. At resample: fallback assigns gap points to nearest plate with majority-voted crust type.
- **Plate rifting** (Section 4.4) — Poisson-triggered, noise-warped Voronoi partition on plate vertex adjacency, boundary vertices get fresh oceanic crust, sub-plates get perturbed rotation axes.
- **Erosion** (Section 4.5) — Continental erosion `z -= (z/zc)*ec*dt`, oceanic damping `z -= (1-z/zt)*eo*dt`, sediment accretion `z += et*dt`. Low-frequency noise modulation.

### Bugs found and fixed

- **Walk sign-to-edge mapping** — `bary[k]` negative means cross edge `(k+1)%3`, not edge `k`. Was causing walk failures.
- **CW triangle handling** — SphericalDelaunay produces mixed CW/CCW triangles. Fixed with `det.signum()` flip in walk and point_in_triangle.
- **Floating-point tolerance** — Vertices exactly on triangle edges have bary weights ~0 that go slightly negative. Added epsilon tolerance `-1e-12`.
- **Velocity units** — `surface_velocity` returned rad/Myr, paper expects mm/yr (= km/Myr). Fixed by multiplying by PLANET_RADIUS in `surface_velocity` itself. This was why Andean orogeny was always zero in both old and new pipelines.
- **Uplift stacking** — Each BoundarySample applied uplift independently. A vertex within 1800km of 200 boundary samples got 200x uplift per step. Fixed: use only the nearest source per vertex.
- **Plate cascade at resample** — Walk gaps (NO_PLATE) caused triangles to be dropped from partition, plates shrank each resample until empty. Fixed with nearest-vertex fallback.
- **Continental speckle noise** — Nearest-vertex fallback could copy crust from adjacent oceanic plate. Fixed with K=6 majority vote on crust type within the assigned plate.
- **Initial continental elevation** — Was 0.3 km (barely above sea level), boundary blending pushed it below 0. Raised to 2.0 km.
- **Resample interval** — Hardcoded 20, now adaptive 10-60 based on max plate speed (matches paper).

### Configuration (matching paper)

- 40 plates, 500k sample points, 0.3 land coverage, 125 steps (250 Myr), dt=2 Myr
- All constants from Appendix A verified against paper

## What still needs doing

### Critical

- **Subduction consumption** — Vertices with `subducted_distance > SURFACE_VISIBILITY_DISTANCE` should be removed from the plate mesh (mesh surgery). Without this, oceanic plates never shrink at convergent boundaries, so the area budget is unbalanced — plates only grow from oceanic generation and never shrink.

### Important

- **Andean flip requires continental neighbor** — Currently flips any oceanic vertex to continental when uplift pushes it above sea level. Should require at least one neighboring continental vertex to prevent isolated single-vertex flips that create speckle noise. Island arcs should form adjacent to existing continents or as coherent chains, not scattered pixels.
- **Walk cold-start improvement** — The walk from triangle 0 fails on concave plate meshes (~4% gap rate). `nearest_start_triangle` helps but is O(vertices). A per-plate spatial grid for triangle lookup would make cold starts O(1).
- **Per-step sample assignment fallback** — `search_candidates` in simulate.rs still returns NO_PLATE when no walk succeeds. Should have the same nearest-vertex fallback as resample. Currently doesn't cause visible issues because assign_samples uses warm-start which rarely fails.

### Nice to have

- **Source clustering** — The old pipeline clustered nearby boundary sources into grid cells to reduce per-vertex work. Current implementation is O(vertices * sources) per plate. Clustering would reduce to O(vertices * clusters).
- **Amplification** (Section 5) — Procedural noise + exemplar-based terrain primitives to add detail to the coarse crust model. This is what makes the paper's results look realistic at close range.
- **Terrane transfer mesh surgery** — Current implementation is "sloppy" (drops triangles, resample rebuilds). Proper mesh surgery would split triangles at the terrane boundary, maintaining valid meshes between resamples.
