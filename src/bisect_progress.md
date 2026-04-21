Here's every piece of logic in the resample pipeline that could cause the bug, including things I don't expect:

1. build_global_lookups — default crust for unowned points

Line 283: unowned points get CrustData::oceanic(7.0, -4.0, 0.0, DVec3::X) as default
If any global point index is NOT written by any plate (ownership gap), it stays oceanic
We validated this with debug_validate (every point owned exactly once), but only after move_plates, not right before resample. If resample itself left a gap in a previous cycle, it would propagate.
Status: NOT fully ruled out across resamples
2. SphereGrid::find_nearest_k — returns wrong nearest point

Could return a point from a different plate whose oceanic point is physically closer than the same-plate continental point after rotation
Has tests against brute-force, but only on clean Fibonacci grids, not on rotated multi-plate grids
Status: NOT tested on rotated points
3. SphereGrid coordinate convention — bin_of uses p.y for lat, p.z.atan2(p.x) for lon

If the simulation uses a different convention somewhere, points could hash to wrong bins
Checked: consistent with simulate.rs:980. Likely fine but...
Status: NOT explicitly tested with rotated points across all quadrants
4. Plate assignment — global nearest picks wrong plate

After large rotation, plate B's oceanic points can be physically closer than plate A's continental points to a position deep inside what was plate A's territory
This would assign the point to plate B with oceanic crust
Status: NOT ruled out — this is the cross-plate contamination theory
5. Crust lookup — nearest point is from wrong plate

interpolate_crust_nearest uses the same global SphereGrid, not filtered by assigned plate
The nearest global point could be from plate B (oceanic) even though the point was assigned to plate A
Wait — both plate assignment and crust lookup use the same find_nearest_k call result, so they should agree... but they're actually SEPARATE calls (plate assignment in resample(), crust in interpolate_crust_nearest). They should return the same result for the same query, but it's technically two independent lookups.
Status: NOT ruled out — verify they agree
6. SphericalFibonacci::new(point_count) — deterministic but identical each time

The new grid is always the same positions. This is by design but means every resample maps rotated points to the exact same target grid. Could create systematic aliasing.
Status: NOT ruled out as source of systematic patterns
7. move_plates — rotation correctness

DQuat::from_axis_angle then mul_vec3 then normalize(). If the rotation axis or speed is wrong, points could drift to unexpected positions.
debug_validate confirms points stay unit-length and owned. But doesn't check that the rotation is correct (right direction, right magnitude).
Status: NOT verified for correctness of rotation direction/magnitude
8. Compounding across resamples

Even if each resample introduces only a few errors at coastlines, over 20 resamples (400 steps), errors propagate inward. A coastal point flipped to oceanic becomes "ground truth" for the next resample.
Status: NOT ruled out — the primary suspect for deep-interior islands
9. Rendering — render_simulation_ex with SphereGrid

The renderer builds its own SphereGrid on post-resample points and does its own nearest-point lookup. If the renderer's lookup disagrees with the resample's assignment, it could show false islands that don't exist in the simulation data.
Status: NOT ruled out


Fibonacci spiral Moiré aliasing was an issue -> Fixed with majority vote among K nearest same-plate old points for crust type instead of copying from the single nearest point.
 Wierd spikes between plates -> Fixed by renderer now uses SphereGrid::find_nearest_k (nearest-point lookup) instead of the global Delaunay + triangle ownership. Each pixel gets the plate and crust type of its nearest simulation point directly — no degenerate triangles involved.