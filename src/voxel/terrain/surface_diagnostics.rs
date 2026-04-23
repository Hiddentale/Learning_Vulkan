/// Terrain smoothness verification.
///
/// These helpers compute the *actual rendered surface* (topmost solid block of
/// each column, after all density and overhang processing) and compute the
/// max delta between adjacent columns. Used to detect cliffs / discontinuities
/// in the height field without needing to run the game.

#[cfg(test)]
mod tests {
    use crate::voxel::biome;
    use crate::voxel::block::BlockType;
    use crate::voxel::sphere::{self, ChunkPos, Face, FACE_SIDE_CHUNKS, PLANET_RADIUS_BLOCKS, SURFACE_RADIUS_BLOCKS};
    use super::super::generate::{generate_column, CHUNK_LAYERS};
    use super::super::height::{sample_params_at_world, sample_density_block};
    use super::super::noises::WorldNoises;
    use crate::voxel::chunk::CHUNK_SIZE;

    /// Side length (in chunks) of the centered window used by the full-face
    /// surface scans below. The cliff and ground-truth tests don't depend on
    /// scanning the *entire* face — any contiguous patch is enough to surface
    /// noise-budget regressions. Bounding the window keeps these tests O(1)
    /// regardless of `FACE_SIDE_CHUNKS`, which scales quadratically.
    const TEST_SCAN_CHUNKS: i32 = 16;
    const _: () = assert!(TEST_SCAN_CHUNKS <= FACE_SIDE_CHUNKS);

    /// Half-open chunk range `[lo, hi)` of size `TEST_SCAN_CHUNKS`, centered
    /// on the face midpoint.
    fn scan_range() -> (i32, i32) {
        let lo = FACE_SIDE_CHUNKS / 2 - TEST_SCAN_CHUNKS / 2;
        (lo, lo + TEST_SCAN_CHUNKS)
    }

    /// **End-to-end ground truth.** Generates an entire chunk column with
    /// `generate_column` (the same function the renderer uses) and walks it
    /// from the highest cy down to find the topmost non-air, non-water block.
    /// Returns its world-space radius. This is what the player actually sees.
    /// Slow (one full column generation per call) — use sparingly.
    fn topmost_solid_radius_via_generator(face: Face, cx: i32, cz: i32, x: usize, z: usize, seed: u32) -> f64 {
        let chunks = generate_column(face, cx, cz, seed, None);
        // chunks[cy] indexes from the bottom up. Walk top→bottom.
        for cy_rev in (0..CHUNK_LAYERS).rev() {
            for ly_rev in (0..CHUNK_SIZE).rev() {
                let block = chunks[cy_rev].get(x, ly_rev, z);
                if block != BlockType::Air && block != BlockType::Water {
                    let cp = sphere::ChunkPos {
                        face,
                        cx,
                        cy: cy_rev as i32,
                        cz,
                    };
                    let world = sphere::chunk_to_world(cp, glam::Vec3::new(x as f32 + 0.5, ly_rev as f32 + 0.5, z as f32 + 0.5));
                    return world.length();
                }
            }
        }
        0.0
    }

    /// Walk a column radially from the maximum possible surface down to the
    /// planet center; return the radial distance of the first solid block.
    /// This measures what the player actually sees, not the cached
    /// `surface_radius` from the per-column probe.
    fn topmost_solid_radius(face: Face, cx: i32, cz: i32, x: i32, z: i32, noises: &WorldNoises) -> Option<f64> {
        let max_radius = SURFACE_RADIUS_BLOCKS as f64 + 100.0;
        let min_radius = PLANET_RADIUS_BLOCKS as f64 - 50.0;
        let probe = sphere::chunk_to_world(ChunkPos { face, cx, cy: 0, cz }, glam::Vec3::new(x as f32 + 0.5, 0.5, z as f32 + 0.5));
        let params = sample_params_at_world(noises, probe, None);
        let surface_radius = PLANET_RADIUS_BLOCKS as f64 + params.height as f64;
        let sea_radius = SURFACE_RADIUS_BLOCKS as f64;
        let surface_block = biome::surface_block(params.biome);
        let subsurface_block = biome::subsurface_block(params.biome);

        // The probe direction (which we sample at every radial step) is
        // identical for the whole column.
        let dir = probe.normalize_or(glam::DVec3::Y);
        let mut r = max_radius;
        while r > min_radius {
            let world = dir * r;
            let block = sample_density_block(world, r, surface_radius, sea_radius, surface_block, subsurface_block, noises);
            if block != BlockType::Air && block != BlockType::Water {
                return Some(r);
            }
            r -= 1.0;
        }
        None
    }

    /// Sample the topmost solid radius for every (lx, lz) in a chunk column.
    fn chunk_surface_grid(face: Face, cx: i32, cz: i32, noises: &WorldNoises) -> [[f64; 16]; 16] {
        let mut g = [[0.0; 16]; 16];
        for (x, row) in g.iter_mut().enumerate() {
            for (z, cell) in row.iter_mut().enumerate() {
                *cell = topmost_solid_radius(face, cx, cz, x as i32, z as i32, noises).unwrap_or(0.0);
            }
        }
        g
    }

    /// Print a 16×16 grid of surface heights (relative to PLANET_RADIUS) for
    /// a single chunk so we can eyeball cliffs vs smooth slopes. Disabled by
    /// default — re-enable with `cargo test ... -- --ignored` for debugging.
    #[test]
    #[ignore]
    fn dump_terrain_surface_for_one_chunk() {
        let noises = WorldNoises::new(42);
        let face = Face::PosY;
        let cx = FACE_SIDE_CHUNKS / 2;
        let cz = FACE_SIDE_CHUNKS / 2;
        let g = chunk_surface_grid(face, cx, cz, &noises);
        eprintln!(
            "\n--- terrain surface heights at face={:?} chunk=({},{}) (radius - PLANET_RADIUS) ---",
            face, cx, cz
        );
        for z in 0..16 {
            let mut row = String::new();
            for col in &g {
                row.push_str(&format!("{:5.1} ", col[z] - PLANET_RADIUS_BLOCKS as f64));
            }
            eprintln!("{}", row);
        }
    }

    /// Sample the entire surface of one face and return the maximum
    /// absolute height delta between any two horizontally-adjacent columns,
    /// plus the (gx, gz) coordinate where it occurred.
    fn measure_max_adjacent_delta(face: Face, noises: &WorldNoises) -> (f64, usize, usize) {
        let (clo, chi) = scan_range();
        let n = (TEST_SCAN_CHUNKS as usize) * 16;
        let mut grid = vec![0.0_f64; n * n];
        for cx in clo..chi {
            for cz in clo..chi {
                for lx in 0..16 {
                    for lz in 0..16 {
                        let gx = (cx - clo) as usize * 16 + lx;
                        let gz = (cz - clo) as usize * 16 + lz;
                        grid[gx * n + gz] = topmost_solid_radius(face, cx, cz, lx as i32, lz as i32, noises).unwrap_or(0.0);
                    }
                }
            }
        }
        let mut max_delta = 0.0_f64;
        let mut worst = (0, 0);
        for gx in 0..n {
            for gz in 0..n {
                let h = grid[gx * n + gz];
                if gx + 1 < n {
                    let d = (grid[(gx + 1) * n + gz] - h).abs();
                    if d > max_delta {
                        max_delta = d;
                        worst = (gx, gz);
                    }
                }
                if gz + 1 < n {
                    let d = (grid[gx * n + gz + 1] - h).abs();
                    if d > max_delta {
                        max_delta = d;
                        worst = (gx, gz);
                    }
                }
            }
        }
        (max_delta, worst.0, worst.1)
    }

    /// Maximum allowed adjacent-column height delta. A 3-block cliff
    /// corresponds to a slope of ~71° per single-column step — steep but
    /// climbable in single jumps. Anything bigger is a wall and indicates
    /// the noise gradient (`amplitude × frequency`) has been over-budgeted.
    ///
    /// To raise this threshold you must justify the visual change. To lower
    /// noise contributions until the test passes, see the amplitude/frequency
    /// budgets in the constants at the top of this file.
    const MAX_ADJACENT_DELTA: f64 = 3.0;

    /// Generate one full chunk via `generate_column`, dump its surface heights
    /// AND simultaneously the radii my analytic `topmost_solid_radius` reports.
    /// If the two grids differ, the analytic test is lying about what the
    /// player sees.
    #[test]
    fn ground_truth_matches_analytic_test() {
        let face = Face::PosY;
        let cx = FACE_SIDE_CHUNKS / 2;
        let cz = FACE_SIDE_CHUNKS / 2;
        let noises = WorldNoises::new(42);
        let mut max_diff = 0.0_f64;
        for x in 0..16 {
            for z in 0..16 {
                let analytic = topmost_solid_radius(face, cx, cz, x as i32, z as i32, &noises).unwrap_or(0.0);
                let truth = topmost_solid_radius_via_generator(face, cx, cz, x, z, 42);
                let diff = (analytic - truth).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
            }
        }
        eprintln!("max disagreement between analytic and ground-truth = {:.1}", max_diff);
        assert!(
            max_diff < 1.5,
            "analytic test disagrees with generate_column by {} blocks — the analytic test is lying",
            max_diff
        );
    }

    /// Probe a specific user-reported cliff. Generates a 3×3 chunk window
    /// centered on the given chunk and prints the topmost solid radius for
    /// every column plus the worst lateral exposure starting from any
    /// surface block.
    #[test]
    fn probe_user_reported_cliff_posz_6_2() {
        let face = Face::PosZ;
        let mid_cx = 6_i32;
        let mid_cz = 2_i32;
        // 5×5 chunk window so the cliff (which may be 1-2 chunks from the
        // player's body) is included.
        const W: usize = 5 * 16;
        let mut grid = vec![vec![vec![BlockType::Air; CHUNK_SIZE * CHUNK_LAYERS]; W]; W];
        for dx in 0..5i32 {
            for dz in 0..5i32 {
                let cx = mid_cx - 2 + dx;
                let cz = mid_cz - 2 + dz;
                if cx < 0 || cz < 0 || cx >= FACE_SIDE_CHUNKS || cz >= FACE_SIDE_CHUNKS {
                    continue;
                }
                let chunks = generate_column(face, cx, cz, 42, None);
                for (cy, chunk) in chunks.iter().enumerate().take(CHUNK_LAYERS) {
                    for ly in 0..CHUNK_SIZE {
                        for lx in 0..16 {
                            for lz in 0..16 {
                                let gx = dx as usize * 16 + lx;
                                let gz = dz as usize * 16 + lz;
                                let gy = cy * CHUNK_SIZE + ly;
                                grid[gx][gz][gy] = chunk.get(lx, ly, lz);
                            }
                        }
                    }
                }
            }
        }
        let solid = |x: usize, y: usize, z: usize| -> bool {
            if x >= W || z >= W || y >= CHUNK_SIZE * CHUNK_LAYERS {
                return false;
            }
            let b = grid[x][z][y];
            b != BlockType::Air && b != BlockType::Water
        };
        // Also build a "solid OR water" view so we can see where water
        // columns are vs land columns. Water columns count as "below sea level".
        let any_solid_or_water = |x: usize, y: usize, z: usize| -> bool {
            if x >= W || z >= W || y >= CHUNK_SIZE * CHUNK_LAYERS {
                return false;
            }
            grid[x][z][y] != BlockType::Air
        };
        // Top-of-column heights for the WxW window. Print TWO grids:
        // 1) topmost solid (= seabed for water columns, surface for land)
        // 2) topmost solid-or-water (= water surface for water cols, surface for land)
        eprintln!(
            "\n--- topmost SOLID (skip water) for face=PosZ chunks ({}..{}, *, {}..{}), player at cx={} cz={} ---",
            mid_cx - 2,
            mid_cx + 2,
            mid_cz - 2,
            mid_cz + 2,
            mid_cx,
            mid_cz
        );
        let mut tops = vec![vec![0usize; W]; W];
        for (x, tops_col) in tops.iter_mut().enumerate().take(W) {
            for (z, top_val) in tops_col.iter_mut().enumerate().take(W) {
                for y in (0..CHUNK_SIZE * CHUNK_LAYERS).rev() {
                    if solid(x, y, z) {
                        *top_val = y;
                        break;
                    }
                }
            }
        }
        for z in 0..W {
            let mut row = String::new();
            for col in tops.iter().take(W) {
                row.push_str(&format!("{:3}", col[z] as i32));
            }
            eprintln!("{}", row);
        }
        eprintln!("\n--- topmost solid-or-water (= visible top including ocean surface) ---");
        let mut wtops = vec![vec![0usize; W]; W];
        for (x, wtops_col) in wtops.iter_mut().enumerate().take(W) {
            for (z, wtop_val) in wtops_col.iter_mut().enumerate().take(W) {
                for y in (0..CHUNK_SIZE * CHUNK_LAYERS).rev() {
                    if any_solid_or_water(x, y, z) {
                        *wtop_val = y;
                        break;
                    }
                }
            }
        }
        for z in 0..W {
            let mut row = String::new();
            for col in wtops.iter().take(W) {
                row.push_str(&format!("{:3}", col[z] as i32));
            }
            eprintln!("{}", row);
        }
        // Max top-delta across this window
        let mut max_top_delta = 0i32;
        let mut worst = (0, 0);
        for x in 0..W - 1 {
            for z in 0..W - 1 {
                let h = tops[x][z] as i32;
                let dx_d = (tops[x + 1][z] as i32 - h).abs();
                let dz_d = (tops[x][z + 1] as i32 - h).abs();
                if dx_d > max_top_delta {
                    max_top_delta = dx_d;
                    worst = (x, z);
                }
                if dz_d > max_top_delta {
                    max_top_delta = dz_d;
                    worst = (x, z);
                }
            }
        }
        // Max VISIBLE lateral exposure (run starting from top, walking down while solid here / air at neighbor)
        let mut max_run = 0usize;
        let mut worst_run = (0, 0, 0);
        for (x, tops_col) in tops.iter().enumerate().take(W - 1).skip(1) {
            for (z, &top) in tops_col.iter().enumerate().take(W - 1).skip(1) {
                if top == 0 {
                    continue;
                }
                for (dir_idx, (dx, dz)) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)].iter().enumerate() {
                    let nx = (x as i32 + dx) as usize;
                    let nz = (z as i32 + dz) as usize;
                    let mut run = 0usize;
                    let mut y = top as i32;
                    while y >= 0 {
                        if solid(x, y as usize, z) && !solid(nx, y as usize, nz) {
                            run += 1;
                            y -= 1;
                        } else {
                            break;
                        }
                    }
                    if run > max_run {
                        max_run = run;
                        worst_run = (x, z, dir_idx);
                    }
                }
            }
        }
        eprintln!("max top-delta = {} at ({},{})", max_top_delta, worst.0, worst.1);
        eprintln!(
            "max VISIBLE lateral exposure = {} at column ({},{}) dir={}",
            max_run, worst_run.0, worst_run.1, worst_run.2
        );

        // ----- Now print a focused 33×33 view centered on the player -----
        // Player is at face_z chunks (mid_cx, mid_cz) local (4.5, 1.7, 0.2).
        // The 5×5 window puts the player chunk at center: grid x range 16..32 → 32..48.
        // Center on grid (40, 40 ish).
        let pgx = (mid_cx - (mid_cx - 2)) as usize * 16 + 4; // = 36
        let pgz = (mid_cz - (mid_cz - 2)) as usize * 16; // = 32
        eprintln!(
            "\n--- 33×33 view of topmost SOLID centered on player (player at grid ({},{})) ---",
            pgx, pgz
        );
        eprintln!("--- player cube_y = {} (cy=4 ly=1.7) ---", 4 * 16 + 2);
        let half = 16i32;
        for dz in -half..=half {
            let z = pgz as i32 + dz;
            if z < 0 || z >= W as i32 {
                continue;
            }
            let mut row = String::new();
            for dx in -half..=half {
                let x = pgx as i32 + dx;
                if x < 0 || x >= W as i32 {
                    row.push_str("   ");
                    continue;
                }
                let v = tops[x as usize][z as usize] as i32;
                let marker = if dx == 0 && dz == 0 { '*' } else { ' ' };
                row.push_str(&format!("{}{:3}", marker, v));
            }
            eprintln!("{}", row);
        }

        // Compute slopes from the player position to the worst nearby surface
        let player_y = 4 * 16 + 2; // cy*16 + ly
        let mut max_slope_to_surface = 0.0_f64;
        let mut steepest_target = (0i32, 0i32, 0i32);
        for dz in -half..=half {
            for dx in -half..=half {
                let x = pgx as i32 + dx;
                let z = pgz as i32 + dz;
                if x < 0 || z < 0 || x >= W as i32 || z >= W as i32 {
                    continue;
                }
                let target_y = tops[x as usize][z as usize] as i32;
                let horiz = ((dx * dx + dz * dz) as f64).sqrt();
                if horiz < 1.0 {
                    continue;
                }
                let vert = (target_y - player_y) as f64;
                let slope = vert / horiz;
                if slope > max_slope_to_surface {
                    max_slope_to_surface = slope;
                    steepest_target = (x, z, target_y);
                }
            }
        }
        eprintln!("\nFrom player at cube_y={}, steepest visible slope to a surface block:", player_y);
        let h = (((steepest_target.0 - pgx as i32).pow(2) + (steepest_target.1 - pgz as i32).pow(2)) as f64).sqrt();
        eprintln!(
            "  target=({},{}, top_y={}) → vertical={}, horizontal={:.1}, slope={:.2} ({:.0}°)",
            steepest_target.0,
            steepest_target.1,
            steepest_target.2,
            steepest_target.2 - player_y,
            h,
            max_slope_to_surface,
            max_slope_to_surface.atan().to_degrees()
        );
    }

    /// Maximum allowed *visible* lateral cliff exposure. A visible cliff is a
    /// contiguous stack of solid-here / air-there blocks that includes the
    /// topmost solid block of `here` (i.e. it's at the surface, not buried
    /// inside a deep cave that the player can't see).
    const MAX_LATERAL_EXPOSURE: usize = 3;

    /// Scan a 3×3 chunk window for the longest lateral exposure across all
    /// four horizontal neighbors. Permanent guard against the
    /// "cave-cuts-into-neighbor exposes a tall wall" failure mode that the
    /// top-of-column delta test cannot see.
    #[test]
    fn terrain_lateral_exposure_within_threshold() {
        let face = Face::PosY;
        let mid = FACE_SIDE_CHUNKS / 2;
        // Generate a 3×3 chunk window so neighbor lookups across chunk
        // boundaries have real data.
        let mut grid = vec![vec![vec![BlockType::Air; CHUNK_SIZE * CHUNK_LAYERS]; 48]; 48];
        for dx in 0..3 {
            for dz in 0..3 {
                let cx = mid - 1 + dx as i32;
                let cz = mid - 1 + dz as i32;
                let chunks = generate_column(face, cx, cz, 42, None);
                for (cy, chunk) in chunks.iter().enumerate().take(CHUNK_LAYERS) {
                    for ly in 0..CHUNK_SIZE {
                        for lx in 0..16 {
                            for lz in 0..16 {
                                let gx = dx * 16 + lx;
                                let gz = dz * 16 + lz;
                                let gy = cy * CHUNK_SIZE + ly;
                                grid[gx][gz][gy] = chunk.get(lx, ly, lz);
                            }
                        }
                    }
                }
            }
        }
        let solid = |x: usize, y: usize, z: usize| -> bool {
            if x >= 48 || z >= 48 || y >= CHUNK_SIZE * CHUNK_LAYERS {
                return false;
            }
            let b = grid[x][z][y];
            b != BlockType::Air && b != BlockType::Water
        };
        // Find the topmost solid block of each column so we can restrict the
        // exposure scan to "starting at or near the surface".
        let topmost = |x: usize, z: usize| -> Option<usize> { (0..CHUNK_SIZE * CHUNK_LAYERS).rev().find(|&y| solid(x, y, z)) };

        let mut max_run = 0usize;
        let mut worst = (0, 0, 0, 0);
        // Skip the outermost ring so neighbor lookups always have data.
        for x in 1..47 {
            for z in 1..47 {
                let top_here = match topmost(x, z) {
                    Some(t) => t,
                    None => continue,
                };
                for (dir_idx, (dx, dz)) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)].iter().enumerate() {
                    let nx = (x as i32 + dx) as usize;
                    let nz = (z as i32 + dz) as usize;
                    // Walk down from `top_here` while solid here AND air at neighbor.
                    // Stop the run as soon as the chain breaks.
                    let mut run = 0usize;
                    let mut y = top_here as i32;
                    while y >= 0 {
                        let here = solid(x, y as usize, z);
                        let neigh = solid(nx, y as usize, nz);
                        if here && !neigh {
                            run += 1;
                            y -= 1;
                        } else {
                            break;
                        }
                    }
                    if run > max_run {
                        max_run = run;
                        worst = (x, top_here, z, dir_idx);
                    }
                }
            }
        }
        eprintln!(
            "max VISIBLE lateral cliff exposure in 3×3 chunk window = {} blocks at column ({},{}) top y={}, dir={}",
            max_run, worst.0, worst.2, worst.1, worst.3
        );
        // Print the column profile at the worst spot so we can see exactly
        // which blocks form the cliff.
        let (wx, _wy, wz, dir_idx) = worst;
        let (dx, dz): (i32, i32) = [(1, 0), (-1, 0), (0, 1), (0, -1)][dir_idx];
        let nx = (wx as i32 + dx) as usize;
        let nz = (wz as i32 + dz) as usize;
        eprintln!("column profile at ({},{}) vs neighbor ({},{}):", wx, wz, nx, nz);
        let top_a = topmost(wx, wz).unwrap();
        let top_b = topmost(nx, nz).unwrap_or(0);
        eprintln!("  top here = {}, top neighbor = {}", top_a, top_b);
        let lo = top_a.saturating_sub(15);
        for y in (lo..=top_a + 2).rev() {
            let a = if solid(wx, y, wz) { '#' } else { '.' };
            let b = if solid(nx, y, nz) { '#' } else { '.' };
            eprintln!("  y={:3}: here={} neigh={}", y, a, b);
        }
        assert!(
            max_run <= MAX_LATERAL_EXPOSURE,
            "max visible lateral cliff = {} blocks, threshold {}. \
             A cave or overhang is exposing too many vertically-stacked block faces. \
             Push CAVE_MIN_DEPTH further from the surface, reduce OVERHANG_STRENGTH, \
             or reduce CAVE_THRESHOLD to make caves rarer.",
            max_run,
            MAX_LATERAL_EXPOSURE
        );
    }

    /// Ground-truth scan over an entire face. For each (cx, cz) chunk, calls
    /// `generate_column` (the renderer's path) and walks every column top→bottom
    /// to find the topmost solid block. Slow (144 column generations per face)
    /// but it's the only thing we can fully trust since the analytic version
    /// might be diverging from reality somewhere. Ignored by default
    /// because each call invokes the full chunk-column generator (~65k
    /// column gens at TEST_SCAN_CHUNKS=16); run with `--ignored` when
    /// chasing analytic-vs-ground-truth divergences.
    #[test]
    #[ignore]
    fn ground_truth_full_face_scan() {
        let face = Face::PosY;
        let (clo, chi) = scan_range();
        let n = (TEST_SCAN_CHUNKS as usize) * 16;
        let mut grid = vec![0.0_f64; n * n];
        for cx in clo..chi {
            for cz in clo..chi {
                for lx in 0..16 {
                    for lz in 0..16 {
                        let gx = (cx - clo) as usize * 16 + lx;
                        let gz = (cz - clo) as usize * 16 + lz;
                        grid[gx * n + gz] = topmost_solid_radius_via_generator(face, cx, cz, lx, lz, 42);
                    }
                }
            }
        }
        let mut max_delta = 0.0_f64;
        let mut worst = (0, 0);
        for gx in 0..n {
            for gz in 0..n {
                let h = grid[gx * n + gz];
                if gx + 1 < n {
                    let d = (grid[(gx + 1) * n + gz] - h).abs();
                    if d > max_delta {
                        max_delta = d;
                        worst = (gx, gz);
                    }
                }
                if gz + 1 < n {
                    let d = (grid[gx * n + gz + 1] - h).abs();
                    if d > max_delta {
                        max_delta = d;
                        worst = (gx, gz);
                    }
                }
            }
        }
        eprintln!(
            "GROUND-TRUTH full-face scan: max adjacent delta = {:.1} at ({},{})",
            max_delta, worst.0, worst.1
        );
        let (wx, wz) = worst;
        eprintln!("--- 9×9 window of TRUE surface heights around worst ---");
        for dz in -4i32..=4 {
            let mut row = String::new();
            for dx in -4i32..=4 {
                let x = wx as i32 + dx;
                let z = wz as i32 + dz;
                if x >= 0 && z >= 0 && (x as usize) < n && (z as usize) < n {
                    row.push_str(&format!("{:6.1} ", grid[x as usize * n + z as usize] - PLANET_RADIUS_BLOCKS as f64));
                } else {
                    row.push_str("   .   ");
                }
            }
            eprintln!("{}", row);
        }
    }

    /// Ground-truth scan: for one center chunk and its 4 axis neighbors, use
    /// `generate_column` to find the actual topmost solid block of every column,
    /// then compute the max adjacent delta across the union. If THIS reports
    /// > 3 the renderer truly is producing cliffs.
    #[test]
    fn ground_truth_height_field_smooth_in_center_chunk_neighborhood() {
        let face = Face::PosY;
        let mid = FACE_SIDE_CHUNKS / 2;
        // 3×3 chunk window of ground-truth heights → 48×48 grid.
        let mut grid = [[0.0_f64; 48]; 48];
        for dx in 0..3 {
            for dz in 0..3 {
                let cx = mid - 1 + dx as i32;
                let cz = mid - 1 + dz as i32;
                for lx in 0..16 {
                    for lz in 0..16 {
                        let h = topmost_solid_radius_via_generator(face, cx, cz, lx, lz, 42);
                        grid[dx * 16 + lx][dz * 16 + lz] = h;
                    }
                }
            }
        }
        let mut max_delta = 0.0_f64;
        let mut worst = (0, 0);
        for x in 0..48 {
            for z in 0..48 {
                if x + 1 < 48 {
                    let d = (grid[x + 1][z] - grid[x][z]).abs();
                    if d > max_delta {
                        max_delta = d;
                        worst = (x, z);
                    }
                }
                if z + 1 < 48 {
                    let d = (grid[x][z + 1] - grid[x][z]).abs();
                    if d > max_delta {
                        max_delta = d;
                        worst = (x, z);
                    }
                }
            }
        }
        eprintln!(
            "ground-truth max adjacent delta in 3×3 chunk window = {:.1} at ({},{})",
            max_delta, worst.0, worst.1
        );
        // Print 9×9 window around the worst spot for inspection.
        let (wx, wz) = worst;
        eprintln!("--- 9×9 window of TRUE surface heights around worst ---");
        for dz in -4i32..=4 {
            let mut row = String::new();
            for dx in -4i32..=4 {
                let x = wx as i32 + dx;
                let z = wz as i32 + dz;
                if x >= 0 && z >= 0 && (x as usize) < 48 && (z as usize) < 48 {
                    row.push_str(&format!("{:6.1} ", grid[x as usize][z as usize] - PLANET_RADIUS_BLOCKS as f64));
                } else {
                    row.push_str("   .   ");
                }
            }
            eprintln!("{}", row);
        }
    }

    #[test]
    fn terrain_height_field_has_no_extreme_cliffs() {
        let noises = WorldNoises::new(42);
        let mut worst_face_delta = 0.0_f64;
        let mut worst_face = Face::PosY;
        for face in sphere::ALL_FACES {
            // Also report the min/max surface height on this face so we can
            // see if there are coastlines (height crossing sea level) that
            // would generate large land/water jumps.
            let (clo, chi) = scan_range();
            let n = (TEST_SCAN_CHUNKS as usize) * 16;
            let mut hi = 0.0_f64;
            let mut lo = 1e9_f64;
            for cx in clo..chi {
                for cz in clo..chi {
                    for lx in 0..16 {
                        for lz in 0..16 {
                            let h = topmost_solid_radius(face, cx, cz, lx, lz, &noises).unwrap_or(0.0);
                            if h > hi {
                                hi = h;
                            }
                            if h < lo {
                                lo = h;
                            }
                        }
                    }
                }
            }
            let _ = n;
            let (d, gx, gz) = measure_max_adjacent_delta(face, &noises);
            eprintln!(
                "face {:?}: max adjacent delta = {:.1} at ({},{}); surface range [{:.0} .. {:.0}] (sea_level = {})",
                face,
                d,
                gx,
                gz,
                lo,
                hi,
                sphere::SURFACE_RADIUS_BLOCKS
            );
            if d > worst_face_delta {
                worst_face_delta = d;
                worst_face = face;
            }
        }
        assert!(
            worst_face_delta <= MAX_ADJACENT_DELTA,
            "max adjacent-column height delta = {:.1} blocks on face {:?}, threshold {:.1}. \
             Reduce noise amplitudes (mountain/detail/weirdness/continental) so that \
             amplitude × 2π × frequency stays under ~0.3 per contribution.",
            worst_face_delta,
            worst_face,
            MAX_ADJACENT_DELTA
        );
    }

    /// For every cube edge, walk along the edge in face A and for each sample
    /// step laterally (in face A's tangent direction) onto face B. Compare the
    /// surface heights of the original column on A and the column it lands on
    /// in B. A large delta here is a cliff at a face seam — the symptom the
    /// within-face test cannot see because it scans only one face at a time.
    #[test]
    fn terrain_height_field_smooth_across_face_seams() {
        let noises = WorldNoises::new(42);
        let mut max_delta = 0.0_f64;
        let mut worst = (Face::PosY, Face::PosY, glam::DVec3::ZERO);
        // 1.5 blocks is enough to push past one column's worth of width without
        // crossing two neighbors at once.
        let nudge = 1.5_f64;
        for face_a in sphere::ALL_FACES {
            let (tu, tv, _) = sphere::face_basis(face_a);
            // The four edges of this face, each parameterised by their normal
            // direction (in face-A tangent space) and the column index inside
            // face A that's right next to the edge.
            let edges: [(glam::DVec3, i32, i32); 4] = [
                (tu.as_dvec3(), FACE_SIDE_CHUNKS - 1, 15), // +u edge
                (-tu.as_dvec3(), 0, 0),                    // -u edge
                (tv.as_dvec3(), FACE_SIDE_CHUNKS - 1, 15), // +v edge (cz)
                (-tv.as_dvec3(), 0, 0),                    // -v edge (cz)
            ];
            // Sample only a centered window of columns along each edge —
            // enough to surface seam discontinuities without paying for the
            // full face length on a 16k-radius planet.
            let (clo, chi) = scan_range();
            for (edge_idx, (out_dir, edge_chunk, edge_block)) in edges.iter().enumerate() {
                for cz_or_cx in clo..chi {
                    for lz_or_lx in 0..16 {
                        let (cx, cz, lx, lz) = match edge_idx {
                            0 | 1 => (*edge_chunk, cz_or_cx, *edge_block, lz_or_lx),
                            2 | 3 => (cz_or_cx, *edge_chunk, lz_or_lx, *edge_block),
                            _ => unreachable!(),
                        };
                        let h_a = topmost_solid_radius(face_a, cx, cz, lx, lz, &noises).unwrap_or(0.0);
                        let probe_a = sphere::chunk_to_world(
                            sphere::ChunkPos { face: face_a, cx, cy: 0, cz },
                            glam::Vec3::new(lx as f32 + 0.5, 0.5, lz as f32 + 0.5),
                        );
                        // Step laterally past the edge in world space.
                        let test_point = probe_a + *out_dir * nudge;
                        if let Some((nb_cp, nlx, _nly, nlz)) = sphere::world_to_chunk_local(test_point) {
                            if nb_cp.face != face_a {
                                let h_b = topmost_solid_radius(nb_cp.face, nb_cp.cx, nb_cp.cz, nlx as i32, nlz as i32, &noises).unwrap_or(0.0);
                                let d = (h_a - h_b).abs();
                                if d > max_delta {
                                    max_delta = d;
                                    worst = (face_a, nb_cp.face, probe_a);
                                }
                            }
                        }
                    }
                }
            }
        }
        eprintln!(
            "max cross-seam delta = {:.1} blocks (faces {:?} → {:?}) at world {:?}",
            max_delta, worst.0, worst.1, worst.2
        );
        assert!(
            max_delta <= MAX_ADJACENT_DELTA + 1.0,
            "max cross-seam height delta = {:.1} blocks, threshold {:.1}. \
             A discontinuity at the cube edge means columns on opposite sides \
             of the seam sample noise at noticeably different directions.",
            max_delta,
            MAX_ADJACENT_DELTA + 1.0
        );
    }
}
