use super::terrain::WorldNoises;
use noise::NoiseFn;
use std::fs;
use std::path::Path;

/// Coarse erosion map generated once at world creation.
/// Stores height deltas (negative = eroded, positive = deposited) that modify
/// the noise-based terrain height during chunk generation.
pub struct ErosionMap {
    grid: Vec<f32>,
    pub size: usize,
    cell_size: f32,
    origin_x: f32,
    origin_z: f32,
}

// Erosion simulation constants — tuned for 32-block cells where 1 unit = 1 block of height
const RAIN_RATE: f32 = 0.002;
const EVAPORATION_RATE: f32 = 0.05;
const SEDIMENT_CAPACITY: f32 = 0.02;
const EROSION_RATE: f32 = 0.05;
const DEPOSITION_RATE: f32 = 0.3;
const GRAVITY: f32 = 4.0;
const PIPE_AREA: f32 = 0.5;
const MIN_TILT: f32 = 0.01;

impl ErosionMap {
    /// Sample erosion delta at world coordinates via bilinear interpolation.
    /// Returns 0.0 for positions outside the map.
    pub fn sample(&self, wx: f64, wz: f64) -> f64 {
        let fx = (wx as f32 - self.origin_x) / self.cell_size;
        let fz = (wz as f32 - self.origin_z) / self.cell_size;

        if fx < 0.0 || fz < 0.0 {
            return 0.0;
        }
        let ix = fx as usize;
        let iz = fz as usize;
        if ix + 1 >= self.size || iz + 1 >= self.size {
            return 0.0;
        }

        let tx = fx - ix as f32;
        let tz = fz - iz as f32;

        let v00 = self.grid[ix + iz * self.size];
        let v10 = self.grid[(ix + 1) + iz * self.size];
        let v01 = self.grid[ix + (iz + 1) * self.size];
        let v11 = self.grid[(ix + 1) + (iz + 1) * self.size];

        let a = v00 + (v10 - v00) * tx;
        let b = v01 + (v11 - v01) * tx;
        (a + (b - a) * tz) as f64
    }

    pub fn save(&self, path: &Path) -> anyhow::Result<()> {
        let mut data = Vec::with_capacity(16 + self.grid.len() * 4);
        data.extend_from_slice(&(self.size as u32).to_le_bytes());
        data.extend_from_slice(&self.cell_size.to_bits().to_le_bytes());
        data.extend_from_slice(&(self.origin_x as i32).to_le_bytes());
        data.extend_from_slice(&(self.origin_z as i32).to_le_bytes());
        for &v in &self.grid {
            data.extend_from_slice(&v.to_bits().to_le_bytes());
        }
        fs::write(path, &data)?;
        Ok(())
    }

    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let data = fs::read(path)?;
        anyhow::ensure!(data.len() >= 16, "erosion map too small");
        let size = u32::from_le_bytes(data[0..4].try_into()?) as usize;
        let cell_size = f32::from_bits(u32::from_le_bytes(data[4..8].try_into()?));
        let origin_x = i32::from_le_bytes(data[8..12].try_into()?) as f32;
        let origin_z = i32::from_le_bytes(data[12..16].try_into()?) as f32;
        anyhow::ensure!(data.len() == 16 + size * size * 4, "erosion map size mismatch");
        let mut grid = Vec::with_capacity(size * size);
        for i in 0..size * size {
            let off = 16 + i * 4;
            grid.push(f32::from_bits(u32::from_le_bytes(data[off..off + 4].try_into()?)));
        }
        Ok(Self {
            grid,
            size,
            cell_size,
            origin_x,
            origin_z,
        })
    }
}

/// Generate a coarse erosion map by running hydraulic erosion on noise-sampled terrain.
pub fn generate_erosion_map(size: usize, cell_size: f32, seed: u32, iterations: usize) -> ErosionMap {
    let half_extent = (size as f32 * cell_size) / 2.0;
    let origin_x = -half_extent;
    let origin_z = -half_extent;

    // Sample base terrain heights from noise
    let noises = WorldNoises::new(seed);
    let mut terrain: Vec<f32> = Vec::with_capacity(size * size);
    for iz in 0..size {
        for ix in 0..size {
            let wx = origin_x as f64 + ix as f64 * cell_size as f64;
            let wz = origin_z as f64 + iz as f64 * cell_size as f64;
            let warped_x = wx + noises.warp_x.get([wx, wz]) * super::terrain::WARP_STRENGTH;
            let warped_z = wz + noises.warp_z.get([wx, wz]) * super::terrain::WARP_STRENGTH;
            let c = noises.continentalness.get([warped_x, warped_z]);
            let e = noises.erosion_noise.get([warped_x, warped_z]);
            let w = noises.weirdness.get([warped_x, warped_z]);
            let h = super::terrain::compute_height_from_params(&noises, warped_x, warped_z, c, e, w);
            terrain.push(h as f32);
        }
    }

    // --- Tectonic uplift ---
    let uplift = generate_tectonic_uplift(size, cell_size, origin_x, origin_z, seed);
    for i in 0..size * size {
        terrain[i] += uplift[i];
    }

    let initial_terrain = terrain.clone();

    // Simulation state
    let n = size;
    let mut water = vec![0.0f32; n * n];
    let mut sediment = vec![0.0f32; n * n];
    let mut flux = vec![[0.0f32; 4]; n * n]; // N, E, S, W

    let dt = 1.0f32;

    for _iter in 0..iterations {
        // 1. Rain
        for w in water.iter_mut() {
            *w += RAIN_RATE * dt;
        }

        // 2. Outflow flux based on height differences
        for iz in 0..n {
            for ix in 0..n {
                let idx = ix + iz * n;
                let h = terrain[idx] + water[idx];
                let mut f = flux[idx];

                // North (iz-1), East (ix+1), South (iz+1), West (ix-1)
                let neighbors = [
                    if iz > 0 { Some(ix + (iz - 1) * n) } else { None },
                    if ix + 1 < n { Some((ix + 1) + iz * n) } else { None },
                    if iz + 1 < n { Some(ix + (iz + 1) * n) } else { None },
                    if ix > 0 { Some((ix - 1) + iz * n) } else { None },
                ];

                for (dir, nb) in neighbors.iter().enumerate() {
                    if let Some(ni) = nb {
                        let nh = terrain[*ni] + water[*ni];
                        let dh = h - nh;
                        f[dir] = (f[dir] + GRAVITY * PIPE_AREA * dh * dt / cell_size).max(0.0);
                    } else {
                        f[dir] = 0.0;
                    }
                }

                // Scale down if total outflow exceeds available water
                let total_out = f[0] + f[1] + f[2] + f[3];
                if total_out > 0.0 {
                    let available = water[idx] * cell_size * cell_size / dt;
                    if total_out > available {
                        let scale = available / total_out;
                        for v in f.iter_mut() {
                            *v *= scale;
                        }
                    }
                }
                flux[idx] = f;
            }
        }

        // 3. Water transport: update water from net flux
        let mut velocity_x = vec![0.0f32; n * n];
        let mut velocity_z = vec![0.0f32; n * n];
        for iz in 0..n {
            for ix in 0..n {
                let idx = ix + iz * n;
                let out = flux[idx][0] + flux[idx][1] + flux[idx][2] + flux[idx][3];

                let in_n = if iz > 0 { flux[ix + (iz - 1) * n][2] } else { 0.0 }; // neighbor's south
                let in_e = if ix + 1 < n { flux[(ix + 1) + iz * n][3] } else { 0.0 }; // neighbor's west
                let in_s = if iz + 1 < n { flux[ix + (iz + 1) * n][0] } else { 0.0 }; // neighbor's north
                let in_w = if ix > 0 { flux[(ix - 1) + iz * n][1] } else { 0.0 }; // neighbor's east

                let inflow = in_n + in_e + in_s + in_w;
                let dv = (inflow - out) * dt / (cell_size * cell_size);
                water[idx] = (water[idx] + dv).max(0.0);

                // 4. Velocity from flux differences
                let avg_water = (water[idx] + 0.001).recip(); // avoid div-by-zero
                velocity_x[idx] = (in_w - flux[idx][3] + flux[idx][1] - in_e) * 0.5 * avg_water;
                velocity_z[idx] = (in_n - flux[idx][0] + flux[idx][2] - in_s) * 0.5 * avg_water;
            }
        }

        // 5. Erosion and deposition
        for iz in 0..n {
            for ix in 0..n {
                let idx = ix + iz * n;
                let speed = (velocity_x[idx] * velocity_x[idx] + velocity_z[idx] * velocity_z[idx]).sqrt();

                // Local tilt approximation
                let tilt = if ix > 0 && ix + 1 < n && iz > 0 && iz + 1 < n {
                    let dx = (terrain[idx + 1] - terrain[idx - 1]) / (2.0 * cell_size);
                    let dz = (terrain[idx + n] - terrain[idx - n]) / (2.0 * cell_size);
                    (dx * dx + dz * dz).sqrt().max(MIN_TILT)
                } else {
                    MIN_TILT
                };

                let capacity = SEDIMENT_CAPACITY * speed * tilt;

                if sediment[idx] > capacity {
                    // Deposit excess sediment
                    let deposit = (sediment[idx] - capacity) * DEPOSITION_RATE * dt;
                    terrain[idx] += deposit;
                    sediment[idx] -= deposit;
                } else {
                    // Erode terrain
                    let erode = (capacity - sediment[idx]) * EROSION_RATE * dt;
                    let erode = erode.min(terrain[idx] - 1.0); // don't erode below bedrock
                    terrain[idx] -= erode;
                    sediment[idx] += erode;
                }
            }
        }

        // 6. Sediment transport (simple advection)
        let mut new_sediment = vec![0.0f32; n * n];
        for iz in 0..n {
            for ix in 0..n {
                let idx = ix + iz * n;
                let sx = ix as f32 - velocity_x[idx] * dt / cell_size;
                let sz = iz as f32 - velocity_z[idx] * dt / cell_size;
                let sx = sx.clamp(0.0, (n - 1) as f32);
                let sz = sz.clamp(0.0, (n - 1) as f32);
                let si = sx as usize;
                let sj = sz as usize;
                let si = si.min(n - 2);
                let sj = sj.min(n - 2);
                let tx = sx - si as f32;
                let tz = sz - sj as f32;
                new_sediment[idx] = sediment[si + sj * n] * (1.0 - tx) * (1.0 - tz)
                    + sediment[(si + 1) + sj * n] * tx * (1.0 - tz)
                    + sediment[si + (sj + 1) * n] * (1.0 - tx) * tz
                    + sediment[(si + 1) + (sj + 1) * n] * tx * tz;
            }
        }
        sediment = new_sediment;

        // 7. Evaporation
        for w in water.iter_mut() {
            *w *= 1.0 - EVAPORATION_RATE * dt;
        }
    }

    // Compute erosion delta (final - initial)
    let mut grid = Vec::with_capacity(n * n);
    for i in 0..n * n {
        grid.push(terrain[i] - initial_terrain[i]);
    }

    ErosionMap {
        grid,
        size,
        cell_size,
        origin_x,
        origin_z,
    }
}

// --- Tectonic plate simulation ---

const NUM_PLATES: usize = 20;
const MAX_UPLIFT: f32 = 250.0; // max height boost at convergent boundaries
const MAX_RIFT: f32 = 30.0; // max depth at divergent boundaries
const BOUNDARY_WIDTH: f32 = 80.0; // cells over which stress spreads

/// Simple deterministic RNG from seed (xorshift32).
fn xorshift(mut state: u32) -> (u32, u32) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    (state, state)
}

/// Returns a float in [-1, 1] from a u32.
fn rand_f32(v: u32) -> f32 {
    (v as f32 / u32::MAX as f32) * 2.0 - 1.0
}

/// Generate tectonic uplift map. Returns per-cell height offset.
fn generate_tectonic_uplift(size: usize, _cell_size: f32, _origin_x: f32, _origin_z: f32, seed: u32) -> Vec<f32> {
    // Generate plate seed points and velocities
    let mut rng = seed.wrapping_mul(2654435761);
    let mut plates: Vec<([f32; 2], [f32; 2])> = Vec::with_capacity(NUM_PLATES);
    for _ in 0..NUM_PLATES {
        let (r, next) = xorshift(rng);
        rng = next;
        let px = (r % size as u32) as f32;
        let (r, next) = xorshift(rng);
        rng = next;
        let pz = (r % size as u32) as f32;
        let (r, next) = xorshift(rng);
        rng = next;
        let vx = rand_f32(r);
        let (r, next) = xorshift(rng);
        rng = next;
        let vz = rand_f32(r);
        plates.push(([px, pz], [vx, vz]));
    }

    // Assign each cell to nearest plate (Voronoi)
    let mut plate_ids = vec![0u16; size * size];
    for iz in 0..size {
        for ix in 0..size {
            let mut best_dist = f32::MAX;
            let mut best_id = 0u16;
            for (id, (center, _)) in plates.iter().enumerate() {
                let dx = ix as f32 - center[0];
                let dz = iz as f32 - center[1];
                let dist = dx * dx + dz * dz;
                if dist < best_dist {
                    best_dist = dist;
                    best_id = id as u16;
                }
            }
            plate_ids[ix + iz * size] = best_id;
        }
    }

    // Compute boundary stress per cell
    let mut uplift = vec![0.0f32; size * size];
    for iz in 1..size - 1 {
        for ix in 1..size - 1 {
            let idx = ix + iz * size;
            let my_plate = plate_ids[idx];

            // Check 4 neighbors for plate boundary
            let neighbors = [
                plate_ids[ix + (iz - 1) * size],
                plate_ids[(ix + 1) + iz * size],
                plate_ids[ix + (iz + 1) * size],
                plate_ids[(ix - 1) + iz * size],
            ];

            for &nb_plate in &neighbors {
                if nb_plate == my_plate {
                    continue;
                }
                // Boundary found — compute stress from relative velocity
                let (_, my_vel) = plates[my_plate as usize];
                let (_, nb_vel) = plates[nb_plate as usize];
                let rel_vx = my_vel[0] - nb_vel[0];
                let rel_vz = my_vel[1] - nb_vel[1];

                // Direction from this cell toward neighbor plate center
                let nb_center = plates[nb_plate as usize].0;
                let dx = nb_center[0] - ix as f32;
                let dz = nb_center[1] - iz as f32;
                let len = (dx * dx + dz * dz).sqrt().max(1.0);
                let nx = dx / len;
                let nz = dz / len;

                // Stress = relative velocity projected onto boundary normal
                // Positive = convergent (plates pushing together) = uplift
                // Negative = divergent (plates pulling apart) = rift
                let stress = rel_vx * nx + rel_vz * nz;

                if stress > 0.0 {
                    uplift[idx] += stress * MAX_UPLIFT;
                } else {
                    uplift[idx] += stress * MAX_RIFT;
                }
                break; // one boundary neighbor is enough
            }
        }
    }

    // Spread boundary stress outward (distance falloff)
    let spread = BOUNDARY_WIDTH;
    let mut spread_uplift = vec![0.0f32; size * size];

    // Collect boundary cells for distance-based spread
    let mut boundary_cells: Vec<(usize, usize, f32)> = Vec::new();
    for iz in 0..size {
        for ix in 0..size {
            let v = uplift[ix + iz * size];
            if v.abs() > 0.1 {
                boundary_cells.push((ix, iz, v));
            }
        }
    }

    // For each cell, accumulate influence from nearby boundary cells
    // Use a coarse pass (check every 4th cell) for performance on 2048²
    let step = 4;
    for iz in (0..size).step_by(step) {
        for ix in (0..size).step_by(step) {
            let mut total = 0.0f32;
            let mut weight_sum = 0.0f32;
            for &(bx, bz, stress) in &boundary_cells {
                let dx = (ix as f32 - bx as f32).abs();
                let dz = (iz as f32 - bz as f32).abs();
                let dist = (dx * dx + dz * dz).sqrt();
                if dist < spread {
                    let w = 1.0 - dist / spread;
                    let w = w * w; // quadratic falloff
                    total += stress * w;
                    weight_sum += w;
                }
            }
            if weight_sum > 0.0 {
                let value = total / weight_sum;
                // Fill the step×step block
                for dz in 0..step {
                    for dx in 0..step {
                        let fx = ix + dx;
                        let fz = iz + dz;
                        if fx < size && fz < size {
                            spread_uplift[fx + fz * size] = value;
                        }
                    }
                }
            }
        }
    }

    spread_uplift
}
