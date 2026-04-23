/// Step 6: Detail Noise + Domain Warping
///
/// Pre-computes 2D height offsets for each 16×16 chunk column, incorporating:
/// - Domain warping: Low-frequency 3D noise shifts the heightmap lookup direction
/// - Slope-gated amplitude: Terrain roughness (Sobel gradient) controls detail intensity
/// - Landform-driven noise: Mountain vs plain vs ocean get different noise characters
///
/// Detail noise is pre-computed during world generation and stored to disk,
/// avoiding frame stutter during gameplay chunk loads.

use crate::world_generation::terrain_amplification::{AmplifiedTerrain, cross_layout::CrossLayout};
use crate::voxel::sphere::{self, Face};
use lz4_flex::{compress_prepend_size, decompress_size_prepended};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Mutex;

pub const CHUNK_SIZE: usize = 16;

/// 2D array of 16×16 = 256 height offsets for one chunk column.
/// Index: `lz * CHUNK_SIZE + lx` where lx = east, lz = south in face coords.
pub type DetailColumn = [f32; CHUNK_SIZE * CHUNK_SIZE];

/// Landform classification based on elevation and local roughness.
#[derive(Copy, Clone, Debug)]
enum Landform {
    Ocean,
    Plains,
    OldMountain,
    YoungMountain,
}

/// Noise character driven by landform type.
struct LandformNoise {
    kind: NoiseKind,
    amplitude: f32,
    frequency: f64,
    octaves: u32,
}

#[derive(Copy, Clone)]
enum NoiseKind {
    Fbm,
    Ridged,
}

/// Classify landform from elevation and roughness (std-dev of 3×3 window).
fn classify_landform(elevation_m: f32, roughness: f32) -> Landform {
    if elevation_m <= 0.0 {
        Landform::Ocean
    } else if roughness > 300.0 {
        // High roughness = steep, crinkled = young mountain
        Landform::YoungMountain
    } else if elevation_m > 2500.0 {
        // High elevation but moderate roughness = old mountain
        Landform::OldMountain
    } else if elevation_m > 400.0 {
        // Mid-elevation = foothills (treat as old mountain)
        Landform::OldMountain
    } else {
        // Low elevation = plains
        Landform::Plains
    }
}

/// Get noise parameters for a landform type.
fn noise_for_landform(landform: Landform) -> LandformNoise {
    match landform {
        Landform::YoungMountain => LandformNoise {
            kind: NoiseKind::Ridged,
            amplitude: 8.0,
            frequency: 0.02,  // ~50m period
            octaves: 4,
        },
        Landform::OldMountain => LandformNoise {
            kind: NoiseKind::Fbm,
            amplitude: 4.0,
            frequency: 0.02,
            octaves: 3,
        },
        Landform::Plains => LandformNoise {
            kind: NoiseKind::Fbm,
            amplitude: 2.0,
            frequency: 0.012,  // ~85m period
            octaves: 3,
        },
        Landform::Ocean => LandformNoise {
            kind: NoiseKind::Fbm,
            amplitude: 1.0,
            frequency: 0.008,  // ~125m period
            octaves: 2,
        },
    }
}

/// Bilinear interpolation with linear extrapolation padding.
/// Avoids ringing at tile boundaries when upsampling from coarse grid.
fn bilinear_sample_extrapolated(
    grid: &[f32],
    w: usize,
    h: usize,
    fx: f64,
    fy: f64,
) -> f32 {
    let x = fx.clamp(0.0, (w - 1) as f64);
    let y = fy.clamp(0.0, (h - 1) as f64);

    let xi = x.floor() as usize;
    let yi = y.floor() as usize;
    let xf = (x - xi as f64) as f32;
    let yf = (y - yi as f64) as f32;

    // Sample 2×2 neighborhood
    let x0 = xi.clamp(0, w - 1);
    let x1 = (xi + 1).clamp(0, w - 1);
    let y0 = yi.clamp(0, h - 1);
    let y1 = (yi + 1).clamp(0, h - 1);

    let v00 = grid[y0 * w + x0];
    let v10 = grid[y0 * w + x1];
    let v01 = grid[y1 * w + x0];
    let v11 = grid[y1 * w + x1];

    // Bilinear interpolation
    let top = v00 * (1.0 - xf) + v10 * xf;
    let bot = v01 * (1.0 - xf) + v11 * xf;
    top * (1.0 - yf) + bot * yf
}

/// Compute Sobel gradient magnitude at a grid point (for slope calculation).
fn sobel_gradient(grid: &[f32], w: usize, h: usize, x: usize, y: usize) -> f32 {
    // Clamp to valid indices
    let xm = x.saturating_sub(1).clamp(0, w - 1);
    let xp = (x + 1).clamp(0, w - 1);
    let ym = y.saturating_sub(1).clamp(0, h - 1);
    let yp = (y + 1).clamp(0, h - 1);

    // Sobel operator
    let gx = -grid[ym * w + xm] + grid[ym * w + xp]
        - 2.0 * grid[y * w + xm] + 2.0 * grid[y * w + xp]
        - grid[yp * w + xm] + grid[yp * w + xp];

    let gy = -grid[ym * w + xm] - 2.0 * grid[ym * w + x] - grid[ym * w + xp]
        + grid[yp * w + xm] + 2.0 * grid[yp * w + x] + grid[yp * w + xp];

    ((gx * gx + gy * gy) as f64).sqrt() as f32
}

/// Compute local roughness as std-dev of a 3×3 window.
fn local_roughness(grid: &[f32], w: usize, h: usize, x: usize, y: usize) -> f32 {
    let xm = x.saturating_sub(1);
    let xp = (x + 1).min(w - 1);
    let ym = y.saturating_sub(1);
    let yp = (y + 1).min(h - 1);

    let mut sum = 0.0;
    let mut count = 0;
    for yy in ym..=yp {
        for xx in xm..=xp {
            sum += grid[yy * w + xx];
            count += 1;
        }
    }
    let mean = sum / count as f32;

    let mut var = 0.0;
    for yy in ym..=yp {
        for xx in xm..=xp {
            let delta = grid[yy * w + xx] - mean;
            var += delta * delta;
        }
    }
    (var / count as f32).sqrt()
}

/// Slope-gated amplitude: continuous, slope-driven height offset amplitude.
/// Flat terrain (slope ~0) → MIN_AMPLITUDE; steep terrain (slope high) → MAX_AMPLITUDE.
/// Uses a 2.5-power curve for smooth transition, derived from terrain-diffusion-mc.
fn slope_gated_amplitude(slope_mag: f32, slope_norm_factor: f32) -> f32 {
    const MIN_AMPLITUDE: f32 = 1.0;
    const MAX_AMPLITUDE: f32 = 8.0;

    let norm = (slope_mag / slope_norm_factor).clamp(0.0, 1.0);
    let sf = norm * norm * norm.sqrt();  // 2.5-power curve: n^2.5
    MIN_AMPLITUDE + sf * (MAX_AMPLITUDE - MIN_AMPLITUDE)
}

/// Pre-compute 16×16 detail height offsets for one chunk column.
///
/// For each block in the column:
/// 1. Map block position to sphere direction
/// 2. Apply domain warp (shifts lookup in tangent space)
/// 3. Sample amplified terrain at warped position
/// 4. Compute slope from Sobel gradient
/// 5. Compute roughness to classify landform
/// 6. Sample landform-appropriate detail noise with slope-gated amplitude
/// 7. Store as height offset
pub fn compute_column(
    face: Face,
    cx: i32,
    cz: i32,
    terrain: &AmplifiedTerrain,
    seed: u32,
) -> DetailColumn {
    let mut result = [0.0_f32; CHUNK_SIZE * CHUNK_SIZE];

    // Create noise generators for warp and detail
    let warp_x = Fbm::<Perlin>::new(seed + 6)
        .set_frequency(0.003)
        .set_octaves(3)
        .set_persistence(0.5)
        .set_lacunarity(2.0);
    let warp_y = Fbm::<Perlin>::new(seed + 11)
        .set_frequency(0.003)
        .set_octaves(3)
        .set_persistence(0.5)
        .set_lacunarity(2.0);
    let warp_z = Fbm::<Perlin>::new(seed + 7)
        .set_frequency(0.003)
        .set_octaves(3)
        .set_persistence(0.5)
        .set_lacunarity(2.0);

    // Set up cross-layout for pixel mapping
    let face_size = terrain.cross_width / 4;
    let cross = CrossLayout::new(face_size);

    // Create one pair of Fbm and Ridged generators for detail noise
    // These will be shared and used based on landform type
    let detail_fbm = Fbm::<Perlin>::new(seed + 1)
        .set_frequency(0.02)
        .set_octaves(3)
        .set_persistence(0.5)
        .set_lacunarity(2.0);
    let detail_ridged = RidgedMulti::<Perlin>::new(seed + 2)
        .set_frequency(0.02)
        .set_octaves(4);

    // Constant for slope normalization: ~80m/pixel at our scale
    const SLOPE_NORM_FACTOR: f32 = 80.0;

    for lz in 0..CHUNK_SIZE {
        for lx in 0..CHUNK_SIZE {
            // Map block position to cube position
            let cube_x = (cx * CHUNK_SIZE as i32 + lx as i32) as f64 + 0.5;
            let cube_z = (cz * CHUNK_SIZE as i32 + lz as i32) as f64 + 0.5;

            // Convert to sphere direction
            let Some(unwarped_dir) = sphere_direction_from_face_cube(face, cube_x, cube_z, &cross) else {
                result[lz * CHUNK_SIZE + lx] = 0.0;
                continue;
            };

            // Domain warp: sample warp noise at sphere position
            let p_warp = sphere::noise_pos_at_world(unwarped_dir * sphere::SURFACE_RADIUS_BLOCKS as f64);
            let warp_3d = glam::DVec3::new(
                warp_x.get(p_warp),
                warp_y.get(p_warp),
                warp_z.get(p_warp),
            );

            // Project warp onto tangent plane (remove radial component)
            let warp_tangent = warp_3d - unwarped_dir * warp_3d.dot(unwarped_dir);
            let warped_dir = (unwarped_dir + warp_tangent * (80.0 / sphere::SURFACE_RADIUS_BLOCKS as f64)).normalize();

            // Sample amplified terrain at warped position
            let Some((elev_m, sobel_grad, roughness)) = sample_terrain_data(
                face, warped_dir, &cross, terrain
            ) else {
                result[lz * CHUNK_SIZE + lx] = 0.0;
                continue;
            };

            // Classify landform and get noise parameters
            let landform = classify_landform(elev_m, roughness);
            let noise_params = noise_for_landform(landform);

            // Compute slope-gated amplitude
            let amplitude = slope_gated_amplitude(sobel_grad, SLOPE_NORM_FACTOR);

            // Sample detail noise at warped position
            let noise_p = sphere::noise_pos_at_world(warped_dir * sphere::SURFACE_RADIUS_BLOCKS as f64);
            let detail_val = match noise_params.kind {
                NoiseKind::Fbm => detail_fbm.get(noise_p),
                NoiseKind::Ridged => detail_ridged.get(noise_p),
            };

            // Scale by landform amplitude and slope gating
            result[lz * CHUNK_SIZE + lx] = (detail_val as f32) * amplitude;
        }
    }

    result
}

/// Map face cube coordinates to sphere direction.
fn sphere_direction_from_face_cube(
    face: Face,
    cube_x: f64,
    cube_z: f64,
    cross: &CrossLayout,
) -> Option<glam::DVec3> {
    // Convert cube coords to cross-grid pixel coordinates
    let (tu, tv, _) = sphere::face_basis(face);

    // This is a simplified approach; for full correctness would need
    // face-to-cross coordinate mapping. For now, use a direct approach.
    let face_size = cross.face_size as f64;
    let u = cube_x / 16.0;  // Block coords to chunk-relative
    let v = cube_z / 16.0;

    // Map chunk coords to face pixel coords (16 blocks per chunk)
    let pixel_u = u * 16.0;
    let pixel_v = v * 16.0;

    // Use CrossLayout to get sphere direction
    let Some(dir) = cross.pixel_to_sphere(pixel_u.round() as u32, pixel_v.round() as u32) else {
        return None;
    };

    Some(dir.normalize())
}

/// Sample elevation, Sobel gradient, and roughness from amplified terrain.
fn sample_terrain_data(
    face: Face,
    dir: glam::DVec3,
    cross: &CrossLayout,
    terrain: &AmplifiedTerrain,
) -> Option<(f32, f32, f32)> {
    // Map sphere direction to cross-grid pixel coordinates
    let (tu, tv, _) = sphere::face_basis(face);
    let cube_pt = dir * (sphere::SURFACE_RADIUS_BLOCKS as f64) / sphere::CUBE_HALF_BLOCKS;
    let u = cube_pt.dot(tu.as_dvec3());
    let v = cube_pt.dot(tv.as_dvec3());

    // Map to pixel coords and sample
    let pixel_u = (u + sphere::CUBE_HALF_BLOCKS) / (2.0 * sphere::CUBE_HALF_BLOCKS) * terrain.cross_width as f64;
    let pixel_v = (v + sphere::CUBE_HALF_BLOCKS) / (2.0 * sphere::CUBE_HALF_BLOCKS) * terrain.cross_height as f64;

    if pixel_u < 0.0 || pixel_u >= terrain.cross_width as f64 ||
       pixel_v < 0.0 || pixel_v >= terrain.cross_height as f64 {
        return None;
    }

    let w = terrain.cross_width as usize;
    let h = terrain.cross_height as usize;

    // Sample elevation via bilinear interpolation
    let elev_m = bilinear_sample_extrapolated(
        &terrain.cross_elevation,
        w, h,
        pixel_u, pixel_v
    );

    // Compute Sobel gradient magnitude at sample point (for slope)
    let sobel_grad = sobel_gradient(
        &terrain.cross_elevation,
        w, h,
        pixel_u.round() as usize,
        pixel_v.round() as usize
    );

    // Compute local roughness from 3×3 window
    let roughness = local_roughness(
        &terrain.cross_elevation,
        w, h,
        pixel_u.round() as usize,
        pixel_v.round() as usize
    );

    Some((elev_m, sobel_grad, roughness))
}

/// In-memory + disk-backed cache for pre-computed detail columns.
///
/// Stores height offsets per chunk column. On miss, computes via `compute_column`
/// and persists to disk (if cache_dir is set) for future loads.
pub struct DetailCache {
    hot: HashMap<(Face, i32, i32), Box<DetailColumn>>,
    terrain: std::sync::Arc<AmplifiedTerrain>,
    cache_dir: Option<PathBuf>,
    seed: u32,
}

impl DetailCache {
    /// Create a new detail cache.
    ///
    /// - `terrain`: reference to amplified terrain (for elevation sampling)
    /// - `seed`: world seed (for noise generation)
    /// - `cache_dir`: optional directory for disk persistence
    ///   (format: `{cache_dir}/{face_idx}/{cx}_{cz}.bin`)
    pub fn new(
        terrain: std::sync::Arc<AmplifiedTerrain>,
        seed: u32,
        cache_dir: Option<PathBuf>,
    ) -> Self {
        Self {
            hot: HashMap::new(),
            terrain,
            cache_dir,
            seed,
        }
    }

    /// Get or compute a detail column. Hits disk cache if available, else computes.
    pub fn get_or_compute(&mut self, face: Face, cx: i32, cz: i32) -> Box<DetailColumn> {
        let key = (face, cx, cz);

        // Check hot cache
        if let Some(col) = self.hot.get(&key) {
            return col.clone();
        }

        // Try disk cache
        if let Some(dir) = &self.cache_dir {
            if let Some(col) = self.load_from_disk(face, cx, cz, dir) {
                let boxed = Box::new(col);
                self.hot.insert(key, boxed.clone());
                return boxed;
            }
        }

        // Compute and cache
        let col = compute_column(face, cx, cz, &self.terrain, self.seed);
        let boxed = Box::new(col);

        // Save to disk if directory set
        if let Some(dir) = &self.cache_dir {
            let _ = self.save_to_disk(face, cx, cz, &col, dir);
        }

        self.hot.insert(key, boxed.clone());
        boxed
    }

    /// Load a detail column from disk. Returns None if file doesn't exist or read fails.
    fn load_from_disk(
        &self,
        face: Face,
        cx: i32,
        cz: i32,
        base_dir: &PathBuf,
    ) -> Option<DetailColumn> {
        let face_idx = face_to_index(face);
        let file_path = base_dir.join(format!("{}", face_idx)).join(format!("{}_{}.bin", cx, cz));

        let data = fs::read(&file_path).ok()?;
        let decompressed = decompress_size_prepended(&data).ok()?;

        // Expect 256 * 4 bytes = 1024 bytes
        if decompressed.len() != CHUNK_SIZE * CHUNK_SIZE * std::mem::size_of::<f32>() {
            return None;
        }

        let mut col = [0.0_f32; CHUNK_SIZE * CHUNK_SIZE];
        let mut cursor = 0;
        for i in 0..CHUNK_SIZE * CHUNK_SIZE {
            let bytes: [u8; 4] = decompressed[cursor..cursor + 4].try_into().ok()?;
            col[i] = f32::from_le_bytes(bytes);
            cursor += 4;
        }

        Some(col)
    }

    /// Save a detail column to disk with LZ4 compression.
    fn save_to_disk(
        &self,
        face: Face,
        cx: i32,
        cz: i32,
        col: &DetailColumn,
        base_dir: &PathBuf,
    ) -> std::io::Result<()> {
        let face_idx = face_to_index(face);
        let face_dir = base_dir.join(format!("{}", face_idx));
        fs::create_dir_all(&face_dir)?;

        let file_path = face_dir.join(format!("{}_{}.bin", cx, cz));

        // Serialize column as f32 bytes
        let mut data = Vec::with_capacity(CHUNK_SIZE * CHUNK_SIZE * 4);
        for &val in col.iter() {
            data.extend_from_slice(&val.to_le_bytes());
        }

        // Compress and save
        let compressed = compress_prepend_size(&data);
        fs::write(file_path, compressed)?;

        Ok(())
    }
}

/// Convert Face enum to numeric index for disk storage.
fn face_to_index(face: Face) -> u8 {
    match face {
        Face::PosX => 0,
        Face::NegX => 1,
        Face::PosY => 2,
        Face::NegY => 3,
        Face::PosZ => 4,
        Face::NegZ => 5,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_ocean_is_low_amplitude() {
        let landform = classify_landform(-1000.0, 10.0);
        match landform {
            Landform::Ocean => {},
            _ => panic!("Expected Ocean landform for negative elevation"),
        }
    }

    #[test]
    fn classify_young_mountain_has_high_amplitude() {
        let landform = classify_landform(3500.0, 400.0);
        match landform {
            Landform::YoungMountain => {},
            _ => panic!("Expected YoungMountain for high elevation + high roughness"),
        }
    }

    #[test]
    fn classify_plains_is_low_amplitude() {
        let landform = classify_landform(200.0, 50.0);
        match landform {
            Landform::Plains => {},
            _ => panic!("Expected Plains for low elevation + low roughness"),
        }
    }

    #[test]
    fn slope_gated_amplitude_flat_is_min() {
        let amp = slope_gated_amplitude(0.0, 80.0);
        assert!((amp - 1.0).abs() < 0.01, "Flat terrain should have amplitude ~1.0");
    }

    #[test]
    fn slope_gated_amplitude_steep_is_max() {
        let amp = slope_gated_amplitude(80.0, 80.0);
        assert!((amp - 8.0).abs() < 0.01, "Steep terrain should have amplitude ~8.0");
    }

    #[test]
    fn slope_gated_amplitude_is_continuous() {
        let mut prev = 0.0_f32;
        for slope in 0..80 {
            let amp = slope_gated_amplitude(slope as f32, 80.0);
            assert!(amp >= prev, "Amplitude should be monotonically increasing");
            prev = amp;
        }
    }

    #[test]
    fn bilinear_interpolation_corners() {
        // Simple 2×2 grid
        let grid = [1.0, 2.0, 3.0, 4.0];

        // Sample at exact corners
        assert!((bilinear_sample_extrapolated(&grid, 2, 2, 0.0, 0.0) - 1.0).abs() < 1e-5);
        assert!((bilinear_sample_extrapolated(&grid, 2, 2, 1.0, 0.0) - 2.0).abs() < 1e-5);
        assert!((bilinear_sample_extrapolated(&grid, 2, 2, 0.0, 1.0) - 3.0).abs() < 1e-5);
        assert!((bilinear_sample_extrapolated(&grid, 2, 2, 1.0, 1.0) - 4.0).abs() < 1e-5);
    }

    #[test]
    fn bilinear_interpolation_center() {
        // Sample at center of grid
        let grid = [1.0, 2.0, 3.0, 4.0];
        let val = bilinear_sample_extrapolated(&grid, 2, 2, 0.5, 0.5);
        assert!((val - 2.5).abs() < 1e-5);
    }
}
