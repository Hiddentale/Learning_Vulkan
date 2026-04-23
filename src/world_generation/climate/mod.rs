use std::collections::VecDeque;
use crate::world_generation::terrain_amplification::cross_layout::CrossLayout;

/// Maximum ocean distance (in pixels) before continentality saturates.
/// At ~19.5 km/pixel cross-grid resolution, 150 pixels ≈ 2900 km.
const MAX_OCEAN_DIST_PIX: f32 = 150.0;

/// Climate data at every pixel on the amplified cross grid.
pub struct ClimateMap {
    /// Temperature in degrees Celsius, row-major.
    pub temperature: Vec<f32>,
    /// Annual precipitation in mm, row-major.
    pub precipitation: Vec<f32>,
    /// Continentality: 0.0 (coast) to 1.0 (interior), row-major.
    pub continentality: Vec<f32>,
    pub width: u32,
    pub height: u32,
}

/// Compute climate at every cross-grid pixel.
///
/// Runs after river networking. Takes elevation (post-carving) and returns
/// temperature, precipitation, and continentality derived from latitude,
/// altitude, and ocean proximity.
pub fn compute(elevation: &[f32], width: u32, height: u32) -> ClimateMap {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;

    // Compute ocean distance via BFS from ocean pixels
    let ocean_dist = ocean_distance(elevation, w, h);

    // Set up cross-layout for pixel-to-sphere mapping
    let face_size = width / 4;
    let cross = CrossLayout::new(face_size);
    let pixel_size_rad = std::f64::consts::PI / 2.0 / face_size as f64;

    let mut temperature = vec![0.0f32; n];
    let mut precipitation = vec![0.0f32; n];
    let mut continentality = vec![0.0f32; n];

    for idx in 0..n {
        let r = (idx / w) as u32;
        let c = (idx % w) as u32;

        // Skip dead-zone pixels (corners of the cross)
        if !cross.is_active(r, c) {
            temperature[idx] = 15.0;
            precipitation[idx] = 600.0;
            continentality[idx] = 0.5;
            continue;
        }

        // Get sphere direction from pixel coordinates
        let Some(dir) = cross.pixel_to_sphere(r, c) else {
            temperature[idx] = 15.0;
            precipitation[idx] = 600.0;
            continentality[idx] = 0.5;
            continue;
        };

        let latitude = dir.y.asin();
        let elev_m = elevation[idx];
        let dist_pix = ocean_dist[idx];

        // Continentality: distance to ocean in pixels, clamped and normalized
        let cont = if dist_pix.is_infinite() {
            1.0
        } else {
            (dist_pix / MAX_OCEAN_DIST_PIX).min(1.0)
        };
        continentality[idx] = cont;

        // Temperature: latitude and altitude with continental modulation
        temperature[idx] = compute_temperature(latitude, elev_m, cont);

        // Precipitation: latitude bands + coastal enhancement
        let dist_ocean_rad = if dist_pix.is_infinite() {
            1e6_f64
        } else {
            dist_pix as f64 * pixel_size_rad
        };
        precipitation[idx] = compute_precipitation(latitude, dist_ocean_rad as f32);
    }

    ClimateMap {
        temperature,
        precipitation,
        continentality,
        width,
        height,
    }
}

/// Compute per-pixel distance to ocean via 8-neighbor BFS.
/// Ocean pixels (elevation <= 0 or NaN) get distance 0.
/// Returns f32::INFINITY for unreachable pixels (dead zones).
fn ocean_distance(elevation: &[f32], w: usize, h: usize) -> Vec<f32> {
    let n = w * h;
    let mut dist = vec![f32::INFINITY; n];
    let mut queue = VecDeque::new();

    // Seed: ocean pixels
    for idx in 0..n {
        if elevation[idx].is_nan() || elevation[idx] <= 0.0 {
            dist[idx] = 0.0;
            queue.push_back(idx);
        }
    }

    // BFS with 8-neighbor connectivity
    const DR: [i32; 8] = [-1, 1, 0, 0, -1, -1, 1, 1];
    const DC: [i32; 8] = [0, 0, -1, 1, -1, 1, -1, 1];

    while let Some(idx) = queue.pop_front() {
        let r = idx / w;
        let c = idx % w;
        let d = dist[idx];

        for k in 0..8 {
            let nr = r as i32 + DR[k];
            let nc = c as i32 + DC[k];
            if nr < 0 || nr >= h as i32 || nc < 0 || nc >= w as i32 {
                continue;
            }
            let nidx = nr as usize * w + nc as usize;
            let new_dist = d + 1.0;

            if new_dist < dist[nidx] {
                dist[nidx] = new_dist;
                queue.push_back(nidx);
            }
        }
    }

    dist
}

/// Compute temperature (°C) from latitude, elevation, and continentality.
fn compute_temperature(latitude: f64, elevation_m: f32, continentality: f32) -> f32 {
    // Base temperature: warmer at equator, colder at poles
    // Formula: 28°C at equator, -27°C at poles (55°C drop)
    let lat_frac = latitude / (std::f64::consts::PI / 2.0);
    let base_temp = 28.0 - 55.0 * lat_frac * lat_frac;

    // Altitude lapse: -6.5°C/km
    let elev_km = (elevation_m / 1000.0).max(0.0) as f64;
    let lapse_temp = base_temp - 6.5 * elev_km;

    // Continental effect: interiors colder and more extreme than coasts
    // Roughly: -15°C at |lat|=60° for fully continental, 0°C effect at equator/poles
    let cont_effect = -15.0 * (continentality as f64) * latitude.sin().abs();
    let temp = lapse_temp + cont_effect;

    temp as f32
}

/// Compute precipitation (mm/year) from latitude and ocean distance.
fn compute_precipitation(latitude: f64, dist_ocean_rad: f32) -> f32 {
    // Latitude-based precipitation bands (ported from coarse_heightmap/climate.rs)
    let abs_lat = latitude.abs();

    // ITCZ: gaussian peak at equator (|lat| < 10°)
    let itcz = (-8.0 * abs_lat * abs_lat).exp();

    // Mid-latitude westerlies: peak at ~50° latitude
    let midlat = 0.6 * (-8.0 * (abs_lat - 1.05).powi(2)).exp();

    // Base factor from latitude
    let lat_factor = itcz + midlat + 0.15;

    // Coastal enhancement: exponential decay with distance from ocean
    // Half-width ~0.15 radians (~1500 km at earth scale)
    let dist_ocean_rad_f64 = dist_ocean_rad as f64;
    let coastal_factor = (-dist_ocean_rad_f64 / 0.15).exp();

    // Combine: inland areas get drying applied as multiplicative factor
    let continental_drying = 1.0 - 0.6 * smoothstep(0.0, 20.0, dist_ocean_rad_f64);
    let combined = lat_factor * (0.4 + 0.6 * coastal_factor + continental_drying * 0.5);

    ((2500.0 * combined).max(50.0)) as f32
}

/// Smoothstep interpolation: smooth from 0 at edge0 to 1 at edge1.
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world_generation::coarse_heightmap;
    use crate::world_generation::sphere_geometry::fibonnaci_spiral::SphericalFibonacci;
    use crate::world_generation::sphere_geometry::plate_seed_placement::{assign_plates, Adjacency};
    use crate::world_generation::sphere_geometry::spherical_delaunay_triangulation::SphericalDelaunay;
    use crate::world_generation::terrain_amplification;
    use crate::world_generation::volcanic_overlay;
    use crate::world_generation::river_networking;

    #[test]
    fn equator_is_warmest() {
        // At sea level and zero continentality, equator should be warmest
        let temp_eq = compute_temperature(0.0, 0.0, 0.0);
        let temp_pole = compute_temperature(std::f64::consts::PI / 2.0, 0.0, 0.0);
        assert!(temp_eq > temp_pole, "equator {temp_eq}°C should be warmer than pole {temp_pole}°C");
    }

    #[test]
    fn altitude_cools() {
        // Higher altitude = colder at same latitude
        let temp_sea = compute_temperature(0.0, 0.0, 0.0);
        let temp_high = compute_temperature(0.0, 3000.0, 0.0);
        assert!(temp_high < temp_sea, "3km altitude {temp_high}°C should be colder than sea level {temp_sea}°C");
    }

    #[test]
    fn continentality_cools() {
        // Continental interior colder than coast at same lat/elev
        let temp_coast = compute_temperature(0.5, 0.0, 0.0);
        let temp_interior = compute_temperature(0.5, 0.0, 1.0);
        assert!(temp_interior < temp_coast, "interior {temp_interior}°C should be colder than coast {temp_coast}°C");
    }

    #[test]
    fn coast_is_wetter() {
        // Low continentality (coast) → higher precipitation than high continentality
        let precip_coast = compute_precipitation(0.3, 1.0); // near coast
        let precip_interior = compute_precipitation(0.3, 50.0); // far inland
        assert!(precip_coast > precip_interior, "coast {precip_coast}mm should be wetter than interior {precip_interior}mm");
    }

    #[test]
    fn ocean_pixel_gets_zero_distance() {
        let elevation = vec![
            -100.0, -100.0, -100.0,
            -100.0,    5.0, -100.0,
            -100.0, -100.0, -100.0,
        ];
        let dist = ocean_distance(&elevation, 3, 3);
        // All ocean cells (corners) should have distance 0
        assert_eq!(dist[0], 0.0);
        assert_eq!(dist[2], 0.0);
        assert_eq!(dist[6], 0.0);
        assert_eq!(dist[8], 0.0);
        // Center land cell should have distance > 0
        assert!(dist[4] > 0.0, "center land should have positive distance");
    }

    #[test]
    fn land_gets_positive_distance() {
        let elevation = vec![
            -100.0, -100.0, -100.0,
            -100.0,    5.0, -100.0,
            -100.0, -100.0, -100.0,
        ];
        let dist = ocean_distance(&elevation, 3, 3);
        let center_land = dist[4]; // (1,1)
        assert!(center_land > 0.0, "land should have positive ocean distance");
        assert!(center_land <= 2.0, "center land should be very close to ocean");
    }

    #[test]
    fn smoothstep_is_smooth() {
        assert_eq!(smoothstep(0.0, 10.0, -5.0), 0.0); // before edge0
        assert_eq!(smoothstep(0.0, 10.0, 15.0), 1.0); // after edge1
        let mid = smoothstep(0.0, 10.0, 5.0);
        assert!(mid > 0.0 && mid < 1.0, "mid should be between 0 and 1");
    }

    /// Full pipeline: coarse → amplify → volcanic → river → climate → PNG export.
    /// Run with: cargo test --release -- climate_ply --ignored --nocapture
    #[test]
    #[ignore]
    fn climate_ply() {
        let seed = 137u64;
        let fib = SphericalFibonacci::new(10_000);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, 40, seed);
        let adjacency = Adjacency::from_delaunay(points.len(), &del);
        let coarse = coarse_heightmap::generate(&points, &assignment, &adjacency, seed);

        let model_dir = std::path::Path::new("data/models/terrain-diffusion-30m");
        let mut terrain = terrain_amplification::amplify(&coarse, &points, &fib, seed, model_dir).expect("amplification failed");

        println!("\n  Applying volcanic overlay on cross grid...");
        volcanic_overlay::overlay(&mut terrain, &coarse, &points, seed);

        println!("  Running river networking on cross-layout ({}×{})...", terrain.cross_width, terrain.cross_height);
        let _flow = river_networking::process(&mut terrain.cross_elevation, terrain.cross_width, terrain.cross_height);

        println!("  Computing climate (Tier 1)...");
        let climate = compute(&terrain.cross_elevation, terrain.cross_width, terrain.cross_height);

        let max_temp = climate.temperature.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min_temp = climate.temperature.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_precip = climate.precipitation.iter().cloned().fold(0.0f32, f32::max);
        let max_cont = climate.continentality.iter().cloned().fold(0.0f32, f32::max);

        println!("  Climate: T=[{min_temp:.1}°C, {max_temp:.1}°C], P=[0, {max_precip:.0}mm], Cont=[0, {max_cont:.2}]");

        // Export temperature as PNG
        println!("  Exporting temperature map...");
        export_temperature_png(
            &climate.temperature,
            terrain.cross_width as usize,
            terrain.cross_height as usize,
            "src/world_generation/climate/cross_temperature.png",
        );

        // Export precipitation as PNG
        println!("  Exporting precipitation map...");
        export_precipitation_png(
            &climate.precipitation,
            terrain.cross_width as usize,
            terrain.cross_height as usize,
            "src/world_generation/climate/cross_precipitation.png",
        );

        println!("\n  Climate computation complete!");
    }

    fn export_temperature_png(temperature: &[f32], w: usize, h: usize, path: &str) {
        let mut img = image::RgbImage::new(w as u32, h as u32);
        for r in 0..h {
            for c in 0..w {
                let t = temperature[r * w + c];
                let rgb = temperature_color(t);
                img.put_pixel(c as u32, r as u32, image::Rgb(rgb));
            }
        }
        img.save(path).expect("failed to save PNG");
        let size = std::fs::metadata(path).unwrap().len() as f64 / 1024.0;
        println!("  Wrote {} ({:.0} KB, {}×{})", path, size, w, h);
    }

    fn export_precipitation_png(precipitation: &[f32], w: usize, h: usize, path: &str) {
        let mut img = image::RgbImage::new(w as u32, h as u32);
        for r in 0..h {
            for c in 0..w {
                let p = precipitation[r * w + c];
                let rgb = precipitation_color(p);
                img.put_pixel(c as u32, r as u32, image::Rgb(rgb));
            }
        }
        img.save(path).expect("failed to save PNG");
        let size = std::fs::metadata(path).unwrap().len() as f64 / 1024.0;
        println!("  Wrote {} ({:.0} KB, {}×{})", path, size, w, h);
    }

    /// Temperature colormap: cold (blue) to hot (red).
    fn temperature_color(temp: f32) -> [u8; 3] {
        if temp < -20.0 {
            [0, 0, 255] // dark blue
        } else if temp < 0.0 {
            let t = (temp + 20.0) / 20.0;
            [0, (t * 128.0) as u8, 255] // cyan
        } else if temp < 15.0 {
            let t = temp / 15.0;
            [0, (255.0 * (1.0 - t * 0.5)) as u8, (255.0 * (1.0 - t)) as u8] // cyan to green
        } else if temp < 25.0 {
            let t = (temp - 15.0) / 10.0;
            [(t * 255.0) as u8, 255, 0] // green to yellow
        } else {
            let t = ((temp - 25.0) / 20.0).min(1.0);
            [255, (255.0 * (1.0 - t * 0.8)) as u8, 0] // yellow to red
        }
    }

    /// Precipitation colormap: dry (yellow) to wet (blue).
    fn precipitation_color(precip: f32) -> [u8; 3] {
        let t = (precip / 2500.0).clamp(0.0, 1.0);
        if t < 0.3 {
            // dry: yellow
            let s = t / 0.3;
            [255, 255, 0]
        } else if t < 0.5 {
            // yellow to green
            let s = (t - 0.3) / 0.2;
            [(255.0 * (1.0 - s * 0.8)) as u8, 255, 0]
        } else if t < 0.8 {
            // green to cyan
            let s = (t - 0.5) / 0.3;
            [0, (255.0 * (1.0 - s)) as u8, (255.0 * s) as u8]
        } else {
            // cyan to blue
            let s = (t - 0.8) / 0.2;
            [0, (100.0 * (1.0 - s)) as u8, 255]
        }
    }
}
