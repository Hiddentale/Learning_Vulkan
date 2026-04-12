use std::f64::consts::PI;
use std::path::Path;

use glam::DVec3;
use image::{Rgb, RgbImage};

use super::fibonnaci_spiral::SphericalFibonacci;
use super::plate_seed_placement::{assign_plates, PlateAssignment};
use super::spherical_delaunay_triangulation::SphericalDelaunay;

const IMAGE_WIDTH: u32 = 2048;
const IMAGE_HEIGHT: u32 = 1024;
const POINT_COUNT: u32 = 10_000;
const PLATE_COUNT: u32 = 40;

/// Distinct, saturated colors for up to 40 plates.
const PALETTE: [[u8; 3]; 40] = [
    [230, 25, 75],   [60, 180, 75],   [255, 225, 25],  [0, 130, 200],
    [245, 130, 48],  [145, 30, 180],  [70, 240, 240],  [240, 50, 230],
    [210, 245, 60],  [250, 190, 212], [0, 128, 128],   [220, 190, 255],
    [170, 110, 40],  [255, 250, 200], [128, 0, 0],     [170, 255, 195],
    [128, 128, 0],   [255, 215, 180], [0, 0, 128],     [128, 128, 128],
    [200, 60, 60],   [60, 200, 120],  [200, 200, 60],  [60, 100, 200],
    [200, 130, 60],  [130, 60, 200],  [60, 200, 200],  [200, 60, 180],
    [160, 210, 80],  [210, 160, 180], [40, 160, 160],  [180, 160, 220],
    [160, 100, 60],  [220, 220, 170], [160, 40, 40],   [130, 220, 170],
    [160, 160, 40],  [220, 180, 160], [40, 40, 160],   [100, 100, 100],
];

fn latlon_to_direction(lat: f64, lon: f64) -> DVec3 {
    let cos_lat = lat.cos();
    DVec3::new(cos_lat * lon.cos(), cos_lat * lon.sin(), lat.sin())
}

/// Renders an equirectangular plate map to an RGB image.
pub fn render_plate_map(
    fibonacci: &SphericalFibonacci,
    assignment: &PlateAssignment,
) -> RgbImage {
    let mut img = RgbImage::new(IMAGE_WIDTH, IMAGE_HEIGHT);

    for py in 0..IMAGE_HEIGHT {
        let lat = PI / 2.0 - (py as f64 + 0.5) / IMAGE_HEIGHT as f64 * PI;
        for px in 0..IMAGE_WIDTH {
            let lon = (px as f64 + 0.5) / IMAGE_WIDTH as f64 * 2.0 * PI - PI;
            let dir = latlon_to_direction(lat, lon);
            let nearest = fibonacci.nearest_index(dir);
            let plate = assignment.plate_ids[nearest as usize];
            let color = PALETTE[plate as usize % PALETTE.len()];
            img.put_pixel(px, py, Rgb(color));
        }
    }

    // Draw seed points as white dots.
    for &seed_idx in &assignment.seeds {
        let p = fibonacci.index_to_point(seed_idx);
        let lat = p.z.asin();
        let lon = p.y.atan2(p.x);
        let px = ((lon + PI) / (2.0 * PI) * IMAGE_WIDTH as f64) as i32;
        let py = ((PI / 2.0 - lat) / PI * IMAGE_HEIGHT as f64) as i32;
        for dy in -2..=2i32 {
            for dx in -2..=2i32 {
                let x = (px + dx).rem_euclid(IMAGE_WIDTH as i32) as u32;
                let y = (py + dy).clamp(0, IMAGE_HEIGHT as i32 - 1) as u32;
                img.put_pixel(x, y, Rgb([255, 255, 255]));
            }
        }
    }

    img
}

/// Generates the full plate pipeline and saves a PNG.
pub fn generate_and_save(seed: u64, output: &Path) {
    let fibonacci = SphericalFibonacci::new(POINT_COUNT);
    let points = fibonacci.all_points();
    let delaunay = SphericalDelaunay::from_points(&points);
    let assignment = assign_plates(&points, &fibonacci, &delaunay, PLATE_COUNT, seed);

    let img = render_plate_map(&fibonacci, &assignment);
    img.save(output).expect("failed to save plate map");
    println!("Saved plate map to {}", output.display());
    println!("  {} points, {} plates, seed {seed}", POINT_COUNT, PLATE_COUNT);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Run with: cargo test --release plate_visualizer -- --ignored --nocapture
    fn plate_visualizer() {
        let output = Path::new("src/tectonic_simulation/plate_map.png");
        generate_and_save(42, output);
    }
}
