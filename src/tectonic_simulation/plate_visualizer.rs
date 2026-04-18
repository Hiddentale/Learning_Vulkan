use std::f64::consts::PI;
use std::path::Path;

use glam::DVec3;
use image::{Rgb, RgbImage};

use super::fibonnaci_spiral::SphericalFibonacci;
use super::plate_seed_placement::{assign_plates, PlateAssignment};
use super::plates::{CrustType, Plate};
use super::simulate::Simulation;
use super::spherical_delaunay_triangulation::SphericalDelaunay;

const IMAGE_WIDTH: u32 = 2048;
const IMAGE_HEIGHT: u32 = 1024;
const POINT_COUNT: u32 = 50_000;
const PLATE_COUNT: u32 = 20;

/// Velocity magnitudes below this are treated as stationary (no arrow drawn).
const MIN_VELOCITY: f64 = 1e-12;
/// Line thickness in pixels for velocity arrows.
const LINE_THICKNESS: u32 = 2;
/// Half-size of the seed marker dot in pixels.
const SEED_DOT_RADIUS: i32 = 2;
/// Offset to sample at pixel centers rather than pixel edges.
const PIXEL_CENTER: f64 = 0.5;

const CONTINENTAL_COLOR: [u8; 3] = [255, 255, 255];
const OCEANIC_COLOR: [u8; 3] = [160, 200, 240];
const BORDER_COLOR: [u8; 3] = [0, 0, 0];
const ARROW_COLOR: [u8; 3] = [0, 0, 0];
const ARROW_LENGTH: f64 = 40.0;
const ARROW_HEAD_LENGTH: f64 = 10.0;
const ARROW_HEAD_HALF_WIDTH: f64 = 5.0;

/// Distinct, saturated colors for up to 40 plates.
const PALETTE: [[u8; 3]; 40] = [
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 212],
    [0, 128, 128],
    [220, 190, 255],
    [170, 110, 40],
    [255, 250, 200],
    [128, 0, 0],
    [170, 255, 195],
    [128, 128, 0],
    [255, 215, 180],
    [0, 0, 128],
    [128, 128, 128],
    [200, 60, 60],
    [60, 200, 120],
    [200, 200, 60],
    [60, 100, 200],
    [200, 130, 60],
    [130, 60, 200],
    [60, 200, 200],
    [200, 60, 180],
    [160, 210, 80],
    [210, 160, 180],
    [40, 160, 160],
    [180, 160, 220],
    [160, 100, 60],
    [220, 220, 170],
    [160, 40, 40],
    [130, 220, 170],
    [160, 160, 40],
    [220, 180, 160],
    [40, 40, 160],
    [100, 100, 100],
];

/// Convert lat/lon to a unit direction vector.
/// Uses visualizer convention: X = east, Y = north (up in lat), Z = toward viewer.
/// This differs from the simulation convention (Y = up) because the equirectangular
/// projection maps latitude to the Z-component of atan2-based lookups.
fn latlon_to_direction(lat: f64, lon: f64) -> DVec3 {
    let cos_lat = lat.cos();
    DVec3::new(cos_lat * lon.cos(), cos_lat * lon.sin(), lat.sin())
}

/// Convert a pixel coordinate to a sphere direction.
fn pixel_to_direction(px: u32, py: u32) -> DVec3 {
    let lat = PI / 2.0 - (py as f64 + PIXEL_CENTER) / IMAGE_HEIGHT as f64 * PI;
    let lon = (px as f64 + PIXEL_CENTER) / IMAGE_WIDTH as f64 * 2.0 * PI - PI;
    latlon_to_direction(lat, lon)
}

/// Draw plate borders: any pixel whose right or bottom neighbor belongs to a different plate.
fn draw_borders(img: &mut RgbImage, pixel_plate: &[u32]) {
    let color = Rgb(BORDER_COLOR);
    for py in 0..IMAGE_HEIGHT {
        for px in 0..IMAGE_WIDTH {
            let idx = (py * IMAGE_WIDTH + px) as usize;
            let plate = pixel_plate[idx];
            let on_border = [(1i32, 0i32), (0, 1)].iter().any(|&(dx, dy)| {
                let nx = (px as i32 + dx).rem_euclid(IMAGE_WIDTH as i32) as u32;
                let ny = (py as i32 + dy).clamp(0, IMAGE_HEIGHT as i32 - 1) as u32;
                let ni = (ny * IMAGE_WIDTH + nx) as usize;
                pixel_plate[ni] != plate
            });
            if on_border {
                img.put_pixel(px, py, color);
            }
        }
    }
}

/// Renders an equirectangular plate map to an RGB image.
pub fn render_plate_map(fibonacci: &SphericalFibonacci, assignment: &PlateAssignment) -> RgbImage {
    let mut img = RgbImage::new(IMAGE_WIDTH, IMAGE_HEIGHT);

    for py in 0..IMAGE_HEIGHT {
        for px in 0..IMAGE_WIDTH {
            let dir = pixel_to_direction(px, py);
            let nearest = fibonacci.nearest_index(dir);
            let plate = assignment.plate_ids[nearest as usize];
            let color = PALETTE[plate as usize % PALETTE.len()];
            img.put_pixel(px, py, Rgb(color));
        }
    }

    for &seed_idx in &assignment.seeds {
        let p = fibonacci.index_to_point(seed_idx);
        let lat = p.z.asin();
        let lon = p.y.atan2(p.x);
        let px = ((lon + PI) / (2.0 * PI) * IMAGE_WIDTH as f64) as i32;
        let py = ((PI / 2.0 - lat) / PI * IMAGE_HEIGHT as f64) as i32;
        for dy in -SEED_DOT_RADIUS..=SEED_DOT_RADIUS {
            for dx in -SEED_DOT_RADIUS..=SEED_DOT_RADIUS {
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

/// Renders plates colored by crust type with velocity arrows at each plate center.
pub fn render_initialized_plates(fibonacci: &SphericalFibonacci, assignment: &PlateAssignment, plates: &[Plate]) -> RgbImage {
    let mut img = RgbImage::new(IMAGE_WIDTH, IMAGE_HEIGHT);

    let mut point_crust = vec![CrustType::Oceanic; fibonacci.point_count() as usize];
    for plate in plates {
        for (i, &pi) in plate.point_indices.iter().enumerate() {
            point_crust[pi as usize] = plate.crust[i].crust_type;
        }
    }

    let mut pixel_plate = vec![0u32; (IMAGE_WIDTH * IMAGE_HEIGHT) as usize];
    for py in 0..IMAGE_HEIGHT {
        for px in 0..IMAGE_WIDTH {
            let dir = pixel_to_direction(px, py);
            let nearest = fibonacci.nearest_index(dir);
            let idx = (py * IMAGE_WIDTH + px) as usize;
            pixel_plate[idx] = assignment.plate_ids[nearest as usize];

            let color = match point_crust[nearest as usize] {
                CrustType::Continental => CONTINENTAL_COLOR,
                CrustType::Oceanic => OCEANIC_COLOR,
            };
            img.put_pixel(px, py, Rgb(color));
        }
    }

    draw_borders(&mut img, &pixel_plate);

    for (plate_idx, plate) in plates.iter().enumerate() {
        let seed_3d = fibonacci.index_to_point(assignment.seeds[plate_idx]);
        let velocity = plate.surface_velocity(seed_3d);
        if velocity.length() < MIN_VELOCITY {
            continue;
        }
        draw_velocity_arrow(&mut img, seed_3d, velocity);
    }

    img
}

/// Project a 3D velocity vector to 2D pixel displacement on the equirectangular map,
/// then draw an arrow (line + head).
fn draw_velocity_arrow(img: &mut RgbImage, origin: DVec3, velocity: DVec3) {
    let lat = origin.z.asin();
    let lon = origin.y.atan2(origin.x);
    let cx = (lon + PI) / (2.0 * PI) * IMAGE_WIDTH as f64;
    let cy = (PI / 2.0 - lat) / PI * IMAGE_HEIGHT as f64;

    let east = DVec3::new(-origin.y, origin.x, 0.0).normalize_or_zero();
    let north = origin.cross(east);
    let ve = velocity.dot(east);
    let vn = velocity.dot(north);
    let mag = (ve * ve + vn * vn).sqrt();
    if mag < MIN_VELOCITY {
        return;
    }

    // Pixel direction: east = +x, north = -y.
    let dx = ve / mag;
    let dy = -vn / mag;

    let tip_x = cx + dx * ARROW_LENGTH;
    let tip_y = cy + dy * ARROW_LENGTH;

    draw_line(img, cx, cy, tip_x, tip_y, ARROW_COLOR);

    let back_x = -dx;
    let back_y = -dy;
    let perp_x = -dy;
    let perp_y = dx;
    for sign in [-1.0, 1.0] {
        let hx = tip_x + back_x * ARROW_HEAD_LENGTH + perp_x * ARROW_HEAD_HALF_WIDTH * sign;
        let hy = tip_y + back_y * ARROW_HEAD_LENGTH + perp_y * ARROW_HEAD_HALF_WIDTH * sign;
        draw_line(img, tip_x, tip_y, hx, hy, ARROW_COLOR);
    }
}

fn draw_line(img: &mut RgbImage, x0: f64, y0: f64, x1: f64, y1: f64, color: [u8; 3]) {
    let dx = x1 - x0;
    let dy = y1 - y0;
    let steps = dx.abs().max(dy.abs()).ceil() as usize;
    if steps == 0 {
        return;
    }
    let sx = dx / steps as f64;
    let sy = dy / steps as f64;
    for i in 0..=steps {
        let x = (x0 + sx * i as f64).round() as i32;
        let y = (y0 + sy * i as f64).round() as i32;
        let px = x.rem_euclid(IMAGE_WIDTH as i32) as u32;
        if y >= 0 && y < IMAGE_HEIGHT as i32 {
            for oy in 0..LINE_THICKNESS as i32 {
                let py = (y + oy).clamp(0, IMAGE_HEIGHT as i32 - 1) as u32;
                img.put_pixel(px, py, Rgb(color));
            }
        }
    }
}

/// Render the simulation state. Must be called right after a resample so that
/// points are a clean Fibonacci grid and `nearest_index` works correctly.
pub fn render_simulation(sim: &Simulation) -> RgbImage {
    let mut img = RgbImage::new(IMAGE_WIDTH, IMAGE_HEIGHT);

    let delaunay = SphericalDelaunay::from_points(&sim.points);

    let n = sim.points.len();
    let mut point_plate = vec![0u32; n];
    let mut point_crust = vec![CrustType::Oceanic; n];
    for (plate_idx, plate) in sim.plates.iter().enumerate() {
        for (local, &global) in plate.point_indices.iter().enumerate() {
            point_plate[global as usize] = plate_idx as u32;
            point_crust[global as usize] = plate.crust[local].crust_type;
        }
    }

    let mut pixel_plate = vec![0u32; (IMAGE_WIDTH * IMAGE_HEIGHT) as usize];
    let mut last_tri = 0;
    for py in 0..IMAGE_HEIGHT {
        for px in 0..IMAGE_WIDTH {
            let dir = pixel_to_direction(px, py);

            let (tri, _b1, _b2, _b3) = delaunay.locate(dir, &sim.points, last_tri);
            last_tri = tri;

            let base = tri * 3;
            let vi = [
                delaunay.triangles[base] as usize,
                delaunay.triangles[base + 1] as usize,
                delaunay.triangles[base + 2] as usize,
            ];
            let nearest = *vi
                .iter()
                .max_by(|&&a, &&b| dir.dot(sim.points[a]).partial_cmp(&dir.dot(sim.points[b])).unwrap())
                .unwrap();

            let idx = (py * IMAGE_WIDTH + px) as usize;
            pixel_plate[idx] = point_plate[nearest];

            let color = match point_crust[nearest] {
                CrustType::Continental => CONTINENTAL_COLOR,
                CrustType::Oceanic => OCEANIC_COLOR,
            };
            img.put_pixel(px, py, Rgb(color));
        }
    }

    draw_borders(&mut img, &pixel_plate);

    for plate in &sim.plates {
        if plate.point_indices.is_empty() {
            continue;
        }
        let centroid: DVec3 = plate
            .point_indices
            .iter()
            .map(|&i| sim.points[i as usize])
            .sum::<DVec3>()
            .normalize_or_zero();
        let velocity = plate.surface_velocity(centroid);
        if velocity.length() < MIN_VELOCITY {
            continue;
        }
        draw_velocity_arrow(&mut img, centroid, velocity);
    }

    img
}

#[cfg(test)]
mod tests {
    use super::super::plate_initializer::{initialize_plates, InitParams};
    use super::*;

    #[test]
    #[ignore] // Run with: cargo test --release plate_visualizer -- --ignored --nocapture
    fn plate_visualizer() {
        let output = Path::new("src/tectonic_simulation/plate_map.png");
        generate_and_save(42, output);
    }

    const TIMELAPSE_POINTS: u32 = 50_000;
    const TIMELAPSE_STEPS: usize = 400;
    /// Render every N resample cycles. Each cycle = RESAMPLE_INTERVAL steps.
    const TIMELAPSE_RENDER_EVERY_N_RESAMPLES: usize = 1;

    #[test]
    #[ignore] // Run with: cargo test --release simulation_timelapse -- --ignored --nocapture
    fn simulation_timelapse() {
        use super::super::resample;
        use std::io::Write;

        let suffix = if TIMELAPSE_POINTS >= 1_000_000 {
            format!("{}m", TIMELAPSE_POINTS / 1_000_000)
        } else if TIMELAPSE_POINTS >= 1_000 {
            format!("{}", TIMELAPSE_POINTS / 1_000)
        } else {
            format!("{}", TIMELAPSE_POINTS)
        };
        let dir_name = format!("src/tectonic_simulation/timelapse_{suffix}");
        let output_dir = Path::new(&dir_name);
        std::fs::create_dir_all(output_dir).expect("failed to create timelapse dir");

        let fibonacci = SphericalFibonacci::new(TIMELAPSE_POINTS);
        let points = fibonacci.all_points();
        let delaunay = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fibonacci, &delaunay, PLATE_COUNT, 42);
        let plates = initialize_plates(&points, &delaunay, &assignment, &InitParams::default());
        let mut sim = Simulation::new(points, plates, &delaunay);

        let log_path = output_dir.join("tectonic_debug.log");
        sim.enable_diagnostics(&log_path, 0, usize::MAX);

        let steps_per_frame = resample::RESAMPLE_INTERVAL * TIMELAPSE_RENDER_EVERY_N_RESAMPLES;
        let total_frames = TIMELAPSE_STEPS / steps_per_frame + 1;
        let mut frame = 0;

        frame += 1;
        print!(
            "\r[{:>3}%] Frame {}/{} (t=0 Myr)        ",
            frame * 100 / total_frames,
            frame,
            total_frames
        );
        std::io::stdout().flush().unwrap();
        render_simulation(&sim).save(output_dir.join("frame_000.png")).unwrap();

        for step in 1..=TIMELAPSE_STEPS {
            sim.step();
            if step % steps_per_frame == 0 {
                frame += 1;
                let pct = (frame * 100).min(100 * total_frames) / total_frames;
                print!("\r[{:>3}%] Frame {}/{} (t={:.0} Myr)        ", pct, frame, total_frames, sim.time);
                std::io::stdout().flush().unwrap();
                render_simulation(&sim).save(output_dir.join(format!("frame_{:03}.png", step))).unwrap();
            }
        }
        println!("\r[100%] Done — {} frames in {}        ", total_frames, output_dir.display());
    }

    const DEBUG_EXPORT_POINTS: u32 = 100_000;
    const DEBUG_EXPORT_STEPS: usize = 200;
    const DEBUG_EXPORT_RECORD_EVERY_N_RESAMPLES: usize = 1;

    #[test]
    #[ignore] // Run with: cargo test --release debug_export -- --ignored --nocapture
    fn debug_export() {
        use super::super::debug_export::DebugRecorder;
        use super::super::resample;
        use std::io::Write;

        let fibonacci = SphericalFibonacci::new(DEBUG_EXPORT_POINTS);
        let points = fibonacci.all_points();
        let delaunay = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fibonacci, &delaunay, PLATE_COUNT, 42);
        let plates = initialize_plates(&points, &delaunay, &assignment, &InitParams::default());
        let mut sim = Simulation::new(points, plates, &delaunay);

        let mut recorder = DebugRecorder::new();
        recorder.set_triangulation(&delaunay);
        recorder.record(&sim);

        let steps_per_frame = resample::RESAMPLE_INTERVAL * DEBUG_EXPORT_RECORD_EVERY_N_RESAMPLES;
        let total_steps = DEBUG_EXPORT_STEPS;

        for step in 1..=total_steps {
            sim.step();
            if step % steps_per_frame == 0 {
                recorder.record(&sim);
                let pct = step * 100 / total_steps;
                print!(
                    "\r[{:>3}%] Recorded frame {} (t={:.0} Myr)        ",
                    pct,
                    recorder.frame_count(),
                    sim.time
                );
                std::io::stdout().flush().unwrap();
            }
        }
        println!();

        let output = Path::new("tools/debug_viewer/sim_data.bin");
        std::fs::create_dir_all(output.parent().unwrap()).unwrap();
        recorder.save(output).expect("failed to save debug export");
    }

    #[test]
    #[ignore] // Run with: cargo test --release initialized_plate_map -- --ignored --nocapture
    fn initialized_plate_map() {
        let fibonacci = SphericalFibonacci::new(POINT_COUNT);
        let points = fibonacci.all_points();
        let delaunay = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fibonacci, &delaunay, PLATE_COUNT, 42);
        let plates = initialize_plates(&points, &delaunay, &assignment, &InitParams::default());

        let img = render_initialized_plates(&fibonacci, &assignment, &plates);
        let output = Path::new("src/tectonic_simulation/initialized_plates.png");
        img.save(output).expect("failed to save initialized plate map");
        println!("Saved initialized plate map to {}", output.display());
    }
}
