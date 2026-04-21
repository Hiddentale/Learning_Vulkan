use std::f64::consts::PI;
use std::path::Path;

use glam::DVec3;
use image::{Rgb, RgbImage};

use super::fibonnaci_spiral::SphericalFibonacci;
use super::plate_seed_placement::{assign_plates, PlateAssignment};
use super::plates::CrustType;
use super::simulate::{Simulation, NO_PLATE};
use super::sphere_grid::SphereGrid;
use super::spherical_delaunay_triangulation::SphericalDelaunay;

const IMAGE_WIDTH: u32 = 2048;
const IMAGE_HEIGHT: u32 = 1024;
const POINT_COUNT: u32 = 50_000;
const PLATE_COUNT: u32 = 40;

const MIN_VELOCITY: f64 = 1e-12;
const LINE_THICKNESS: u32 = 2;
const SEED_DOT_RADIUS: i32 = 2;
const PIXEL_CENTER: f64 = 0.5;

const CONTINENTAL_COLOR: [u8; 3] = [255, 255, 255];
const OCEANIC_COLOR: [u8; 3] = [160, 200, 240];
const BORDER_COLOR: [u8; 3] = [0, 0, 0];
const ARROW_COLOR: [u8; 3] = [0, 0, 0];
const ARROW_LENGTH: f64 = 40.0;
const ARROW_HEAD_LENGTH: f64 = 10.0;
const ARROW_HEAD_HALF_WIDTH: f64 = 5.0;

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

fn pixel_to_direction(px: u32, py: u32) -> DVec3 {
    let lat = PI / 2.0 - (py as f64 + PIXEL_CENTER) / IMAGE_HEIGHT as f64 * PI;
    let lon = (px as f64 + PIXEL_CENTER) / IMAGE_WIDTH as f64 * 2.0 * PI - PI;
    let cos_lat = lat.cos();
    DVec3::new(cos_lat * lon.cos(), cos_lat * lon.sin(), lat.sin())
}

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

/// Render the simulation state using the sample grid.
/// Each pixel finds the nearest sample point and uses its cached plate/crust.
pub fn render_simulation(sim: &Simulation) -> RgbImage {
    let mut img = RgbImage::new(IMAGE_WIDTH, IMAGE_HEIGHT);
    let grid = SphereGrid::build(&sim.sample_points);
    let mut pixel_plate = vec![0u32; (IMAGE_WIDTH * IMAGE_HEIGHT) as usize];

    for py in 0..IMAGE_HEIGHT {
        for px in 0..IMAGE_WIDTH {
            let dir = pixel_to_direction(px, py);
            let nearest = grid.find_nearest_k(dir, &sim.sample_points, 1);
            let vi = nearest[0].0 as usize;
            let cache = &sim.sample_cache[vi];
            let idx = (py * IMAGE_WIDTH + px) as usize;
            pixel_plate[idx] = cache.plate;

            let crust_type = if cache.plate != NO_PLATE {
                // Majority vote among triangle's 3 vertices to suppress
                // speckle noise at continent-ocean boundaries.
                let plate = &sim.plates[cache.plate as usize];
                let [va, vb, vc] = plate.triangles[cache.triangle as usize];
                let cont = [va, vb, vc]
                    .iter()
                    .filter(|&&v| plate.crust[v as usize].crust_type == CrustType::Continental)
                    .count();
                if cont >= 2 {
                    CrustType::Continental
                } else {
                    CrustType::Oceanic
                }
            } else {
                CrustType::Oceanic
            };

            let color = match crust_type {
                CrustType::Continental => CONTINENTAL_COLOR,
                CrustType::Oceanic => OCEANIC_COLOR,
            };
            img.put_pixel(px, py, Rgb(color));
        }
    }

    //draw_borders(&mut img, &pixel_plate);

    for plate in &sim.plates {
        if plate.reference_points.is_empty() {
            continue;
        }
        let centroid: DVec3 = plate
            .reference_points
            .iter()
            .map(|&p| plate.to_world(p))
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

#[cfg(test)]
mod tests {
    use super::super::plate_initializer::{initialize_plates, InitParams};
    use super::super::plate_seed_placement::Adjacency;
    use super::super::resample;
    use super::*;

    fn make_sim(point_count: u32, plate_count: u32) -> Simulation {
        let fib = SphericalFibonacci::new(point_count);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, plate_count, 42);
        let (plates, cache) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        let adj = Adjacency::from_delaunay(points.len(), &del);
        Simulation::new(points, cache, adj, plates)
    }

    #[test]
    #[ignore]
    fn plate_visualizer() {
        let output = Path::new("src/tectonic_simulation/plate_map.png");
        generate_and_save(42, output);
    }

    const TIMELAPSE_POINTS: u32 = 500_000;
    const TIMELAPSE_STEPS: usize = 125;
    const TIMELAPSE_RENDER_EVERY_N_RESAMPLES: usize = 1;

    #[test]
    #[ignore]
    fn simulation_timelapse() {
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

        let mut sim = make_sim(TIMELAPSE_POINTS, PLATE_COUNT);

        let log_path = output_dir.join("tectonic_debug.log");
        sim.enable_diagnostics(&log_path, 0, usize::MAX);

        let steps_per_frame = resample::resample_interval(&sim.plates) * TIMELAPSE_RENDER_EVERY_N_RESAMPLES;
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
                resample::resample(&mut sim);
                frame += 1;
                let pct = (frame * 100).min(100 * total_frames) / total_frames;
                print!("\r[{:>3}%] Frame {}/{} (t={:.0} Myr)        ", pct, frame, total_frames, sim.time);
                std::io::stdout().flush().unwrap();
                render_simulation(&sim).save(output_dir.join(format!("frame_{:03}.png", step))).unwrap();
            }
        }
        println!("\r[100%] Done — {} frames in {}        ", total_frames, output_dir.display());
    }
}
