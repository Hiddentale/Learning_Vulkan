pub mod climate;
pub mod coarse_heightmap;
pub mod detail_noise;
pub mod river_networking;
pub mod sphere_geometry;
pub mod terrain_amplification;
pub mod volcanic_overlay;

use std::path::Path;
use std::sync::{Arc, Mutex};

/// Result of running the full world generation pipeline (Steps 0-6).
pub struct WorldTerrainData {
    pub amplified: Arc<terrain_amplification::AmplifiedTerrain>,
    pub climate: Arc<climate::ClimateMap>,
    pub flow: Arc<river_networking::FlowData>,
    pub detail_cache: Arc<Mutex<detail_noise::DetailCache>>,
}

/// Run the complete world generation pipeline (Steps 0-6).
///
/// Parameters:
/// - `seed`: World seed for all noise and random processes
/// - `model_dir`: Path to directory containing ONNX models (coarse_model.onnx, base_model.onnx, decoder_model.onnx)
/// - `point_count`: Number of Fibonacci sphere points (higher = more detail, slower; typical: 10,000)
/// - `plate_count`: Number of tectonic plates (typical: 40)
///
/// Returns: WorldTerrainData bundle with AmplifiedTerrain, ClimateMap, FlowData
///
/// This function runs all computation steps in sequence:
/// 1. Coarse heightmap from tectonic simulation
/// 2. Terrain amplification via ONNX diffusion model
/// 3. Volcanic overlay
/// 4. River networking
/// 5. Climate computation
pub fn generate_world_terrain(
    seed: u64,
    model_dir: &Path,
    point_count: usize,
    plate_count: usize,
) -> anyhow::Result<WorldTerrainData> {
    use sphere_geometry::fibonnaci_spiral::SphericalFibonacci;
    use sphere_geometry::spherical_delaunay_triangulation::SphericalDelaunay;
    use sphere_geometry::plate_seed_placement::{assign_plates, Adjacency};

    println!("[Pipeline] Starting world generation pipeline with seed={}, points={}, plates={}", seed, point_count, plate_count);

    // Step 0: Sphere geometry - point distribution and triangulation
    println!("[Pipeline] Generating Fibonacci sphere distribution ({} points)...", point_count);
    let fib = SphericalFibonacci::new(point_count as u32);
    let points = fib.all_points();
    println!("[Pipeline] Triangulating sphere...");
    let del = SphericalDelaunay::from_points(&points);

    // Step 1: Tectonic simulation and coarse heightmap
    println!("[Pipeline] Assigning tectonic plates ({} plates)...", plate_count);
    let assignment = assign_plates(&points, &fib, &del, plate_count as u32, seed);
    let adjacency = Adjacency::from_delaunay(points.len(), &del);
    println!("[Pipeline] Generating coarse heightmap...");
    let coarse = coarse_heightmap::generate(&points, &assignment, &adjacency, seed);

    // Step 2: Terrain amplification (ONNX diffusion)
    println!("[Pipeline] Amplifying terrain with diffusion model...");
    let mut amplified = terrain_amplification::amplify(&coarse, &points, &fib, seed, model_dir)?;

    // Step 3: Volcanic overlay
    println!("[Pipeline] Applying volcanic overlay...");
    volcanic_overlay::overlay(&mut amplified, &coarse, &points, seed);

    // Step 4: River networking
    println!("[Pipeline] Running river networking ({}×{})...", amplified.cross_width, amplified.cross_height);
    let flow = river_networking::process(&mut amplified.cross_elevation, amplified.cross_width, amplified.cross_height);

    // Step 5: Climate computation
    println!("[Pipeline] Computing climate (Tier 1)...");
    let climate = climate::compute(&amplified.cross_elevation, amplified.cross_width, amplified.cross_height);

    // Log climate statistics
    let max_temp = climate.temperature.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let min_temp = climate.temperature.iter().cloned().fold(f32::INFINITY, f32::min);
    let max_precip = climate.precipitation.iter().cloned().fold(0.0f32, f32::max);
    println!("[Pipeline] Climate: T=[{:.1}°C, {:.1}°C], P=[0, {:.0}mm]", min_temp, max_temp, max_precip);

    // Step 6: Detail noise pre-computation
    println!("[Pipeline] Pre-computing detail noise (slope-gated, landform-driven)...");
    let amplified_arc = Arc::new(amplified);
    let detail_cache = detail_noise::precompute_all(
        amplified_arc.clone(),
        (seed & 0xFFFFFFFF) as u32,
        None,  // TODO: wire up cache_dir from args if needed
    );

    println!("[Pipeline] World generation complete!");

    Ok(WorldTerrainData {
        amplified: amplified_arc,
        climate: Arc::new(climate),
        flow: Arc::new(flow),
        detail_cache: Arc::new(Mutex::new(detail_cache)),
    })
}
