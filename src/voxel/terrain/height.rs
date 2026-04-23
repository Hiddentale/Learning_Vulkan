use super::noises::{WorldNoises, WARP_STRENGTH, MOUNTAIN_AMPLITUDE, DETAIL_AMPLITUDE, WEIRDNESS_AMPLITUDE, SEA_LEVEL, MIN_HEIGHT, MAX_HEIGHT, CAVE_THRESHOLD, CAVE_MIN_DEPTH, CAVE_SCALE_FACTOR, DIRT_DEPTH};
use super::super::biome::{self, Biome};
use super::super::block::BlockType;
use super::super::sphere::{self, Face};
use noise::NoiseFn;

/// All terrain parameters for a single world direction/position.
/// Combines noise-driven shape (continentalness, erosion, weirdness) with
/// climate data (temperature, precipitation, continentality) and river presence.
#[allow(dead_code)]
pub(crate) struct TerrainParams {
    /// Noise-derived continentality [-1, 1], drives base height shape.
    pub(crate) continentalness: f64,
    /// Noise-derived erosion [-1, 1], modulates mountain/detail amplitude.
    pub(crate) erosion: f64,
    /// Noise-derived weirdness [-1, 1], gates fantasy features (overhangs, caves).
    pub(crate) weirdness: f64,
    /// Temperature in degrees Celsius (from climate map or noise fallback).
    pub(crate) temperature: f64,
    /// Precipitation in mm/year (from climate map or noise fallback).
    pub(crate) precipitation: f64,
    /// Continentality [0, 1] from ocean distance (from climate map or noise fallback).
    pub(crate) continentality: f64,
    /// River Strahler order (0 = not a river, 1+ = stream order).
    pub(crate) river_order: u8,
    /// Final surface height in blocks above sea level.
    pub(crate) height: usize,
    /// Biome classification from temperature, precipitation, elevation, river presence.
    pub(crate) biome: Biome,
}

/// Sample all terrain parameters at face-local cube coordinates, applying
/// domain warping. The (u, v) tangent coordinates are projected onto the
/// planet sphere via [`sphere::noise_pos_on_face`] and 3D noise is sampled
/// at the resulting sphere point — this gives seamless terrain across face
/// edges for free (3D noise has no seams).
pub(crate) fn sample_params(noises: &WorldNoises, face: Face, u: f64, v: f64, erosion_map: Option<&super::super::erosion::ErosionMap>) -> TerrainParams {
    let warp_p = sphere::noise_pos_on_face(face, u, v);
    let warped_u = u + noises.warp_x.get(warp_p) * WARP_STRENGTH;
    let warped_v = v + noises.warp_z.get(warp_p) * WARP_STRENGTH;
    let p = sphere::noise_pos_on_face(face, warped_u, warped_v);

    let continentalness = noises.continentalness.get(p);
    let erosion = noises.erosion_noise.get(p);
    let weirdness = noises.weirdness.get(p);
    let temperature = noises.temperature.get(p);
    let humidity = noises.humidity.get(p);
    let mut height = compute_height_from_params(noises, face, warped_u, warped_v, continentalness, erosion, weirdness);

    // Apply hydraulic erosion delta — Phase C: erosion map is still flat,
    // so feed it the face-local cube coordinates. Will be revisited when
    // erosion is rebuilt for the sphere surface.
    if let Some(emap) = erosion_map {
        let delta = emap.sample(u, v);
        height = (height as f64 + delta).clamp(MIN_HEIGHT as f64, MAX_HEIGHT as f64) as usize;
    }

    let temperature_c = temperature * 35.0 + 10.0;
    let precipitation_mm = (humidity + 1.0) * 750.0;
    let continentality_norm = (continentalness + 1.0) * 0.5;

    let biome = biome::determine_biome(temperature_c, precipitation_mm, continentality_norm, height, SEA_LEVEL, 0, weirdness);

    TerrainParams {
        continentalness,
        erosion,
        weirdness,
        temperature: temperature_c,
        precipitation: precipitation_mm,
        continentality: continentality_norm,
        river_order: 0,
        height,
        biome,
    }
}

/// Maps continentalness [-1, 1] to a base height offset from sea level.
/// Piecewise linear: deep ocean → shelf → coast → lowland → highland.
pub(crate) fn continental_curve(c: f64) -> f64 {
    if c < -0.4 {
        // Deep ocean: -40 at c=-1.0 to -10 at c=-0.4
        lerp(-40.0, -10.0, (c + 1.0) / 0.6)
    } else if c < -0.2 {
        // Ocean shelf: -10 to 0
        lerp(-10.0, 0.0, (c + 0.4) / 0.2)
    } else if c < 0.0 {
        // Coast: 0 to +5
        lerp(0.0, 5.0, (c + 0.2) / 0.2)
    } else if c < 0.5 {
        // Lowland: +5 to +30
        lerp(5.0, 30.0, c / 0.5)
    } else {
        // Highland: +30 to +80
        lerp(30.0, 80.0, (c - 0.5) / 0.5)
    }
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

pub fn compute_height_from_params(
    noises: &WorldNoises,
    face: Face,
    u: f64,
    v: f64,
    continentalness: f64,
    erosion: f64,
    weirdness: f64,
) -> usize {
    let base = continental_curve(continentalness);
    let p = sphere::noise_pos_on_face(face, u, v);

    // Erosion controls terrain roughness: high erosion = full mountains, low = flat
    let erosion_factor = (0.3 + erosion * 0.7).clamp(0.3, 1.0);
    let mountain = noises.mountain.get(p) * MOUNTAIN_AMPLITUDE * erosion_factor;
    let detail = noises.detail.get(p) * DETAIL_AMPLITUDE * erosion_factor;
    let weirdness_offset = weirdness * WEIRDNESS_AMPLITUDE;

    let height = SEA_LEVEL as f64 + base + mountain + detail + weirdness_offset;
    height.clamp(MIN_HEIGHT as f64, MAX_HEIGHT as f64) as usize
}

/// Direction-only sample of all terrain parameters. Replaces the old
/// `sample_params(face, u, v)` for the density-based pipeline. Two world
/// points sharing a radial direction produce identical params, which is
/// what makes the terrain seamless across cube faces.
/// Constants for elevation-to-blocks conversion.
const ELEV_SCALE: f64 = 0.07;  // blocks per meter
const RIVER_CARVE_MAX_BLOCKS: f64 = 12.0;  // max depth river carves

fn elev_to_height(elev_m: f32) -> usize {
    let blocks = (elev_m as f64) * ELEV_SCALE;
    ((SEA_LEVEL as f64 + blocks).clamp(MIN_HEIGHT as f64, MAX_HEIGHT as f64)) as usize
}

fn apply_river_carving(height: usize, order: u8) -> usize {
    if order > 0 {
        let carve_depth = (RIVER_CARVE_MAX_BLOCKS * (order as f64 / 10.0).min(1.0)) as usize;
        height.saturating_sub(carve_depth).max(MIN_HEIGHT as usize)
    } else {
        height
    }
}

pub(crate) fn sample_params_at_world(noises: &WorldNoises, world: glam::DVec3, terrain: Option<&super::terrain_data::TerrainData>) -> TerrainParams {
    // Continuous 3D domain warp. Sampling three noise channels gives a vector
    // offset in world space; subtracting its radial component projects it onto
    // the local tangent plane without ever picking a basis (no hairy-ball
    // discontinuity). Equivalent to a tangent-plane warp but defined for every
    // direction.
    let p_warp = sphere::noise_pos_at_world(world);
    let dir = world.normalize_or(glam::DVec3::Y);
    let warp_3d = glam::DVec3::new(noises.warp_x.get(p_warp), noises.warp_y.get(p_warp), noises.warp_z.get(p_warp));
    let warp_tangent = warp_3d - dir * warp_3d.dot(dir);
    let warped_dir = (dir + warp_tangent * (WARP_STRENGTH / sphere::SURFACE_RADIUS_BLOCKS as f64)).normalize_or(dir);
    let p = [
        warped_dir.x * sphere::SURFACE_RADIUS_BLOCKS as f64,
        warped_dir.y * sphere::SURFACE_RADIUS_BLOCKS as f64,
        warped_dir.z * sphere::SURFACE_RADIUS_BLOCKS as f64,
    ];

    // Always sample noise-derived shape parameters
    let continentalness = noises.continentalness.get(p);
    let erosion = noises.erosion_noise.get(p);
    let weirdness = noises.weirdness.get(p);
    let detail_noise = noises.detail.get(p);

    // Sample climate and height from terrain data if available; fallback to noise
    let (temperature, precipitation, continentality, river_order, mut height) = if let Some(terrain_data) = terrain {
        let (temp, precip, cont) = terrain_data.climate.sample_at_dir(warped_dir);
        let order = terrain_data.flow.stream_order_at_dir(warped_dir);
        let mut h = elev_to_height(terrain_data.amplified.elevation_at_dir(warped_dir));

        // Apply procedural detail noise to add surface variation (gated by erosion)
        let erosion_factor = (0.3 + erosion * 0.7).clamp(0.3, 1.0);
        let detail_blocks = (detail_noise * DETAIL_AMPLITUDE * erosion_factor) as i32;
        h = ((h as i32 + detail_blocks).clamp(MIN_HEIGHT as i32, MAX_HEIGHT as i32)) as usize;

        let h = apply_river_carving(h, order);
        (temp as f64, precip as f64, cont as f64, order, h)
    } else {
        // Noise fallback: map noise values to realistic ranges
        let temp_noise = noises.temperature.get(p);
        let humid_noise = noises.humidity.get(p);
        let temp = temp_noise * 35.0 + 10.0;  // Celsius
        let precip = (humid_noise + 1.0) * 750.0;  // mm/yr
        let cont = (continentalness + 1.0) * 0.5;  // [0, 1]

        let base = continental_curve(continentalness);
        let erosion_factor = (0.3 + erosion * 0.7).clamp(0.3, 1.0);
        let mountain = noises.mountain.get(p) * MOUNTAIN_AMPLITUDE * erosion_factor;
        let detail = noises.detail.get(p) * DETAIL_AMPLITUDE * erosion_factor;
        let weirdness_offset = weirdness * WEIRDNESS_AMPLITUDE;
        let h = (SEA_LEVEL as f64 + base + mountain + detail + weirdness_offset).clamp(MIN_HEIGHT as f64, MAX_HEIGHT as f64) as usize;

        (temp, precip, cont, 0, h)
    };

    height = height;

    let biome = biome::determine_biome(temperature, precipitation, continentality, height, SEA_LEVEL, river_order, weirdness);
    TerrainParams {
        continentalness,
        erosion,
        weirdness,
        temperature,
        precipitation,
        continentality,
        river_order,
        height,
        biome,
    }
}

/// Sample the params-derived surface radius and surface block at an arbitrary
/// world cartesian point. The returned radius is `PLANET_RADIUS + height` from
/// the same `sample_params_at_world` path that drives chunk density terrain,
/// so heightmap tiles using this stay consistent with the mesh-shader chunks
/// at every world point. The returned block is the surface biome block at
/// the same point.
pub fn surface_radius_at_world(noises: &WorldNoises, world: glam::DVec3, terrain: Option<&super::terrain_data::TerrainData>) -> (f64, BlockType) {
    let params = sample_params_at_world(noises, world, terrain);
    let radius = sphere::PLANET_RADIUS_BLOCKS as f64 + params.height as f64;
    let block = biome::surface_block(params.biome);
    (radius, block)
}

/// Per-block density evaluation. Direction-dependent values are passed in.
///
/// **Surface contract**: a block is solid iff `r <= surface_radius`. There is
/// no 3D overhang noise carving the surface — that would create a per-block
/// height field that diverges from the analytical `surface_radius_at_world`,
/// breaking LOD parity with the heightmap tile path. Caves are still allowed
/// strictly below the surface (`depth_from_surface > CAVE_MIN_DEPTH`) so they
/// never punch through the visible top.
///
/// Pinned by `heightmap_top_matches_chunked_top_within_one_block` in
/// `voxel::heightmap_generator::tests`.
pub(crate) fn sample_density_block(
    world: glam::DVec3,
    r: f64,
    surface_radius: f64,
    sea_radius: f64,
    surface: BlockType,
    subsurface: BlockType,
    noises: &WorldNoises,
) -> BlockType {
    if r > surface_radius {
        return if r < sea_radius { BlockType::Water } else { BlockType::Air };
    }

    // Below the surface — pick stone / subsurface / surface based on depth.
    let depth_from_surface = (surface_radius - r).max(0.0);
    let block = if depth_from_surface < 1.0 {
        surface
    } else if depth_from_surface < DIRT_DEPTH as f64 {
        subsurface
    } else {
        BlockType::Stone
    };

    // 3D cave carving — spheres of air punched out of the solid mass.
    // Caves must stay well below the surface so they don't expose tall walls
    // when an adjacent column happens to be solid right where this column
    // has a cave. With CAVE_SCALE=0.05 (period ~20 blocks) the cave features
    // are ~10 blocks across; a depth-from-surface threshold of CAVE_MIN_DEPTH
    // ensures even the topmost cave block sits well under the surface band.
    if depth_from_surface > CAVE_MIN_DEPTH as f64 {
        let cave_val = noises.cave.get([world.x * CAVE_SCALE_FACTOR, world.y * CAVE_SCALE_FACTOR, world.z * CAVE_SCALE_FACTOR]);
        if cave_val > CAVE_THRESHOLD {
            return BlockType::Air;
        }
    }

    block
}
