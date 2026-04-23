use super::noises::{WorldNoises, WARP_STRENGTH, MOUNTAIN_AMPLITUDE, DETAIL_AMPLITUDE, WEIRDNESS_AMPLITUDE, SEA_LEVEL, MIN_HEIGHT, MAX_HEIGHT, CAVE_THRESHOLD, CAVE_MIN_DEPTH, CAVE_SCALE_FACTOR, DIRT_DEPTH, OVERHANG_SCALE_FACTOR, OVERHANG_STRENGTH_FACTOR, OVERHANG_BAND_SIZE};
use super::super::biome::{self, Biome};
use super::super::block::BlockType;
use super::super::sphere::{self, Face};
use noise::NoiseFn;

/// All noise router parameters for a single (x, z) position.
#[allow(dead_code)]
pub(crate) struct TerrainParams {
    pub(crate) continentalness: f64,
    pub(crate) erosion: f64,
    pub(crate) weirdness: f64,
    pub(crate) temperature: f64,
    pub(crate) humidity: f64,
    pub(crate) height: usize,
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

    let biome = biome::determine_biome(continentalness, temperature, humidity, erosion, weirdness, height, SEA_LEVEL);

    TerrainParams {
        continentalness,
        erosion,
        weirdness,
        temperature,
        humidity,
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
pub(crate) fn sample_params_at_world(noises: &WorldNoises, world: glam::DVec3, erosion_map: Option<&super::super::erosion::ErosionMap>) -> TerrainParams {
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

    let continentalness = noises.continentalness.get(p);
    let erosion = noises.erosion_noise.get(p);
    let weirdness = noises.weirdness.get(p);
    let temperature = noises.temperature.get(p);
    let humidity = noises.humidity.get(p);

    let base = continental_curve(continentalness);
    let erosion_factor = (0.3 + erosion * 0.7).clamp(0.3, 1.0);
    let mountain = noises.mountain.get(p) * MOUNTAIN_AMPLITUDE * erosion_factor;
    let detail = noises.detail.get(p) * DETAIL_AMPLITUDE * erosion_factor;
    let weirdness_offset = weirdness * WEIRDNESS_AMPLITUDE;
    let mut height = (SEA_LEVEL as f64 + base + mountain + detail + weirdness_offset).clamp(MIN_HEIGHT as f64, MAX_HEIGHT as f64) as usize;

    if let Some(emap) = erosion_map {
        // Erosion map is still indexed in face-local cube coords; sample with
        // the dominant face's projection of the warped direction.
        let face = sphere::face_for_cube_point(warped_dir);
        let (tu, tv, _) = sphere::face_basis(face);
        let cube_pt = warped_dir * sphere::CUBE_HALF_BLOCKS;
        let u = cube_pt.dot(tu.as_dvec3());
        let v = cube_pt.dot(tv.as_dvec3());
        let delta = emap.sample(u, v);
        height = (height as f64 + delta).clamp(MIN_HEIGHT as f64, MAX_HEIGHT as f64) as usize;
    }

    let biome = biome::determine_biome(continentalness, temperature, humidity, erosion, weirdness, height, SEA_LEVEL);
    TerrainParams {
        continentalness,
        erosion,
        weirdness,
        temperature,
        humidity,
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
pub fn surface_radius_at_world(noises: &WorldNoises, world: glam::DVec3, erosion_map: Option<&super::super::erosion::ErosionMap>) -> (f64, BlockType) {
    let params = sample_params_at_world(noises, world, erosion_map);
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
