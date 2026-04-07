use super::biome::{self, Biome};
use super::block::BlockType;
use super::chunk::{Chunk, CHUNK_SIZE};
use super::sphere::{self, Face};
use super::world::{TERRAIN_MAX_CY, TERRAIN_MIN_CY};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti};

pub(crate) const SEA_LEVEL: usize = 64;
const DIRT_DEPTH: usize = 4;
const MIN_HEIGHT: usize = 4;
const MAX_HEIGHT: usize = 700;
const CAVE_THRESHOLD: f64 = 0.55;
const CHUNK_LAYERS: usize = (TERRAIN_MAX_CY - TERRAIN_MIN_CY + 1) as usize;

// Noise scales — each parameter operates at a different spatial frequency
const CONTINENTALNESS_SCALE: f64 = 0.0008;
const EROSION_SCALE: f64 = 0.002;
const WEIRDNESS_SCALE: f64 = 0.004;
const DETAIL_SCALE: f64 = 0.02;
const MOUNTAIN_SCALE: f64 = 0.005;
const CAVE_SCALE: f64 = 0.05;
const TEMPERATURE_SCALE: f64 = 0.001;
const HUMIDITY_SCALE: f64 = 0.001;
const WARP_SCALE: f64 = 0.003;
pub(crate) const WARP_STRENGTH: f64 = 80.0;
const OVERHANG_SCALE: f64 = 0.04;
const OVERHANG_STRENGTH: f64 = 8.0;
const OVERHANG_BAND: usize = 20;

// Height contributions
const MOUNTAIN_AMPLITUDE: f64 = 50.0;
const DETAIL_AMPLITUDE: f64 = 8.0;
const WEIRDNESS_AMPLITUDE: f64 = 15.0;

pub(crate) struct WorldNoises {
    pub(crate) continentalness: Fbm<Perlin>,
    pub(crate) erosion_noise: Fbm<Perlin>,
    pub(crate) weirdness: Fbm<Perlin>,
    detail: Fbm<Perlin>,
    mountain: RidgedMulti<Perlin>,
    cave: Perlin,
    temperature: Fbm<Perlin>,
    humidity: Fbm<Perlin>,
    pub(crate) warp_x: Fbm<Perlin>,
    pub(crate) warp_z: Fbm<Perlin>,
    overhang: Perlin,
}

impl WorldNoises {
    pub(crate) fn new(seed: u32) -> Self {
        Self {
            continentalness: Fbm::<Perlin>::new(seed)
                .set_frequency(CONTINENTALNESS_SCALE)
                .set_octaves(5)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            erosion_noise: Fbm::<Perlin>::new(seed + 9)
                .set_frequency(EROSION_SCALE)
                .set_octaves(4)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            weirdness: Fbm::<Perlin>::new(seed + 10)
                .set_frequency(WEIRDNESS_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            detail: Fbm::<Perlin>::new(seed + 1)
                .set_frequency(DETAIL_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            mountain: RidgedMulti::<Perlin>::new(seed + 2).set_frequency(MOUNTAIN_SCALE).set_octaves(4),
            cave: Perlin::new(seed + 3),
            temperature: Fbm::<Perlin>::new(seed + 4)
                .set_frequency(TEMPERATURE_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            humidity: Fbm::<Perlin>::new(seed + 5)
                .set_frequency(HUMIDITY_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            warp_x: Fbm::<Perlin>::new(seed + 6)
                .set_frequency(WARP_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            warp_z: Fbm::<Perlin>::new(seed + 7)
                .set_frequency(WARP_SCALE)
                .set_octaves(3)
                .set_persistence(0.5)
                .set_lacunarity(2.0),
            overhang: Perlin::new(seed + 8),
        }
    }
}

/// All noise router parameters for a single (x, z) position.
#[allow(dead_code)]
struct TerrainParams {
    continentalness: f64,
    erosion: f64,
    weirdness: f64,
    temperature: f64,
    humidity: f64,
    height: usize,
    biome: Biome,
}

/// Sample all terrain parameters at face-local cube coordinates, applying
/// domain warping. The (u, v) tangent coordinates are projected onto the
/// planet sphere via [`sphere::noise_pos_on_face`] and 3D noise is sampled
/// at the resulting sphere point — this gives seamless terrain across face
/// edges for free (3D noise has no seams).
fn sample_params(noises: &WorldNoises, face: Face, u: f64, v: f64, erosion_map: Option<&super::erosion::ErosionMap>) -> TerrainParams {
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
fn continental_curve(c: f64) -> f64 {
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

pub(crate) fn compute_height_from_params(noises: &WorldNoises, face: Face, u: f64, v: f64, continentalness: f64, erosion: f64, weirdness: f64) -> usize {
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

/// Sample the surface block type at face-local coordinates.
pub(crate) fn sample_surface(noises: &WorldNoises, face: Face, u: f64, v: f64, erosion_map: Option<&super::erosion::ErosionMap>) -> (usize, BlockType) {
    let params = sample_params(noises, face, u, v, erosion_map);
    let surface = biome::surface_block(params.biome);
    (params.height, surface)
}

/// Generates a full column of chunks via 3D density sampling. For each block
/// in the column, we compute the world cartesian position via the cube-to-
/// sphere projection and evaluate a density function at that point. Density
/// > 0 → solid; density ≤ 0 with `|world| < surface_radius_at_sea_level` →
/// water; otherwise air. Because density is purely a function of world
/// position (and noise on direction), terrain is seamless across face edges.
pub fn generate_column(face: Face, chunk_x: i32, chunk_z: i32, seed: u32, erosion_map: Option<&super::erosion::ErosionMap>) -> Vec<Chunk> {
    let noises = WorldNoises::new(seed);
    let mut chunks: Vec<Chunk> = (0..CHUNK_LAYERS).map(|_| Chunk::new(BlockType::Air)).collect();

    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            fill_density_column(&mut chunks, face, chunk_x, chunk_z, x, z, &noises, erosion_map);
        }
    }

    chunks
}

/// Per-(x, z) column fill: walks every radial layer, evaluates density at the
/// block center, and writes the resulting block type.
fn fill_density_column(
    chunks: &mut [Chunk],
    face: Face,
    chunk_x: i32,
    chunk_z: i32,
    x: usize,
    z: usize,
    noises: &WorldNoises,
    erosion_map: Option<&super::erosion::ErosionMap>,
) {
    let sea_level_radius = sphere::SURFACE_RADIUS_BLOCKS as f64;
    let max_radius_seen = sea_level_radius + MOUNTAIN_AMPLITUDE + WEIRDNESS_AMPLITUDE + 50.0;
    for cy in 0..CHUNK_LAYERS {
        for ly in 0..CHUNK_SIZE {
            let cp = sphere::ChunkPos { face, cx: chunk_x, cy: cy as i32, cz: chunk_z };
            let local = glam::Vec3::new(x as f32 + 0.5, ly as f32 + 0.5, z as f32 + 0.5);
            let world = sphere::chunk_to_world(cp, local);
            let r = world.length();
            // Above any possible mountain → cheap air.
            if r > max_radius_seen + 1.0 {
                continue;
            }
            let block = sample_density_block(world, r, noises, erosion_map);
            if block != BlockType::Air {
                chunks[cy].set(x, ly, z, block);
            }
        }
    }
}

/// Density-based block lookup. `world` is the cartesian world position of the
/// block center; `r = |world|`. Samples per-direction noise once and decides
/// solid / water / air based on the radial distance.
fn sample_density_block(world: glam::DVec3, r: f64, noises: &WorldNoises, erosion_map: Option<&super::erosion::ErosionMap>) -> BlockType {
    let params = sample_params_at_world(noises, world, erosion_map);
    let surface_radius = sphere::PLANET_RADIUS_BLOCKS as f64 + params.height as f64;
    let sea_radius = sphere::SURFACE_RADIUS_BLOCKS as f64;

    if r > surface_radius + OVERHANG_BAND as f64 {
        return if r < sea_radius { BlockType::Water } else { BlockType::Air };
    }

    // Inside the overhang band: smoothly fade between solid and air via 3D noise.
    if r > surface_radius - OVERHANG_BAND as f64 {
        let base_density = (surface_radius - r) / OVERHANG_BAND as f64;
        let noise_val = noises.overhang.get([world.x * OVERHANG_SCALE, world.y * OVERHANG_SCALE, world.z * OVERHANG_SCALE]);
        let density = base_density + noise_val * (OVERHANG_STRENGTH / OVERHANG_BAND as f64);
        if density <= 0.0 {
            return if r < sea_radius { BlockType::Water } else { BlockType::Air };
        }
    }

    // Below the surface — pick stone / subsurface / surface based on depth.
    let depth_from_surface = (surface_radius - r).max(0.0);
    let block = if depth_from_surface < 1.0 {
        biome::surface_block(params.biome)
    } else if depth_from_surface < DIRT_DEPTH as f64 {
        biome::subsurface_block(params.biome)
    } else {
        BlockType::Stone
    };

    // 3D cave carving — spheres of air punched out of the solid mass.
    if depth_from_surface > 5.0 {
        let cave_val = noises.cave.get([world.x * CAVE_SCALE, world.y * CAVE_SCALE, world.z * CAVE_SCALE]);
        if cave_val > CAVE_THRESHOLD {
            return BlockType::Air;
        }
    }

    block
}

/// Direction-only sample of all terrain parameters. Replaces the old
/// `sample_params(face, u, v)` for the density-based pipeline. Two world
/// points sharing a radial direction produce identical params, which is
/// what makes the terrain seamless across cube faces.
fn sample_params_at_world(noises: &WorldNoises, world: glam::DVec3, erosion_map: Option<&super::erosion::ErosionMap>) -> TerrainParams {
    let p_warp = sphere::noise_pos_at_world(world);
    let warp_x = noises.warp_x.get(p_warp) * WARP_STRENGTH;
    let warp_z = noises.warp_z.get(p_warp) * WARP_STRENGTH;
    // Apply warp by rotating the direction slightly along two tangent axes.
    let dir = world.normalize_or(glam::DVec3::Y);
    let tangent_a = if dir.y.abs() < 0.9 { dir.cross(glam::DVec3::Y).normalize_or(glam::DVec3::X) } else { dir.cross(glam::DVec3::X).normalize_or(glam::DVec3::Z) };
    let tangent_b = dir.cross(tangent_a).normalize_or(glam::DVec3::Z);
    let warped_dir = (dir + tangent_a * (warp_x / sphere::SURFACE_RADIUS_BLOCKS as f64) + tangent_b * (warp_z / sphere::SURFACE_RADIUS_BLOCKS as f64)).normalize_or(dir);
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
    let mut height = (SEA_LEVEL as f64 + base + mountain + detail + weirdness_offset)
        .clamp(MIN_HEIGHT as f64, MAX_HEIGHT as f64) as usize;

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

/// Generate a 64³ LOD super-chunk by sampling terrain noise at `voxel_size` spacing.
pub fn generate_lod_super_chunk(origin: [i32; 3], voxel_size: u32, seed: u32, erosion_map: Option<&super::erosion::ErosionMap>) -> LodVoxelGrid {
    let noises = WorldNoises::new(seed);
    let vs = voxel_size as f64;
    let grid_size = CHUNK_SIZE * 4; // 64
    let mut blocks = vec![BlockType::Air; grid_size * grid_size * grid_size];

    for gz in 0..grid_size {
        for gx in 0..grid_size {
            let wx = origin[0] as f64 + gx as f64 * vs;
            let wz = origin[2] as f64 + gz as f64 * vs;
            // Phase C: LOD super-chunk path is disabled. Hardcode +Y face.
            let params = sample_params(&noises, Face::PosY, wx, wz, erosion_map);
            let surface = biome::surface_block(params.biome);
            let subsurface = biome::subsurface_block(params.biome);

            for gy in 0..grid_size {
                let wy = origin[1] as f64 + gy as f64 * vs;
                let y_top = (wy + vs - 1.0) as usize;
                let block = sample_block(y_top, params.height, surface, subsurface, &noises, wx, wy + vs * 0.5, wz);
                blocks[gx + gz * grid_size + gy * grid_size * grid_size] = block;
            }
        }
    }

    // Strip underground: keep only top SURFACE_DEPTH solid voxels per column.
    const SURFACE_DEPTH: usize = 2;
    for gz in 0..grid_size {
        for gx in 0..grid_size {
            let col = gx + gz * grid_size;
            let mut top = 0;
            for gy in (0..grid_size).rev() {
                if blocks[col + gy * grid_size * grid_size] != BlockType::Air {
                    top = gy;
                    break;
                }
            }
            if top >= SURFACE_DEPTH {
                for gy in 0..top - SURFACE_DEPTH {
                    blocks[col + gy * grid_size * grid_size] = BlockType::Air;
                }
            }
        }
    }

    LodVoxelGrid { blocks, size: grid_size }
}

fn sample_block(y: usize, height: usize, surface: BlockType, subsurface: BlockType, noises: &WorldNoises, wx: f64, wy: f64, wz: f64) -> BlockType {
    if y > height && y <= SEA_LEVEL {
        return BlockType::Water;
    }
    if y > height + OVERHANG_BAND {
        return BlockType::Air;
    }

    let band_bottom = height.saturating_sub(OVERHANG_BAND);
    let band_top = height + OVERHANG_BAND;
    if y >= band_bottom && y <= band_top {
        let base_density = (height as f64 - y as f64) / OVERHANG_BAND as f64;
        let noise_val = noises.overhang.get([wx * OVERHANG_SCALE, wy * OVERHANG_SCALE, wz * OVERHANG_SCALE]);
        let density = base_density + noise_val * (OVERHANG_STRENGTH / OVERHANG_BAND as f64);
        if density <= 0.0 {
            return BlockType::Air;
        }
    }

    let block = if y >= height {
        surface
    } else if y + DIRT_DEPTH > height {
        subsurface
    } else {
        BlockType::Stone
    };

    if y >= 1 && y + 5 <= height {
        let cave_val = noises.cave.get([wx * CAVE_SCALE, wy * CAVE_SCALE, wz * CAVE_SCALE]);
        if cave_val > CAVE_THRESHOLD {
            return BlockType::Air;
        }
    }

    block
}

/// A flat 64³ voxel grid for LOD super-chunk generation.
pub struct LodVoxelGrid {
    blocks: Vec<BlockType>,
    size: usize,
}

impl super::svdag::VoxelSource for LodVoxelGrid {
    fn get(&self, x: usize, y: usize, z: usize) -> BlockType {
        self.blocks[x + z * self.size + y * self.size * self.size]
    }
}
