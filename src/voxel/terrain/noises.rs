use noise::{Fbm, MultiFractal, NoiseFn, Perlin, RidgedMulti};

pub(crate) const SEA_LEVEL: usize = 64;
pub const CHUNK_LAYERS: usize = 48;

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

// Overhang noise is the main *gradient* source in the surface band: a 3D
// Perlin field that displaces the apparent surface by up to ±STRENGTH blocks.
// For two adjacent columns the displacement can differ by up to ~2×STRENGTH
// in the worst case, so STRENGTH directly bounds the cliff height it produces.
// Keep small for walkable terrain.
const OVERHANG_SCALE: f64 = 0.04;
const OVERHANG_STRENGTH: f64 = 1.5;
const OVERHANG_BAND: usize = 20;

// Height contributions. Each amplitude is budgeted against its frequency so
// that `2 × amplitude × frequency × octaves` stays under ~1 block of slope
// per block of horizontal — pinned by `terrain_height_field_has_no_extreme_cliffs`.
// Octave count comes from the FBM/RidgedMulti config below; persistence=0.5
// and lacunarity=2 keep `amplitude × frequency` constant across octaves, so
// each octave contributes equally to the total max gradient.
pub(crate) const MOUNTAIN_AMPLITUDE: f64 = 25.0;
pub(crate) const DETAIL_AMPLITUDE: f64 = 4.0;
pub(crate) const WEIRDNESS_AMPLITUDE: f64 = 10.0;

// Overhang and cave constants exported for use in other modules
pub(crate) const CAVE_SCALE_FACTOR: f64 = CAVE_SCALE;
pub(crate) const CAVE_THRESHOLD: f64 = 0.55;
pub(crate) const CAVE_MIN_DEPTH: usize = 20;
pub(crate) const OVERHANG_SCALE_FACTOR: f64 = OVERHANG_SCALE;
pub(crate) const OVERHANG_STRENGTH_FACTOR: f64 = OVERHANG_STRENGTH;
pub(crate) const OVERHANG_BAND_SIZE: usize = OVERHANG_BAND;
pub(crate) const DIRT_DEPTH: usize = 4;
pub(crate) const MIN_HEIGHT: usize = 4;
pub(crate) const MAX_HEIGHT: usize = 760;

pub(crate) struct WorldNoises {
    pub(crate) continentalness: Fbm<Perlin>,
    pub(crate) erosion_noise: Fbm<Perlin>,
    pub(crate) weirdness: Fbm<Perlin>,
    pub(crate) detail: Fbm<Perlin>,
    pub(crate) mountain: RidgedMulti<Perlin>,
    pub(crate) cave: Perlin,
    pub(crate) temperature: Fbm<Perlin>,
    pub(crate) humidity: Fbm<Perlin>,
    pub(crate) warp_x: Fbm<Perlin>,
    pub(crate) warp_y: Fbm<Perlin>,
    pub(crate) warp_z: Fbm<Perlin>,
    pub(crate) overhang: Perlin,
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
            warp_y: Fbm::<Perlin>::new(seed + 11)
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
