/// Synthetic conditioning generator matching terrain-diffusion's SyntheticMapFactory.
///
/// Generates 5-channel conditioning maps from Perlin FBm noise, quantile-matched
/// to real Earth climate/elevation distributions. Our coarse heightmap data is
/// then merged in (elevation, temperature, precipitation channels), while temp_std
/// and precip_cv come from the synthetic distribution.

use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use super::rasterize::FaceGrid;

// ── Data quantile tables (from WorldClim/ETOPO via pipeline_data.json) ─────

const N_QUANTILES: usize = 64;

/// Elevation quantiles (meters) — from ETOPO global bathymetry+topography.
const DATA_Q_ELEV: [f64; N_QUANTILES] = [
    -8908.572708, -5859.425968, -5680.062500, -5556.000000, -5457.177844,
    -5374.312500, -5299.125000, -5225.500000, -5157.000000, -5090.937500,
    -5023.302496, -4955.750000, -4887.250000, -4819.437500, -4749.500000,
    -4681.910365, -4618.062500, -4553.947791, -4491.625000, -4430.500000,
    -4369.187500, -4307.250000, -4243.437500, -4176.312500, -4108.812500,
    -4039.562500, -3969.312500, -3896.562500, -3820.812500, -3742.375000,
    -3660.812500, -3577.750000, -3489.000000, -3391.062500, -3284.250000,
    -3163.437500, -3018.750580, -2845.187500, -2634.875000, -2340.437500,
    -1959.437500, -1470.437500,  -886.562500,  -270.270923,   -96.500000,
      -46.437500,    -5.687500,    31.473268,    75.718885,   116.210139,
      155.429860,   200.263131,   251.869130,   303.879387,   360.770172,
      423.069257,   498.589531,   597.614121,   735.122289,   921.481520,
     1128.382972,  1409.797512,  2069.464369,  5725.387708,
];

/// Temperature quantiles (°C) — from WorldClim BIO1 (annual mean).
const DATA_Q_TEMP: [f64; N_QUANTILES] = [
     -7.480439,  -3.040520,  -1.534993,  -0.305050,   0.748339,   1.682985,
      2.611660,   3.453644,   4.173894,   4.928205,   5.739545,   6.471782,
      7.171846,   7.972068,   8.723493,   9.449371,  10.184601,  10.935276,
     11.777507,  12.718906,  13.634449,  14.524478,  15.352643,  16.122212,
     16.804365,  17.497226,  18.239329,  19.020263,  19.798532,  20.545255,
     21.255982,  21.938234,  22.569903,  23.177217,  23.713912,  24.209672,
     24.644518,  25.024180,  25.342167,  25.614439,  25.836527,  26.028795,
     26.193630,  26.329586,  26.451043,  26.565504,  26.680805,  26.798440,
     26.921566,  27.048303,  27.183987,  27.342892,  27.522454,  27.730871,
     27.954905,  28.218535,  28.549502,  28.897528,  29.258546,  29.617545,
     30.013476,  30.514685,  31.287339,  35.577648,
];

/// Temperature seasonality quantiles — WorldClim BIO4 (std × 100).
const DATA_Q_TEMP_STD: [f64; N_QUANTILES] = [
    -1189.496463, -587.439450, -503.396615, -448.562572, -396.060428,
     -355.428057, -325.977221, -303.496501, -287.104973, -272.858219,
     -261.387864, -251.564437, -243.049454, -234.568460, -225.934447,
     -216.705458, -206.715007, -195.448638, -183.813790, -172.531911,
     -161.008176, -148.777967, -135.136217, -121.132120, -107.175605,
      -92.574378,  -78.915826,  -64.986228,  -51.172946,  -36.970333,
      -23.123696,   -8.370727,    6.936053,   21.514557,   35.332688,
       49.717945,   64.204089,   78.149468,   91.538829,  104.222613,
      116.357192,  127.472280,  138.447326,  149.835746,  161.555390,
      173.576563,  186.802237,  199.932344,  213.104929,  226.719433,
      240.782153,  254.911386,  268.991527,  283.818430,  300.975508,
      323.291020,  346.144232,  368.320274,  394.089274,  423.163447,
      452.450927,  484.692652,  526.894494,  757.595864,
];

/// Precipitation quantiles (mm/year) — WorldClim BIO12.
const DATA_Q_PRECIP: [f64; N_QUANTILES] = [
       0.0,    11.0,    24.0,    36.0,    56.0,    75.0,    97.0,   116.0,
     133.0,   153.0,   174.0,   196.0,   217.0,   239.0,   259.0,   280.0,
     301.0,   321.0,   342.0,   361.0,   380.0,   398.0,   416.0,   434.0,
     452.0,   470.0,   489.0,   506.0,   522.0,   540.0,   559.0,   578.0,
     598.0,   617.0,   637.0,   660.0,   687.0,   718.0,   752.0,   788.0,
     823.0,   863.0,   905.0,   949.0,   993.0,  1039.0,  1085.0,  1134.0,
    1187.0,  1243.0,  1305.0,  1373.0,  1441.0,  1512.0,  1583.0,  1661.0,
    1755.0,  1890.0,  2050.0,  2244.0,  2453.0,  2717.0,  3126.0,  6444.692,
];

/// Precipitation seasonality quantiles — WorldClim BIO15 (CV).
const DATA_Q_PRECIP_CV: [f64; N_QUANTILES] = [
      0.000000,  12.661659,  15.258313,  17.282962,  19.123285,  20.701857,
     22.219669,  23.798007,  25.369890,  26.730541,  28.289891,  29.896200,
     31.372248,  32.771257,  33.967217,  35.457512,  36.933733,  38.568454,
     40.240753,  41.939310,  43.566444,  45.219568,  46.838436,  48.243933,
     49.609552,  50.917416,  52.181744,  53.506839,  54.976086,  56.517271,
     58.169158,  59.769242,  61.409418,  63.109088,  64.836380,  66.562249,
     68.332195,  70.276683,  72.411980,  74.551362,  76.810668,  78.906723,
     80.955401,  82.978699,  84.954300,  87.001171,  89.085048,  91.126549,
     93.207723,  95.381776,  97.613502, 100.078773, 102.574386, 105.145791,
    107.743195, 110.365249, 112.870621, 115.338885, 118.236280, 121.787155,
    127.492922, 137.723237, 150.279064, 199.415089,
];

const ALL_DATA_Q: [&[f64; N_QUANTILES]; 5] = [
    &DATA_Q_ELEV, &DATA_Q_TEMP, &DATA_Q_TEMP_STD, &DATA_Q_PRECIP, &DATA_Q_PRECIP_CV,
];

// Temperature std regression coefficients (from WorldClim BIO4 vs BIO1)
const A_TEMP_STD: f64 = -34.7552214802;
const B_TEMP_STD: f64 = 1189.4964628859;
const TEMP_STD_P1: f64 = -900.9432288232;
const TEMP_STD_P99: f64 = 630.0148360773;

// ── Noise configuration (matches FastNoiseLite settings) ───────────────────

const BASE_FREQUENCY: f64 = 0.05;
const LACUNARITY: f64 = 2.0;
const GAIN: f64 = 0.5;
const OCTAVES: [usize; 5] = [4, 2, 4, 4, 4];
const QUANTILE_GRID_SIZE: usize = 1024;
const QUANTILE_GRID_STEP: f64 = 32.0;
const QUANTILE_EPS: f64 = 1e-4;

/// Pre-computed synthetic conditioning state for a given world seed.
pub(super) struct SyntheticConditioner {
    noises: [Fbm<Perlin>; 5],
    noise_quantiles: [[f64; N_QUANTILES]; 5],
}

impl SyntheticConditioner {
    /// Build conditioner for a given seed. Computes noise quantiles (~50ms).
    pub fn new(seed: u64) -> Self {
        let mut noises: [Fbm<Perlin>; 5] = std::array::from_fn(|ch| {
            let ch_seed = ((seed.wrapping_add(ch as u64 + 1)) & 0x7FFFFFFF) as u32;
            Fbm::new(ch_seed)
                .set_frequency(BASE_FREQUENCY)
                .set_lacunarity(LACUNARITY)
                .set_persistence(GAIN)
                .set_octaves(OCTAVES[ch])
        });

        let noise_quantiles = std::array::from_fn(|ch| {
            build_noise_quantiles(&noises[ch])
        });

        Self { noises, noise_quantiles }
    }

    /// Generate 5-channel synthetic conditioning for a tile within a grid.
    /// The grid can be a face or a cross layout — uses (tx, ty) as tile origin
    /// and grid.resolution as row width.
    /// Returns (5 * tile_h * tile_w) flat array: [elev_sqrt, temp, temp_std, precip, precip_cv].
    ///
    /// Channels 0 (elevation), 1 (temperature), 3 (precipitation) are taken from
    /// the coarse heightmap. Channels 2 (temp_std) and 4 (precip_cv) come from
    /// the synthetic distribution.
    pub fn generate_conditioning(
        &self,
        face: &FaceGrid,
        tx: u32, ty: u32,
        tile_w: u32, tile_h: u32,
    ) -> Vec<f32> {
        let tw = tile_w as usize;
        let th = tile_h as usize;
        let grid_w = face.resolution as usize;
        let grid_h = face.height as usize;
        let pixels = tw * th;
        let mut raw = vec![[0.0f64; 5]; pixels];

        // Sample synthetic noise for all 5 channels using cross-layout coordinates.
        // This gives continuous noise across the entire cross.
        for r in 0..th {
            for c in 0..tw {
                let px = r * tw + c;
                let x = (tx as f64 + c as f64) * 1.0;
                let y = (ty as f64 + r as f64) * 1.0;

                for ch in 0..5 {
                    let noise_val = self.noises[ch].get([x, y]);
                    raw[px][ch] = quantile_transform(
                        noise_val,
                        &self.noise_quantiles[ch],
                        ALL_DATA_Q[ch],
                    );
                }
            }
        }

        // Override channels 0, 1, 3 with our coarse heightmap data.
        for r in 0..th {
            for c in 0..tw {
                let gy = (ty as usize + r).min(grid_h - 1);
                let gx = (tx as usize + c).min(grid_w - 1);
                let src = gy * grid_w + gx;
                let px = r * tw + c;

                let our_elev_m = face.elevation[src] as f64 * 1000.0;
                raw[px][0] = our_elev_m;

                raw[px][1] = face.temperature[src] as f64;
                raw[px][3] = face.precipitation[src] as f64;
            }
        }

        // Finalize: post-processing matching SyntheticMapFactory
        finalize(&mut raw);

        // Pack into flat (5, tile_h, tile_w) output
        let mut out = vec![0.0f32; 5 * pixels];
        for px in 0..pixels {
            for ch in 0..5 {
                out[ch * pixels + px] = raw[px][ch] as f32;
            }
        }
        out
    }
}

// ── Noise quantile computation ─────────────────────────────────────────────

fn build_noise_quantiles(noise: &Fbm<Perlin>) -> [f64; N_QUANTILES] {
    let n = QUANTILE_GRID_SIZE;
    let mut values = Vec::with_capacity(n * n);
    for r in 0..n {
        for c in 0..n {
            let x = c as f64 * QUANTILE_GRID_STEP;
            let y = r as f64 * QUANTILE_GRID_STEP;
            values.push(noise.get([x, y]));
        }
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let total = values.len();
    let mut q = [0.0f64; N_QUANTILES];
    for i in 0..N_QUANTILES {
        let pct = QUANTILE_EPS + i as f64 * (1.0 - 2.0 * QUANTILE_EPS) / (N_QUANTILES - 1) as f64;
        let idx = pct * (total - 1) as f64;
        let lo = (idx as usize).min(total - 1);
        let hi = (lo + 1).min(total - 1);
        let frac = idx - lo as f64;
        q[i] = values[lo] + frac * (values[hi] - values[lo]);
    }

    // Ensure strictly increasing
    let mut min_diff = f64::MAX;
    for i in 1..N_QUANTILES {
        if q[i] > q[i - 1] {
            min_diff = min_diff.min(q[i] - q[i - 1]);
        }
    }
    if min_diff == f64::MAX {
        min_diff = 1e-10;
    }
    for i in 1..N_QUANTILES {
        if q[i] <= q[i - 1] {
            q[i] = q[i - 1] + min_diff * 0.1;
        }
    }
    q
}

// ── Quantile transform (piecewise linear interpolation) ────────────────────

fn quantile_transform(val: f64, noise_q: &[f64; N_QUANTILES], data_q: &[f64; N_QUANTILES]) -> f64 {
    if val <= noise_q[0] {
        return data_q[0];
    }
    if val >= noise_q[N_QUANTILES - 1] {
        return data_q[N_QUANTILES - 1];
    }
    // Binary search
    let mut lo = 0;
    let mut hi = N_QUANTILES - 1;
    while hi - lo > 1 {
        let mid = (lo + hi) / 2;
        if noise_q[mid] <= val {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = (val - noise_q[lo]) / (noise_q[hi] - noise_q[lo]);
    data_q[lo] + t * (data_q[hi] - data_q[lo])
}

// ── Finalize post-processing ───────────────────────────────────────────────

fn finalize(raw: &mut [[f64; 5]]) {
    for px in raw.iter_mut() {
        let elev = px[0];
        let precip = px[3];

        // Temperature: lapse rate correction + clamp + cold-stretch
        let lapse = ((-6.5 + 0.0015 * precip) / 1000.0).clamp(-9.8 / 1000.0, -4.0 / 1000.0);
        let temp = px[1] + lapse * elev.max(0.0);
        let temp = temp.clamp(-10.0, 40.0);
        px[1] = if temp > 20.0 { temp } else { (temp - 20.0) * 1.25 + 20.0 };

        // Temp std: de-correlate, rescale, re-correlate, floor at 20
        let temp_std_raw = px[2];
        let t01 = (temp_std_raw - TEMP_STD_P1) / (TEMP_STD_P99 - TEMP_STD_P1);
        let baseline = TEMP_STD_P1.max(-(A_TEMP_STD * px[1] + B_TEMP_STD));
        let temp_std = t01 * (TEMP_STD_P99 - baseline) + baseline + (A_TEMP_STD * px[1] + B_TEMP_STD);
        px[2] = temp_std.max(20.0);

        // Precip std: dampen at high precipitation
        px[4] *= ((185.0 - 0.04111 * precip) / 185.0).max(0.0);

        // Elevation: signed sqrt transform (applied last)
        px[0] = elev.signum() * elev.abs().sqrt();
    }
}
