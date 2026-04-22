use glam::DVec3;

use super::elevation::gaussian;

const HOTSPOT_COUNT: usize = 8;
const HOTSPOT_OCEANIC_SIGMA: f64 = 0.06;
const HOTSPOT_OCEANIC_STRENGTH: f64 = 5.5;
const HOTSPOT_CONTINENTAL_SIGMA: f64 = 0.10;
const HOTSPOT_CONTINENTAL_STRENGTH: f64 = 1.0;

struct Hotspot {
    center: DVec3,
    sigma: f64,
    strength: f64,
}

pub(super) fn generate_and_apply(
    points: &[DVec3],
    is_continental: &[bool],
    seed: u64,
    elevation: &mut [f32],
) {
    let hotspots = generate(points, is_continental, seed);
    apply(points, &hotspots, elevation);
}

fn generate(points: &[DVec3], is_continental: &[bool], seed: u64) -> Vec<Hotspot> {
    let mut rng = super::splitmix64(seed.wrapping_add(999));
    let mut hotspots = Vec::with_capacity(HOTSPOT_COUNT);

    for _ in 0..HOTSPOT_COUNT {
        rng = super::splitmix64(rng);
        let z = (rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
        rng = super::splitmix64(rng);
        let theta = (rng as f64 / u64::MAX as f64) * std::f64::consts::TAU;
        let r = (1.0 - z * z).sqrt();
        let center = DVec3::new(r * theta.cos(), r * theta.sin(), z);

        let nearest = points
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.dot(center).partial_cmp(&b.dot(center)).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let (sigma, strength) = if is_continental[nearest] {
            (HOTSPOT_CONTINENTAL_SIGMA, HOTSPOT_CONTINENTAL_STRENGTH)
        } else {
            (HOTSPOT_OCEANIC_SIGMA, HOTSPOT_OCEANIC_STRENGTH)
        };

        hotspots.push(Hotspot { center, sigma, strength });
    }

    hotspots
}

fn apply(points: &[DVec3], hotspots: &[Hotspot], elevation: &mut [f32]) {
    for hs in hotspots {
        for (i, &p) in points.iter().enumerate() {
            let dist = p.dot(hs.center).clamp(-1.0, 1.0).acos();
            let dome = hs.strength * gaussian(dist, hs.sigma);
            if dome > 0.01 {
                elevation[i] += dome as f32;
            }
        }
    }
}
