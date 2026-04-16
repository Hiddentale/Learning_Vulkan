use glam::DVec3;

const LAT_BINS: usize = 64;
const LON_BINS: usize = 128;
const BIN_COUNT: usize = LAT_BINS * LON_BINS;

/// Lat/lon bin grid on the unit sphere for spatial nearest-neighbor queries.
/// Stores point indices (u32) into an external positions array.
pub struct SphereGrid {
    bins: Vec<Vec<u32>>,
}

impl SphereGrid {
    /// Build a grid from a set of unit-sphere positions.
    pub fn build(positions: &[DVec3]) -> Self {
        let mut bins: Vec<Vec<u32>> = (0..BIN_COUNT).map(|_| Vec::new()).collect();
        for (i, p) in positions.iter().enumerate() {
            let (lat, lon) = bin_of(*p);
            bins[lat * LON_BINS + lon].push(i as u32);
        }
        Self { bins }
    }

    /// Find the k nearest points to `dir`. Returns up to k `(index, arc_distance)`
    /// pairs sorted by ascending distance. `positions` is the same array passed
    /// to `build`.
    pub fn find_nearest_k(
        &self,
        dir: DVec3,
        positions: &[DVec3],
        k: usize,
    ) -> Vec<(u32, f64)> {
        if positions.is_empty() || k == 0 {
            return Vec::new();
        }
        // Average angular spacing for N points on a unit sphere: sqrt(4π/N).
        // Start with 3× that as the search radius.
        let avg_spacing = (4.0 * std::f64::consts::PI / positions.len() as f64).sqrt();
        let mut radius = avg_spacing * 3.0;

        loop {
            let candidates = self.candidates_within(dir, radius, positions);
            if candidates.len() >= k || radius >= std::f64::consts::PI {
                return top_k(candidates, k);
            }
            radius = (radius * 2.0).min(std::f64::consts::PI);
        }
    }

    fn candidates_within(
        &self,
        dir: DVec3,
        radius: f64,
        positions: &[DVec3],
    ) -> Vec<(u32, f64)> {
        let (plat, plon) = bin_of(dir);
        let lat_width = std::f64::consts::PI / LAT_BINS as f64;
        let lon_width = std::f64::consts::TAU / LON_BINS as f64;
        let lat_span = (radius / lat_width).ceil() as usize + 1;
        let lat = dir.y.clamp(-1.0, 1.0).asin();
        let cos_lat = lat.cos().max(0.01);
        let lon_span = (radius / (cos_lat * lon_width)).ceil() as usize + 1;

        let lat_lo = plat.saturating_sub(lat_span);
        let lat_hi = (plat + lat_span).min(LAT_BINS - 1);

        let cos_radius = radius.cos();
        let mut result = Vec::new();

        for lat_bin in lat_lo..=lat_hi {
            for offset in 0..=(2 * lon_span) {
                let lon_bin = (plon + LON_BINS + offset - lon_span) % LON_BINS;
                for &idx in &self.bins[lat_bin * LON_BINS + lon_bin] {
                    let dot = dir.dot(positions[idx as usize]);
                    if dot >= cos_radius {
                        let dist = dot.clamp(-1.0, 1.0).acos();
                        result.push((idx, dist));
                    }
                }
            }
        }

        result
    }
}

fn bin_of(p: DVec3) -> (usize, usize) {
    let lat = p.y.clamp(-1.0, 1.0).asin();
    let lon = p.z.atan2(p.x);
    let lat_bin = ((lat / std::f64::consts::PI + 0.5) * LAT_BINS as f64)
        .max(0.0)
        .min(LAT_BINS as f64 - 1.0) as usize;
    let lon_bin = ((lon / std::f64::consts::TAU + 0.5) * LON_BINS as f64)
        .max(0.0)
        .min(LON_BINS as f64 - 1.0) as usize;
    (lat_bin, lon_bin)
}

fn top_k(mut candidates: Vec<(u32, f64)>, k: usize) -> Vec<(u32, f64)> {
    if candidates.len() <= k {
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        return candidates;
    }
    candidates.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
    candidates.truncate(k);
    candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    candidates
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tectonic_simulation::fibonnaci_spiral::SphericalFibonacci;

    fn fibonacci_points(n: u32) -> Vec<DVec3> {
        SphericalFibonacci::new(n).all_points()
    }

    fn brute_nearest(dir: DVec3, positions: &[DVec3]) -> u32 {
        positions
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.dot(dir).partial_cmp(&b.dot(dir)).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap()
    }

    #[test]
    fn nearest_1_matches_brute_force() {
        let points = fibonacci_points(10_000);
        let grid = SphereGrid::build(&points);
        // Test 200 random query directions spread across the sphere.
        let queries = fibonacci_points(200);
        for q in &queries {
            let result = grid.find_nearest_k(*q, &points, 1);
            assert_eq!(result.len(), 1);
            let expected = brute_nearest(*q, &points);
            assert_eq!(
                result[0].0, expected,
                "grid nearest {} != brute {}",
                result[0].0, expected
            );
        }
    }

    #[test]
    fn nearest_6_sorted_ascending() {
        let points = fibonacci_points(10_000);
        let grid = SphereGrid::build(&points);
        let queries = fibonacci_points(50);
        for q in &queries {
            let result = grid.find_nearest_k(*q, &points, 6);
            assert_eq!(result.len(), 6);
            for w in result.windows(2) {
                assert!(
                    w[0].1 <= w[1].1,
                    "not sorted: {} > {}",
                    w[0].1, w[1].1
                );
            }
        }
    }

    #[test]
    fn uniform_sphere_all_points_have_6_neighbors() {
        let points = fibonacci_points(5_000);
        let grid = SphereGrid::build(&points);
        for p in &points {
            let result = grid.find_nearest_k(*p, &points, 6);
            assert_eq!(result.len(), 6, "point has fewer than 6 neighbors");
        }
    }

    #[test]
    fn antipodal_query_returns_results() {
        let points = fibonacci_points(1_000);
        let grid = SphereGrid::build(&points);
        // Query at both poles.
        for dir in [DVec3::Y, DVec3::NEG_Y, DVec3::X, DVec3::NEG_X] {
            let result = grid.find_nearest_k(dir, &points, 6);
            assert_eq!(result.len(), 6, "failed for dir {dir}");
        }
    }
}
