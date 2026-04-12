use std::f64::consts::PI;

use glam::DVec3;

const GOLDEN_RATIO: f64 = 1.618_033_988_749_895;
const GOLDEN_RATIO_INVERSE: f64 = GOLDEN_RATIO - 1.0;
const SQRT_5: f64 = 2.236_067_977_499_790;

/// The zone formula divides by sin^2(theta), which blows up at the poles.
const POLAR_EPSILON: f64 = 1e-30;

/// Below 8 points the Voronoi topology at the poles hasn't converged yet.
const MIN_POINT_COUNT: u32 = 8;

/// The first two basis vectors are degenerate; usable grids start at index 2.
const MIN_ZONE: u32 = 2;

/// Fractional part of a*b using fused multiply-add to avoid a second rounding step.
fn frac_mul(a: f64, b: f64) -> f64 {
    let product = a.mul_add(b, 0.0);
    a.mul_add(b, -product.floor())
}

/// The spiral's azimuth step for a given Fibonacci-number spacing.
fn basis_vector_azimuth(fibonacci_number: f64) -> f64 {
    2.0 * PI * frac_mul(fibonacci_number + 1.0, GOLDEN_RATIO_INVERSE)
        - 2.0 * PI * GOLDEN_RATIO_INVERSE
}

/// Which pair of basis vectors produces the most square grid cells at this latitude.
fn dominant_zone(cos_polar: f64, point_count: f64) -> u32 {
    let sin_squared = 1.0 - cos_polar * cos_polar;
    if sin_squared < POLAR_EPSILON {
        return MIN_ZONE;
    }
    let raw = (SQRT_5 * point_count * PI * sin_squared).ln()
        / (GOLDEN_RATIO * GOLDEN_RATIO).ln();
    (raw.floor() as u32).max(MIN_ZONE)
}

/// Binet's closed-form approximation, rounded to the nearest integer.
fn approx_fibonacci(k: u32) -> f64 {
    (GOLDEN_RATIO.powi(k as i32) / SQRT_5).round()
}

/// The spiral distributes points linearly in z; invert that to recover an index.
fn z_to_index(z: f64, point_count: f64) -> u32 {
    (point_count * 0.5 - z * point_count * 0.5) as u32
}

fn to_cartesian(azimuth: f64, z: f64) -> DVec3 {
    let sin_polar = (1.0 - z * z).sqrt();
    DVec3::new(
        azimuth.cos() * sin_polar,
        azimuth.sin() * sin_polar,
        z,
    )
}

/// Nearly uniform point distribution on the unit sphere using a golden-ratio spiral.
///
/// Both forward (index -> point) and inverse (point -> nearest index) run in O(1)
/// with no precomputed tables.
pub struct SphericalFibonacci {
    point_count: u32,
}

impl SphericalFibonacci {
    pub fn new(point_count: u32) -> Self {
        assert!(point_count >= MIN_POINT_COUNT);
        Self { point_count }
    }

    pub fn point_count(&self) -> u32 {
        self.point_count
    }

    pub fn index_to_point(&self, index: u32) -> DVec3 {
        let count = self.point_count as f64;
        let i = index as f64;
        let z = 1.0 - (2.0 * i + 1.0) / count;
        let azimuth = 2.0 * PI * frac_mul(i, GOLDEN_RATIO_INVERSE);
        to_cartesian(azimuth, z)
    }

    pub fn all_points(&self) -> Vec<DVec3> {
        (0..self.point_count).map(|i| self.index_to_point(i)).collect()
    }

    /// O(1) — no precomputation or table lookup.
    pub fn nearest_index(&self, direction: DVec3) -> u32 {
        let count = self.point_count as f64;
        let zone = dominant_zone(direction.z, count);

        let fib_k = approx_fibonacci(zone);
        let fib_k1 = approx_fibonacci(zone + 1);

        let basis = GridBasis::from_fibonacci_pair(fib_k, fib_k1, count);
        let cell = basis.containing_cell(direction);

        self.find_nearest_in_cell(direction, &basis, &cell)
    }

    /// The largest angular radius guaranteed to fit inside every Voronoi cell.
    pub fn min_cell_radius(&self) -> f64 {
        0.5 * self.index_to_point(3).dot(self.index_to_point(0)).acos()
    }

    /// The worst-case angular distance from any sphere point to its nearest sample.
    pub fn max_cell_radius(&self) -> f64 {
        let p1 = self.index_to_point(1);
        let circumcenter = (self.index_to_point(2) - p1).cross(self.index_to_point(4) - p1);
        circumcenter.normalize().dot(p1).acos()
    }

    fn find_nearest_in_cell(&self, direction: DVec3, basis: &GridBasis, cell: &GridCell) -> u32 {
        let mut best_distance = f64::INFINITY;
        let mut best_index = 0u32;

        for corner in 0..4u32 {
            let offset_u = (corner % 2) as f64;
            let offset_v = (corner / 2) as f64;

            // Near the poles a corner can land outside [-1, 1]; reflecting it back
            // still picks a valid candidate without needing a special polar branch.
            let mut z = basis.z_at_offset(cell, offset_u, offset_v);
            z = z.clamp(-1.0, 1.0) * 2.0 - z;

            let candidate = z_to_index(z, self.point_count as f64).min(self.point_count - 1);

            // Using the forward mapping here instead of the grid coordinates
            // avoids accumulating floating-point error through the basis matrix.
            let candidate_point = self.index_to_point(candidate);
            let distance = (candidate_point - direction).length_squared();
            if distance < best_distance {
                best_distance = distance;
                best_index = candidate;
            }
        }

        best_index
    }
}

/// The spiral's local 2D grid at a given latitude, used to narrow the
/// nearest-neighbor search to four candidate points.
struct GridBasis {
    z_k: f64,
    z_k1: f64,
    z_offset: f64,
    inv_00: f64,
    inv_01: f64,
    inv_10: f64,
    inv_11: f64,
}

struct GridCell {
    u: f64,
    v: f64,
}

impl GridBasis {
    fn from_fibonacci_pair(fib_k: f64, fib_k1: f64, point_count: f64) -> Self {
        let azimuth_k = basis_vector_azimuth(fib_k);
        let azimuth_k1 = basis_vector_azimuth(fib_k1);
        let z_k = -2.0 * fib_k / point_count;
        let z_k1 = -2.0 * fib_k1 / point_count;

        let det = azimuth_k * z_k1 - azimuth_k1 * z_k;

        Self {
            z_k,
            z_k1,
            // Offsets so that grid coordinate (0,0) lands on the first spiral point.
            z_offset: 1.0 - 1.0 / point_count,
            inv_00: z_k1 / det,
            inv_01: -azimuth_k1 / det,
            inv_10: -z_k / det,
            inv_11: azimuth_k / det,
        }
    }

    fn containing_cell(&self, point: DVec3) -> GridCell {
        let azimuth = point.y.atan2(point.x).min(PI);
        let dz = point.z - self.z_offset;
        GridCell {
            u: (self.inv_00 * azimuth + self.inv_01 * dz).floor(),
            v: (self.inv_10 * azimuth + self.inv_11 * dz).floor(),
        }
    }

    fn z_at_offset(&self, cell: &GridCell, offset_u: f64, offset_v: f64) -> f64 {
        self.z_k * (cell.u + offset_u) + self.z_k1 * (cell.v + offset_v) + self.z_offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_points_are_on_unit_sphere() {
        let sf = SphericalFibonacci::new(1024);
        for i in 0..sf.point_count() {
            let len = sf.index_to_point(i).length();
            assert!((len - 1.0).abs() < 1e-14, "point {i}: length {len}");
        }
    }

    #[test]
    fn forward_z_range_excludes_poles() {
        let sf = SphericalFibonacci::new(256);
        assert!(sf.index_to_point(0).z < 1.0);
        assert!(sf.index_to_point(sf.point_count() - 1).z > -1.0);
    }

    #[test]
    fn inverse_recovers_all_indices() {
        let sf = SphericalFibonacci::new(512);
        for i in 0..sf.point_count() {
            let point = sf.index_to_point(i);
            assert_eq!(i, sf.nearest_index(point), "roundtrip failed for {i}");
        }
    }

    #[test]
    fn inverse_matches_brute_force() {
        let sf = SphericalFibonacci::new(256);
        let probes = [
            DVec3::X,
            DVec3::Y,
            DVec3::Z,
            DVec3::NEG_Z,
            DVec3::new(1.0, 1.0, 1.0).normalize(),
            DVec3::new(-0.3, 0.7, -0.5).normalize(),
            DVec3::new(0.9, -0.1, 0.2).normalize(),
        ];
        for probe in &probes {
            let found = sf.nearest_index(*probe);
            let found_dist = (sf.index_to_point(found) - *probe).length_squared();
            for i in 0..sf.point_count() {
                let dist = (sf.index_to_point(i) - *probe).length_squared();
                assert!(found_dist <= dist + 1e-12, "{probe}: {found} not nearest, {i} closer");
            }
        }
    }

    #[test]
    fn voronoi_bounds_are_ordered() {
        let sf = SphericalFibonacci::new(1024);
        let min = sf.min_cell_radius();
        let max = sf.max_cell_radius();
        assert!(0.0 < min);
        assert!(min < max);
        assert!(max < PI / 2.0);
    }
}
