use glam::DVec3;

use super::fibonnaci_spiral::SphericalFibonacci;
use super::plate_seed_placement::Adjacency;
use super::plates::{CrustData, Plate};
use super::simulate::Simulation;
use super::spherical_delaunay_triangulation::SphericalDelaunay;

/// How many steps between resamples.
pub const RESAMPLE_INTERVAL: usize = 20;

/// Resample the simulation onto a fresh Fibonacci grid.
///
/// For each new point, finds the nearest old point and copies its plate
/// ownership and crust data. Then rebuilds the Delaunay and adjacency.
pub fn resample(sim: &mut Simulation, point_count: u32) {
    let old_points = &sim.points;

    // Build per-point lookups from current plate state.
    let mut point_plate = vec![0u32; old_points.len()];
    let mut point_local = vec![0usize; old_points.len()];
    for (plate_idx, plate) in sim.plates.iter().enumerate() {
        for (local, &global) in plate.point_indices.iter().enumerate() {
            point_plate[global as usize] = plate_idx as u32;
            point_local[global as usize] = local;
        }
    }

    // Spatial hash on old moved positions for fast nearest lookup.
    let hash = SphereHash::new(old_points);

    // Fresh Fibonacci grid.
    let fib = SphericalFibonacci::new(point_count);
    let new_points = fib.all_points();

    // For each new point: find nearest old point, take its plate and crust.
    let plate_count = sim.plates.len();
    let mut new_plate_points: Vec<Vec<u32>> = vec![Vec::new(); plate_count];
    let mut new_plate_crust: Vec<Vec<CrustData>> = vec![Vec::new(); plate_count];

    for (new_idx, &new_p) in new_points.iter().enumerate() {
        let nearest_old = hash.nearest(new_p, old_points);
        let owner = point_plate[nearest_old as usize];
        let local = point_local[nearest_old as usize];
        let crust = sim.plates[owner as usize].crust[local].clone();

        new_plate_points[owner as usize].push(new_idx as u32);
        new_plate_crust[owner as usize].push(crust);
    }

    // Rebuild plates with new point assignments.
    for (plate_idx, plate) in sim.plates.iter_mut().enumerate() {
        plate.point_indices = new_plate_points[plate_idx].clone();
        plate.crust = new_plate_crust[plate_idx].clone();
    }

    // Replace simulation state.
    let new_delaunay = SphericalDelaunay::from_points(&new_points);
    sim.adjacency = Adjacency::from_delaunay(new_points.len(), &new_delaunay);
    sim.points = new_points;
}

/// Spatial hash over a lat/lon grid for fast nearest-point lookup.
struct SphereHash {
    bins: Vec<Vec<u32>>,
    lat_bins: usize,
    lon_bins: usize,
}

impl SphereHash {
    fn new(points: &[DVec3]) -> Self {
        let lat_bins = 90;
        let lon_bins = 180;
        let mut bins = vec![Vec::new(); lat_bins * lon_bins];
        for (i, &p) in points.iter().enumerate() {
            let (lb, lob) = to_bin(p, lat_bins, lon_bins);
            bins[lb * lon_bins + lob].push(i as u32);
        }
        Self { bins, lat_bins, lon_bins }
    }

    fn nearest(&self, dir: DVec3, points: &[DVec3]) -> u32 {
        let (lb, lob) = to_bin(dir, self.lat_bins, self.lon_bins);
        let mut best = 0u32;
        let mut best_dot = f64::NEG_INFINITY;

        // 3×3 neighborhood covers ~27 points on average — always enough.
        for dlat in -1i32..=1 {
            let rlat = (lb as i32 + dlat).clamp(0, self.lat_bins as i32 - 1) as usize;
            for dlon in -1i32..=1 {
                let rlon = (lob as i32 + dlon).rem_euclid(self.lon_bins as i32) as usize;
                for &idx in &self.bins[rlat * self.lon_bins + rlon] {
                    let d = dir.dot(points[idx as usize]);
                    if d > best_dot {
                        best_dot = d;
                        best = idx;
                    }
                }
            }
        }

        // Fallback for empty bins near poles.
        if best_dot == f64::NEG_INFINITY {
            for radius in 2..self.lat_bins as i32 {
                for dlat in -radius..=radius {
                    let rlat = (lb as i32 + dlat).clamp(0, self.lat_bins as i32 - 1) as usize;
                    for dlon in -radius..=radius {
                        let rlon = (lob as i32 + dlon).rem_euclid(self.lon_bins as i32) as usize;
                        for &idx in &self.bins[rlat * self.lon_bins + rlon] {
                            let d = dir.dot(points[idx as usize]);
                            if d > best_dot {
                                best_dot = d;
                                best = idx;
                            }
                        }
                    }
                }
                if best_dot > f64::NEG_INFINITY {
                    break;
                }
            }
        }

        best
    }
}

fn to_bin(p: DVec3, lat_bins: usize, lon_bins: usize) -> (usize, usize) {
    let lat = p.z.clamp(-1.0, 1.0).asin();
    let lon = p.y.atan2(p.x);
    let lb = ((lat + std::f64::consts::FRAC_PI_2) / std::f64::consts::PI * lat_bins as f64) as usize;
    let lob = ((lon + std::f64::consts::PI) / (2.0 * std::f64::consts::PI) * lon_bins as f64) as usize;
    (lb.min(lat_bins - 1), lob.min(lon_bins - 1))
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::plate_initializer::{initialize_plates, InitParams};
    use super::super::plate_seed_placement::{assign_plates, WarpParams};

    fn setup(point_count: u32, plate_count: u32) -> Simulation {
        let fib = SphericalFibonacci::new(point_count);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, plate_count, 42, &WarpParams::default());
        let plates = initialize_plates(&points, &del, &assignment, &InitParams::default());
        Simulation::new(points, plates, &del)
    }

    #[test]
    fn resample_preserves_point_count() {
        let mut sim = setup(500, 8);
        sim.run(5);
        resample(&mut sim, 500);
        let total: usize = sim.plates.iter().map(|p| p.point_count()).sum();
        assert_eq!(total, 500);
    }

    #[test]
    fn resample_preserves_plate_count() {
        let mut sim = setup(500, 8);
        sim.run(5);
        let plate_count_before = sim.plates.len();
        resample(&mut sim, 500);
        assert_eq!(sim.plates.len(), plate_count_before);
    }

    #[test]
    fn resample_points_on_unit_sphere() {
        let mut sim = setup(500, 8);
        sim.run(5);
        resample(&mut sim, 500);
        for &p in &sim.points {
            assert!((p.length() - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn resample_most_plates_have_points() {
        let mut sim = setup(500, 8);
        sim.run(5);
        resample(&mut sim, 500);
        let non_empty = sim.plates.iter().filter(|p| !p.point_indices.is_empty()).count();
        assert!(non_empty >= 6, "too many empty plates: only {non_empty}/8 have points");
    }

    #[test]
    fn resample_crust_parallel_to_indices() {
        let mut sim = setup(500, 8);
        sim.run(5);
        resample(&mut sim, 500);
        for plate in &sim.plates {
            assert_eq!(plate.crust.len(), plate.point_indices.len());
        }
    }

    #[test]
    fn resample_after_movement_still_works() {
        let mut sim = setup(1000, 12);
        sim.run(20);
        resample(&mut sim, 1000);
        sim.run(5);
        let total: usize = sim.plates.iter().map(|p| p.point_count()).sum();
        assert_eq!(total, 1000);
    }
}
