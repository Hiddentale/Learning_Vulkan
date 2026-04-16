use std::io::{self, Read, Write};
use std::path::Path;

use glam::DVec3;

use super::plates::{CrustData, CrustType, OrogenyType, Plate};
use super::sphere_grid::SphereGrid;

/// Number of nearest neighbors for IDW interpolation.
const IDW_K: usize = 6;
/// Exponent for inverse-distance weighting.
const IDW_POWER: f64 = 2.0;
/// Blocks per km of tectonic elevation.
pub const ELEVATION_SCALE: f64 = 50.0;

const MAGIC: [u8; 4] = *b"TECT";
const VERSION: u32 = 5;

/// Compact runtime representation of one simulation point.
#[derive(Clone, Debug)]
pub struct CrustPoint {
    pub pos: DVec3,
    pub crust_type: CrustType,
    pub thickness: f32,
    pub elevation: f32,
    pub age: f32,
    pub local_direction: [f32; 3],
    pub orogeny_type: Option<OrogenyType>,
}

/// IDW-interpolated result at an arbitrary sphere direction.
#[derive(Clone, Debug)]
pub struct InterpolatedCrust {
    pub elevation: f64,
    pub thickness: f64,
    pub age: f64,
    pub local_direction: DVec3,
    pub crust_type: CrustType,
    pub orogeny_type: Option<OrogenyType>,
}

/// Runtime query structure for tectonic simulation output.
pub struct TectonicMap {
    points: Vec<CrustPoint>,
    positions: Vec<DVec3>,
    grid: SphereGrid,
}

impl TectonicMap {
    /// Build from completed simulation output.
    pub fn from_simulation(sim_points: &[DVec3], plates: &[Plate]) -> Self {
        let total: usize = plates.iter().map(|p| p.point_count()).sum();
        let mut points = Vec::with_capacity(total);
        let mut positions = Vec::with_capacity(total);

        for plate in plates {
            for (local_idx, &global_idx) in plate.point_indices.iter().enumerate() {
                let pos = sim_points[global_idx as usize].normalize();
                let crust = &plate.crust[local_idx];
                positions.push(pos);
                points.push(crust_point_from(pos, crust));
            }
        }

        let grid = SphereGrid::build(&positions);
        Self { points, positions, grid }
    }

    /// IDW interpolation from the 6 nearest points.
    pub fn sample_at(&self, dir: DVec3) -> InterpolatedCrust {
        let dir = dir.normalize_or(DVec3::Y);
        let neighbors = self.grid.find_nearest_k(dir, &self.positions, IDW_K);

        if neighbors.is_empty() {
            return default_crust();
        }

        // If coincident with a point, return it directly.
        if neighbors[0].1 < 1e-10 {
            return interpolated_from(&self.points[neighbors[0].0 as usize]);
        }

        let mut total_weight = 0.0;
        let mut elevation = 0.0;
        let mut thickness = 0.0;
        let mut age = 0.0;
        let mut direction = DVec3::ZERO;

        for &(idx, dist) in &neighbors {
            let w = 1.0 / dist.powf(IDW_POWER);
            let pt = &self.points[idx as usize];
            total_weight += w;
            elevation += pt.elevation as f64 * w;
            thickness += pt.thickness as f64 * w;
            age += pt.age as f64 * w;
            let ld = pt.local_direction;
            direction += DVec3::new(ld[0] as f64, ld[1] as f64, ld[2] as f64) * w;
        }

        let inv_w = 1.0 / total_weight;
        let nearest = &self.points[neighbors[0].0 as usize];

        let local_direction = direction * inv_w;
        let local_direction = local_direction
            .try_normalize()
            .unwrap_or_else(|| {
                let ld = nearest.local_direction;
                DVec3::new(ld[0] as f64, ld[1] as f64, ld[2] as f64)
                    .normalize_or(DVec3::X)
            });

        InterpolatedCrust {
            elevation: elevation * inv_w * ELEVATION_SCALE,
            thickness: thickness * inv_w,
            age: age * inv_w,
            local_direction,
            crust_type: nearest.crust_type,
            orogeny_type: nearest.orogeny_type,
        }
    }

    /// Nearest-point lookup for discrete properties.
    pub fn nearest_at(&self, dir: DVec3) -> &CrustPoint {
        let neighbors = self.grid.find_nearest_k(dir, &self.positions, 1);
        &self.points[neighbors[0].0 as usize]
    }

    pub fn point_count(&self) -> usize {
        self.points.len()
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(&MAGIC)?;
        f.write_all(&VERSION.to_le_bytes())?;
        f.write_all(&(self.points.len() as u32).to_le_bytes())?;

        for pt in &self.points {
            write_f64(&mut f, pt.pos.x)?;
            write_f64(&mut f, pt.pos.y)?;
            write_f64(&mut f, pt.pos.z)?;
            f.write_all(&[crust_type_to_u8(pt.crust_type)])?;
            write_f32(&mut f, pt.thickness)?;
            write_f32(&mut f, pt.elevation)?;
            write_f32(&mut f, pt.age)?;
            write_f32(&mut f, pt.local_direction[0])?;
            write_f32(&mut f, pt.local_direction[1])?;
            write_f32(&mut f, pt.local_direction[2])?;
            f.write_all(&[orogeny_to_u8(pt.orogeny_type)])?;
        }

        Ok(())
    }

    pub fn load(path: &Path) -> io::Result<Self> {
        let data = std::fs::read(path)?;
        let mut cursor = &data[..];

        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "bad TECT magic"));
        }

        let version = read_u32(&mut cursor)?;
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported TECT version {version}"),
            ));
        }

        let count = read_u32(&mut cursor)? as usize;
        let mut points = Vec::with_capacity(count);
        let mut positions = Vec::with_capacity(count);

        for _ in 0..count {
            let x = read_f64(&mut cursor)?;
            let y = read_f64(&mut cursor)?;
            let z = read_f64(&mut cursor)?;
            let pos = DVec3::new(x, y, z);

            let mut ct_byte = [0u8; 1];
            cursor.read_exact(&mut ct_byte)?;
            let crust_type = u8_to_crust_type(ct_byte[0]);

            let thickness = read_f32(&mut cursor)?;
            let elevation = read_f32(&mut cursor)?;
            let age = read_f32(&mut cursor)?;
            let ld0 = read_f32(&mut cursor)?;
            let ld1 = read_f32(&mut cursor)?;
            let ld2 = read_f32(&mut cursor)?;

            let mut ot_byte = [0u8; 1];
            cursor.read_exact(&mut ot_byte)?;
            let orogeny_type = u8_to_orogeny(ot_byte[0]);

            positions.push(pos);
            points.push(CrustPoint {
                pos,
                crust_type,
                thickness,
                elevation,
                age,
                local_direction: [ld0, ld1, ld2],
                orogeny_type,
            });
        }

        let grid = SphereGrid::build(&positions);
        Ok(Self { points, positions, grid })
    }
}

fn crust_point_from(pos: DVec3, crust: &CrustData) -> CrustPoint {
    CrustPoint {
        pos,
        crust_type: crust.crust_type,
        thickness: crust.thickness as f32,
        elevation: crust.elevation as f32,
        age: crust.age as f32,
        local_direction: [
            crust.local_direction.x as f32,
            crust.local_direction.y as f32,
            crust.local_direction.z as f32,
        ],
        orogeny_type: crust.orogeny_type,
    }
}

fn interpolated_from(pt: &CrustPoint) -> InterpolatedCrust {
    let ld = pt.local_direction;
    InterpolatedCrust {
        elevation: pt.elevation as f64 * ELEVATION_SCALE,
        thickness: pt.thickness as f64,
        age: pt.age as f64,
        local_direction: DVec3::new(ld[0] as f64, ld[1] as f64, ld[2] as f64)
            .normalize_or(DVec3::X),
        crust_type: pt.crust_type,
        orogeny_type: pt.orogeny_type,
    }
}

fn default_crust() -> InterpolatedCrust {
    InterpolatedCrust {
        elevation: 0.0,
        thickness: 0.0,
        age: 0.0,
        local_direction: DVec3::X,
        crust_type: CrustType::Oceanic,
        orogeny_type: None,
    }
}

fn crust_type_to_u8(ct: CrustType) -> u8 {
    match ct {
        CrustType::Oceanic => 0,
        CrustType::Continental => 1,
    }
}

fn u8_to_crust_type(b: u8) -> CrustType {
    match b {
        1 => CrustType::Continental,
        _ => CrustType::Oceanic,
    }
}

fn orogeny_to_u8(ot: Option<OrogenyType>) -> u8 {
    match ot {
        None => 0,
        Some(OrogenyType::Andean) => 1,
        Some(OrogenyType::Himalayan) => 2,
    }
}

fn u8_to_orogeny(b: u8) -> Option<OrogenyType> {
    match b {
        1 => Some(OrogenyType::Andean),
        2 => Some(OrogenyType::Himalayan),
        _ => None,
    }
}

fn write_f64(w: &mut impl Write, v: f64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_f32(w: &mut impl Write, v: f32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

#[allow(dead_code)]
fn read_u32(r: &mut &[u8]) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

#[allow(dead_code)]
fn read_f64(r: &mut &[u8]) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

#[allow(dead_code)]
fn read_f32(r: &mut &[u8]) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tectonic_simulation::fibonnaci_spiral::SphericalFibonacci;

    fn make_test_plates(n: u32) -> (Vec<DVec3>, Vec<Plate>) {
        let fib = SphericalFibonacci::new(n);
        let points = fib.all_points();
        let half = n / 2;

        let plate_a = Plate {
            point_indices: (0..half).collect(),
            crust: (0..half)
                .map(|i| {
                    CrustData::continental(
                        30.0,
                        2.0 + (i as f64) * 0.001,
                        100.0,
                        DVec3::new(1.0, 0.0, 0.0),
                        OrogenyType::Himalayan,
                    )
                })
                .collect(),
            rotation_axis: DVec3::Y,
            angular_speed: 0.01,
        };

        let plate_b = Plate {
            point_indices: (half..n).collect(),
            crust: (half..n)
                .map(|_| CrustData::oceanic(7.0, -4.0, 50.0, DVec3::new(0.0, 0.0, 1.0)))
                .collect(),
            rotation_axis: DVec3::NEG_Y,
            angular_speed: 0.02,
        };

        (points, vec![plate_a, plate_b])
    }

    #[test]
    fn from_simulation_preserves_point_count() {
        let (points, plates) = make_test_plates(1000);
        let map = TectonicMap::from_simulation(&points, &plates);
        let expected: usize = plates.iter().map(|p| p.point_count()).sum();
        assert_eq!(map.point_count(), expected);
    }

    #[test]
    fn sample_at_returns_nearest_for_exact_point() {
        let (points, plates) = make_test_plates(1000);
        let map = TectonicMap::from_simulation(&points, &plates);
        // Query at the exact position of point 0 (a continental point).
        let result = map.sample_at(points[0]);
        assert_eq!(result.crust_type, CrustType::Continental);
        assert!(result.elevation > 0.0, "continental point should have positive elevation");
    }

    #[test]
    fn sample_at_interpolates_between_neighbors() {
        let (points, plates) = make_test_plates(1000);
        let map = TectonicMap::from_simulation(&points, &plates);
        // Query at a point midway between two known points.
        let mid = (points[0] + points[1]).normalize();
        let result = map.sample_at(mid);
        // The interpolated elevation should be between the two source elevations.
        let e0 = plates[0].crust[0].elevation * ELEVATION_SCALE;
        let e1 = plates[0].crust[1].elevation * ELEVATION_SCALE;
        let lo = e0.min(e1);
        let hi = e0.max(e1);
        // Allow some slack since 6-point IDW includes more than just these two.
        assert!(
            result.elevation >= lo * 0.5 && result.elevation <= hi * 2.0,
            "elevation {} outside plausible range [{}, {}]",
            result.elevation, lo * 0.5, hi * 2.0,
        );
    }

    #[test]
    fn crust_type_is_nearest_not_interpolated() {
        let (points, plates) = make_test_plates(1000);
        let map = TectonicMap::from_simulation(&points, &plates);
        // Point 0 is continental.
        let result = map.sample_at(points[0]);
        assert_eq!(result.crust_type, CrustType::Continental);
        // Point 500 is oceanic (in plate_b).
        let result = map.sample_at(points[500]);
        assert_eq!(result.crust_type, CrustType::Oceanic);
    }

    #[test]
    fn save_load_roundtrip() {
        let (points, plates) = make_test_plates(500);
        let map = TectonicMap::from_simulation(&points, &plates);

        let dir = std::env::temp_dir().join("tectonic_map_test.bin");
        map.save(&dir).expect("save failed");
        let loaded = TectonicMap::load(&dir).expect("load failed");
        let _ = std::fs::remove_file(&dir);

        assert_eq!(map.point_count(), loaded.point_count());
        // Verify a few points roundtrip.
        for i in [0, 100, 249] {
            let a = &map.points[i];
            let b = &loaded.points[i];
            assert!((a.pos - b.pos).length() < 1e-10);
            assert_eq!(a.crust_type, b.crust_type);
            assert!((a.elevation - b.elevation).abs() < 1e-6);
            assert!((a.thickness - b.thickness).abs() < 1e-6);
            assert_eq!(a.orogeny_type, b.orogeny_type);
        }
    }

    #[test]
    fn local_direction_normalized_after_interpolation() {
        let (points, plates) = make_test_plates(1000);
        let map = TectonicMap::from_simulation(&points, &plates);
        let queries = SphericalFibonacci::new(50).all_points();
        for q in &queries {
            let result = map.sample_at(*q);
            let len = result.local_direction.length();
            assert!(
                (len - 1.0).abs() < 1e-6,
                "direction not normalized: length = {len}"
            );
        }
    }
}
