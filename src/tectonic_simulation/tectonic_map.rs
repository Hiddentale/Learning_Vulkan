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
/// Distance below which a query point is treated as coincident with a data point.
const COINCIDENT_THRESHOLD: f64 = 1e-10;

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

impl CrustPoint {
    pub fn direction(&self) -> DVec3 {
        let ld = self.local_direction;
        DVec3::new(ld[0] as f64, ld[1] as f64, ld[2] as f64)
    }
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
    /// Build from plates with per-plate reference meshes.
    /// Transforms each plate's reference-frame vertices to world space.
    pub fn from_plates(plates: &[Plate]) -> Self {
        let total: usize = plates.iter().map(|p| p.point_count()).sum();
        let mut points = Vec::with_capacity(total);
        let mut positions = Vec::with_capacity(total);

        for plate in plates {
            for (i, &ref_pos) in plate.reference_points.iter().enumerate() {
                let world_pos = plate.to_world(ref_pos).normalize();
                let crust = &plate.crust[i];
                positions.push(world_pos);
                points.push(crust_point_from(world_pos, crust));
            }
        }

        let grid = SphereGrid::build(&positions);
        Self {
            points,
            positions,
            grid,
        }
    }

    pub fn sample_at(&self, dir: DVec3) -> InterpolatedCrust {
        let dir = dir.normalize_or(DVec3::Y);
        let neighbors = self.grid.find_nearest_k(dir, &self.positions, IDW_K);

        if neighbors.is_empty() {
            return default_crust();
        }

        if neighbors[0].1 < COINCIDENT_THRESHOLD {
            return interpolated_from(&self.points[neighbors[0].0 as usize]);
        }

        let nearest = &self.points[neighbors[0].0 as usize];
        let (elevation, thickness, age, direction) = self.accumulate_idw(&neighbors);
        let local_direction = direction
            .try_normalize()
            .unwrap_or_else(|| nearest.direction().normalize_or(DVec3::X));

        InterpolatedCrust {
            elevation: elevation * ELEVATION_SCALE,
            thickness,
            age,
            local_direction,
            crust_type: nearest.crust_type,
            orogeny_type: nearest.orogeny_type,
        }
    }

    fn accumulate_idw(&self, neighbors: &[(u32, f64)]) -> (f64, f64, f64, DVec3) {
        let mut total_weight = 0.0;
        let mut elevation = 0.0;
        let mut thickness = 0.0;
        let mut age = 0.0;
        let mut direction = DVec3::ZERO;

        for &(idx, dist) in neighbors {
            let w = 1.0 / dist.powf(IDW_POWER);
            let pt = &self.points[idx as usize];
            total_weight += w;
            elevation += pt.elevation as f64 * w;
            thickness += pt.thickness as f64 * w;
            age += pt.age as f64 * w;
            direction += pt.direction() * w;
        }

        let inv_w = 1.0 / total_weight;
        (
            elevation * inv_w,
            thickness * inv_w,
            age * inv_w,
            direction * inv_w,
        )
    }

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
        Ok(Self {
            points,
            positions,
            grid,
        })
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
    InterpolatedCrust {
        elevation: pt.elevation as f64 * ELEVATION_SCALE,
        thickness: pt.thickness as f64,
        age: pt.age as f64,
        local_direction: pt.direction().normalize_or(DVec3::X),
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

fn read_u32(r: &mut &[u8]) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_f64(r: &mut &[u8]) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_f32(r: &mut &[u8]) -> io::Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(f32::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tectonic_simulation::fibonnaci_spiral::SphericalFibonacci;
    use crate::tectonic_simulation::plate_initializer::{initialize_plates, InitParams};
    use crate::tectonic_simulation::plate_seed_placement::assign_plates;
    use crate::tectonic_simulation::spherical_delaunay_triangulation::SphericalDelaunay;

    fn make_test_plates() -> Vec<Plate> {
        let fib = SphericalFibonacci::new(1000);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, 2, 42);
        let (plates, _) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        plates
    }

    #[test]
    fn from_plates_preserves_point_count() {
        let plates = make_test_plates();
        let map = TectonicMap::from_plates(&plates);
        let expected: usize = plates.iter().map(|p| p.point_count()).sum();
        assert_eq!(map.point_count(), expected);
    }

    #[test]
    fn sample_at_returns_valid_crust() {
        let plates = make_test_plates();
        let map = TectonicMap::from_plates(&plates);
        let result = map.sample_at(DVec3::X);
        assert!(result.thickness > 0.0);
    }

    #[test]
    fn save_load_roundtrip() {
        let plates = make_test_plates();
        let map = TectonicMap::from_plates(&plates);

        let dir = std::env::temp_dir().join("tectonic_map_test.bin");
        map.save(&dir).expect("save failed");
        let loaded = TectonicMap::load(&dir).expect("load failed");
        let _ = std::fs::remove_file(&dir);

        assert_eq!(map.point_count(), loaded.point_count());
        for i in [0, 100, 249] {
            let a = &map.points[i];
            let b = &loaded.points[i];
            assert!((a.pos - b.pos).length() < 1e-10);
            assert_eq!(a.crust_type, b.crust_type);
        }
    }
}
