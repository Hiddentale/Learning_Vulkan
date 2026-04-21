use std::io::{self, Write};
use std::path::Path;

use super::plates::CrustType;
use super::plates::OrogenyType;
use super::simulate::{Simulation, NO_PLATE};
use super::spherical_delaunay_triangulation::SphericalDelaunay;

const MAGIC: [u8; 4] = *b"DBGV";
const VERSION: u32 = 2;

/// Accumulates simulation snapshots and writes a single binary file
/// that the Three.js debug viewer can load.
pub struct DebugRecorder {
    frames: Vec<FrameData>,
    triangles: Vec<u32>,
}

struct FrameData {
    time: f64,
    points: Vec<PackedPoint>,
    plates: Vec<PlateMeta>,
}

struct PackedPoint {
    x: f32,
    y: f32,
    z: f32,
    elevation: f32,
    thickness: f32,
    age: f32,
    plate_id: u16,
    crust_type: u8,
    orogeny: u8,
}

struct PlateMeta {
    axis_x: f32,
    axis_y: f32,
    axis_z: f32,
    angular_speed: f32,
    point_count: u32,
}

impl DebugRecorder {
    pub fn new() -> Self {
        Self {
            frames: Vec::new(),
            triangles: Vec::new(),
        }
    }

    pub fn set_triangulation(&mut self, delaunay: &SphericalDelaunay) {
        self.triangles = delaunay.triangles.clone();
    }

    /// Record a snapshot from the simulation's sample grid.
    /// Each sample point gets its plate assignment and interpolated crust
    /// from the warm-start cache.
    pub fn record(&mut self, sim: &Simulation) {
        let n = sim.sample_points.len();
        let mut points = Vec::with_capacity(n);

        for i in 0..n {
            let p = sim.sample_points[i];
            let cache = &sim.sample_cache[i];

            let (elevation, thickness, age, plate_id, ct, orog) = if cache.plate != NO_PLATE {
                let crust = sim.interpolated_crust(cache);
                (
                    crust.elevation as f32,
                    crust.thickness as f32,
                    crust.age as f32,
                    cache.plate as u16,
                    crust_type_byte(crust.crust_type),
                    orogeny_byte(crust.orogeny_type),
                )
            } else {
                (-4.0f32, 7.0f32, 0.0f32, u16::MAX, 0, 0)
            };

            points.push(PackedPoint {
                x: p.x as f32,
                y: p.y as f32,
                z: p.z as f32,
                elevation,
                thickness,
                age,
                plate_id,
                crust_type: ct,
                orogeny: orog,
            });
        }

        let plates = sim
            .plates
            .iter()
            .map(|p| PlateMeta {
                axis_x: p.rotation_axis.x as f32,
                axis_y: p.rotation_axis.y as f32,
                axis_z: p.rotation_axis.z as f32,
                angular_speed: p.angular_speed as f32,
                point_count: p.point_count() as u32,
            })
            .collect();

        self.frames.push(FrameData {
            time: sim.time,
            points,
            plates,
        });
    }

    pub fn save(&self, path: &Path) -> io::Result<()> {
        let mut f = std::fs::File::create(path)?;
        f.write_all(&MAGIC)?;
        f.write_all(&VERSION.to_le_bytes())?;
        f.write_all(&(self.frames.len() as u32).to_le_bytes())?;

        let tri_count = self.triangles.len() / 3;
        f.write_all(&(tri_count as u32).to_le_bytes())?;
        for &idx in &self.triangles {
            f.write_all(&idx.to_le_bytes())?;
        }

        for frame in &self.frames {
            f.write_all(&frame.time.to_le_bytes())?;
            f.write_all(&(frame.points.len() as u32).to_le_bytes())?;
            f.write_all(&(frame.plates.len() as u32).to_le_bytes())?;

            for plate in &frame.plates {
                f.write_all(&plate.axis_x.to_le_bytes())?;
                f.write_all(&plate.axis_y.to_le_bytes())?;
                f.write_all(&plate.axis_z.to_le_bytes())?;
                f.write_all(&plate.angular_speed.to_le_bytes())?;
                f.write_all(&plate.point_count.to_le_bytes())?;
            }

            for pt in &frame.points {
                f.write_all(&pt.x.to_le_bytes())?;
                f.write_all(&pt.y.to_le_bytes())?;
                f.write_all(&pt.z.to_le_bytes())?;
                f.write_all(&pt.elevation.to_le_bytes())?;
                f.write_all(&pt.thickness.to_le_bytes())?;
                f.write_all(&pt.age.to_le_bytes())?;
                f.write_all(&pt.plate_id.to_le_bytes())?;
                f.write_all(&[pt.crust_type])?;
                f.write_all(&[pt.orogeny])?;
            }
        }

        let size_mb = f.metadata()?.len() as f64 / (1024.0 * 1024.0);
        println!(
            "Saved {} frames ({} points/frame, {} triangles) to {} ({:.1} MB)",
            self.frames.len(),
            self.frames.first().map_or(0, |f| f.points.len()),
            tri_count,
            path.display(),
            size_mb
        );
        Ok(())
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }
}

fn crust_type_byte(ct: CrustType) -> u8 {
    match ct {
        CrustType::Oceanic => 0,
        CrustType::Continental => 1,
    }
}

fn orogeny_byte(ot: Option<OrogenyType>) -> u8 {
    match ot {
        None => 0,
        Some(OrogenyType::Andean) => 1,
        Some(OrogenyType::Himalayan) => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tectonic_simulation::fibonnaci_spiral::SphericalFibonacci;
    use crate::tectonic_simulation::plate_initializer::{initialize_plates, InitParams};
    use crate::tectonic_simulation::plate_seed_placement::{assign_plates, Adjacency};

    #[test]
    fn record_and_save_smoke_test() {
        let fib = SphericalFibonacci::new(5_000);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, 10, 42);
        let (plates, cache) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        let adj = Adjacency::from_delaunay(points.len(), &del);
        let sim = Simulation::new(points, cache, adj, plates);

        let mut recorder = DebugRecorder::new();
        recorder.set_triangulation(&del);
        recorder.record(&sim);
        assert_eq!(recorder.frame_count(), 1);

        let path = std::env::temp_dir().join("debug_export_test.bin");
        recorder.save(&path).expect("save failed");
        let meta = std::fs::metadata(&path).expect("file missing");
        assert!(meta.len() > 0);
        let _ = std::fs::remove_file(&path);
    }
}
