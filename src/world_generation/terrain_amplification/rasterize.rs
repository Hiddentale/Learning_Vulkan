/// Project the coarse heightmap (on Fibonacci sphere points) onto 6 cube-face
/// grids for terrain-diffusion conditioning. Uses the Catmull/Everitt projection
/// from voxel::sphere and IDW interpolation from the nearest sphere points.

use glam::DVec3;

use crate::voxel::sphere::{self, Face};
use crate::world_generation::coarse_heightmap::CoarseHeightmap;
use crate::world_generation::sphere_geometry::fibonnaci_spiral::SphericalFibonacci;
use crate::world_generation::sphere_geometry::sphere_grid::SphereGrid;

const ALL_FACES: [Face; 6] = [
    Face::PosX,
    Face::NegX,
    Face::PosY,
    Face::NegY,
    Face::PosZ,
    Face::NegZ,
];

const IDW_NEIGHBORS: usize = 6;
const IDW_POWER: f64 = 2.0;

/// Per-face rasterized grid of conditioning channels.
pub(super) struct FaceGrid {
    pub elevation: Vec<f32>,
    pub temperature: Vec<f32>,
    pub precipitation: Vec<f32>,
    pub resolution: u32,
}

/// Rasterize a coarse heightmap onto 6 cube-face grids at the given resolution.
pub(super) fn rasterize(
    coarse: &CoarseHeightmap,
    points: &[DVec3],
    fibonacci: &SphericalFibonacci,
    resolution: u32,
) -> [FaceGrid; 6] {
    let grid = SphereGrid::build(points);

    ALL_FACES.map(|face| rasterize_face(face, resolution, coarse, points, &grid))
}

fn rasterize_face(
    face: Face,
    resolution: u32,
    coarse: &CoarseHeightmap,
    points: &[DVec3],
    grid: &SphereGrid,
) -> FaceGrid {
    let n = (resolution * resolution) as usize;
    let mut elevation = vec![0.0f32; n];
    let mut temperature = vec![0.0f32; n];
    let mut precipitation = vec![0.0f32; n];

    let (tu, tv, normal) = sphere::face_basis(face);
    let tu = DVec3::new(tu.x as f64, tu.y as f64, tu.z as f64);
    let tv = DVec3::new(tv.x as f64, tv.y as f64, tv.z as f64);
    let normal = DVec3::new(normal.x as f64, normal.y as f64, normal.z as f64);

    for row in 0..resolution {
        for col in 0..resolution {
            let idx = (row * resolution + col) as usize;

            // UV in [-1, 1] across the face
            let u = (col as f64 + 0.5) / resolution as f64 * 2.0 - 1.0;
            let v = (row as f64 + 0.5) / resolution as f64 * 2.0 - 1.0;

            // Cube point → sphere direction
            let cube_pt = tu * u + tv * v + normal;
            let dir = sphere::cube_to_sphere_unit(cube_pt).normalize();

            // IDW interpolation from nearest coarse points
            let neighbors = grid.find_nearest_k(dir, points, IDW_NEIGHBORS);
            let (e, t, p) = idw_interpolate(coarse, &neighbors);

            elevation[idx] = e;
            temperature[idx] = t;
            precipitation[idx] = p;
        }
    }

    FaceGrid {
        elevation,
        temperature,
        precipitation,
        resolution,
    }
}

fn idw_interpolate(coarse: &CoarseHeightmap, neighbors: &[(u32, f64)]) -> (f32, f32, f32) {
    let mut w_sum = 0.0f64;
    let mut e_sum = 0.0f64;
    let mut t_sum = 0.0f64;
    let mut p_sum = 0.0f64;

    for &(idx, dist) in neighbors {
        let w = if dist < 1e-12 {
            // Exact match — return directly
            return (
                coarse.elevation[idx as usize],
                coarse.temperature[idx as usize],
                coarse.precipitation[idx as usize],
            );
        } else {
            1.0 / dist.powf(IDW_POWER)
        };
        w_sum += w;
        e_sum += w * coarse.elevation[idx as usize] as f64;
        t_sum += w * coarse.temperature[idx as usize] as f64;
        p_sum += w * coarse.precipitation[idx as usize] as f64;
    }

    if w_sum < 1e-12 {
        return (0.0, 0.0, 0.0);
    }

    (
        (e_sum / w_sum) as f32,
        (t_sum / w_sum) as f32,
        (p_sum / w_sum) as f32,
    )
}

/// Pad a face grid from `resolution` to `target` by edge-extending border pixels.
pub(super) fn pad_to_size(grid: &FaceGrid, target: u32) -> FaceGrid {
    let src = grid.resolution;
    if src >= target {
        return FaceGrid {
            elevation: grid.elevation.clone(),
            temperature: grid.temperature.clone(),
            precipitation: grid.precipitation.clone(),
            resolution: src,
        };
    }

    let pad = |channel: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; (target * target) as usize];
        // Center the source in the padded output
        let offset = (target - src) / 2;
        for r in 0..target {
            for c in 0..target {
                let sr = (r as i32 - offset as i32).clamp(0, src as i32 - 1) as u32;
                let sc = (c as i32 - offset as i32).clamp(0, src as i32 - 1) as u32;
                out[(r * target + c) as usize] = channel[(sr * src + sc) as usize];
            }
        }
        out
    };

    FaceGrid {
        elevation: pad(&grid.elevation),
        temperature: pad(&grid.temperature),
        precipitation: pad(&grid.precipitation),
        resolution: target,
    }
}
