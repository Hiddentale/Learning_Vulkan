/// Cross-layout mapping for seamless cube-sphere terrain generation.
///
/// The cross layout arranges 6 cube faces in a 3×4 grid:
/// ```text
///          ┌──────┐
///          │ +Y   │  (row 0, col 1)
///     ┌────┼──────┼────┬──────┐
///     │ -X │ +Z   │ +X │ -Z   │  (row 1, cols 0-3)
///     └────┼──────┼────┴──────┘
///          │ -Y   │  (row 2, col 1)
///          └──────┘
/// ```
///
/// Diffusion tiles naturally overlap across face boundaries in this layout,
/// producing seamless terrain. The j-axis wraps with period = 4 * face_size.

use glam::DVec3;

/// Face positions in the cross layout: (row, col) for faces 0..5.
/// Matches sphere_export.py _cross_face_atlas.
const FACE_POSITIONS: [(u32, u32); 6] = [
    (1, 2), // face 0: +X
    (1, 0), // face 1: -X
    (1, 1), // face 2: +Z  (note: different from our Face enum order!)
    (1, 1), // face 3: unused placeholder — see CROSS_FACE_MAP below
    (1, 1), // face 4: unused placeholder
    (1, 1), // face 5: unused placeholder
];

/// Maps cross (row, col) to (face_id, is_active).
/// face_id uses the sphere_export.py convention: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.
const CROSS_FACE_MAP: [(u32, u32, i32); 6] = [
    (1, 2, 0),  // +X at row 1, col 2
    (1, 0, 1),  // -X at row 1, col 0
    (0, 1, 2),  // +Y at row 0, col 1
    (2, 1, 3),  // -Y at row 2, col 1
    (1, 1, 4),  // +Z at row 1, col 1
    (1, 3, 5),  // -Z at row 1, col 3
];

pub(super) struct CrossLayout {
    pub face_size: u32,
    pub width: u32,  // 4 * face_size
    pub height: u32, // 3 * face_size
}

impl CrossLayout {
    pub fn new(face_size: u32) -> Self {
        Self {
            face_size,
            width: 4 * face_size,
            height: 3 * face_size,
        }
    }

    /// Get the (i_start, j_start) offset for a given cross face_id (0..5).
    /// face_id follows sphere_export convention: 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z.
    pub fn face_offset(&self, face_id: usize) -> (u32, u32) {
        let (row, col, _) = CROSS_FACE_MAP[face_id];
        (row * self.face_size, col * self.face_size)
    }

    /// Map a cross-layout pixel (i, j) to a 3D sphere direction.
    /// Returns None for dead zones (pixels not in any face region).
    pub fn pixel_to_sphere(&self, i: u32, j: u32) -> Option<DVec3> {
        let fs = self.face_size;
        let row = i / fs;
        let col = j / fs;

        let face_id = match (row, col) {
            (1, 0) => 1,  // -X
            (1, 1) => 4,  // +Z
            (1, 2) => 0,  // +X
            (1, 3) => 5,  // -Z
            (0, 1) => 2,  // +Y
            (2, 1) => 3,  // -Y
            _ => return None,
        };

        let local_i = (i % fs) as f64;
        let local_j = (j % fs) as f64;
        let s = (fs - 1).max(1) as f64;

        // Compute u, v ∈ [0, 1] and map to cube point.
        // Inverse of _cross_face_atlas in sphere_export.py.
        let (x, y, z) = match face_id {
            0 => { // +X: j = 2*fs + u*s, i = (2*fs-1) - v*s
                let u = (local_j) / s;
                let v = (s - local_i) / s;
                (1.0, 2.0 * v - 1.0, -(2.0 * u - 1.0))
            }
            1 => { // -X: j = u*s, i = (2*fs-1) - v*s
                let u = local_j / s;
                let v = (s - local_i) / s;
                (-1.0, 2.0 * v - 1.0, 2.0 * u - 1.0)
            }
            2 => { // +Y: j = fs + u*s, i = v*s
                let u = local_j / s;
                let v = local_i / s;
                (2.0 * u - 1.0, 1.0, 2.0 * v - 1.0)
            }
            3 => { // -Y: j = fs + u*s, i = 2*fs + v*s
                let u = local_j / s;
                let v = local_i / s;
                (2.0 * u - 1.0, -1.0, -(2.0 * v - 1.0))
            }
            4 => { // +Z: j = fs + u*s, i = (2*fs-1) - v*s
                let u = local_j / s;
                let v = (s - local_i) / s;
                (2.0 * u - 1.0, 2.0 * v - 1.0, 1.0)
            }
            5 => { // -Z: j = 3*fs + u*s, i = (2*fs-1) - v*s
                let u = local_j / s;
                let v = (s - local_i) / s;
                (-(2.0 * u - 1.0), 2.0 * v - 1.0, -1.0)
            }
            _ => unreachable!(),
        };

        let len = (x * x + y * y + z * z).sqrt();
        Some(DVec3::new(x / len, y / len, z / len))
    }

    /// Check if pixel (i, j) is in an active face region.
    pub fn is_active(&self, i: u32, j: u32) -> bool {
        let row = i / self.face_size;
        let col = j / self.face_size;
        matches!((row, col), (1, 0) | (1, 1) | (1, 2) | (1, 3) | (0, 1) | (2, 1))
    }

    /// Resample a face from cross-layout data into face_basis coordinates.
    ///
    /// For each output pixel (r, c), computes the sphere direction using the
    /// engine's face_basis, then projects into cross-layout coordinates via
    /// sphere_export's atlas mapping, and bilinear-samples the cross data.
    ///
    /// `data` is (channels * height * width) in channel-first layout.
    /// `engine_face`: 0=PosX, 1=NegX, 2=PosY, 3=NegY, 4=PosZ, 5=NegZ.
    /// `out_res`: resolution of the output face grid.
    pub fn resample_face(
        &self,
        data: &[f32],
        channels: u32,
        engine_face: usize,
        out_res: u32,
    ) -> Vec<f32> {
        use crate::voxel::sphere::{self, Face};

        let faces = [Face::PosX, Face::NegX, Face::PosY, Face::NegY, Face::PosZ, Face::NegZ];
        let face = faces[engine_face];
        let (tu, tv, normal) = sphere::face_basis(face);
        let tu = DVec3::new(tu.x as f64, tu.y as f64, tu.z as f64);
        let tv = DVec3::new(tv.x as f64, tv.y as f64, tv.z as f64);
        let n = DVec3::new(normal.x as f64, normal.y as f64, normal.z as f64);

        let fs = self.face_size as f64;
        let s = (self.face_size - 1).max(1) as f64;
        let w = self.width as usize;
        let h = self.height as usize;
        let ch = channels as usize;
        let or = out_res as usize;

        let mut out = vec![0.0f32; ch * or * or];

        for r in 0..or {
            for c in 0..or {
                // Compute face_basis sphere direction for this pixel
                let u_fb = (c as f64 + 0.5) / or as f64 * 2.0 - 1.0;
                let v_fb = (r as f64 + 0.5) / or as f64 * 2.0 - 1.0;
                let cube_pt = tu * u_fb + tv * v_fb + n;
                let x = cube_pt.x;
                let y = cube_pt.y;
                let z = cube_pt.z;

                // Map to cross-layout (i, j) using sphere_export atlas
                let (ci, cj) = sphere_to_cross_atlas(engine_face, x, y, z, fs, s);

                // Bilinear sample from cross data
                let dst_base = r * or + c;
                for k in 0..ch {
                    out[k * or * or + dst_base] =
                        bilinear_sample_cross(data, k, w, h, ci, cj);
                }
            }
        }
        out
    }
}

/// Map a 3D direction to cross-layout (i_float, j_float).
/// Port of sphere_export.py _cross_face_atlas.
pub(super) fn sphere_to_cross_atlas(
    face_id: usize,
    x: f64, y: f64, z: f64,
    fs: f64, s: f64,
) -> (f64, f64) {
    let eps = 1e-8;
    match face_id {
        0 => { // +X
            let d = x.abs().max(eps);
            let u = (-z / d + 1.0) * 0.5;
            let v = (y / d + 1.0) * 0.5;
            ((2.0 * fs - 1.0) - v * s, 2.0 * fs + u * s)
        }
        1 => { // -X
            let d = x.abs().max(eps);
            let u = (z / d + 1.0) * 0.5;
            let v = (y / d + 1.0) * 0.5;
            ((2.0 * fs - 1.0) - v * s, u * s)
        }
        2 => { // +Y
            let d = y.abs().max(eps);
            let u = (x / d + 1.0) * 0.5;
            let v = (z / d + 1.0) * 0.5;
            (v * s, fs + u * s)
        }
        3 => { // -Y
            let d = y.abs().max(eps);
            let u = (x / d + 1.0) * 0.5;
            let v = (1.0 - z / d) * 0.5;
            (2.0 * fs + v * s, fs + u * s)
        }
        4 => { // +Z
            let d = z.abs().max(eps);
            let u = (x / d + 1.0) * 0.5;
            let v = (y / d + 1.0) * 0.5;
            ((2.0 * fs - 1.0) - v * s, fs + u * s)
        }
        5 => { // -Z
            let d = z.abs().max(eps);
            let u = (-x / d + 1.0) * 0.5;
            let v = (y / d + 1.0) * 0.5;
            ((2.0 * fs - 1.0) - v * s, 3.0 * fs + u * s)
        }
        _ => (0.0, 0.0),
    }
}

/// Bilinear sample from channel-first cross data at fractional (i, j).
fn bilinear_sample_cross(
    data: &[f32],
    channel: usize,
    width: usize,
    height: usize,
    i: f64,
    j: f64,
) -> f32 {
    let i0 = (i.floor() as usize).min(height - 1);
    let i1 = (i0 + 1).min(height - 1);
    let j0 = (j.floor() as usize).min(width - 1);
    let j1 = (j0 + 1).min(width - 1);
    let fi = (i - i0 as f64) as f32;
    let fj = (j - j0 as f64) as f32;

    let base = channel * height * width;
    let v00 = data[base + i0 * width + j0];
    let v01 = data[base + i0 * width + j1];
    let v10 = data[base + i1 * width + j0];
    let v11 = data[base + i1 * width + j1];

    v00 * (1.0 - fi) * (1.0 - fj)
        + v01 * (1.0 - fi) * fj
        + v10 * fi * (1.0 - fj)
        + v11 * fi * fj
}
