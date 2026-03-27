use glam::{Mat4, Vec3, Vec4};

const PLANE_COUNT: usize = 6;

/// A view frustum defined by 6 planes, extracted from a view-projection matrix.
/// Each plane normal points inward (toward visible space).
pub struct Frustum {
    planes: [Vec4; PLANE_COUNT],
}

impl Frustum {
    /// Extracts frustum planes from a view-projection matrix (Gribb-Hartmann method).
    /// Each plane is stored as (nx, ny, nz, d) where nx*x + ny*y + nz*z + d >= 0 is inside.
    pub fn from_view_projection(vp: &Mat4) -> Self {
        let row0 = vp.row(0);
        let row1 = vp.row(1);
        let row2 = vp.row(2);
        let row3 = vp.row(3);

        let mut planes = [
            row3 + row0, // left
            row3 - row0, // right
            row3 + row1, // bottom
            row3 - row1, // top
            row3 + row2, // near
            row3 - row2, // far
        ];

        // Normalize each plane so distance tests are in world units
        for plane in &mut planes {
            let length = Vec3::new(plane.x, plane.y, plane.z).length();
            if length > 0.0 {
                *plane /= length;
            }
        }

        Self { planes }
    }

    /// Returns true if an axis-aligned bounding box is at least partially inside the frustum.
    /// Uses the "p-vertex" test: for each plane, find the box corner most aligned with
    /// the plane normal. If that corner is outside, the entire box is outside.
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            let normal = Vec3::new(plane.x, plane.y, plane.z);

            // p-vertex: the corner of the AABB furthest along the plane normal
            let p = Vec3::new(
                if normal.x >= 0.0 { max.x } else { min.x },
                if normal.y >= 0.0 { max.y } else { min.y },
                if normal.z >= 0.0 { max.z } else { min.z },
            );

            // If the most-positive corner is behind the plane, the box is fully outside
            if normal.dot(p) + plane.w < 0.0 {
                return false;
            }
        }
        true
    }
}
