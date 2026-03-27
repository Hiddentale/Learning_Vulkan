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

    /// Returns a frustum plane as `[nx, ny, nz, d]` for passing to the GPU.
    pub fn plane(&self, index: usize) -> [f32; 4] {
        self.planes[index].to_array()
    }

    /// Returns true if an axis-aligned bounding box is at least partially inside the frustum.
    /// Uses the "p-vertex" test: for each plane, find the box corner most aligned with
    /// the plane normal. If that corner is outside, the entire box is outside.
    pub fn intersects_aabb(&self, min: Vec3, max: Vec3) -> bool {
        for plane in &self.planes {
            let normal = Vec3::new(plane.x, plane.y, plane.z);
            let p = Vec3::new(
                if normal.x >= 0.0 { max.x } else { min.x },
                if normal.y >= 0.0 { max.y } else { min.y },
                if normal.z >= 0.0 { max.z } else { min.z },
            );
            if normal.dot(p) + plane.w < 0.0 {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vp_matrix() -> Mat4 {
        let proj = Mat4::perspective_rh(90_f32.to_radians(), 16.0 / 9.0, 0.1, 500.0);
        let view = Mat4::look_at_rh(Vec3::new(0.0, 30.0, 0.0), Vec3::new(0.0, 30.0, -1.0), Vec3::Y);
        proj * view
    }

    #[test]
    fn planes_are_normalized() {
        let frustum = Frustum::from_view_projection(&test_vp_matrix());
        for i in 0..6 {
            let p = frustum.plane(i);
            let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-5, "plane {i} normal not unit length: {len}");
        }
    }

    #[test]
    fn aabb_in_front_of_camera_is_visible() {
        let frustum = Frustum::from_view_projection(&test_vp_matrix());
        let visible = frustum.intersects_aabb(Vec3::new(-8.0, 22.0, -20.0), Vec3::new(8.0, 38.0, -4.0));
        assert!(visible);
    }

    #[test]
    fn aabb_behind_camera_is_culled() {
        let frustum = Frustum::from_view_projection(&test_vp_matrix());
        let visible = frustum.intersects_aabb(Vec3::new(-8.0, 22.0, 10.0), Vec3::new(8.0, 38.0, 26.0));
        assert!(!visible);
    }

    #[test]
    fn aabb_far_left_is_culled() {
        let frustum = Frustum::from_view_projection(&test_vp_matrix());
        let visible = frustum.intersects_aabb(Vec3::new(-500.0, 22.0, -20.0), Vec3::new(-490.0, 38.0, -4.0));
        assert!(!visible);
    }

    #[test]
    fn aabb_straddling_near_plane_is_visible() {
        let frustum = Frustum::from_view_projection(&test_vp_matrix());
        let visible = frustum.intersects_aabb(Vec3::new(-1.0, 29.0, -1.0), Vec3::new(1.0, 31.0, 1.0));
        assert!(visible);
    }
}
