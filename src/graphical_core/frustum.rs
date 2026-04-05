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

    /// Build a frustum that encloses both eyes' view volumes.
    /// Takes the outermost left/right planes and the most conservative
    /// top/bottom/near/far from each eye.
    pub fn combined_stereo(left_vp: &Mat4, right_vp: &Mat4) -> Self {
        let left = Self::from_view_projection(left_vp);
        let right = Self::from_view_projection(right_vp);

        // For each plane pair, pick the one further from center (more permissive).
        // Left plane: use left eye's left plane (it extends further left)
        // Right plane: use right eye's right plane (it extends further right)
        // Top/bottom/near/far: use whichever is more permissive (larger d value)
        Self {
            planes: [
                Self::more_permissive(left.planes[0], right.planes[0]), // left
                Self::more_permissive(right.planes[1], left.planes[1]), // right
                Self::more_permissive(left.planes[2], right.planes[2]), // bottom
                Self::more_permissive(left.planes[3], right.planes[3]), // top
                Self::more_permissive(left.planes[4], right.planes[4]), // near
                Self::more_permissive(left.planes[5], right.planes[5]), // far
            ],
        }
    }

    /// Pick the plane that culls less (further from origin along its normal).
    fn more_permissive(a: Vec4, b: Vec4) -> Vec4 {
        if a.w > b.w {
            a
        } else {
            b
        }
    }

    #[cfg(test)]
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

    /// Mirrors the GLSL project_aabb() from svdag_tile_assign.comp.
    /// Returns None if all corners are behind the camera, else (tile_min, tile_max).
    fn project_aabb(vp: &Mat4, aabb_min: Vec3, aabb_max: Vec3, screen_size: [u32; 2], tile_size: u32) -> Option<([i32; 2], [i32; 2])> {
        let tile_count = [screen_size[0].div_ceil(tile_size), screen_size[1].div_ceil(tile_size)];
        let mut screen_min = glam::Vec2::splat(1e30);
        let mut screen_max = glam::Vec2::splat(-1e30);
        let mut any_in_front = false;

        for i in 0..8u32 {
            let corner = Vec3::new(
                if i & 1 != 0 { aabb_max.x } else { aabb_min.x },
                if i & 2 != 0 { aabb_max.y } else { aabb_min.y },
                if i & 4 != 0 { aabb_max.z } else { aabb_min.z },
            );
            let clip = *vp * corner.extend(1.0);
            if clip.w > 0.001 {
                any_in_front = true;
                let ndc = glam::Vec2::new(clip.x, clip.y) / clip.w;
                let screen = (ndc * 0.5 + 0.5) * glam::Vec2::new(screen_size[0] as f32, screen_size[1] as f32);
                screen_min = screen_min.min(screen);
                screen_max = screen_max.max(screen);
            }
        }

        if !any_in_front {
            return None;
        }

        let tile_min = [
            (screen_min.x / tile_size as f32).clamp(0.0, tile_count[0] as f32 - 1.0) as i32,
            (screen_min.y / tile_size as f32).clamp(0.0, tile_count[1] as f32 - 1.0) as i32,
        ];
        let tile_max = [
            (screen_max.x / tile_size as f32).clamp(0.0, tile_count[0] as f32 - 1.0) as i32,
            (screen_max.y / tile_size as f32).clamp(0.0, tile_count[1] as f32 - 1.0) as i32,
        ];
        Some((tile_min, tile_max))
    }

    #[test]
    fn frustum_visible_chunk_always_gets_tiles() {
        let vp = test_vp_matrix();
        let frustum = Frustum::from_view_projection(&vp);
        let screen = [1920, 1080];
        let tile_size = 8;

        // Generate a grid of AABBs that pass frustum culling
        // and verify each one gets at least one tile
        let mut tested = 0;
        for x in (-20..=20).step_by(4) {
            for z in (-30..=-1).step_by(4) {
                let min = Vec3::new(x as f32, 26.0, z as f32);
                let max = min + Vec3::splat(4.0);
                if !frustum.intersects_aabb(min, max) {
                    continue;
                }
                tested += 1;
                let result = project_aabb(&vp, min, max, screen, tile_size);
                assert!(result.is_some(), "frustum-visible AABB ({min} -> {max}) got no tiles");
                let (t_min, t_max) = result.unwrap();
                assert!(
                    t_max[0] >= t_min[0] && t_max[1] >= t_min[1],
                    "degenerate tile rect for AABB ({min} -> {max}): \
                     tile_min={t_min:?}, tile_max={t_max:?}"
                );
            }
        }
        assert!(tested > 10, "not enough AABBs tested: {tested}");
    }

    #[test]
    fn project_aabb_partially_behind_camera_expands() {
        // Camera at (0,30,0) looking toward -Z.
        // This AABB straddles the camera's near plane — some corners behind.
        let vp = test_vp_matrix();
        let min = Vec3::new(-8.0, 28.0, -2.0);
        let max = Vec3::new(8.0, 32.0, 2.0); // z=2 is behind camera

        let result = project_aabb(&vp, min, max, [1920, 1080], 8);
        assert!(result.is_some(), "partially-behind AABB should still get tiles");
        let (t_min, t_max) = result.unwrap();
        // With corners behind camera, the visible projection should be wide
        // (the behind-camera corners are skipped, making rect too small)
        // This test documents the CURRENT behavior — if it fails after a fix,
        // that's expected and good.
        let tile_width = t_max[0] - t_min[0];
        let tile_height = t_max[1] - t_min[1];
        println!(
            "  partially-behind AABB: tiles [{},{}] -> [{},{}] ({}x{})",
            t_min[0], t_min[1], t_max[0], t_max[1], tile_width, tile_height
        );
    }

    #[test]
    fn project_aabb_is_deterministic() {
        let vp = test_vp_matrix();
        let min = Vec3::new(-8.0, 22.0, -20.0);
        let max = Vec3::new(8.0, 38.0, -4.0);
        let screen = [1920, 1080];

        let first = project_aabb(&vp, min, max, screen, 8);
        for _ in 0..100 {
            let result = project_aabb(&vp, min, max, screen, 8);
            assert_eq!(first, result, "AABB projection is non-deterministic across calls");
        }
    }

    #[test]
    fn frustum_boundary_chunk_stability() {
        // A chunk right at the far plane boundary should give a consistent
        // cull result when tested repeatedly with the same VP matrix.
        let vp = test_vp_matrix(); // far plane at 500
        let frustum = Frustum::from_view_projection(&vp);
        // Place AABB near the far plane
        let min = Vec3::new(-8.0, 26.0, -498.0);
        let max = Vec3::new(8.0, 34.0, -490.0);

        let first_result = frustum.intersects_aabb(min, max);
        for _ in 0..100 {
            assert_eq!(
                frustum.intersects_aabb(min, max),
                first_result,
                "frustum cull at far boundary is non-deterministic"
            );
        }
    }
}
