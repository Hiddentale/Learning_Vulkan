use glam::{Mat4, Quat, Vec3};

pub struct Transform {
    pub position: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub fn model_matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.position)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

/// An object in the scene: draw parameters and culling bounds for one chunk.
pub struct SceneObject {
    /// Offset into the shared index buffer.
    pub first_index: u32,
    pub index_count: u32,
    /// Added to each index value to locate vertices in the shared vertex buffer.
    pub vertex_offset: i32,
    /// Index into the transform SSBO (used as firstInstance in indirect draws).
    pub transform_index: u32,
    /// World-space axis-aligned bounding box for frustum culling.
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
}
