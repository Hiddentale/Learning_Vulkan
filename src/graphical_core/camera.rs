use crate::graphical_core::buffers::allocate_buffer;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::voxel::player::Player;
use glam::{Mat4, Vec3};
use vulkan_rust::{vk, Device, Instance};

pub const FOV_DEGREES: f32 = 90.0;
/// Reverse-Z near plane. With infinite-far reverse-Z this is the *only* depth
/// constant that affects precision: smaller near = more precision near the
/// camera, no penalty far away.
const NEAR_PLANE: f32 = 0.1;

/// Phase D': camera is a *derived* view of the player. `position` and
/// `forward` are written from `Player` once per frame and never mutated
/// independently. The player's cube-space coordinates are the source of
/// truth; cartesian state lives here only because the rendering pipeline
/// (UBO upload, frustum, push constants) expects `Vec3`.
pub struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
}

impl Camera {
    pub fn from_player(player: &Player) -> Self {
        Self {
            position: player.eye_pos(),
            forward: player.forward,
            right: player.right,
        }
    }

    /// Re-derive view state from the player. Call once per frame after
    /// physics has settled.
    pub fn sync_from_player(&mut self, player: &Player) {
        self.position = player.eye_pos();
        self.forward = player.forward;
        self.right = player.right;
    }

    pub fn view_matrix(&self) -> Mat4 {
        let view_up = self.right.cross(self.forward).normalize_or(Vec3::Y);
        Mat4::look_at_rh(self.position, self.position + self.forward, view_up)
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::from_player(&Player::new())
    }
}

/// Eye count for the VP matrix arrays. Desktop uses both slots with the
/// same matrix; VR fills left eye at index 0, right eye at index 1.
pub const MAX_VIEWS: usize = 2;

/// Per-eye view/projection matrices — the camera abstraction's output.
/// Desktop: both entries identical. VR: left eye at 0, right eye at 1.
#[derive(Copy, Clone, Debug)]
pub struct EyeMatrices {
    pub view_projection: [Mat4; MAX_VIEWS],
    pub inverse_view_projection: [Mat4; MAX_VIEWS],
}

impl EyeMatrices {
    /// Build from a desktop camera — same matrix for both eyes.
    pub fn from_camera(camera: &Camera, extent: vk::Extent2D) -> Self {
        let vp = view_projection_matrix(camera, extent);
        let inv = vp.inverse();
        Self {
            view_projection: [vp, vp],
            inverse_view_projection: [inv, inv],
        }
    }

    /// Build from per-eye view and projection matrices (for VR).
    #[allow(dead_code)]
    pub fn from_stereo(left_vp: Mat4, right_vp: Mat4) -> Self {
        Self {
            view_projection: [left_vp, right_vp],
            inverse_view_projection: [left_vp.inverse(), right_vp.inverse()],
        }
    }

    /// The primary VP matrix (eye 0). Used for frustum culling on desktop.
    pub fn primary_vp(&self) -> Mat4 {
        self.view_projection[0]
    }

    /// Whether both eyes have distinct matrices (stereo VR).
    pub fn is_stereo(&self) -> bool {
        self.view_projection[0] != self.view_projection[1]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct UniformBufferObject {
    view_projection: [[[f32; 4]; 4]; MAX_VIEWS],
    inverse_view_projection: [[[f32; 4]; 4]; MAX_VIEWS],
    light_direction: [f32; 3],
    ambient_strength: f32,
    planet_radius: f32,
    cube_half: f32,
    _pad: [f32; 2],
}

/// Allocates a persistently mapped uniform buffer for camera matrices.
pub fn create_uniform_buffer(device: &Device, instance: &Instance, vulkan_application_data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let buffer_size_in_bytes = std::mem::size_of::<UniformBufferObject>() as u64;
    let buffer_usage_flags = vk::BufferUsageFlags::UNIFORM_BUFFER;
    let mut properties = super::host_visible_coherent();
    if !cfg!(target_os = "macos") {
        properties |= vk::MemoryPropertyFlags::DEVICE_LOCAL;
    }

    let (uniform_buffer, uniform_memory, uniform_ptr) = unsafe {
        allocate_buffer::<UniformBufferObject>(
            buffer_size_in_bytes,
            buffer_usage_flags,
            device,
            instance,
            vulkan_application_data,
            properties,
        )?
    };
    vulkan_application_data.uniform_buffer = uniform_buffer;
    vulkan_application_data.uniform_buffer_memory = uniform_memory;
    vulkan_application_data.uniform_buffer_ptr = uniform_ptr;
    Ok(())
}

/// Writes per-eye VP matrices and lighting params to the mapped UBO.
pub fn update_uniform_buffer(data: &VulkanApplicationData, eyes: &EyeMatrices) -> anyhow::Result<()> {
    let sun_direction = Vec3::new(0.3, -1.0, 0.5).normalize();
    let ubo = UniformBufferObject {
        view_projection: [eyes.view_projection[0].to_cols_array_2d(), eyes.view_projection[1].to_cols_array_2d()],
        inverse_view_projection: [
            eyes.inverse_view_projection[0].to_cols_array_2d(),
            eyes.inverse_view_projection[1].to_cols_array_2d(),
        ],
        light_direction: sun_direction.to_array(),
        ambient_strength: 0.15,
        planet_radius: crate::voxel::sphere::PLANET_RADIUS_BLOCKS as f32,
        cube_half: crate::voxel::sphere::CUBE_HALF_BLOCKS as f32,
        _pad: [0.0; 2],
    };

    unsafe {
        std::ptr::copy_nonoverlapping(&ubo, data.uniform_buffer_ptr, 1);
    }
    Ok(())
}

/// Unmaps, destroys, and frees the uniform buffer and its memory.
pub fn destroy_uniform_buffer(device: &Device, vulkan_application_data: &mut VulkanApplicationData) {
    unsafe {
        device.unmap_memory(vulkan_application_data.uniform_buffer_memory);
        device.destroy_buffer(vulkan_application_data.uniform_buffer, None);
        device.free_memory(vulkan_application_data.uniform_buffer_memory, None);
    }
}

/// Computes the view-projection matrix for the current camera and swapchain extent.
/// Used by both UBO upload and frustum extraction.
pub fn view_projection_matrix(camera: &Camera, extent: vk::Extent2D) -> Mat4 {
    let width = extent.width as f32;
    let height = extent.height as f32;
    let projection = reverse_z_infinite_perspective(FOV_DEGREES.to_radians(), width / height, NEAR_PLANE);
    projection * camera.view_matrix()
}

/// Right-handed Vulkan projection with infinite-far and reverse-Z depth.
/// Maps view-space `z = -near` → NDC `z = 1`, `z = -∞` → NDC `z = 0`. The
/// reversed range puts depth precision where it matters (close to the camera)
/// and the infinite far plane removes the second precision-eating divide.
///
/// Y axis is negated to match Vulkan's Y-down NDC.
pub fn reverse_z_infinite_perspective(fov_y: f32, aspect: f32, near: f32) -> Mat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    let mut m = Mat4::ZERO;
    m.x_axis.x = f / aspect;
    m.y_axis.y = -f;
    m.z_axis.z = 0.0;
    m.z_axis.w = -1.0;
    m.w_axis.z = near;
    m
}

#[cfg(test)]
mod reverse_z_tests {
    use super::*;

    fn project(m: &Mat4, view_z: f32) -> f32 {
        let clip = *m * glam::Vec4::new(0.0, 0.0, view_z, 1.0);
        clip.z / clip.w
    }

    #[test]
    fn near_plane_maps_to_one() {
        let m = reverse_z_infinite_perspective(90_f32.to_radians(), 16.0 / 9.0, 0.1);
        let ndc_z = project(&m, -0.1);
        assert!((ndc_z - 1.0).abs() < 1e-5, "ndc.z at near = {}, expected 1.0", ndc_z);
    }

    #[test]
    fn far_point_maps_near_zero() {
        let m = reverse_z_infinite_perspective(90_f32.to_radians(), 16.0 / 9.0, 0.1);
        let ndc_z = project(&m, -1.0e6);
        assert!(ndc_z.abs() < 1e-6, "ndc.z at ~infinity = {}, expected ~0", ndc_z);
    }

    #[test]
    fn monotonic_farther_is_smaller_ndc_z() {
        let m = reverse_z_infinite_perspective(90_f32.to_radians(), 1.0, 0.1);
        let z1 = project(&m, -1.0);
        let z10 = project(&m, -10.0);
        let z100 = project(&m, -100.0);
        assert!(z1 > z10 && z10 > z100, "expected monotonic decreasing, got {} {} {}", z1, z10, z100);
        assert!(z1 > 0.0 && z100 > 0.0, "ndc.z should stay in (0, 1]");
    }

    #[test]
    fn matrix_is_invertible() {
        let m = reverse_z_infinite_perspective(90_f32.to_radians(), 16.0 / 9.0, 0.1);
        let inv = m.inverse();
        let p = inv * glam::Vec4::new(0.0, 0.0, 1.0, 1.0); // near plane at screen center
        let w = p / p.w;
        // Should land on the near plane (view z ≈ -near; since we have no view
        // transform here, that's world z ≈ -0.1).
        assert!((w.z + 0.1).abs() < 1e-4, "inverse * near = {:?}", w);
    }
}
