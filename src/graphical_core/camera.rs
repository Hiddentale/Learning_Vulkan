use crate::graphical_core::buffers::allocate_buffer;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use glam::{Mat4, Vec3};
use vulkan_rust::{vk, Device, Instance};

const FOV_DEGREES: f32 = 90.0;
const NEAR_PLANE: f32 = 0.1;
const FAR_PLANE: f32 = 10000.0;
pub const MAX_PITCH: f32 = 89.0_f32 * (std::f32::consts::PI / 180.0);

// Phase D: spawn comfortably above the +Y pole surface (planet radius 48,
// max terrain ~140-160 above origin) looking horizontally along +X. The
// radial up at this position is (0, 1, 0).
const DEFAULT_POSITION: Vec3 = Vec3::new(0.0, 200.0, 0.0);
const DEFAULT_FORWARD: Vec3 = Vec3::new(1.0, 0.0, 0.0);

/// Phase D: forward-as-state camera. The "up" direction is the radial
/// vector from the planet centre, recomputed every frame from `position`,
/// so the camera tilt automatically follows the curved surface as the
/// player walks. Yaw is rotation of `forward` around `up`; pitch is
/// rotation of `forward` around the local `right`. Both are applied as
/// deltas — there is no global yaw/pitch state.
pub struct Camera {
    pub position: Vec3,
    pub forward: Vec3,
}

impl Camera {
    pub fn new(position: Vec3, forward: Vec3) -> Self {
        Self {
            position,
            forward: forward.normalize_or(Vec3::X),
        }
    }

    /// Radial outward direction at the camera position. Falls back to
    /// global +Y at the planet centre.
    pub fn up(&self) -> Vec3 {
        self.position.normalize_or(Vec3::Y)
    }

    /// Direction the camera is currently facing.
    pub fn front(&self) -> Vec3 {
        self.forward
    }

    /// Right-hand axis perpendicular to forward in the local tangent plane.
    pub fn right(&self) -> Vec3 {
        self.forward.cross(self.up()).normalize_or(Vec3::X)
    }

    pub fn view_matrix(&self) -> Mat4 {
        let up = self.up();
        Mat4::look_at_rh(self.position, self.position + self.forward, up)
    }

    /// Apply a yaw delta (rotation of forward around the local up).
    pub fn rotate_yaw(&mut self, radians: f32) {
        self.forward = rotate_around(self.forward, self.up(), radians).normalize_or(self.forward);
    }

    /// Apply a pitch delta (rotation of forward around the local right),
    /// clamped so that forward stays at most MAX_PITCH off the tangent plane.
    pub fn rotate_pitch(&mut self, radians: f32) {
        let up = self.up();
        let right = self.right();
        let new_forward = rotate_around(self.forward, right, radians).normalize_or(self.forward);
        // Clamp: angle between new_forward and tangent plane = asin(new_forward · up).
        let sin_pitch = new_forward.dot(up).clamp(-1.0, 1.0);
        let max_sin = MAX_PITCH.sin();
        if sin_pitch.abs() <= max_sin {
            self.forward = new_forward;
        } else {
            // Project new_forward onto the cone of max pitch around up.
            let horizontal = (new_forward - up * sin_pitch).normalize_or(self.right());
            let cos_max = MAX_PITCH.cos();
            self.forward = (horizontal * cos_max + up * sin_pitch.signum() * max_sin).normalize_or(self.forward);
        }
    }

    /// Re-orthogonalize `forward` against the current `up` (call after
    /// translating `position` so the camera stays tangent to the new local
    /// frame). Preserves the yaw direction projected onto the new tangent
    /// plane and re-applies the previous pitch.
    pub fn reorthogonalize(&mut self) {
        let up = self.up();
        let dot = self.forward.dot(up);
        let tangent = (self.forward - up * dot).normalize_or(self.forward.normalize_or(Vec3::X));
        // Restore pitch component.
        self.forward = (tangent * (1.0 - dot * dot).max(0.0).sqrt() + up * dot).normalize_or(tangent);
    }
}

/// Rodrigues' rotation: rotate `v` around unit `axis` by `angle` radians.
fn rotate_around(v: Vec3, axis: Vec3, angle: f32) -> Vec3 {
    let c = angle.cos();
    let s = angle.sin();
    v * c + axis.cross(v) * s + axis * axis.dot(v) * (1.0 - c)
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(DEFAULT_POSITION, DEFAULT_FORWARD)
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
    let projection = compute_projection_matrix(FOV_DEGREES, NEAR_PLANE, FAR_PLANE, width, height);
    projection * camera.view_matrix()
}

fn compute_projection_matrix(fov_degrees: f32, near: f32, far: f32, width: f32, height: f32) -> Mat4 {
    let fov_radians = fov_degrees.to_radians();
    let aspect_ratio = width / height;
    let mut proj = Mat4::perspective_rh(fov_radians, aspect_ratio, near, far);
    // Vulkan NDC has Y pointing down; glam's perspective_rh assumes Y up (OpenGL).
    proj.y_axis.y *= -1.0;
    proj
}
