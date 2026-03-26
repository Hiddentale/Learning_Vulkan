use crate::graphical_core::buffers::allocate_buffer;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use glam::{Mat4, Vec3};
use vulkanalia::vk::{self, DeviceV1_0};
use vulkanalia::{Device, Instance};

const FOV_DEGREES: f32 = 90.0;
const NEAR_PLANE: f32 = 0.1;
const FAR_PLANE: f32 = 500.0;
const WORLD_UP: Vec3 = Vec3::Y;
const MAX_PITCH: f32 = 89.0_f32 * (std::f32::consts::PI / 180.0);

const DEFAULT_POSITION: Vec3 = Vec3::new(8.0, 20.0, 30.0);
const DEFAULT_YAW_DEGREES: f32 = 0.0;
const DEFAULT_PITCH_DEGREES: f32 = -30.0;

pub struct Camera {
    pub position: Vec3,
    pub yaw: f32,
    pub pitch: f32,
}

impl Camera {
    pub fn new(position: Vec3, yaw_degrees: f32, pitch_degrees: f32) -> Self {
        Self {
            position,
            yaw: yaw_degrees.to_radians(),
            pitch: pitch_degrees.to_radians().clamp(-MAX_PITCH, MAX_PITCH),
        }
    }

    pub fn front(&self) -> Vec3 {
        Vec3::new(self.pitch.cos() * self.yaw.sin(), self.pitch.sin(), -self.pitch.cos() * self.yaw.cos()).normalize()
    }

    pub fn right(&self) -> Vec3 {
        self.front().cross(WORLD_UP).normalize()
    }

    pub fn view_matrix(&self) -> Mat4 {
        let front = self.front();
        Mat4::look_at_rh(self.position, self.position + front, WORLD_UP)
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::new(DEFAULT_POSITION, DEFAULT_YAW_DEGREES, DEFAULT_PITCH_DEGREES)
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct UniformBufferObject {
    view_projection_matrix: [[f32; 4]; 4],
}

/// Allocates a persistently mapped uniform buffer for camera matrices.
pub fn create_uniform_buffer(device: &Device, instance: &Instance, vulkan_application_data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let buffer_size_in_bytes = std::mem::size_of::<UniformBufferObject>() as u64;
    let buffer_usage_flags = vk::BufferUsageFlags::UNIFORM_BUFFER;
    let properties = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::DEVICE_LOCAL;

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

/// Computes model/view/projection matrices and writes them to the mapped UBO.
pub fn update_uniform_buffer(vulkan_application_data: &VulkanApplicationData, camera: &Camera) -> anyhow::Result<()> {
    let extent = vulkan_application_data.swapchain_extent;
    let width = extent.width as f32;
    let height = extent.height as f32;

    let projection = compute_projection_matrix(FOV_DEGREES, NEAR_PLANE, FAR_PLANE, width, height);
    let view = camera.view_matrix();
    let view_projection = projection * view;

    let ubo = UniformBufferObject {
        view_projection_matrix: view_projection.to_cols_array_2d(),
    };

    unsafe {
        std::ptr::copy_nonoverlapping(&ubo, vulkan_application_data.uniform_buffer_ptr, 1);
    }
    Ok(())
}

/// Unmaps, destroys, and frees the uniform buffer and its memory.
pub fn destroy_uniform_buffer(device: &vulkanalia::Device, vulkan_application_data: &mut VulkanApplicationData) {
    unsafe {
        device.unmap_memory(vulkan_application_data.uniform_buffer_memory);
        device.destroy_buffer(vulkan_application_data.uniform_buffer, None);
        device.free_memory(vulkan_application_data.uniform_buffer_memory, None);
    }
}

fn compute_projection_matrix(fov_degrees: f32, near: f32, far: f32, width: f32, height: f32) -> Mat4 {
    let fov_radians = fov_degrees.to_radians();
    let aspect_ratio = width / height;
    Mat4::perspective_rh(fov_radians, aspect_ratio, near, far)
}
