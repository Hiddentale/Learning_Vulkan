use crate::graphical_core::buffers::allocate_buffer;
use crate::graphical_core::{buffers::allocate_and_fill_buffer, vulkan_object::VulkanApplicationData};
use glam::{Mat4, Quat, Vec3};
use vulkanalia::vk::{self, DeviceV1_0};
use vulkanalia::{vk::BufferUsageFlags, Device, Instance};

/*
Initialization:
    Projection = compute_projection(fov, aspect, near, far)

Each frame:
    if camera_moved:
        View = compute_view(camera_pos, camera_target, up)

    ViewProjection = Projection × View
    Send ViewProjection to GPU

    For each object:
        Model = compute_model(object_pos, object_rotation)
        Send Model to GPU
        GPU: gl_Position = ViewProjection × Model × vertex
 */

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct UniformBufferObject {
    model_matrix: [[f32; 4]; 4],
    view_projection_matrix: [[f32; 4]; 4],
}

pub fn create_uniform_buffer(device: &vulkanalia::Device, instance: &vulkanalia::Instance, vulkan_application_data: &mut VulkanApplicationData) {
    buffer_size_in_bytes = std::mem::size_of::<UniformBufferObject>() as u64;
    buffer_usage_flags = vk::BufferUsageFlags::UNIFORM_BUFFER;
    properties = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::DEVICE_LOCAL;

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
}

/*
pub fn update_camera() {
    let fov_degrees = 90;
    let near = 1;
    let far = 100;
    let width = ;
    let height = ;
    let projection_matrix = compute_projection_matrix(fov_degrees, near, far, width, height)
}
*/

pub fn destroy_uniform_buffer(device: &vulkanalia::Device, vulkan_application_data: &mut VulkanApplicationData) {
    unsafe {
        device.unmap_memory(vulkan_application_data.uniform_buffer_memory);
        device.destroy_buffer(vulkan_application_data.uniform_buffer, None);
        device.free_memory(vulkan_application_data.uniform_buffer_memory, None);
    }
}

fn compute_projection_matrix(fov_degrees: f32, near: f32, far: f32, width: f32, height: f32) -> anyhow::Result<Mat4> {
    let fov_radians = fov_degrees.to_radians();
    let aspect_ratio = width / height;
    let projection_matrix = Mat4::perspective_rh(fov_radians, aspect_ratio, near, far);
    Ok(projection_matrix)
}

fn compute_view_matrix(camera_position: Vec3, camera_target: Vec3, up: Vec3) -> anyhow::Result<Mat4> {
    let view_matrix = Mat4::look_at_rh(camera_position, camera_target, up);
    Ok(view_matrix)
}

fn compute_model_matrix(object_scale: Vec3, object_rotation: Quat, object_position: Vec3) -> anyhow::Result<Mat4> {
    let model_matrix = Mat4::from_scale_rotation_translation(object_scale, object_rotation, object_position);
    Ok(model_matrix)
}
