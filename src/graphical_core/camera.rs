use crate::graphical_core::{buffers::allocate_and_fill_buffer, vulkan_object::VulkanApplicationData};
use glam::{Mat4, Vec3};
use vulkanalia::vk;
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

fn create_and_fill_uniform_buffer(
    matrices_bytes: Vec<u8>,
    vulkan_logical_device: &Device,
    instance: &Instance,
    vulkan_application_data: &mut VulkanApplicationData,
) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
    let uniform_buffer_size_in_bytes = 128 as u64;
    let uniform_buffer = unsafe {
        allocate_and_fill_buffer(
            &matrices_bytes,
            uniform_buffer_size_in_bytes,
            BufferUsageFlags::UNIFORM_BUFFER,
            vulkan_logical_device,
            instance,
            vulkan_application_data,
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?
    };
    Ok(uniform_buffer)
}

fn create_projection_matrix(fov_degrees: f32, near: f32, far: f32, width: f32, height:f32) -> anyhow::Result<Mat4> {
    let fov_radians = fov_degrees.to_radians();
    let aspect_ratio = width / height;
    let projection_matrix = Mat4::perspective_rh(fov_radians, aspect_ratio, near, far);
    Ok(projection_matrix)
}

fn create_view_matrix() -> anyhow::Result<Mat4> {
    let view_matrix = Mat4::look_at_rh(eye, center, up);
    Ok(view_matrix)
}