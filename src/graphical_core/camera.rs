use crate::graphical_core::buffers::allocate_buffer;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use glam::{Mat4, Quat, Vec3};
use vulkanalia::vk::{self, DeviceV1_0};
use vulkanalia::{Device, Instance};

const FOV_DEGREES: f32 = 90.0;
const NEAR_PLANE: f32 = 0.1;
const FAR_PLANE: f32 = 100.0;
const CAMERA_POSITION: Vec3 = Vec3::new(0.0, 0.0, 2.0);
const CAMERA_TARGET: Vec3 = Vec3::ZERO;
const UP: Vec3 = Vec3::Y;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct UniformBufferObject {
    model_matrix: [[f32; 4]; 4],
    view_projection_matrix: [[f32; 4]; 4],
}

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

pub fn update_uniform_buffer(vulkan_application_data: &VulkanApplicationData) -> anyhow::Result<()> {
    let extent = vulkan_application_data.swapchain_accepted_images_width_and_height;
    let width = extent.width as f32;
    let height = extent.height as f32;

    let projection = compute_projection_matrix(FOV_DEGREES, NEAR_PLANE, FAR_PLANE, width, height)?;
    let view = compute_view_matrix(CAMERA_POSITION, CAMERA_TARGET, UP)?;
    let view_projection = projection * view;

    let rotation = Quat::from_rotation_y(30.0_f32.to_radians()) * Quat::from_rotation_x(-20.0_f32.to_radians());
    let model = compute_model_matrix(Vec3::ONE, rotation, Vec3::ZERO)?;

    let ubo = UniformBufferObject {
        model_matrix: model.to_cols_array_2d(),
        view_projection_matrix: view_projection.to_cols_array_2d(),
    };

    unsafe {
        std::ptr::copy_nonoverlapping(&ubo, vulkan_application_data.uniform_buffer_ptr, 1);
    }
    Ok(())
}

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
