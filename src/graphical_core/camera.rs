use crate::graphical_core::{buffers::allocate_and_fill_buffer, vulkan_object::VulkanApplicationData};
use vulkanalia::vk;
use vulkanalia::{vk::BufferUsageFlags, Device, Instance};

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
            vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::,
        )?
    };
    Ok(uniform_buffer)
}