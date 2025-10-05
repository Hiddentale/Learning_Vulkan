/*
Plan:
    Allocate a chunk of GPU memory (vertex buffer):
        1. Create a buffer with usage VERTEX_BUFFER
        2. Allocate memory with properties HOST_VISIBLE | HOST_COHERENT
        3. Bind buffer to memory

    Upload vertex data from CPU → GPU:
        4. Map the memory (get a raw pointer)
        5. memcpy your vertex data into it
        6. Unmap the memory

    Bind that buffer when  ready to draw
    Tell the GPU how to interpret the data
 */

use anyhow;
use vulkanalia::{vk::{self, DeviceV1_0}, Device, Instance};
use crate::graphical_core::vulkan_object::VulkanApplicationData;

//#[repr(C)]
//#[derive(Copy, Clone)]
unsafe fn temp(vulkan_logical_device: &Device, instance: &Instance, vulkan_application_data: &mut VulkanApplicationData) -> anyhow::Result<()> {

    let vertex_buffer_create_info = &vk::BufferCreateInfo {
        size: 12, // Change this to actual vertex data list size calculation: vertices.len() × size_of::<Vertex>()
        usage: vk::BufferUsageFlags::VERTEX_BUFFER,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };

    let vertex_buffer = vulkan_logical_device.create_buffer(vertex_buffer_create_info, None)?;
    
    let v_buffer_mem_requirement = vulkan_logical_device.get_buffer_memory_requirements(vertex_buffer);

    let v_buffer_memory_type = find_memory_type(
        memory_properties: instance.get_physical_device_memory_properties(physical_device),
        type_filter: v_buffer_mem_requirement,
        properties: vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
    );
    Ok(())
}

fn find_memory_type(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    type_filter: u32,
    properties: vk::MemoryPropertyFlags,
) -> Option<u32> {
    Ok
}
