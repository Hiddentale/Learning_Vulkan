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
/* */
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use anyhow;
use vulkanalia::{
    vk::{self, DeviceV1_0, InstanceV1_0},
    Device, Instance,
};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: [f32; 2],
    color: [f32; 3],
}

const VERTICES: [Vertex; 3] = [
    Vertex {
        pos: [0.0, -0.5],
        color: [1.0, 0.0, 0.0],
    },
    Vertex {
        pos: [0.5, 0.5],
        color: [0.0, 1.0, 0.0],
    },
    Vertex {
        pos: [-0.5, 0.5],
        color: [0.0, 0.0, 1.0],
    },
];

unsafe fn temp(vulkan_logical_device: &Device, instance: &Instance, vulkan_application_data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let vertex_buffer_create_info = &vk::BufferCreateInfo {
        size: (VERTICES.len() * size_of::<Vertex>()) as u64,
        usage: vk::BufferUsageFlags::VERTEX_BUFFER,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };

    let vertex_buffer = vulkan_logical_device.create_buffer(vertex_buffer_create_info, None)?;

    let v_buffer_mem_requirement = vulkan_logical_device.get_buffer_memory_requirements(vertex_buffer);

    let memory_properties = instance.get_physical_device_memory_properties(vulkan_application_data.physical_device);
    let type_filter = v_buffer_mem_requirement.memory_type_bits;
    let desired_properties = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    let v_buffer_memory_type = find_memory_type(&memory_properties, type_filter, desired_properties);
    Ok(())
}

fn find_memory_type(memory_properties: &vk::PhysicalDeviceMemoryProperties, type_filter: u32, properties: vk::MemoryPropertyFlags) -> Option<u32> {
    let number_of_different_memory_types = memory_properties.memory_type_count;
    for i in 0..(number_of_different_memory_types - 1) {
        if (type_filter & (1 << i)) != 0 { // Need to understand this better
        }
    }
    Some(2)
}
