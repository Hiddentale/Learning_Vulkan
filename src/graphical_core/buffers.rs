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
    let buffer_create_info = &vk::BufferCreateInfo {
        size: (VERTICES.len() * size_of::<Vertex>()) as u64,
        usage: vk::BufferUsageFlags::VERTEX_BUFFER,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };

    let buffer = unsafe { vulkan_logical_device.create_buffer(buffer_create_info, None)? };

    let buffer_mem_requirement = unsafe { vulkan_logical_device.get_buffer_memory_requirements(buffer) };

    let memory_properties = instance.get_physical_device_memory_properties(vulkan_application_data.physical_device);
    let allowed_memory_types = buffer_mem_requirement.memory_type_bits;
    let desired_properties = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    let buffer_memory_type = find_memory_type(&memory_properties, allowed_memory_types, desired_properties)?;

    let allocation_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        next: std::ptr::null(),
        allocation_size: buffer_mem_requirement.size,
        memory_type_index: buffer_memory_type,
    };

    let memory_allocated = unsafe { vulkan_logical_device.allocate_memory(&allocation_info, None)? };

    let allocate_buffer = unsafe { vulkan_logical_device.bind_buffer_memory(buffer, memory_allocated, 0)? };

    let map_memory = unsafe { vulkan_logical_device.map_memory(memory_allocated, vk::DeviceSize::default(), buffer_mem_requirement.size, vk::MemoryMapFlags::empty())? };

    let m

    unmap_memory = unsafe { vulkan_logical_device.unmap_memory(memory_allocated) };

    Ok(())
}

fn find_memory_type(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    allowed_memory_types: u32,
    desired_properties: vk::MemoryPropertyFlags,
) -> anyhow::Result<u32> {
    let number_of_different_memory_types = memory_properties.memory_type_count;
    for i in 0..number_of_different_memory_types {
        if (allowed_memory_types & (1 << i)) != 0 {
            if (memory_properties.memory_types[i as usize].property_flags & desired_properties) == desired_properties {
                return Ok(i);
            }
        }
    }
    anyhow::bail!("Couldn't find a suitable memory type for given properties: {:?}", desired_properties);
}
