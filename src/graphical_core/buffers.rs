use crate::graphical_core::vulkan_object::VulkanApplicationData;
use anyhow;
use std::{mem::size_of, ptr::copy_nonoverlapping};
use vulkanalia::{
    vk::{self, DeviceV1_0, InstanceV1_0},
    Device, Instance,
};

/// Represents a single vertex with position and color data.
///
/// # Memory Layout
/// `#[repr(C)]` ensures the struct has a predictable memory layout matching C/Vulkan expectations.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
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

/// Allocates GPU buffer memory, copies data from CPU to GPU, and returns handles.
///
/// # Process Overview
/// 1. Create a buffer object (defines size and usage, but no memory backing yet)
/// 2. Query memory requirements (alignment, size, compatible memory types)
/// 3. Find a suitable memory type (HOST_VISIBLE so CPU can write to it)
/// 4. Allocate GPU memory
/// 5. Bind the buffer to the allocated memory (connect buffer handle to actual memory)
/// 6. Map the memory (get a CPU pointer to the GPU memory)
/// 7. Copy data from CPU to GPU via the mapped pointer
/// 8. Unmap the memory (invalidate the CPU pointer, but data persists in GPU memory)
///
/// # Memory Type Strategy
/// Uses `HOST_VISIBLE | HOST_COHERENT` memory:
/// - `HOST_VISIBLE`: CPU can map and write to this memory
/// - `HOST_COHERENT`: CPU writes are immediately visible to GPU (no manual flushing needed)
///
/// # Parameters
/// - `data_slice`: The data to upload (vertices, indices, etc.)
/// - `buffer_usage_flags`: What the buffer will be used for (VERTEX_BUFFER, INDEX_BUFFER, etc.)
/// - `vulkan_logical_device`: The logical device to create resources with
/// - `instance`: Needed to query physical device properties
/// - `vulkan_application_data`: Contains the physical device handle
///
/// # Returns
/// A tuple containing:
/// - `vk::Buffer`: The buffer handle (used to bind for drawing)
/// - `vk::DeviceMemory`: The memory allocation backing the buffer (needed for cleanup)
///
/// # Safety
/// This function is marked `unsafe` because it:
/// - Uses raw pointers via `map_memory`
/// - Performs unchecked memory operations via `copy_nonoverlapping`
/// - Requires the caller to ensure proper cleanup (destroy buffer and free memory)
///
/// # Errors
/// Returns an error if:
/// - Buffer creation fails
/// - Memory allocation fails
/// - No suitable memory type is found
/// - Memory mapping fails
unsafe fn allocate_and_fill_buffer<T>(
    data_slice: &[T],
    buffer_usage_flags: vk::BufferUsageFlags,
    vulkan_logical_device: &Device,
    instance: &Instance,
    vulkan_application_data: &mut VulkanApplicationData,
) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_size_in_bytes = (data_slice.len() * size_of::<T>()) as u64;

    let buffer_create_info = vk::BufferCreateInfo {
        size: (VERTICES.len() * size_of::<Vertex>()) as u64,
        usage: vk::BufferUsageFlags::VERTEX_BUFFER,
        sharing_mode: vk::SharingMode::EXCLUSIVE,
        ..Default::default()
    };

    let buffer = unsafe { vulkan_logical_device.create_buffer(&buffer_create_info, None)? };

    let buffer_mem_requirements = unsafe { vulkan_logical_device.get_buffer_memory_requirements(buffer) };

    let memory_properties = instance.get_physical_device_memory_properties(vulkan_application_data.physical_device);

    let allowed_memory_types = buffer_mem_requirements.memory_type_bits;
    let desired_properties = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    let buffer_memory_type_index = find_memory_type(&memory_properties, allowed_memory_types, desired_properties)?;

    let allocation_info = vk::MemoryAllocateInfo {
        s_type: vk::StructureType::MEMORY_ALLOCATE_INFO,
        next: std::ptr::null(),
        allocation_size: buffer_mem_requirements.size,
        memory_type_index: buffer_memory_type_index,
    };

    let allocated_memory = unsafe { vulkan_logical_device.allocate_memory(&allocation_info, None)? };

    unsafe { vulkan_logical_device.bind_buffer_memory(buffer, allocated_memory, 0)? };

    let pointer_to_mapped_memory = unsafe {
        vulkan_logical_device.map_memory(
            allocated_memory,
            vk::DeviceSize::default(),    // Start at the beginning of the allocation
            buffer_mem_requirements.size, // Map the entire allocation
            vk::MemoryMapFlags::empty(),  // No special flags needed
        )?
    };

    let vertex_pointer = pointer_to_mapped_memory as *mut Vertex;

    unsafe {
        copy_nonoverlapping(
            VERTICES.as_ptr(), // Source: CPU memory containing our data
            vertex_pointer,    // Destination: Mapped pointer to GPU memory
            VERTICES.len(),    // Number of elements to copy
        )
    };

    unsafe { vulkan_logical_device.unmap_memory(allocated_memory) };

    Ok((buffer, allocated_memory))
}

/// Finds a memory type that satisfies both hardware requirements and desired properties.
/// # Parameters
/// - `memory_properties`: The GPU's available memory types and their properties
/// - `allowed_memory_types`: Bitmask of which memory types the buffer supports
/// - `desired_properties`: The properties we need
///
/// # Returns
/// The index of the first suitable memory type found.
///
/// # Errors
/// Returns an error if no memory type satisfies both requirements.
fn find_memory_type(
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    allowed_memory_types: u32,
    requested_properties: vk::MemoryPropertyFlags,
) -> anyhow::Result<u32> {
    let number_of_different_memory_types = memory_properties.memory_type_count;

    for memory_type_index in 0..number_of_different_memory_types {
        let memory_type_is_allowed = (allowed_memory_types & (1 << memory_type_index)) != 0;

        if memory_type_is_allowed {
            let memory_type_properties = memory_properties.memory_types[memory_type_index as usize].property_flags;

            let has_all_desired_properties = (memory_type_properties & requested_properties) == requested_properties;
            if has_all_desired_properties {
                return Ok(memory_type_index);
            }
        }
    }
    anyhow::bail!(
        "Failed to find a suitable memory type for requested properties: {:?}",
        requested_properties
    );
}
