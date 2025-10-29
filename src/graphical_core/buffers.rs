use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::graphical_core::memory::find_memory_type;
use anyhow;
use std::ptr::copy_nonoverlapping;
use vulkanalia::{
    vk::{self, DeviceV1_0, HasBuilder, InstanceV1_0},
    Device, Instance,
};
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
pub unsafe fn allocate_and_fill_buffer<T>(
    data_slice: &[T],
    buffer_size_in_bytes: u64,
    buffer_usage_flags: vk::BufferUsageFlags,
    vulkan_logical_device: &Device,
    instance: &Instance,
    vulkan_application_data: &mut VulkanApplicationData,
) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
    let buffer_create_info = vk::BufferCreateInfo::builder()
        .size(buffer_size_in_bytes)
        .usage(buffer_usage_flags)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = unsafe { vulkan_logical_device.create_buffer(&buffer_create_info, None)? };

    let buffer_mem_requirements = unsafe { vulkan_logical_device.get_buffer_memory_requirements(buffer) };

    let memory_properties = instance.get_physical_device_memory_properties(vulkan_application_data.physical_device);

    let allowed_memory_types = buffer_mem_requirements.memory_type_bits;
    let desired_properties = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    let buffer_memory_type_index = find_memory_type(&memory_properties, allowed_memory_types, desired_properties)?;

    let allocation_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(buffer_mem_requirements.size)
        .memory_type_index(buffer_memory_type_index);

    let allocated_memory = unsafe { vulkan_logical_device.allocate_memory(&allocation_info, None)? };

    unsafe { vulkan_logical_device.bind_buffer_memory(buffer, allocated_memory, 0)? };

    let pointer_to_mapped_memory = unsafe {
        vulkan_logical_device.map_memory(
            allocated_memory,             // Which GPU memory to map
            vk::DeviceSize::default(),    // Start at the beginning of the allocation
            buffer_mem_requirements.size, // Map the entire allocation
            vk::MemoryMapFlags::empty(),  // No special flags needed
        )?
    };

    let vertex_pointer = pointer_to_mapped_memory as *mut T;

    unsafe {
        copy_nonoverlapping(
            data_slice.as_ptr(), // Source: CPU memory containing our data
            vertex_pointer,      // Destination: Mapped pointer to GPU memory
            data_slice.len(),    // Number of elements to copy
        )
    };

    unsafe { vulkan_logical_device.unmap_memory(allocated_memory) };

    Ok((buffer, allocated_memory))
}
