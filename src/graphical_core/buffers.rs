use crate::graphical_core::{memory::find_memory_type, vulkan_object::VulkanApplicationData};
use std::ptr::copy_nonoverlapping;
use vulkan_rust::{vk, Device, Instance};

/// Allocates a GPU buffer with persistently mapped memory.
///
/// Returns the buffer handle, backing memory, and a CPU pointer to the mapped region.
/// The caller is responsible for unmapping and freeing resources at cleanup.
pub unsafe fn allocate_buffer<T>(
    buffer_size_in_bytes: u64,
    buffer_usage_flags: vk::BufferUsageFlags,
    device: &Device,
    instance: &Instance,
    data: &mut VulkanApplicationData,
    properties: vk::MemoryPropertyFlags,
) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory, *mut T)> {
    let buffer_info = vk::BufferCreateInfo::builder()
        .size(buffer_size_in_bytes)
        .usage(buffer_usage_flags)
        .sharing_mode(vk::SharingMode::EXCLUSIVE);

    let buffer = device.create_buffer(&buffer_info, None)?;

    let mem_requirements = device.get_buffer_memory_requirements(buffer);
    let mem_properties = instance.get_physical_device_memory_properties(data.physical_device);
    let mem_type_index = find_memory_type(&mem_properties, mem_requirements.memory_type_bits, properties)?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(mem_type_index);

    let memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_buffer_memory(buffer, memory, 0)?;

    let mapped_ptr = device.map_memory(memory, 0, mem_requirements.size, vk::MemoryMapFlags::empty())? as *mut T;

    Ok((buffer, memory, mapped_ptr))
}

/// Allocates a GPU buffer, copies data into it, then unmaps.
///
/// For one-time uploads (vertex/index data) where the CPU doesn't need
/// ongoing access. The buffer is filled and unmapped immediately.
pub unsafe fn allocate_and_fill_buffer<T>(
    data_slice: &[T],
    buffer_size_in_bytes: u64,
    buffer_usage_flags: vk::BufferUsageFlags,
    device: &Device,
    instance: &Instance,
    data: &mut VulkanApplicationData,
    properties: vk::MemoryPropertyFlags,
) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
    let (buffer, memory, mapped_ptr) = allocate_buffer::<T>(buffer_size_in_bytes, buffer_usage_flags, device, instance, data, properties)?;

    copy_nonoverlapping(data_slice.as_ptr(), mapped_ptr, data_slice.len());
    device.unmap_memory(memory);

    Ok((buffer, memory))
}
