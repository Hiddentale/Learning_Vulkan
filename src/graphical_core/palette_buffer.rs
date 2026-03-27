use crate::graphical_core::buffers::allocate_and_fill_buffer;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::voxel::material::{default_palette, MaterialPalette};
use vulkanalia::vk::{self, DeviceV1_0};
use vulkanalia::{Device, Instance};

/// Allocates a GPU buffer and uploads the default material palette.
///
/// The palette is static — uploaded once and never updated at runtime.
pub unsafe fn create_palette_buffer(device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let palette = default_palette();
    let buffer_size = std::mem::size_of::<MaterialPalette>() as u64;
    let properties = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    let (buffer, memory) = allocate_and_fill_buffer(
        std::slice::from_ref(&palette),
        buffer_size,
        vk::BufferUsageFlags::UNIFORM_BUFFER,
        device,
        instance,
        data,
        properties,
    )?;

    data.palette_buffer = buffer;
    data.palette_buffer_memory = memory;
    Ok(())
}

/// Destroys the palette buffer and frees its backing memory.
pub unsafe fn destroy_palette_buffer(device: &Device, data: &mut VulkanApplicationData) {
    device.destroy_buffer(data.palette_buffer, None);
    device.free_memory(data.palette_buffer_memory, None);
}
