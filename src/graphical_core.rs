pub mod buffers;
pub mod camera;
pub mod commands;
pub mod compute_cull;
pub mod depth;
pub mod descriptors;
pub mod frustum;
pub mod gpu;
pub mod input;
pub mod instance;
pub mod memory;
pub mod mesh_pipeline;
pub mod palette_buffer;
pub mod pipeline;
pub mod queue_families;
pub mod render_pass;
mod shaders;
pub mod svdag_pipeline;
pub mod svdag_pool;
pub mod swapchain;
pub mod texture_mapping;
pub mod ui_pipeline;
pub mod voxel_pool;
pub mod vulkan_object;

const MAX_FRAMES_IN_FLIGHT: usize = 2;

/// Builds an `ImageSubresourceRange` for a single base array layer.
fn subresource_range(aspect: vulkan_rust::vk::ImageAspectFlags, level_count: u32) -> vulkan_rust::vk::ImageSubresourceRange {
    *vulkan_rust::vk::ImageSubresourceRange::builder()
        .aspect_mask(aspect)
        .level_count(level_count)
        .layer_count(1)
}

/// Builds an `ImageSubresourceRange` starting at a specific mip level.
fn subresource_range_mip(aspect: vulkan_rust::vk::ImageAspectFlags, base_mip: u32, level_count: u32) -> vulkan_rust::vk::ImageSubresourceRange {
    *vulkan_rust::vk::ImageSubresourceRange::builder()
        .aspect_mask(aspect)
        .base_mip_level(base_mip)
        .level_count(level_count)
        .layer_count(1)
}

/// Memory visible and coherent from the CPU — used for persistently mapped buffers.
fn host_visible_coherent() -> vulkan_rust::vk::MemoryPropertyFlags {
    vulkan_rust::vk::MemoryPropertyFlags::HOST_VISIBLE | vulkan_rust::vk::MemoryPropertyFlags::HOST_COHERENT
}
