use crate::graphical_core::memory::find_memory_type;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use vulkanalia::vk::{DeviceV1_0, HasBuilder, InstanceV1_0};
use vulkanalia::{vk, Device, Instance};

const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// Returns the depth buffer format used across the pipeline.
pub fn depth_format() -> vk::Format {
    DEPTH_FORMAT
}

/// Creates a depth image, allocates device memory, and creates its image view.
pub unsafe fn create_depth_image(device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let extent = data.swapchain_extent;

    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .format(DEPTH_FORMAT)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    data.depth_image = device.create_image(&image_info, None)?;

    let mem_requirements = device.get_image_memory_requirements(data.depth_image);
    let mem_properties = instance.get_physical_device_memory_properties(data.physical_device);
    let mem_type_index = find_memory_type(&mem_properties, mem_requirements.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(mem_type_index);

    data.depth_image_memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_image_memory(data.depth_image, data.depth_image_memory, 0)?;

    let view_info = vk::ImageViewCreateInfo::builder()
        .image(data.depth_image)
        .view_type(vk::ImageViewType::_2D)
        .format(DEPTH_FORMAT)
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );

    data.depth_image_view = device.create_image_view(&view_info, None)?;

    Ok(())
}

/// Destroys the depth image view, image, and frees its device memory.
pub unsafe fn destroy_depth_image(device: &Device, data: &mut VulkanApplicationData) {
    device.destroy_image_view(data.depth_image_view, None);
    device.destroy_image(data.depth_image, None);
    device.free_memory(data.depth_image_memory, None);
}

/// Computes the number of mip levels for a depth pyramid covering the given extent.
pub fn pyramid_mip_count(extent: vk::Extent2D) -> u32 {
    (extent.width.max(extent.height) as f32).log2().floor() as u32 + 1
}

/// Creates the depth pyramid: an R32_SFLOAT image with a full mip chain,
/// one image view per mip level, and a nearest-filter sampler.
pub unsafe fn create_depth_pyramid(device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let extent = data.swapchain_extent;
    let mip_count = pyramid_mip_count(extent);

    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .format(vk::Format::R32_SFLOAT)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(mip_count)
        .array_layers(1)
        .samples(vk::SampleCountFlags::_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    data.depth_pyramid_image = device.create_image(&image_info, None)?;

    let mem_requirements = device.get_image_memory_requirements(data.depth_pyramid_image);
    let mem_properties = instance.get_physical_device_memory_properties(data.physical_device);
    let mem_type_index = find_memory_type(&mem_properties, mem_requirements.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(mem_type_index);
    data.depth_pyramid_memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_image_memory(data.depth_pyramid_image, data.depth_pyramid_memory, 0)?;

    // One image view per mip level (compute shader writes one mip at a time)
    let mut mip_views = Vec::with_capacity(mip_count as usize);
    for mip in 0..mip_count {
        let view_info = vk::ImageViewCreateInfo::builder()
            .image(data.depth_pyramid_image)
            .view_type(vk::ImageViewType::_2D)
            .format(vk::Format::R32_SFLOAT)
            .subresource_range(
                vk::ImageSubresourceRange::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(mip)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        mip_views.push(device.create_image_view(&view_info, None)?);
    }
    data.depth_pyramid_mip_views = mip_views;

    // Full-mip view for textureLod() sampling in the cull shader
    let full_view_info = vk::ImageViewCreateInfo::builder()
        .image(data.depth_pyramid_image)
        .view_type(vk::ImageViewType::_2D)
        .format(vk::Format::R32_SFLOAT)
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(mip_count)
                .base_array_layer(0)
                .layer_count(1),
        );
    data.depth_pyramid_full_view = device.create_image_view(&full_view_info, None)?;

    // Nearest-filter sampler with clamp-to-edge (min-reduction reads)
    let sampler_info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .max_lod(mip_count as f32);
    data.depth_pyramid_sampler = device.create_sampler(&sampler_info, None)?;
    data.depth_pyramid_mip_count = mip_count;

    Ok(())
}

/// Destroys the depth pyramid image, all mip views, sampler, and frees memory.
pub unsafe fn destroy_depth_pyramid(device: &Device, data: &mut VulkanApplicationData) {
    device.destroy_sampler(data.depth_pyramid_sampler, None);
    device.destroy_image_view(data.depth_pyramid_full_view, None);
    for &view in &data.depth_pyramid_mip_views {
        device.destroy_image_view(view, None);
    }
    data.depth_pyramid_mip_views.clear();
    device.destroy_image(data.depth_pyramid_image, None);
    device.free_memory(data.depth_pyramid_memory, None);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mip_count_1920x1080() {
        let extent = vk::Extent2D { width: 1920, height: 1080 };
        assert_eq!(pyramid_mip_count(extent), 11);
    }

    #[test]
    fn mip_count_power_of_two() {
        let extent = vk::Extent2D { width: 1024, height: 1024 };
        assert_eq!(pyramid_mip_count(extent), 11);
    }

    #[test]
    fn mip_count_small() {
        let extent = vk::Extent2D { width: 2, height: 2 };
        assert_eq!(pyramid_mip_count(extent), 2);
    }

    #[test]
    fn mip_count_1x1() {
        let extent = vk::Extent2D { width: 1, height: 1 };
        assert_eq!(pyramid_mip_count(extent), 1);
    }
}
