use crate::graphical_core::memory::find_memory_type;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use vulkanalia::vk::{DeviceV1_0, HasBuilder, InstanceV1_0};
use vulkanalia::{vk, Device, Instance};

const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

pub fn depth_format() -> vk::Format {
    DEPTH_FORMAT
}

pub unsafe fn create_depth_image(
    device: &Device,
    instance: &Instance,
    data: &mut VulkanApplicationData,
) -> anyhow::Result<()> {
    let extent = data.swapchain_accepted_images_width_and_height;

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
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    data.depth_image = device.create_image(&image_info, None)?;

    let mem_requirements = device.get_image_memory_requirements(data.depth_image);
    let mem_properties = instance.get_physical_device_memory_properties(data.physical_device);
    let mem_type_index = find_memory_type(
        &mem_properties,
        mem_requirements.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

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

pub unsafe fn destroy_depth_image(device: &Device, data: &mut VulkanApplicationData) {
    device.destroy_image_view(data.depth_image_view, None);
    device.destroy_image(data.depth_image, None);
    device.free_memory(data.depth_image_memory, None);
}
