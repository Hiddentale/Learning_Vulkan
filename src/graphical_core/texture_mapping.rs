use crate::graphical_core::buffers::allocate_and_fill_buffer;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use image;
use vulkanalia::vk;
use vulkanalia::vk::{BufferUsageFlags, MemoryPropertyFlags};
use vulkanalia::{Device, DeviceV1_0, Instance};

fn load_texture_from_disk(path_to_texture: &str) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    let texture = image::ImageReader::open(path_to_texture)?.decode()?;

    let width = texture.width();
    let height = texture.height();

    let image_bytes = texture.to_rgba8().into_raw();

    Ok((image_bytes, width, height))
}

fn create_staging_buffer(
    image_bytes: Vec<u8>,
    width: u32,
    height: u32,
    vulkan_logical_device: &Device,
    instance: &Instance,
    vulkan_application_data: &mut VulkanApplicationData,
) -> anyhow::Result<()> {
    let staging_buffer_size_in_bytes = (width * height * 4) as u64;
    let staging_buffer = unsafe {
        allocate_and_fill_buffer(
            &image_bytes,
            staging_buffer_size_in_bytes,
            BufferUsageFlags::TRANSFER_SRC,
            vulkan_logical_device,
            instance,
            vulkan_application_data,
        )?
    };
    Ok(())
}

fn create_image(device: &Device, width: u32, height: u32) -> anyhow::Result<vk::Image> {
    let image_info = vk::ImageCreateInfo {
        s_type: vk::StructureType::IMAGE_CREATE_INFO,
        flags: None,
        next: std::ptr::null(),
        image_type: vk::ImageType::_2D,
        format: vk::Format::R8G8B8A8_SRGB,
        extent: vk::Extent3D { width, height },
        mip_levels: None,
        array_layers: None,
        samples: None,
        tiling: vk::ImageTiling::OPTIMAL,
        usage: vk::ImageUsageFlags::TRANSFER_SRC,
        sharing_mode: None,
        queue_family_index_count: None,
        queue_family_indices: None,
        initial_layout: None,
    };
    device.create_image(&image_info, allocator)
}

fn create_image_view() {}

fn create_sampler() {}

fn transfer_image_data() {}

fn create_descriptor_set_layout() {}

fn update_graphics_pipeline() {}

fn create_descriptor_pool() {}

fn allocate_descriptor_set() {}

fn update_descriptor_set() {}

pub fn check_working() -> anyhow::Result<()> {
    let (image_bytes, width, height) = load_texture_from_disk("textures/red_grass.png")?;
    let staging_buffer = create_staging_buffer(image_bytes, width, height, vulkan_logical_device, instance, vulkan_application_data);
    Ok(())
}
