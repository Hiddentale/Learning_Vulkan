use crate::graphical_core::buffers::allocate_and_fill_buffer;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use image;
use vulkanalia::vk;
use vulkanalia::vk::DeviceV1_0;
use vulkanalia::vk::{BufferUsageFlags, HasBuilder};
use vulkanalia::{Device, Instance};

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
    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .format(vk::Format::R8G8B8A8_SRGB)
        .extent(vk::Extent3D { width, height, depth: 1 }) // Every 2D texture exists in 3D space conceptually
        .mip_levels(1) // How many mipmaps, 1 means no mipmaps
        .array_layers(1) //Number of texture layers
        .samples(vk::SampleCountFlags::_1) //Multisampling/anti-aliasing (number of samples per pixel)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    Ok(unsafe { device.create_image(&image_info, None)? })
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
