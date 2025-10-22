use crate::graphical_core::buffers::allocate_and_fill_buffer;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use image;
use vulkanalia::vk::BufferUsageFlags;
use vulkanalia::{Device, Instance};

fn load_texture_from_disk(path_to_texture: &str) -> anyhow::Result<Vec<u8>> {
    let texture = image::ImageReader::open(path_to_texture)?.decode()?;

    let image_bytes = texture.to_rgba8().into_raw();
    Ok(image_bytes)
}

fn create_staging_buffer(
    image_bytes: Vec<u8>,
    width: i32,
    height: i32,
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

fn create_image() {}

fn create_image_view() {}

fn create_sampler() {}

fn transfer_image_data() {}

fn create_descriptor_set_layout() {}

fn update_graphics_pipeline() {}

fn create_descriptor_pool() {}

fn allocate_descriptor_set() {}

fn update_descriptor_set() {}

pub fn check_working() -> anyhow::Result<()> {
    let image_bytes = load_texture_from_disk("textures/red_grass.png")?;
    let staging_buffer = create_staging_buffer(image_bytes, width, height, vulkan_logical_device, instance, vulkan_application_data);
    Ok(())
}
