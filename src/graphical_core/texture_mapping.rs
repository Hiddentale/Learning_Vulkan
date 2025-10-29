use crate::graphical_core::buffers::allocate_and_fill_buffer;
use crate::graphical_core::memory::find_memory_type;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use image;
use vulkanalia::vk;
use vulkanalia::vk::DeviceV1_0;
use vulkanalia::vk::InstanceV1_0;
use vulkanalia::vk::{BufferUsageFlags, HasBuilder};
use vulkanalia::{Device, Instance};

fn load_texture_from_disk(path_to_texture: &str) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    let texture = image::ImageReader::open(path_to_texture)?.decode()?;

    let width = texture.width();
    let height = texture.height();

    let image_bytes = texture.to_rgba8().into_raw();

    Ok((image_bytes, width, height))
}

fn create_and_fill_staging_buffer(
    image_bytes: Vec<u8>,
    width: u32,
    height: u32,
    vulkan_logical_device: &Device,
    instance: &Instance,
    vulkan_application_data: &mut VulkanApplicationData,
) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
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
    Ok(staging_buffer)
}
/// Create a GPU resource handle that describes the pixel data and what you can do with it.
/// # Parameters
/// - `device`: The Vulkan physical device that communicates with the GPU
/// - `width`: The width of the pixel data before it was converted to bytes
/// - `height`: The height of the pixel data before it was converted to bytes
///
/// Note that this creates an Image in GPU-only memory, with a special type of layout, it doesn't contain any data yet!
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

fn allocate_and_bind_image_device_memory(
    device: &Device,
    image: vk::Image,
    instance: &Instance,
    vulkan_application_data: &mut VulkanApplicationData,
) -> anyhow::Result<vk::DeviceMemory> {
    let image_memory_requirements = unsafe { device.get_image_memory_requirements(image) };

    let memory_properties = unsafe { instance.get_physical_device_memory_properties(vulkan_application_data.physical_device) };

    let allowed_memory_types = image_memory_requirements.memory_type_bits;

    let desired_properties = vk::MemoryPropertyFlags::DEVICE_LOCAL;

    let memory_type_index = find_memory_type(&memory_properties, allowed_memory_types, desired_properties)?;

    let allocate_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(image_memory_requirements.size)
        .memory_type_index(memory_type_index);

    let allocated_memory = unsafe { device.allocate_memory(&allocate_info, None)? };

    unsafe { device.bind_image_memory(image, allocated_memory, 0)? };

    Ok(allocated_memory)
}

fn create_image_view(device: &Device, image: vk::Image) -> anyhow::Result<vk::ImageView> {
    let normal_rgba_values = vk::ComponentSwizzle::IDENTITY;
    let components = vk::ComponentMapping::builder()
        .r(normal_rgba_values)
        .g(normal_rgba_values)
        .b(normal_rgba_values)
        .a(normal_rgba_values);
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let image_view_create_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .format(vk::Format::R8G8B8A8_SRGB)
        .view_type(vk::ImageViewType::_2D)
        .components(components)
        .subresource_range(subresource_range);

    Ok(unsafe { device.create_image_view(&image_view_create_info, None)? })
}

fn create_sampler() {}

fn transfer_image_data() {} // Command buffer recording

fn create_descriptor_set_layout() {} // Define what resources shaders expect

fn update_graphics_pipeline() {}

fn create_descriptor_pool() {}

fn allocate_descriptor_set() {}

fn update_descriptor_set() {}

pub fn check_working() -> anyhow::Result<()> {
    let (image_bytes, width, height) = load_texture_from_disk("textures/red_grass.png")?;
    let staging_buffer = create_and_fill_staging_buffer(image_bytes, width, height, vulkan_logical_device, instance, vulkan_application_data);
    Ok(())
}
