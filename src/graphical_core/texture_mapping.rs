use crate::graphical_core::buffers::allocate_and_fill_buffer;
use crate::graphical_core::memory::find_memory_type;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use image;
use vulkanalia::vk;
use vulkanalia::vk::DeviceV1_0;
use vulkanalia::vk::Handle;
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
    image_width: u32,
    image_height: u32,
    vulkan_logical_device: &Device,
    instance: &Instance,
    vulkan_application_data: &mut VulkanApplicationData,
) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
    let staging_buffer_size_in_bytes = (image_width * image_height * 4) as u64;
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

fn create_sampler(device: &Device) -> anyhow::Result<vk::Sampler> {
    let sampler_create_info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .anisotropy_enable(false)
        .max_anisotropy(1.0)
        .min_lod(0.0)
        .max_lod(0.0)
        .mip_lod_bias(0.0);

    Ok(unsafe { device.create_sampler(&sampler_create_info, None)? })
}

fn transfer_image_data(
    device: &Device,
    image: vk::Image,
    image_width: u32,
    image_height: u32,
    staging_buffer: vk::Buffer,
    instance: &Instance,
    vulkan_application_data: &mut VulkanApplicationData,
) -> anyhow::Result<()> {
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_buffer_count(1)
        .command_pool(vulkan_application_data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY);

    let command_buffer = unsafe { device.allocate_command_buffers(&allocate_info)? };

    let command_buffer_begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    unsafe { device.begin_command_buffer(command_buffer[0], &command_buffer_begin_info)? };

    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .base_array_layer(0)
        .level_count(1)
        .layer_count(1);

    let initial_image_memory_barriers = vk::ImageMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .image(image)
        .subresource_range(subresource_range)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .build();

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer[0],
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[initial_image_memory_barriers],
        );
    }

    let image_subresource = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1);

    let offset = vk::Offset3D::builder().x(0).y(0).z(0);
    let image_extent = vk::Extent3D::builder().width(image_width).height(image_height).depth(1);

    let regions = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(image_subresource)
        .image_offset(offset)
        .image_extent(image_extent);

    unsafe {
        device.cmd_copy_buffer_to_image(
            command_buffer[0],
            staging_buffer,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[regions],
        );
    }

    let image_memory_barriers = vk::ImageMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    unsafe {
        device.cmd_pipeline_barrier(
            command_buffer[0],
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[image_memory_barriers],
        );
    }

    unsafe { device.end_command_buffer(command_buffer[0])? }

    let command_buffers = [command_buffer[0]];

    let submit_info = vk::SubmitInfo::builder()
        .command_buffers(&command_buffers)
        .wait_semaphores(&[])
        .signal_semaphores(&[])
        .wait_dst_stage_mask(&[]);

    unsafe { device.queue_submit(vulkan_application_data.graphics_queue, &[submit_info], vk::Fence::null())? }

    unsafe { device.queue_wait_idle(vulkan_application_data.graphics_queue)? }

    unsafe {
        device.free_command_buffers(vulkan_application_data.command_pool, &[command_buffer[0]]);
    }

    Ok(())
}

pub fn create_descriptor_set_layout(device: &Device, vulkan_application_data: &mut VulkanApplicationData,) -> anyhow::Result<()> {
    let layout_info = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .build();
    let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[layout_info]).build();
    vulkan_application_data.descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&create_info, None)? };
    Ok(())
}

fn create_descriptor_pool() {}

fn allocate_descriptor_set() {}

fn update_descriptor_set() {}
