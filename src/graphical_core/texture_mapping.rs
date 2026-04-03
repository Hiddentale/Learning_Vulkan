use crate::graphical_core::{buffers::allocate_and_fill_buffer, memory::find_memory_type, vulkan_object::VulkanApplicationData};
use vk::Handle;
use vulkan_rust::{vk, Device, Instance};

const TEXTURE_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;
const BYTES_PER_PIXEL: u32 = 4;

fn get_texture_path(texture_name: &str) -> String {
    let project_root = env!("CARGO_MANIFEST_DIR");
    format!("{}/textures/{}", project_root, texture_name)
}

/// Loads a texture from disk and returns the GPU image, memory, view, and sampler.
pub fn create_texture_image(
    device: &Device,
    instance: &Instance,
    vulkan_application_data: &mut VulkanApplicationData,
) -> anyhow::Result<(vk::Image, vk::DeviceMemory, vk::ImageView, vk::Sampler)> {
    let texture_path = get_texture_path("red_grass.png");
    let (image_bytes, width, height) = load_texture_from_disk(&texture_path)?;
    let (staging_buffer, staging_buffer_memory) =
        create_and_fill_staging_buffer(image_bytes, width, height, device, instance, vulkan_application_data)?;
    let image = create_image(device, width, height)?;
    let device_memory = allocate_and_bind_image_device_memory(device, image, instance, vulkan_application_data)?;
    transfer_image_data(device, image, width, height, staging_buffer, vulkan_application_data)?;
    let image_view = create_image_view(device, image)?;
    let sampler = create_sampler(device)?;
    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
    }
    Ok((image, device_memory, image_view, sampler))
}

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
    device: &Device,
    instance: &Instance,
    vulkan_application_data: &mut VulkanApplicationData,
) -> anyhow::Result<(vk::Buffer, vk::DeviceMemory)> {
    let staging_buffer_size_in_bytes = (image_width * image_height * BYTES_PER_PIXEL) as u64;
    let staging_buffer = unsafe {
        allocate_and_fill_buffer(
            &image_bytes,
            staging_buffer_size_in_bytes,
            vk::BufferUsageFlags::TRANSFER_SRC,
            device,
            instance,
            vulkan_application_data,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?
    };
    Ok(staging_buffer)
}

fn create_image(device: &Device, width: u32, height: u32) -> anyhow::Result<vk::Image> {
    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .format(TEXTURE_FORMAT)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::_1)
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
    let mem_requirements = unsafe { device.get_image_memory_requirements(image) };
    let mem_properties = unsafe { instance.get_physical_device_memory_properties(vulkan_application_data.physical_device) };
    let mem_type_index = find_memory_type(&mem_properties, mem_requirements.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_requirements.size)
        .memory_type_index(mem_type_index);

    let allocated_memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    unsafe { device.bind_image_memory(image, allocated_memory, 0)? };
    Ok(allocated_memory)
}

fn transfer_image_data(
    device: &Device,
    image: vk::Image,
    image_width: u32,
    image_height: u32,
    staging_buffer: vk::Buffer,
    data: &mut VulkanApplicationData,
) -> anyhow::Result<()> {
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_buffer_count(1)
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY);

    let cmd = unsafe { device.allocate_command_buffers(&alloc_info)? }[0];

    let begin_info = vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe { device.begin_command_buffer(cmd, &begin_info)? };
    unsafe { record_image_transfer(device, cmd, image, image_width, image_height, staging_buffer) };
    unsafe { device.end_command_buffer(cmd)? }

    let command_buffers = [cmd];
    let submit_info = vk::SubmitInfo::builder()
        .command_buffers(&command_buffers)
        .wait_semaphores(&[])
        .signal_semaphores(&[])
        .wait_dst_stage_mask(&[]);

    unsafe { device.queue_submit(data.graphics_queue, &[*submit_info], vk::Fence::null())? }
    unsafe { device.queue_wait_idle(data.graphics_queue)? }
    unsafe { device.free_command_buffers(data.command_pool, &[cmd]) }
    Ok(())
}

unsafe fn record_image_transfer(device: &Device, cmd: vk::CommandBuffer, image: vk::Image, width: u32, height: u32, staging_buffer: vk::Buffer) {
    transition_image_layout(device, cmd, image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
    copy_buffer_to_image(device, cmd, staging_buffer, image, width, height);
    transition_image_layout(
        device,
        cmd,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    );
}

unsafe fn transition_image_layout(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let (src_access, dst_access, src_stage, dst_stage) = match (old_layout, new_layout) {
        (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
            vk::AccessFlags::empty(),
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
        ),
        (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        ),
        _ => panic!("Unsupported layout transition: {:?} -> {:?}", old_layout, new_layout),
    };

    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .base_array_layer(0)
        .level_count(1)
        .layer_count(1);

    let barrier = vk::ImageMemoryBarrier::builder()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(*subresource_range)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

    device.cmd_pipeline_barrier(
        cmd,
        src_stage,
        dst_stage,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[*barrier],
    );
}

unsafe fn copy_buffer_to_image(device: &Device, cmd: vk::CommandBuffer, buffer: vk::Buffer, image: vk::Image, width: u32, height: u32) {
    let subresource = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1);

    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(*subresource)
        .image_offset(*vk::Offset3D::builder().x(0).y(0).z(0))
        .image_extent(*vk::Extent3D::builder().width(width).height(height).depth(1));

    device.cmd_copy_buffer_to_image(cmd, buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[*region]);
}

fn create_image_view(device: &Device, image: vk::Image) -> anyhow::Result<vk::ImageView> {
    let identity = vk::ComponentSwizzle::IDENTITY;
    let components = vk::ComponentMapping::builder().r(identity).g(identity).b(identity).a(identity);
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let view_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .format(TEXTURE_FORMAT)
        .view_type(vk::ImageViewType::_2D)
        .components(*components)
        .subresource_range(*subresource_range);

    Ok(unsafe { device.create_image_view(&view_info, None)? })
}

fn create_sampler(device: &Device) -> anyhow::Result<vk::Sampler> {
    let sampler_info = vk::SamplerCreateInfo::builder()
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

    Ok(unsafe { device.create_sampler(&sampler_info, None)? })
}

/// Destroys the texture sampler, image view, image, and frees its device memory.
pub fn destroy_textures(device: &Device, data: &mut VulkanApplicationData) {
    unsafe {
        device.destroy_sampler(data.texture_sampler, None);
        device.destroy_image_view(data.texture_image_view, None);
        device.destroy_image(data.texture_image, None);
        device.free_memory(data.texture_memory, None);
    }
}
