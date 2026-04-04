use crate::graphical_core::{buffers::allocate_and_fill_buffer, memory::find_memory_type, vulkan_object::VulkanApplicationData};
use vk::Handle;
use vulkan_rust::{vk, Device, Instance};

const TEXTURE_FORMAT: vk::Format = vk::Format::R8G8B8A8_SRGB;
const BYTES_PER_PIXEL: u32 = 4;

/// Atlas dimensions: each layer is 32x16 (left half = side, right half = top).
const ATLAS_WIDTH: u32 = 32;
const ATLAS_HEIGHT: u32 = 16;

/// Block textures in material_id order. None means no texture (use palette color).
const TEXTURE_FILES: &[Option<&str>] = &[
    None,              // 0: Air
    Some("grass.png"), // 1: Grass (32x16 atlas: side + top)
    Some("dirt.png"),  // 2: Dirt
    Some("stone.png"), // 3: Stone
    Some("water.png"), // 4: Water
    Some("sand.png"),  // 5: Sand
    None,              // 6: Snow
    None,              // 7: Gravel
];

fn get_texture_path(texture_name: &str) -> String {
    let project_root = env!("CARGO_MANIFEST_DIR");
    format!("{}/textures/{}", project_root, texture_name)
}

/// Loads all block textures into a 2D array image (one layer per material).
/// Layers without textures are filled with white (shader uses palette color instead).
pub fn create_texture_image(
    device: &Device,
    instance: &Instance,
    data: &mut VulkanApplicationData,
) -> anyhow::Result<(vk::Image, vk::DeviceMemory, vk::ImageView, vk::Sampler)> {
    let layer_count = TEXTURE_FILES.len() as u32;
    let layer_bytes = (ATLAS_WIDTH * ATLAS_HEIGHT * BYTES_PER_PIXEL) as usize;
    let mut all_bytes = vec![255u8; layer_bytes * layer_count as usize]; // white default

    for (i, entry) in TEXTURE_FILES.iter().enumerate() {
        if let Some(filename) = entry {
            let path = get_texture_path(filename);
            let (pixels, w, h) = load_texture_from_disk(&path)?;
            let dst = &mut all_bytes[i * layer_bytes..(i + 1) * layer_bytes];
            blit_to_atlas(dst, &pixels, w, h);
        }
    }

    let total_size = (layer_count * ATLAS_WIDTH * ATLAS_HEIGHT * BYTES_PER_PIXEL) as u64;
    let (staging_buffer, staging_buffer_memory) = unsafe {
        allocate_and_fill_buffer(
            &all_bytes,
            total_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            device,
            instance,
            data,
            super::host_visible_coherent(),
        )?
    };

    let image = create_array_image(device, layer_count)?;
    let device_memory = allocate_and_bind_image_memory(device, image, instance, data)?;
    transfer_array_image(device, image, layer_count, staging_buffer, data)?;
    let image_view = create_array_image_view(device, image, layer_count)?;
    let sampler = create_sampler(device)?;

    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
    }

    Ok((image, device_memory, image_view, sampler))
}

/// Copies source pixels into a 32x16 atlas layer.
/// If source is 32x16, copies directly. If 16x16, duplicates into both halves.
fn blit_to_atlas(dst: &mut [u8], src: &[u8], src_w: u32, src_h: u32) {
    let bpp = BYTES_PER_PIXEL as usize;
    if src_w == ATLAS_WIDTH && src_h == ATLAS_HEIGHT {
        dst[..src.len()].copy_from_slice(src);
    } else if src_w == ATLAS_HEIGHT && src_h == ATLAS_HEIGHT {
        // 16x16 -> duplicate into left and right halves of 32x16
        for row in 0..ATLAS_HEIGHT as usize {
            let src_row_start = row * src_w as usize * bpp;
            let src_row = &src[src_row_start..src_row_start + src_w as usize * bpp];
            let dst_row_start = row * ATLAS_WIDTH as usize * bpp;
            // Left half
            dst[dst_row_start..dst_row_start + src_w as usize * bpp].copy_from_slice(src_row);
            // Right half
            let right_start = dst_row_start + src_w as usize * bpp;
            dst[right_start..right_start + src_w as usize * bpp].copy_from_slice(src_row);
        }
    } else {
        log::warn!(
            "Texture {}x{} doesn't match atlas {}x{}, skipping",
            src_w,
            src_h,
            ATLAS_WIDTH,
            ATLAS_HEIGHT
        );
    }
}

fn load_texture_from_disk(path: &str) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    let texture = image::ImageReader::open(path)?.decode()?;
    let width = texture.width();
    let height = texture.height();
    let bytes = texture.to_rgba8().into_raw();
    Ok((bytes, width, height))
}

fn create_array_image(device: &Device, layer_count: u32) -> anyhow::Result<vk::Image> {
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .format(TEXTURE_FORMAT)
        .extent(vk::Extent3D {
            width: ATLAS_WIDTH,
            height: ATLAS_HEIGHT,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(layer_count)
        .samples(vk::SampleCountFlags::_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    Ok(unsafe { device.create_image(&info, None)? })
}

fn allocate_and_bind_image_memory(
    device: &Device,
    image: vk::Image,
    instance: &Instance,
    data: &mut VulkanApplicationData,
) -> anyhow::Result<vk::DeviceMemory> {
    let requirements = unsafe { device.get_image_memory_requirements(image) };
    let properties = unsafe { instance.get_physical_device_memory_properties(data.physical_device) };
    let type_index = find_memory_type(&properties, requirements.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(type_index);

    let memory = unsafe { device.allocate_memory(&alloc_info, None)? };
    unsafe { device.bind_image_memory(image, memory, 0)? };
    Ok(memory)
}

fn transfer_array_image(
    device: &Device,
    image: vk::Image,
    layer_count: u32,
    staging_buffer: vk::Buffer,
    data: &mut VulkanApplicationData,
) -> anyhow::Result<()> {
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_buffer_count(1)
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY);

    let cmd = unsafe { device.allocate_command_buffers(&alloc_info)? }[0];
    unsafe {
        device.begin_command_buffer(
            cmd,
            &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?;

        // Transition all layers: UNDEFINED -> TRANSFER_DST
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            layer_count,
        );

        // Copy each layer from staging buffer
        let layer_size = (ATLAS_WIDTH * ATLAS_HEIGHT * BYTES_PER_PIXEL) as u64;
        let regions: Vec<vk::BufferImageCopy> = (0..layer_count)
            .map(|layer| {
                let subresource = *vk::ImageSubresourceLayers::builder()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(layer)
                    .layer_count(1);

                *vk::BufferImageCopy::builder()
                    .buffer_offset(layer as u64 * layer_size)
                    .image_subresource(subresource)
                    .image_extent(*vk::Extent3D::builder().width(ATLAS_WIDTH).height(ATLAS_HEIGHT).depth(1))
            })
            .collect();

        device.cmd_copy_buffer_to_image(cmd, staging_buffer, image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &regions);

        // Transition all layers: TRANSFER_DST -> SHADER_READ_ONLY
        transition_image_layout(
            device,
            cmd,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            layer_count,
        );

        device.end_command_buffer(cmd)?;
    }

    let command_buffers = [cmd];
    let submit_info = vk::SubmitInfo::builder()
        .command_buffers(&command_buffers)
        .wait_semaphores(&[])
        .signal_semaphores(&[])
        .wait_dst_stage_mask(&[]);

    unsafe {
        device.queue_submit(data.graphics_queue, &[*submit_info], vk::Fence::null())?;
        device.queue_wait_idle(data.graphics_queue)?;
        device.free_command_buffers(data.command_pool, &[cmd]);
    }
    Ok(())
}

unsafe fn transition_image_layout(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    layer_count: u32,
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

    let subresource_range = *vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(layer_count);

    let barrier = vk::ImageMemoryBarrier::builder()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(subresource_range)
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

fn create_array_image_view(device: &Device, image: vk::Image, layer_count: u32) -> anyhow::Result<vk::ImageView> {
    let subresource_range = *vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .level_count(1)
        .layer_count(layer_count);

    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .format(TEXTURE_FORMAT)
        .view_type(vk::ImageViewType::_2D_ARRAY)
        .subresource_range(subresource_range);

    Ok(unsafe { device.create_image_view(&info, None)? })
}

fn create_sampler(device: &Device) -> anyhow::Result<vk::Sampler> {
    let info = vk::SamplerCreateInfo::builder()
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

    Ok(unsafe { device.create_sampler(&info, None)? })
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
