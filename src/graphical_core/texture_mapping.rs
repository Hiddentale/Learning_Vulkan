use crate::graphical_core::{buffers::allocate_and_fill_buffer, memory::find_memory_type, vulkan_object::VulkanApplicationData};
use image;
use vulkanalia::vk;
use vulkanalia::{
    vk::{
        BufferUsageFlags, CopyDescriptorSet, DescriptorPool, DescriptorSet, DescriptorSetLayout, DeviceMemory, DeviceV1_0, Handle, HasBuilder, Image,
        ImageView, InstanceV1_0, Sampler,
    },
    Device, Instance,
};

/// Resolves a texture filename to an absolute path at compile time.
///
/// # Process Overview
/// Uses the `CARGO_MANIFEST_DIR` environment variable (set by Cargo during compilation)
/// to construct an absolute path to the project's textures directory. This ensures
/// the texture can be found regardless of where the executable is run from.
///
/// # Parameters
/// - `texture_name`: Filename of the texture (e.g., "dirt.png")
///
/// # Returns
/// Absolute path string like "/home/user/project/textures/dirt.png"
///
/// # Note
/// The path is baked into the binary at compile time. If you move the project
/// directory, you'll need to recompile.
fn get_texture_path(texture_name: &str) -> String {
    let project_root = env!("CARGO_MANIFEST_DIR");
    format!("{}/textures/{}", project_root, texture_name)
}

/// Orchestrates the complete texture loading pipeline from disk to GPU.
///
/// # Process Overview
/// This is the main public function that coordinates all texture creation steps:
/// 1. Load PNG/JPEG from disk into RAM
/// 2. Create CPU-accessible staging buffer and copy pixel data into it
/// 3. Create GPU image handle (no memory yet)
/// 4. Allocate GPU-only memory and bind image to it
/// 5. Transfer data: staging buffer → GPU image (with layout transitions)
/// 6. Create image view (how shaders interpret the image)
/// 7. Create sampler (how shaders sample the image)
///
/// # Parameters
/// - `device`: The logical device for all resource creation
/// - `instance`: Needed for memory type queries
/// - `vulkan_application_data`: Contains physical device and command pool
///
/// # Returns
/// A tuple containing all texture resources needed for rendering:
/// - `vk::Image`: The GPU image handle (must be destroyed at shutdown)
/// - `vk::DeviceMemory`: The GPU memory backing the image (must be freed at shutdown)
/// - `vk::ImageView`: View into the image for shader access (must be destroyed)
/// - `vk::Sampler`: Sampler defining filtering/addressing (must be destroyed)
///
/// # Errors
/// Returns an error if any step fails (file not found, out of memory, etc.)
///
/// # Note
/// Currently hardcoded to load "red_grass.png".
pub fn create_texture_image(
    device: &Device,
    instance: &Instance,
    vulkan_application_data: &mut VulkanApplicationData,
) -> anyhow::Result<(Image, DeviceMemory, ImageView, Sampler)> {
    let texture_path = get_texture_path("red_grass.png");
    let (image_bytes, width, height) = load_texture_from_disk(&texture_path)?;
    let (staging_buffer, staging_buffer_memory) =
        create_and_fill_staging_buffer(image_bytes, width, height, device, instance, vulkan_application_data)?;
    let image = create_image(device, width, height)?;
    let device_memory = allocate_and_bind_image_device_memory(device, image, instance, vulkan_application_data)?;
    transfer_image_data(device, image, width, height, staging_buffer, instance, vulkan_application_data)?;
    let image_view = create_image_view(device, image)?;
    let sampler = create_sampler(device)?;
    unsafe {
        device.destroy_buffer(staging_buffer, None);
        device.free_memory(staging_buffer_memory, None);
    }
    Ok((image, device_memory, image_view, sampler))
}

/// Loads an image file from disk and decodes it into raw RGBA8 pixel data.
///
/// # Process Overview
/// 1. Open the image file using the `image` crate
/// 2. Decode the image format (PNG, JPEG, etc.)
/// 3. Convert to RGBA8 format (4 bytes per pixel: red, green, blue, alpha)
/// 4. Extract dimensions and raw pixel bytes
///
/// # Parameters
/// - `path_to_texture`: File path to the image
///
/// # Returns
/// A tuple containing:
/// - `Vec<u8>`: Raw pixel data in RGBA8 format (width × height × 4 bytes)
/// - `u32`: Image width in pixels
/// - `u32`: Image height in pixels
///
/// # Errors
/// Returns an error if:
/// - File cannot be opened
/// - Image format is unsupported or corrupted
/// - File is not a valid image
fn load_texture_from_disk(path_to_texture: &str) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    let texture = image::ImageReader::open(path_to_texture)?.decode()?;

    let width = texture.width();
    let height = texture.height();

    let image_bytes = texture.to_rgba8().into_raw();

    Ok((image_bytes, width, height))
}

/// Creates a CPU-accessible staging buffer and fills it with texture pixel data.
///
/// # Purpose
/// Staging buffers act as an intermediate workspace for transferring data to GPU-only
/// memory. The CPU can write to staging buffers, but they're not optimal for GPU access.
/// After filling the staging buffer, we'll copy its contents to a DEVICE_LOCAL image.
///
/// # Process Overview
/// 1. Calculate buffer size (width × height × 4 bytes per RGBA pixel)
/// 2. Create buffer with TRANSFER_SRC usage (can be source of copy operations)
/// 3. Allocate HOST_VISIBLE memory (CPU can write to it)
/// 4. Map memory and copy pixel data from CPU to GPU
///
/// # Parameters
/// - `image_bytes`: Raw RGBA8 pixel data from `load_texture_from_disk`
/// - `image_width`: Width of the texture in pixels
/// - `image_height`: Height of the texture in pixels
/// - `vulkan_logical_device`: Device to create buffer with
/// - `instance`: Needed to query memory properties
/// - `vulkan_application_data`: Contains physical device handle
///
/// # Returns
/// A tuple containing:
/// - `vk::Buffer`: The staging buffer handle
/// - `vk::DeviceMemory`: The memory backing the buffer
///
/// # Errors
/// Returns an error if buffer creation or memory allocation fails.
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
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        )?
    };
    Ok(staging_buffer)
}

/// Creates a GPU image resource with optimal layout for texture sampling.
///
/// # Key Concept
/// This creates an image HANDLE but doesn't allocate any memory yet. Think of it as
/// declaring a variable without initializing it. Memory allocation happens separately
/// in `allocate_and_bind_image_device_memory`.
///
/// # Image Configuration
/// - **Type**: 2D (even though stored in 3D space conceptually)
/// - **Format**: R8G8B8A8_SRGB (4 bytes per pixel, sRGB color space)
/// - **Tiling**: OPTIMAL (GPU-specific layout for fast sampling, not CPU-readable)
/// - **Usage**: TRANSFER_DST (can receive data) | SAMPLED (can be sampled in shaders)
/// - **Initial Layout**: UNDEFINED (contents are garbage until we transfer data)
///
/// # Parameters
/// - `device`: The logical device to create the image with
/// - `width`: Image width in pixels
/// - `height`: Image height in pixels
///
/// # Returns
/// An image handle that must later be bound to actual GPU memory
///
/// # Errors
/// Returns an error if image creation fails (unlikely unless parameters are invalid).
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

/// Allocates GPU-only memory for an image and binds the image handle to it.
///
/// # Process Overview
/// 1. Query the image's memory requirements (size, alignment, compatible types)
/// 2. Find a DEVICE_LOCAL memory type (fastest for GPU access, CPU cannot touch it)
/// 3. Allocate the memory from the GPU
/// 4. Bind the image handle to the allocated memory (connect them)
///
/// # Parameters
/// - `device`: The logical device to allocate memory with
/// - `image`: The image handle created by `create_image`
/// - `instance`: Needed to query physical device memory properties
/// - `vulkan_application_data`: Contains the physical device handle
///
/// # Returns
/// The allocated GPU memory that backs the image
///
/// # Errors
/// Returns an error if:
/// - Memory allocation fails (out of VRAM)
/// - No suitable DEVICE_LOCAL memory type exists
/// - Memory binding fails
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

/// Copies texture data from a staging buffer to a GPU-only image with layout transitions.
///
/// # Process Overview
/// 1. Allocate a temporary command buffer (one-time use)
/// 2. Begin recording commands
/// 3. **Transition**: UNDEFINED → TRANSFER_DST_OPTIMAL (prepare image to receive data)
/// 4. **Copy**: Staging buffer → Image (the actual pixel data transfer)
/// 5. **Transition**: TRANSFER_DST_OPTIMAL → SHADER_READ_ONLY_OPTIMAL (prepare for sampling)
/// 6. End recording and submit to GPU queue
/// 7. Wait for completion
/// 8. Free the temporary command buffer
///
/// # Parameters
/// - `device`: The logical device for command execution
/// - `image`: The destination image (GPU-only memory)
/// - `image_width`: Width in pixels (for defining copy region)
/// - `image_height`: Height in pixels (for defining copy region)
/// - `staging_buffer`: Source buffer containing pixel data (CPU-visible memory)
/// - `instance`: Needed for physical device queries
/// - `vulkan_application_data`: Contains command pool and graphics queue
///
/// # Synchronization
/// This function performs a **blocking wait** (`queue_wait_idle`). The CPU stalls
/// until the GPU finishes copying. For initialization, this is fine. For runtime
/// texture streaming, we'd want asynchronous transfers with fences/semaphores.
///
/// # Errors
/// Returns an error if:
/// - Command buffer allocation fails
/// - Command recording fails
/// - Queue submission fails
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
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

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
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .image(image)
        .subresource_range(subresource_range)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);

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

/// Creates a view into an image that describes how shaders should interpret it.
///
/// # Purpose
/// Images are raw memory. Image views add interpretation: "This is a 2D RGBA texture,
/// access the color channel, don't swizzle components, use all mip levels."
///
/// # Configuration
/// - **View Type**: 2D
/// - **Format**: R8G8B8A8_SRGB
/// - **Component Mapping**: Identity
/// - **Subresource**: All color data, no mipmaps, single layer
///
/// # Parameters
/// - `device`: The logical device to create the view with
/// - `image`: The image to create a view into
///
/// # Returns
/// An image view handle that can be bound to descriptor sets
///
/// # Errors
/// Returns an error if view creation fails.
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

/// Creates a sampler that defines how textures are filtered and addressed.
///
/// # Purpose
/// Samplers control HOW the GPU reads texture data:
/// - What happens when UVs go outside [0, 1]? (address mode)
/// - How to interpolate between pixels? (filtering)
/// - How to blend between mipmap levels? (mipmapping)
///
/// # Configuration
/// - **Filtering**: NEAREST
/// - **Address Mode**: REPEAT
/// - **Mipmapping**: Disabled
/// - **Anisotropic Filtering**: Disabled
///
/// # Parameters
/// - `device`: The logical device to create the sampler with
///
/// # Returns
/// A sampler handle that can be combined with image views in descriptor sets
///
/// # Errors
/// Returns an error if sampler creation fails.
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

/// Defines the structure of descriptor sets: what resources shaders can access.
///
/// # Layout Structure
/// - **Binding 0**: One combined image sampler (image view + sampler in one)
/// - **Shader Stage**: Fragment shader only
/// - **Descriptor Count**: 1
///
/// # Parameters
/// - `device`: The logical device to create the layout with
/// - `vulkan_application_data`: Stores the created layout for later use
///
/// # Errors
/// Returns an error if layout creation fails.
pub fn create_descriptor_set_layout(device: &Device, vulkan_application_data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let layout_info = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_count(1)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);
    let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[layout_info]).build();
    vulkan_application_data.descriptor_set_layout = unsafe { device.create_descriptor_set_layout(&create_info, None)? };
    Ok(())
}

/// Creates a pool that can allocate descriptor sets.
///
/// # Pool Configuration
/// - **Max Sets**: 1
/// - **Pool Sizes**: 1 combined image sampler
///
/// # Parameters
/// - `device`: The logical device to create the pool with
/// - `vulkan_application_data`: Stores the pool for later allocation and cleanup
/// # Errors
/// Returns an error if pool creation fails.
pub fn create_descriptor_pool(device: &Device, vulkan_application_data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let descriptor_pool_size = vk::DescriptorPoolSize::builder()
        .descriptor_count(1)
        .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER);

    let pool_info = vk::DescriptorPoolCreateInfo::builder()
        .flags(vk::DescriptorPoolCreateFlags::empty())
        .max_sets(1)
        .pool_sizes(&[descriptor_pool_size])
        .build();

    vulkan_application_data.descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None)? };
    Ok(())
}

/// Allocates a descriptor set from a pool, matching the provided layout.
///
/// # Parameters
/// - `device`: The logical device to allocate with
/// - `descriptor_pool`: The pool to allocate from (must have capacity remaining)
/// - `layouts`: The descriptor set layout defining the set's structure
///
/// # Returns
/// A handle to a vector of  descriptor sets
///
/// # Errors
/// Returns an error if:
/// - Pool is exhausted (no capacity left)
/// - Pool doesn't support the requested layout
/// - Allocation fails
pub fn allocate_descriptor_set(device: &Device, descriptor_pool: DescriptorPool, layouts: DescriptorSetLayout) -> anyhow::Result<Vec<DescriptorSet>> {
    let allocate_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(descriptor_pool)
        .set_layouts(&[layouts])
        .build();
    Ok(unsafe { device.allocate_descriptor_sets(&allocate_info)? })
}

/// Writes actual texture resources (image view + sampler) into a descriptor set.
///
/// # Parameters
/// - `device`: The logical device to execute the write with
/// - `descriptor_set`: The descriptor set to write into
/// - `image_view`: The texture image view to bind
/// - `sampler`: The sampler to bind
pub fn update_descriptor_set(device: &Device, descriptor_set: DescriptorSet, image_view: ImageView, sampler: Sampler) {
    let image_info = vk::DescriptorImageInfo::builder()
        .image_view(image_view)
        .sampler(sampler)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

    let descriptor_writes = vk::WriteDescriptorSet::builder()
        .dst_set(descriptor_set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&[image_info])
        .build();

    unsafe {
        device.update_descriptor_sets(&[descriptor_writes], &[] as &[CopyDescriptorSet]);
    }
}
