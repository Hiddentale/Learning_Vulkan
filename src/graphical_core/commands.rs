use crate::graphical_core::compute_cull::{CullPushConstants, DepthPyramidResources, DepthReducePush};
use crate::graphical_core::svdag_pipeline::{CullPush, RaymarchPush, SvdagPipeline, TileAssignPush};
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::graphical_core::{self, MAX_FRAMES_IN_FLIGHT};
use vk::Handle;
use vulkan_rust::{vk, Device, Instance};

/// Creates a framebuffer for each swapchain image view, attaching color and depth.
pub unsafe fn create_frame_buffers(device: &Device, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    data.framebuffers = data
        .swapchain_image_views
        .iter()
        .map(|i| {
            let attachments = &[*i, data.depth_image_view];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(data.render_pass)
                .attachments(attachments)
                .width(data.swapchain_extent.width)
                .height(data.swapchain_extent.height)
                .layers(1);

            device.create_framebuffer(&create_info, None)
        })
        .collect::<anyhow::Result<Vec<_>, _>>()?;

    Ok(())
}

/// Creates a command pool for the graphics queue family.
pub unsafe fn create_command_pool(instance: &Instance, device: &Device, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let indices = graphical_core::queue_families::RequiredQueueFamilies::get(instance, data, data.physical_device)?;
    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
        .queue_family_index(indices.graphics_queue_index);

    data.command_pool = device.create_command_pool(&info, None)?;
    Ok(())
}

/// Allocates one command buffer per framebuffer without recording.
pub unsafe fn allocate_command_buffers(device: &Device, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let allocate_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(data.framebuffers.len() as u32);

    data.command_buffers = device.allocate_command_buffers(&allocate_info)?;
    Ok(())
}

unsafe fn transition_pyramid_undefined_to_general(device: &Device, cmd: vk::CommandBuffer, data: &VulkanApplicationData) {
    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .image(data.depth_pyramid_image)
        .subresource_range(super::subresource_range(vk::ImageAspectFlags::COLOR, data.depth_pyramid_mip_count));
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[*barrier],
    );
}

/// Draws a fullscreen triangle with the sky shader before voxel geometry.
pub unsafe fn draw_sky(device: &Device, cmd: vk::CommandBuffer, data: &VulkanApplicationData) {
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, data.sky_pipeline);
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        data.sky_pipeline_layout,
        0,
        &[data.descriptor_set],
        &[],
    );
    let screen_size: [f32; 2] = [data.swapchain_extent.width as f32, data.swapchain_extent.height as f32];
    let push_bytes: &[u8] = std::slice::from_raw_parts(screen_size.as_ptr() as *const u8, std::mem::size_of::<[f32; 2]>());
    device.cmd_push_constants(cmd, data.sky_pipeline_layout, vk::ShaderStageFlags::FRAGMENT, 0, push_bytes);
    device.cmd_draw(cmd, 3, 1, 0, 0);
}

pub unsafe fn begin_render_pass(device: &Device, cmd: vk::CommandBuffer, data: &VulkanApplicationData, framebuffer_index: usize) {
    let clear_values = &[vk::ClearValue::color_f32([0.0, 0.0, 0.0, 1.0]), vk::ClearValue::depth_stencil(1.0, 0)];
    let info = vk::RenderPassBeginInfo::builder()
        .render_pass(data.render_pass)
        .framebuffer(data.framebuffers[framebuffer_index])
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: data.swapchain_extent,
        })
        .clear_values(clear_values);
    device.cmd_begin_render_pass(cmd, &info, vk::SubpassContents::INLINE);
}

unsafe fn begin_render_pass_no_clear(device: &Device, cmd: vk::CommandBuffer, data: &VulkanApplicationData, framebuffer_index: usize) {
    let info = vk::RenderPassBeginInfo::builder()
        .render_pass(data.render_pass_load)
        .framebuffer(data.framebuffers[framebuffer_index])
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent: data.swapchain_extent,
        });
    device.cmd_begin_render_pass(cmd, &info, vk::SubpassContents::INLINE);
}

/// Generates the depth pyramid from the depth buffer after the render pass.
/// Transitions the depth buffer to shader-read, then dispatches one reduction per mip level.
unsafe fn record_depth_pyramid_generation(device: &Device, cmd: vk::CommandBuffer, data: &VulkanApplicationData, pyramid: &DepthPyramidResources) {
    let mip_count = data.depth_pyramid_mip_count;
    let extent = data.swapchain_extent;

    // Transition depth buffer: DEPTH_ATTACHMENT → SHADER_READ_ONLY
    let depth_barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .new_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        .src_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .image(data.depth_image)
        .subresource_range(super::subresource_range(vk::ImageAspectFlags::DEPTH, 1));
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[*depth_barrier],
    );

    // Pyramid is already in GENERAL layout; ensure prior reads complete before writes
    let pyramid_barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags::SHADER_READ)
        .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
        .image(data.depth_pyramid_image)
        .subresource_range(super::subresource_range(vk::ImageAspectFlags::COLOR, mip_count));
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[*pyramid_barrier],
    );

    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, pyramid.pipeline);

    for mip in 0..mip_count {
        let dst_width = (extent.width >> mip).max(1);
        let dst_height = (extent.height >> mip).max(1);

        device.cmd_bind_descriptor_sets(
            cmd,
            vk::PipelineBindPoint::COMPUTE,
            pyramid.pipeline_layout,
            0,
            &[pyramid.descriptor_sets[mip as usize]],
            &[],
        );

        let push = DepthReducePush {
            dst_size: [dst_width, dst_height],
            is_copy: if mip == 0 { 1 } else { 0 },
            _pad: 0,
        };
        let push_bytes: &[u8] = std::slice::from_raw_parts(&push as *const DepthReducePush as *const u8, std::mem::size_of::<DepthReducePush>());
        device.cmd_push_constants(cmd, pyramid.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes);

        let wg_x = dst_width.div_ceil(16);
        let wg_y = dst_height.div_ceil(16);
        device.cmd_dispatch(cmd, wg_x, wg_y, 1);

        // Barrier between mip passes: previous write must complete before next read
        if mip + 1 < mip_count {
            let mip_barrier = vk::ImageMemoryBarrier::builder()
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ)
                .image(data.depth_pyramid_image)
                .subresource_range(super::subresource_range_mip(vk::ImageAspectFlags::COLOR, mip, 1));
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier],
                &[] as &[vk::BufferMemoryBarrier],
                &[*mip_barrier],
            );
        }
    }

    // Transition depth buffer back to DEPTH_ATTACHMENT for next frame
    let depth_restore = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
        .src_access_mask(vk::AccessFlags::SHADER_READ)
        .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
        .image(data.depth_image)
        .subresource_range(super::subresource_range(vk::ImageAspectFlags::DEPTH, 1));
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[*depth_restore],
    );
}

/// Records the two-phase mesh shader rendering pipeline:
/// 1. Phase 1: previously visible chunks (frustum cull only)
/// 2. Build depth pyramid
/// 3. Phase 2: previously invisible chunks (frustum + Hi-Z occlusion)
pub unsafe fn record_mesh_shader_command_buffer(
    device: &Device,
    data: &VulkanApplicationData,
    image_index: usize,
    mesh_pipeline: &crate::graphical_core::mesh_pipeline::MeshShaderPipeline,
    depth_pyramid: &DepthPyramidResources,
    cull_push: &CullPushConstants,
    pyramid_needs_init: bool,
    svdag: Option<(&SvdagPipeline, &CullPush, &TileAssignPush, &RaymarchPush)>,
) -> anyhow::Result<()> {
    let cmd = data.command_buffers[image_index];
    device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
    device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder())?;

    if pyramid_needs_init {
        transition_pyramid_undefined_to_general(device, cmd, data);
    }

    // Workaround: vulkan-rust missing TASK|MESH ShaderStageFlags bits
    let task_mesh_flags = vk::ShaderStageFlags::from_raw(0x40 | 0x80);

    // === Phase 1: previously visible chunks (no occlusion test) ===
    begin_render_pass(device, cmd, data, image_index);
    draw_sky(device, cmd, data);

    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, mesh_pipeline.pipeline);
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        mesh_pipeline.pipeline_layout,
        0,
        &[mesh_pipeline.descriptor_set],
        &[],
    );

    let mut push1 = *cull_push;
    push1.phase = 1;
    let push1_bytes: &[u8] = std::slice::from_raw_parts(&push1 as *const CullPushConstants as *const u8, std::mem::size_of::<CullPushConstants>());
    device.cmd_push_constants(cmd, mesh_pipeline.pipeline_layout, task_mesh_flags, 0, push1_bytes);
    device.cmd_draw_mesh_tasks_ext(cmd, push1.chunk_count, 1, 1);

    device.cmd_end_render_pass(cmd);

    // === Build depth pyramid from phase 1 depth ===
    record_depth_pyramid_generation(device, cmd, data, depth_pyramid);

    // === Phase 2: previously invisible chunks (with occlusion test) ===
    begin_render_pass_no_clear(device, cmd, data, image_index);

    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, mesh_pipeline.pipeline);
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        mesh_pipeline.pipeline_layout,
        0,
        &[mesh_pipeline.descriptor_set],
        &[],
    );

    let mut push2 = *cull_push;
    push2.phase = 2;
    let push2_bytes: &[u8] = std::slice::from_raw_parts(&push2 as *const CullPushConstants as *const u8, std::mem::size_of::<CullPushConstants>());
    device.cmd_push_constants(cmd, mesh_pipeline.pipeline_layout, task_mesh_flags, 0, push2_bytes);
    device.cmd_draw_mesh_tasks_ext(cmd, push2.chunk_count, 1, 1);

    device.cmd_end_render_pass(cmd);

    // === SVDAG 3-pass compute pipeline + composite ===
    if let Some((sp, cull_pc, tile_pc, march_pc)) = svdag {
        if cull_pc.total_chunks > 0 {
            record_svdag_passes(device, cmd, data, image_index, sp, cull_pc, tile_pc, march_pc);
        }
    }

    device.end_command_buffer(cmd)?;
    Ok(())
}

unsafe fn push_bytes<T>(val: &T) -> &[u8] {
    std::slice::from_raw_parts(val as *const T as *const u8, std::mem::size_of::<T>())
}

/// Records the Aokana-style 3-pass SVDAG compute pipeline + composite fragment pass.
unsafe fn record_svdag_passes(
    device: &Device,
    cmd: vk::CommandBuffer,
    data: &VulkanApplicationData,
    image_index: usize,
    sp: &SvdagPipeline,
    cull_pc: &CullPush,
    tile_pc: &TileAssignPush,
    march_pc: &RaymarchPush,
) {
    let empty_mem: &[vk::MemoryBarrier] = &[];
    let _empty_buf: &[vk::BufferMemoryBarrier] = &[];
    let color_range = super::subresource_range(vk::ImageAspectFlags::COLOR, 1);

    // Reset visible count to 0 on the GPU timeline (no CPU/GPU race)
    device.cmd_fill_buffer(cmd, sp.visible_count_buffer, 0, 4, 0);
    let count_reset = *vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)
        .buffer(sp.visible_count_buffer)
        .size(vk::WHOLE_SIZE);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        empty_mem,
        &[count_reset],
        &[] as &[vk::ImageMemoryBarrier],
    );

    // --- Pass 1: Frustum cull ---
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, sp.cull_pipeline);
    device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, sp.cull_layout, 0, &[sp.cull_desc_set], &[]);
    device.cmd_push_constants(cmd, sp.cull_layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes(cull_pc));
    device.cmd_dispatch(cmd, cull_pc.total_chunks.div_ceil(64), 1, 1);

    // Barrier: cull writes → tile reads
    let buf_barrier = *vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .buffer(sp.visible_buffer)
        .size(vk::WHOLE_SIZE);
    let count_barrier = *vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .buffer(sp.visible_count_buffer)
        .size(vk::WHOLE_SIZE);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        empty_mem,
        &[buf_barrier, count_barrier],
        &[] as &[vk::ImageMemoryBarrier],
    );

    // --- Pass 2: Tile assignment ---
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, sp.tile_pipeline);
    device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, sp.tile_layout, 0, &[sp.tile_desc_set], &[]);
    device.cmd_push_constants(cmd, sp.tile_layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes(tile_pc));
    device.cmd_dispatch(cmd, sp.tile_count[0].div_ceil(8), sp.tile_count[1].div_ceil(8), 1);

    // Barrier 1: tile writes → ray march reads (buffer only, COMPUTE → COMPUTE)
    let tile_barrier = *vk::BufferMemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .buffer(sp.tile_buffer)
        .size(vk::WHOLE_SIZE);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        empty_mem,
        &[tile_barrier],
        &[] as &[vk::ImageMemoryBarrier],
    );

    // Barrier 2: transition output images for clear (COMPUTE → TRANSFER)
    let color_to_general = *vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .image(sp.color_image)
        .subresource_range(color_range);
    let depth_to_general = *vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .image(sp.depth_image)
        .subresource_range(color_range);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::TRANSFER,
        vk::DependencyFlags::empty(),
        empty_mem,
        &[] as &[vk::BufferMemoryBarrier],
        &[color_to_general, depth_to_general],
    );

    // Clear output images to transparent / far depth
    let clear_color = vk::ClearColorValue::from_float32([0.0, 0.0, 0.0, 0.0]);
    let clear_depth = vk::ClearColorValue::from_float32([1.0, 0.0, 0.0, 0.0]);
    device.cmd_clear_color_image(cmd, sp.color_image, vk::ImageLayout::GENERAL, &clear_color, &[color_range]);
    device.cmd_clear_color_image(cmd, sp.depth_image, vk::ImageLayout::GENERAL, &clear_depth, &[color_range]);

    // Barrier 3: clear → compute write (TRANSFER → COMPUTE)
    let clear_to_compute = *vk::MemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[clear_to_compute],
        &[] as &[vk::BufferMemoryBarrier],
        &[] as &[vk::ImageMemoryBarrier],
    );

    // --- Pass 3: Ray march ---
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, sp.march_pipeline);
    device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::COMPUTE, sp.march_layout, 0, &[sp.march_desc_set], &[]);
    device.cmd_push_constants(cmd, sp.march_layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes(march_pc));
    let extent = data.swapchain_extent;
    device.cmd_dispatch(cmd, extent.width.div_ceil(8), extent.height.div_ceil(8), 1);

    // Transition output images: GENERAL → SHADER_READ_ONLY for composite sampling
    let color_to_read = *vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .image(sp.color_image)
        .subresource_range(color_range);
    let depth_to_read = *vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .image(sp.depth_image)
        .subresource_range(color_range);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        empty_mem,
        &[] as &[vk::BufferMemoryBarrier],
        &[color_to_read, depth_to_read],
    );

    // --- Composite: fullscreen triangle sampling compute output, depth-tested + alpha-blended ---
    begin_render_pass_no_clear(device, cmd, data, image_index);
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, sp.composite_pipeline);
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        sp.composite_layout,
        0,
        &[sp.composite_desc_set],
        &[],
    );
    device.cmd_draw(cmd, 3, 1, 0, 0);
    device.cmd_end_render_pass(cmd);
}

/// Creates semaphores and fences for each frame in flight.
pub unsafe fn create_sync_objects(device: &Device, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        data.image_available_semaphores.push(device.create_semaphore(&semaphore_info, None)?);
        data.render_finished_semaphores.push(device.create_semaphore(&semaphore_info, None)?);
        data.in_flight_fences.push(device.create_fence(&fence_info, None)?);
    }
    data.images_in_flight = data.swapchain_images.iter().map(|_| vk::Fence::null()).collect();
    Ok(())
}
