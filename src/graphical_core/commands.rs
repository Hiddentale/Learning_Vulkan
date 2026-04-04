use crate::graphical_core::compute_cull::{CullPushConstants, DepthPyramidResources, DepthReducePush};
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
unsafe fn draw_sky(device: &Device, cmd: vk::CommandBuffer, data: &VulkanApplicationData) {
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
    device.end_command_buffer(cmd)?;
    Ok(())
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
