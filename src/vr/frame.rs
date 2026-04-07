use super::session::{view_to_vp, VrSession};
use super::swapchain::VrSwapchain;
use crate::graphical_core::camera::EyeMatrices;
use crate::graphical_core::mesh_pipeline::MeshShaderPipeline;
use crate::graphical_core::voxel_pool::VoxelPool;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use anyhow::Context;
use openxr as xr;
use vulkan_rust::{vk, Device};

/// Run one VR frame: wait for timing, locate eyes, render to the headset,
/// and submit the composition layer.
///
/// Returns the `EyeMatrices` so the desktop spectator view can reuse them.
///
/// # Safety
/// Calls unsafe Vulkan command recording and queue submission APIs.
pub unsafe fn render_vr_frame(
    device: &Device,
    data: &VulkanApplicationData,
    session: &mut VrSession,
    swapchain: &mut VrSwapchain,
    mesh_pipeline: &MeshShaderPipeline,
    voxel_pool: &VoxelPool,
) -> anyhow::Result<Option<EyeMatrices>> {
    if !session.is_running() {
        return Ok(None);
    }

    let frame_state = session.frame_waiter.wait()?;
    session.frame_stream.begin()?;

    if !frame_state.should_render {
        session
            .frame_stream
            .end(frame_state.predicted_display_time, xr::EnvironmentBlendMode::OPAQUE, &[])?;
        return Ok(None);
    }

    let (_, views) = session
        .session
        .locate_views(
            xr::ViewConfigurationType::PRIMARY_STEREO,
            frame_state.predicted_display_time,
            &session.stage_space,
        )
        .context("failed to locate views")?;

    let eyes = EyeMatrices::from_stereo(view_to_vp(&views[0]), view_to_vp(&views[1]));

    let image_index = swapchain.handle.acquire_image()?;
    swapchain.handle.wait_image(xr::Duration::INFINITE)?;

    record_vr_frame(device, data, swapchain, image_index as usize, mesh_pipeline, voxel_pool)?;
    submit_and_wait(device, data.graphics_queue, swapchain.command_buffer)?;

    swapchain.handle.release_image()?;

    submit_composition_layer(session, swapchain, &views, frame_state.predicted_display_time)?;

    Ok(Some(eyes))
}

/// Record a full render pass: sky + voxel geometry using mesh shaders.
unsafe fn record_vr_frame(
    device: &Device,
    data: &VulkanApplicationData,
    swapchain: &VrSwapchain,
    image_index: usize,
    mesh_pipeline: &MeshShaderPipeline,
    voxel_pool: &VoxelPool,
) -> anyhow::Result<()> {
    let cmd = swapchain.command_buffer;
    let extent = vk::Extent2D {
        width: swapchain.config.width,
        height: swapchain.config.height,
    };

    device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
    device.begin_command_buffer(
        cmd,
        &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
    )?;

    // Begin render pass (clears color + depth)
    let clear_values = &[vk::ClearValue::color_f32([0.0, 0.0, 0.0, 1.0]), vk::ClearValue::depth_stencil(1.0, 0)];
    let rp_info = vk::RenderPassBeginInfo::builder()
        .render_pass(swapchain.render_pass)
        .framebuffer(swapchain.framebuffers[image_index])
        .render_area(vk::Rect2D {
            offset: vk::Offset2D::default(),
            extent,
        })
        .clear_values(clear_values);
    device.cmd_begin_render_pass(cmd, &rp_info, vk::SubpassContents::INLINE);

    // Sky
    draw_sky(device, cmd, data, extent);

    // Voxel geometry via mesh shaders (simple single-phase dispatch for VR)
    let chunk_count = voxel_pool.chunk_count();
    if chunk_count > 0 {
        draw_voxels_mesh_shader(device, cmd, mesh_pipeline, chunk_count);
    }

    device.cmd_end_render_pass(cmd);
    device.end_command_buffer(cmd)?;
    Ok(())
}

unsafe fn draw_sky(device: &Device, cmd: vk::CommandBuffer, data: &VulkanApplicationData, extent: vk::Extent2D) {
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, data.sky_pipeline);
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        data.sky_pipeline_layout,
        0,
        &[data.descriptor_set],
        &[],
    );
    let screen_size: [f32; 2] = [extent.width as f32, extent.height as f32];
    let push_bytes: &[u8] = std::slice::from_raw_parts(screen_size.as_ptr() as *const u8, std::mem::size_of::<[f32; 2]>());
    device.cmd_push_constants(cmd, data.sky_pipeline_layout, vk::ShaderStageFlags::FRAGMENT, 0, push_bytes);
    device.cmd_draw(cmd, 3, 1, 0, 0);
}

unsafe fn draw_voxels_mesh_shader(device: &Device, cmd: vk::CommandBuffer, mesh_pipeline: &MeshShaderPipeline, chunk_count: u32) {
    let task_mesh_flags = vk::ShaderStageFlags::from_raw(0x40 | 0x80);

    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, mesh_pipeline.pipeline);
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::GRAPHICS,
        mesh_pipeline.pipeline_layout,
        0,
        &[mesh_pipeline.descriptor_set],
        &[],
    );

    // Single-phase dispatch — no culling push constants needed for VR (all chunks visible)
    let push = crate::graphical_core::compute_cull::CullPushConstants {
        planes: [[0.0; 4]; 6],
        camera_pos: [0.0; 3],
        chunk_count,
        screen_size: [0.0; 2],
        phase: 1,
        draw_offset: crate::voxel::block::BlockType::opaque_mask(),
        planet_radius: crate::voxel::sphere::PLANET_RADIUS_BLOCKS as f32,
        stereo: 1,
        _pad: [0.0; 2],
    };
    let push_bytes: &[u8] = std::slice::from_raw_parts(
        &push as *const crate::graphical_core::compute_cull::CullPushConstants as *const u8,
        std::mem::size_of::<crate::graphical_core::compute_cull::CullPushConstants>(),
    );
    device.cmd_push_constants(cmd, mesh_pipeline.pipeline_layout, task_mesh_flags, 0, push_bytes);
    device.cmd_draw_mesh_tasks_ext(cmd, chunk_count, 1, 1);
}

fn submit_composition_layer(session: &mut VrSession, swapchain: &VrSwapchain, views: &[xr::View], display_time: xr::Time) -> anyhow::Result<()> {
    let extent = xr::Extent2Di {
        width: swapchain.config.width as i32,
        height: swapchain.config.height as i32,
    };
    let rect = xr::Rect2Di {
        offset: xr::Offset2Di { x: 0, y: 0 },
        extent,
    };

    let projection_views = [
        xr::CompositionLayerProjectionView::new().pose(views[0].pose).fov(views[0].fov).sub_image(
            xr::SwapchainSubImage::new()
                .swapchain(&swapchain.handle)
                .image_array_index(0)
                .image_rect(rect),
        ),
        xr::CompositionLayerProjectionView::new().pose(views[1].pose).fov(views[1].fov).sub_image(
            xr::SwapchainSubImage::new()
                .swapchain(&swapchain.handle)
                .image_array_index(1)
                .image_rect(rect),
        ),
    ];

    let projection_layer = xr::CompositionLayerProjection::new().space(&session.stage_space).views(&projection_views);

    session
        .frame_stream
        .end(display_time, xr::EnvironmentBlendMode::OPAQUE, &[&projection_layer])?;

    Ok(())
}

unsafe fn submit_and_wait(device: &Device, queue: vk::Queue, cmd: vk::CommandBuffer) -> anyhow::Result<()> {
    let command_buffers = &[cmd];
    let submit_info = vk::SubmitInfo::builder().command_buffers(command_buffers);
    let fence = device.create_fence(&vk::FenceCreateInfo::builder(), None)?;
    device.queue_submit(queue, &[*submit_info], fence)?;
    device.wait_for_fences(&[fence], true, u64::MAX)?;
    device.destroy_fence(fence, None);
    Ok(())
}
