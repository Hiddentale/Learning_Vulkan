use super::session::{view_to_vp, VrSession};
use super::swapchain::VrSwapchain;
use crate::graphical_core::camera::EyeMatrices;
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
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    indirect_buffer: vk::Buffer,
    draw_count: u32,
) -> anyhow::Result<Option<EyeMatrices>> {
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

    record_vr_frame(
        device,
        data,
        swapchain,
        image_index as usize,
        vertex_buffer,
        index_buffer,
        indirect_buffer,
        draw_count,
    )?;
    submit_and_wait(device, data.graphics_queue, swapchain.command_buffer)?;

    swapchain.handle.release_image()?;

    submit_composition_layer(session, swapchain, &views, frame_state.predicted_display_time)?;

    Ok(Some(eyes))
}

/// Record a full render pass: sky + voxel geometry using the shared pipelines.
unsafe fn record_vr_frame(
    device: &Device,
    data: &VulkanApplicationData,
    swapchain: &VrSwapchain,
    image_index: usize,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    indirect_buffer: vk::Buffer,
    draw_count: u32,
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

    // Voxel geometry (simple indirect draw — no two-phase culling for VR yet)
    if draw_count > 0 {
        draw_voxels(device, cmd, data, vertex_buffer, index_buffer, indirect_buffer, draw_count);
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

unsafe fn draw_voxels(
    device: &Device,
    cmd: vk::CommandBuffer,
    data: &VulkanApplicationData,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    indirect_buffer: vk::Buffer,
    draw_count: u32,
) {
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, data.pipeline);
    device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, data.pipeline_layout, 0, &[data.descriptor_set], &[]);
    device.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buffer], &[0]);
    device.cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32);

    let stride = std::mem::size_of::<crate::graphical_core::vulkan_object::DrawIndexedIndirectCommand>() as u32;
    device.cmd_draw_indexed_indirect(cmd, indirect_buffer, 0, draw_count, stride);
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
