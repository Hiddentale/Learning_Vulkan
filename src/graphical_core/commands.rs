use crate::graphical_core::compute_cull::{ComputeCullResources, CullPushConstants, DepthPyramidResources, DepthReducePush};
use crate::graphical_core::vulkan_object::{DrawIndexedIndirectCommand, VulkanApplicationData, MAX_INDIRECT_DRAWS};
use crate::graphical_core::{self, MAX_FRAMES_IN_FLIGHT};
#[cfg(not(target_os = "macos"))]
use vulkanalia::vk::DeviceV1_2;
use vulkanalia::vk::{DeviceV1_0, Handle, HasBuilder};
use vulkanalia::{vk, Device, Instance};

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

const PHASE2_DRAW_OFFSET: u32 = (crate::graphical_core::vulkan_object::MAX_INDIRECT_DRAWS / 2) as u32;

/// Records the two-phase GPU-driven rendering pipeline:
/// 1. Cull phase 1 (previously visible chunks, frustum+face only)
/// 2. Draw phase 1
/// 3. Build depth pyramid from phase 1 depth
/// 4. Cull phase 2 (previously invisible chunks, frustum+face+occlusion)
/// 5. Draw phase 2
pub unsafe fn record_command_buffer(
    device: &Device,
    data: &VulkanApplicationData,
    image_index: usize,
    compute: &ComputeCullResources,
    depth_pyramid: &DepthPyramidResources,
    cull_push: &CullPushConstants,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
    pyramid_needs_init: bool,
) -> anyhow::Result<()> {
    let cmd = data.command_buffers[image_index];
    device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
    device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder())?;

    if pyramid_needs_init {
        transition_pyramid_undefined_to_general(device, cmd, data);
    }

    // Reset draw counts (and on macOS, the indirect buffer) to 0
    device.cmd_fill_buffer(cmd, compute.draw_count_buffer, 0, 8, 0);
    if cfg!(target_os = "macos") {
        let indirect_size = (MAX_INDIRECT_DRAWS * std::mem::size_of::<DrawIndexedIndirectCommand>()) as u64;
        device.cmd_fill_buffer(cmd, data.indirect_buffer, 0, indirect_size, 0);
    }
    let fill_barrier = vk::MemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[fill_barrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[] as &[vk::ImageMemoryBarrier],
    );

    // Bind cull compute pipeline for phase 1
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, compute.pipeline);
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        compute.pipeline_layout,
        0,
        &[compute.descriptor_set],
        &[],
    );

    let stride = std::mem::size_of::<DrawIndexedIndirectCommand>() as u32;
    let phase2_byte_offset = PHASE2_DRAW_OFFSET as u64 * stride as u64;

    // === Phase 1: previously visible chunks (no occlusion test) ===
    let mut push1 = *cull_push;
    push1.phase = 1;
    push1.draw_offset = 0;
    dispatch_cull(device, cmd, compute, &push1);

    // Barrier: compute → draw indirect
    compute_to_draw_barrier(device, cmd);

    // Draw phase 1 (begins render pass with clear)
    begin_render_pass(device, cmd, data, image_index);
    bind_graphics_state(device, cmd, data, vertex_buffer, index_buffer);
    draw_indirect(device, cmd, data.indirect_buffer, 0, compute.draw_count_buffer, 0, PHASE2_DRAW_OFFSET, stride);
    device.cmd_end_render_pass(cmd);

    // === Build depth pyramid from phase 1 depth ===
    record_depth_pyramid_generation(device, cmd, data, depth_pyramid);

    // === Phase 2: previously invisible chunks (with occlusion test) ===
    // Re-bind cull pipeline (pyramid generation bound a different compute pipeline)
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, compute.pipeline);
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        compute.pipeline_layout,
        0,
        &[compute.descriptor_set],
        &[],
    );

    let mut push2 = *cull_push;
    push2.phase = 2;
    push2.draw_offset = PHASE2_DRAW_OFFSET;
    dispatch_cull(device, cmd, compute, &push2);

    // Barrier: compute → draw indirect
    compute_to_draw_barrier(device, cmd);

    // Draw phase 2 (resume render pass WITHOUT clearing — append to existing depth+color)
    begin_render_pass_no_clear(device, cmd, data, image_index);
    bind_graphics_state(device, cmd, data, vertex_buffer, index_buffer);
    draw_indirect(device, cmd, data.indirect_buffer, phase2_byte_offset, compute.draw_count_buffer, 4, PHASE2_DRAW_OFFSET, stride);
    device.cmd_end_render_pass(cmd);

    device.end_command_buffer(cmd)?;
    Ok(())
}

unsafe fn transition_pyramid_undefined_to_general(device: &Device, cmd: vk::CommandBuffer, data: &VulkanApplicationData) {
    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::UNDEFINED)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .image(data.depth_pyramid_image)
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(data.depth_pyramid_mip_count)
                .base_array_layer(0)
                .layer_count(1),
        );
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::TOP_OF_PIPE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );
}

unsafe fn dispatch_cull(device: &Device, cmd: vk::CommandBuffer, compute: &ComputeCullResources, push: &CullPushConstants) {
    let push_bytes: &[u8] = std::slice::from_raw_parts(push as *const CullPushConstants as *const u8, std::mem::size_of::<CullPushConstants>());
    device.cmd_push_constants(cmd, compute.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes);
    device.cmd_dispatch(cmd, ComputeCullResources::workgroup_count(push.chunk_count), 1, 1);
}

unsafe fn compute_to_draw_barrier(device: &Device, cmd: vk::CommandBuffer) {
    let barrier = vk::MemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        &[barrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[] as &[vk::ImageMemoryBarrier],
    );
}

/// On platforms with `draw_indirect_count`, uses the GPU-written count buffer.
/// On macOS (MoltenVK), falls back to `cmd_draw_indexed_indirect` with a fixed max
/// draw count — the indirect buffer is zeroed each frame so unused slots are no-ops.
unsafe fn draw_indirect(
    device: &Device,
    cmd: vk::CommandBuffer,
    buffer: vk::Buffer,
    offset: u64,
    _count_buffer: vk::Buffer,
    _count_offset: u64,
    max_draws: u32,
    stride: u32,
) {
    #[cfg(target_os = "macos")]
    {
        let _ = (_count_buffer, _count_offset);
        device.cmd_draw_indexed_indirect(cmd, buffer, offset, max_draws, stride);
    }
    #[cfg(not(target_os = "macos"))]
    {
        device.cmd_draw_indexed_indirect_count(cmd, buffer, offset, _count_buffer, _count_offset, max_draws, stride);
    }
}

unsafe fn begin_render_pass(device: &Device, cmd: vk::CommandBuffer, data: &VulkanApplicationData, framebuffer_index: usize) {
    let clear_values = &[
        vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        },
        vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
        },
    ];
    let info = vk::RenderPassBeginInfo::builder()
        .render_pass(data.render_pass)
        .framebuffer(data.framebuffers[framebuffer_index])
        .render_area(vk::Rect2D::builder().offset(vk::Offset2D::default()).extent(data.swapchain_extent))
        .clear_values(clear_values);
    device.cmd_begin_render_pass(cmd, &info, vk::SubpassContents::INLINE);
}

unsafe fn begin_render_pass_no_clear(device: &Device, cmd: vk::CommandBuffer, data: &VulkanApplicationData, framebuffer_index: usize) {
    let info = vk::RenderPassBeginInfo::builder()
        .render_pass(data.render_pass_load)
        .framebuffer(data.framebuffers[framebuffer_index])
        .render_area(vk::Rect2D::builder().offset(vk::Offset2D::default()).extent(data.swapchain_extent));
    device.cmd_begin_render_pass(cmd, &info, vk::SubpassContents::INLINE);
}

unsafe fn bind_graphics_state(
    device: &Device,
    cmd: vk::CommandBuffer,
    data: &VulkanApplicationData,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
) {
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, data.pipeline);
    device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, data.pipeline_layout, 0, &[data.descriptor_set], &[]);
    device.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buffer], &[0]);
    device.cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32);
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
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[depth_barrier],
    );

    // Pyramid is already in GENERAL layout; ensure prior reads complete before writes
    let pyramid_barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::GENERAL)
        .new_layout(vk::ImageLayout::GENERAL)
        .src_access_mask(vk::AccessFlags::SHADER_READ)
        .dst_access_mask(vk::AccessFlags::SHADER_WRITE)
        .image(data.depth_pyramid_image)
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(mip_count)
                .base_array_layer(0)
                .layer_count(1),
        );
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[pyramid_barrier],
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
                .subresource_range(
                    vk::ImageSubresourceRange::builder()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .base_mip_level(mip)
                        .level_count(1)
                        .base_array_layer(0)
                        .layer_count(1),
                );
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[] as &[vk::MemoryBarrier],
                &[] as &[vk::BufferMemoryBarrier],
                &[mip_barrier],
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
        .subresource_range(
            vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[depth_restore],
    );
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
