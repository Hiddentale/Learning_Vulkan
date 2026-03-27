use crate::graphical_core::compute_cull::{ComputeCullResources, CullPushConstants};
use crate::graphical_core::vulkan_object::{DrawIndexedIndirectCommand, VulkanApplicationData};
use crate::graphical_core::{self, MAX_FRAMES_IN_FLIGHT};
use vulkanalia::vk::{DeviceV1_0, DeviceV1_2, Handle, HasBuilder};
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

/// Records compute culling dispatch + graphics draw into a command buffer.
pub unsafe fn record_command_buffer(
    device: &Device,
    data: &VulkanApplicationData,
    image_index: usize,
    compute: &ComputeCullResources,
    cull_push: &CullPushConstants,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
) -> anyhow::Result<()> {
    let cmd = data.command_buffers[image_index];
    device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;

    let begin_info = vk::CommandBufferBeginInfo::builder();
    device.begin_command_buffer(cmd, &begin_info)?;

    record_compute_cull(device, cmd, compute, cull_push);
    record_draw_commands(device, cmd, data, image_index, compute, vertex_buffer, index_buffer);

    device.end_command_buffer(cmd)?;
    Ok(())
}

/// Resets draw count, dispatches the cull compute shader, and inserts a barrier.
unsafe fn record_compute_cull(device: &Device, cmd: vk::CommandBuffer, compute: &ComputeCullResources, cull_push: &CullPushConstants) {
    // Reset draw count to 0
    device.cmd_fill_buffer(cmd, compute.draw_count_buffer, 0, 4, 0);

    // Barrier: transfer write → compute read/write
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

    // Dispatch compute
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, compute.pipeline);
    device.cmd_bind_descriptor_sets(
        cmd,
        vk::PipelineBindPoint::COMPUTE,
        compute.pipeline_layout,
        0,
        &[compute.descriptor_set],
        &[],
    );
    let push_bytes: &[u8] = std::slice::from_raw_parts(
        cull_push as *const CullPushConstants as *const u8,
        std::mem::size_of::<CullPushConstants>(),
    );
    device.cmd_push_constants(cmd, compute.pipeline_layout, vk::ShaderStageFlags::COMPUTE, 0, push_bytes);

    let workgroups = ComputeCullResources::workgroup_count(cull_push.chunk_count);
    device.cmd_dispatch(cmd, workgroups, 1, 1);

    // Barrier: compute shader writes → indirect draw reads
    let compute_barrier = vk::MemoryBarrier::builder()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::INDIRECT_COMMAND_READ);
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::DRAW_INDIRECT,
        vk::DependencyFlags::empty(),
        &[compute_barrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[] as &[vk::ImageMemoryBarrier],
    );
}

unsafe fn record_draw_commands(
    device: &Device,
    cmd: vk::CommandBuffer,
    data: &VulkanApplicationData,
    framebuffer_index: usize,
    compute: &ComputeCullResources,
    vertex_buffer: vk::Buffer,
    index_buffer: vk::Buffer,
) {
    let render_area = vk::Rect2D::builder().offset(vk::Offset2D::default()).extent(data.swapchain_extent);
    let color_clear_value = vk::ClearValue {
        color: vk::ClearColorValue {
            float32: [0.0, 0.0, 0.0, 1.0],
        },
    };
    let depth_clear_value = vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue { depth: 1.0, stencil: 0 },
    };
    let clear_values = &[color_clear_value, depth_clear_value];
    let info = vk::RenderPassBeginInfo::builder()
        .render_pass(data.render_pass)
        .framebuffer(data.framebuffers[framebuffer_index])
        .render_area(render_area)
        .clear_values(clear_values);

    device.cmd_begin_render_pass(cmd, &info, vk::SubpassContents::INLINE);
    device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, data.pipeline);
    device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, data.pipeline_layout, 0, &[data.descriptor_set], &[]);

    device.cmd_bind_vertex_buffers(cmd, 0, &[vertex_buffer], &[0]);
    device.cmd_bind_index_buffer(cmd, index_buffer, 0, vk::IndexType::UINT32);

    let stride = std::mem::size_of::<DrawIndexedIndirectCommand>() as u32;
    let max_draws = (crate::graphical_core::vulkan_object::MAX_INDIRECT_DRAWS) as u32;
    device.cmd_draw_indexed_indirect_count(cmd, data.indirect_buffer, 0, compute.draw_count_buffer, 0, max_draws, stride);

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
