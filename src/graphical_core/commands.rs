use crate::graphical_core::scene::SceneObject;
use crate::graphical_core::{vulkan_object::VulkanApplicationData, MAX_FRAMES_IN_FLIGHT};
use vulkanalia::vk::{DeviceV1_0, Handle, HasBuilder};
use vulkanalia::{vk, Device, Instance};

use crate::graphical_core;

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

/// Records draw commands into a single command buffer for the given framebuffer index.
///
/// Called once per frame. The command buffer must have been allocated and
/// must not be in use by the GPU (caller ensures this via fence waits).
pub unsafe fn record_command_buffer(device: &Device, data: &VulkanApplicationData, image_index: usize, scene: &[SceneObject]) -> anyhow::Result<()> {
    let cmd = data.command_buffers[image_index];
    device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;

    let begin_info = vk::CommandBufferBeginInfo::builder();
    device.begin_command_buffer(cmd, &begin_info)?;
    record_draw_commands(device, cmd, data, image_index, scene);
    device.end_command_buffer(cmd)?;
    Ok(())
}

unsafe fn record_draw_commands(
    device: &Device,
    cmd: vk::CommandBuffer,
    data: &VulkanApplicationData,
    framebuffer_index: usize,
    scene: &[SceneObject],
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

    for object in scene {
        let mesh = &data.meshes[object.mesh_index];

        device.cmd_bind_vertex_buffers(cmd, 0, &[mesh.vertex_buffer], &[0]);
        device.cmd_bind_index_buffer(cmd, mesh.index_buffer, 0, vk::IndexType::UINT32);

        let model_bytes = object.transform.model_matrix().to_cols_array();
        let byte_slice = std::slice::from_raw_parts(model_bytes.as_ptr() as *const u8, 64);
        device.cmd_push_constants(cmd, data.pipeline_layout, vk::ShaderStageFlags::VERTEX, 0, byte_slice);

        device.cmd_draw_indexed(cmd, mesh.index_count, 1, 0, 0, 0);
    }

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
