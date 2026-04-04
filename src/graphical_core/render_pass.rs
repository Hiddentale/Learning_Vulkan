use crate::graphical_core::depth::depth_format;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use vulkan_rust::{vk, Device, Instance};

/// Creates both render passes: one that clears (phase 1) and one that loads (phase 2).
pub unsafe fn create_render_pass(_instance: &Instance, device: &Device, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    data.render_pass = create_multiview_render_pass(
        device,
        data.swapchain_format,
        1,
        vk::AttachmentLoadOp::CLEAR,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::UNDEFINED,
    )?;

    data.render_pass_load = create_multiview_render_pass(
        device,
        data.swapchain_format,
        1,
        vk::AttachmentLoadOp::LOAD,
        vk::ImageLayout::PRESENT_SRC,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    )?;

    Ok(())
}

/// Creates a multiview-enabled render pass.
///
/// `view_count` controls how many views (array layers) are rendered
/// simultaneously: 1 for desktop, 2 for stereo VR.
pub unsafe fn create_multiview_render_pass(
    device: &Device,
    color_format: vk::Format,
    view_count: u32,
    load_op: vk::AttachmentLoadOp,
    color_initial_layout: vk::ImageLayout,
    depth_initial_layout: vk::ImageLayout,
) -> anyhow::Result<vk::RenderPass> {
    let final_color_layout = if view_count > 1 {
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    } else {
        vk::ImageLayout::PRESENT_SRC
    };

    let color_attachment = vk::AttachmentDescription::builder()
        .format(color_format)
        .samples(vk::SampleCountFlags::_1)
        .load_op(load_op)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(color_initial_layout)
        .final_layout(final_color_layout);

    let depth_attachment = vk::AttachmentDescription::builder()
        .format(depth_format())
        .samples(vk::SampleCountFlags::_1)
        .load_op(load_op)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(depth_initial_layout)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_attachment_ref = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_attachments = &[*color_attachment_ref];
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments)
        .depth_stencil_attachment(&depth_attachment_ref);

    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL)
        .dst_subpass(0)
        .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
        .src_access_mask(vk::AccessFlags::empty())
        .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS)
        .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE);

    let view_mask = (1u32 << view_count) - 1; // 0b1 for desktop, 0b11 for stereo
    let correlation_mask = view_mask;
    let view_masks = [view_mask];
    let correlation_masks = [correlation_mask];
    let mut multiview_info = vk::RenderPassMultiviewCreateInfo::builder()
        .view_masks(&view_masks)
        .correlation_masks(&correlation_masks);

    let attachments = &[*color_attachment, *depth_attachment];
    let subpasses = &[*subpass];
    let dependencies = &[*dependency];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies)
        .push_next(&mut *multiview_info);

    Ok(device.create_render_pass(&info, None)?)
}
