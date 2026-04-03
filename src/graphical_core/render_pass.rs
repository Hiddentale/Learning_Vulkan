use crate::graphical_core::depth::depth_format;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use vulkan_rust::{vk, Device, Instance};

/// Creates both render passes: one that clears (phase 1) and one that loads (phase 2).
pub unsafe fn create_render_pass(_instance: &Instance, device: &Device, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    data.render_pass = create_render_pass_with_load_op(
        device,
        data.swapchain_format,
        vk::AttachmentLoadOp::CLEAR,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::UNDEFINED,
    )?;

    data.render_pass_load = create_render_pass_with_load_op(
        device,
        data.swapchain_format,
        vk::AttachmentLoadOp::LOAD,
        vk::ImageLayout::PRESENT_SRC,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    )?;

    Ok(())
}

unsafe fn create_render_pass_with_load_op(
    device: &Device,
    swapchain_format: vk::Format,
    load_op: vk::AttachmentLoadOp,
    color_initial_layout: vk::ImageLayout,
    depth_initial_layout: vk::ImageLayout,
) -> anyhow::Result<vk::RenderPass> {
    let color_attachment = vk::AttachmentDescription::builder()
        .format(swapchain_format)
        .samples(vk::SampleCountFlags::_1)
        .load_op(load_op)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(color_initial_layout)
        .final_layout(vk::ImageLayout::PRESENT_SRC);

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

    let attachments = &[*color_attachment, *depth_attachment];
    let subpasses = &[*subpass];
    let dependencies = &[*dependency];
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    Ok(device.create_render_pass(&info, None)?)
}
