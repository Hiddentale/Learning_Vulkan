use vulkanalia::{Device, vk};
use vulkanalia::vk::{DeviceV1_0, Handle, HasBuilder};
use crate::graphical_core::extra::create_shader_module;
use crate::graphical_core::vulkan_object::VulkanApplicationData;

pub unsafe fn create_pipeline(device: &Device, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let vertex_shader = include_bytes!("../shaders/vert.spv");
    let fragment_shader = include_bytes!("../shaders/frag.spv");

    let vertex_shader_module = create_shader_module(device, &vertex_shader[..])?;
    let fragment_shader_module = create_shader_module(device, &fragment_shader[..])?;

    let vertex_stage = vk::PipelineShaderStageCreateInfo::builder().stage(vk::ShaderStageFlags::VERTEX).module(vertex_shader_module).name(b"main\0");
    let fragment_stage = vk::PipelineShaderStageCreateInfo::builder().stage(vk::ShaderStageFlags::FRAGMENT).module(fragment_shader_module).name(b"main\0");
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder().topology(vk::PrimitiveTopology::TRIANGLE_LIST).primitive_restart_enable(false);
    let viewport = vk::Viewport::builder().x(0.0).y(0.0).width(data.swapchain_accepted_images_width_and_height.width as f32).height(data.swapchain_accepted_images_width_and_height.height as f32).min_depth(0.0).max_depth(1.0);
    let scissor = vk::Rect2D::builder().offset(vk::Offset2D { x: 0, y: 0 }).extent(data.swapchain_accepted_images_width_and_height);
    let viewports = &[viewport];
    let scissors = &[scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder().viewports(viewports).scissors(scissors);
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder().depth_clamp_enable(false).rasterizer_discard_enable(false).polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0).cull_mode(vk::CullModeFlags::BACK).front_face(vk::FrontFace::CLOCKWISE).depth_bias_enable(false);
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder().sample_shading_enable(false).rasterization_samples(vk::SampleCountFlags::_1);
    let attachment = vk::PipelineColorBlendAttachmentState::builder().color_write_mask(vk::ColorComponentFlags::all()).blend_enable(true).src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA).color_blend_op(vk::BlendOp::ADD).src_alpha_blend_factor(vk::BlendFactor::ONE).dst_alpha_blend_factor(vk::BlendFactor::ZERO).alpha_blend_op(vk::BlendOp::ADD);
    let attachments = &[attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder().logic_op_enable(false).logic_op(vk::LogicOp::COPY).attachments(attachments).blend_constants([0.0, 0.0, 0.0, 0.0]);
    let layout_info = vk::PipelineLayoutCreateInfo::builder();

    data.pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

    let stages = &[vertex_stage, fragment_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder().stages(stages).vertex_input_state(&vertex_input_state).input_assembly_state(&input_assembly_state).viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state).multisample_state(&multisample_state).color_blend_state(&color_blend_state).layout(data.pipeline_layout).render_pass(data.render_pass).subpass(0);

    data.pipeline = device.create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?.0[0];


    device.destroy_shader_module(vertex_shader_module, None);
    device.destroy_shader_module(fragment_shader_module, None);
    Ok(())
}