use crate::graphical_core::mesh::Vertex;
use crate::graphical_core::shaders::create_shader_module;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use vulkanalia::vk::{DeviceV1_0, Handle, HasBuilder};
use vulkanalia::{vk, Device};

/// Creates the graphics pipeline and its layout from the current render pass and descriptor layout.
///
/// # Safety
/// Calls unsafe Vulkan APIs. The caller must destroy the pipeline and layout before the
/// render pass or descriptor set layout they reference.
pub unsafe fn create_pipeline(vulkan_logical_device: &Device, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let (vertex_binding, vertex_attributes) = vertex_input_descriptions();
    let bindings = &[vertex_binding];
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(bindings)
        .vertex_attribute_descriptions(&vertex_attributes);

    let vertex_shader_module = create_shader_module(vulkan_logical_device, &include_bytes!("../shaders/shader.vert.spv")[..])?;
    let fragment_shader_module = create_shader_module(vulkan_logical_device, &include_bytes!("../shaders/shader.frag.spv")[..])?;

    let vertex_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vertex_shader_module)
        .name(b"main\0");
    let fragment_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(fragment_shader_module)
        .name(b"main\0");
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);
    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);
    let scissor = vk::Rect2D::builder().offset(vk::Offset2D { x: 0, y: 0 }).extent(data.swapchain_extent);
    let viewports = &[viewport];
    let scissors = &[scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder().viewports(viewports).scissors(scissors);
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false);
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(false)
        .rasterization_samples(vk::SampleCountFlags::_1);
    let blend_attachment = color_blend_attachment();
    let blend_attachments = &[blend_attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY)
        .attachments(blend_attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);
    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    let layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&[data.descriptor_set_layout]).build();

    data.pipeline_layout = vulkan_logical_device.create_pipeline_layout(&layout_info, None)?;

    let stages = &[vertex_stage, fragment_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .layout(data.pipeline_layout)
        .render_pass(data.render_pass)
        .subpass(0);

    data.pipeline = vulkan_logical_device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
        .0[0];

    vulkan_logical_device.destroy_shader_module(vertex_shader_module, None);
    vulkan_logical_device.destroy_shader_module(fragment_shader_module, None);
    Ok(())
}

fn vertex_input_descriptions() -> (vk::VertexInputBindingDescription, [vk::VertexInputAttributeDescription; 2]) {
    let binding = vk::VertexInputBindingDescription::builder()
        .binding(0)
        .stride(std::mem::size_of::<Vertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)
        .build();

    let position = vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(0)
        .format(vk::Format::R32G32B32_SFLOAT)
        .offset(0)
        .build();

    let uv_coordinate = vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(1)
        .format(vk::Format::R32G32_SFLOAT)
        .offset(std::mem::size_of::<[f32; 3]>() as u32)
        .build();

    (binding, [position, uv_coordinate])
}

fn color_blend_attachment() -> vk::PipelineColorBlendAttachmentState {
    vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)
        .build()
}
