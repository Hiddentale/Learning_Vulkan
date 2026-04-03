use crate::graphical_core::mesh::Vertex;
use crate::graphical_core::shaders::create_shader_module;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use vk::Handle;
use vulkan_rust::{vk, Device};

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
        .name(c"main");
    let fragment_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(fragment_shader_module)
        .name(c"main");
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
    let viewports = &[*viewport];
    let scissors = &[*scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder().viewports(viewports).scissors(scissors);
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        .front_face(vk::FrontFace::CLOCKWISE)
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

    let set_layouts = [data.descriptor_set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);

    data.pipeline_layout = vulkan_logical_device.create_pipeline_layout(&layout_info, None)?;

    let stages = &[*vertex_stage, *fragment_stage];
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

    data.pipeline = vulkan_logical_device.create_graphics_pipeline(vk::PipelineCache::null(), &info, None)?;

    vulkan_logical_device.destroy_shader_module(vertex_shader_module, None);
    vulkan_logical_device.destroy_shader_module(fragment_shader_module, None);
    Ok(())
}

fn vertex_input_descriptions() -> (vk::VertexInputBindingDescription, [vk::VertexInputAttributeDescription; 4]) {
    let binding = *vk::VertexInputBindingDescription::builder()
        .binding(0)
        .stride(std::mem::size_of::<Vertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);

    let position = *vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(0)
        .format(vk::Format::R32G32B32_SFLOAT)
        .offset(std::mem::offset_of!(Vertex, position) as u32);

    let uv_coordinate = *vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(1)
        .format(vk::Format::R32G32_SFLOAT)
        .offset(std::mem::offset_of!(Vertex, uv_coordinate) as u32);

    let normal = *vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(2)
        .format(vk::Format::R32G32B32_SFLOAT)
        .offset(std::mem::offset_of!(Vertex, normal) as u32);

    let material_id = *vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(3)
        .format(vk::Format::R32_UINT)
        .offset(std::mem::offset_of!(Vertex, material_id) as u32);

    (binding, [position, uv_coordinate, normal, material_id])
}

/// Creates the sky pipeline: fullscreen triangle, no vertex input, no depth write.
pub unsafe fn create_sky_pipeline(device: &Device, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let vert_module = create_shader_module(device, &include_bytes!("../shaders/sky.vert.spv")[..])?;
    let frag_module = create_shader_module(device, &include_bytes!("../shaders/sky.frag.spv")[..])?;

    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_module)
        .name(c"main");
    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_module)
        .name(c"main");

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();
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
    let viewports = &[*viewport];
    let scissors = &[*scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder().viewports(viewports).scissors(scissors);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::CLOCKWISE)
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
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::ALWAYS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    let push_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(std::mem::size_of::<[f32; 2]>() as u32);
    let push_ranges = &[*push_range];
    let set_layouts = [data.descriptor_set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&set_layouts)
        .push_constant_ranges(push_ranges);

    data.sky_pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

    let stages = &[*vert_stage, *frag_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .layout(data.sky_pipeline_layout)
        .render_pass(data.render_pass)
        .subpass(0);

    data.sky_pipeline = device.create_graphics_pipeline(vk::PipelineCache::null(), &info, None)?;

    device.destroy_shader_module(vert_module, None);
    device.destroy_shader_module(frag_module, None);
    Ok(())
}

fn color_blend_attachment() -> vk::PipelineColorBlendAttachmentState {
    *vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD)
}
