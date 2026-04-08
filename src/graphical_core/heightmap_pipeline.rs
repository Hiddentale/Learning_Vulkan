use crate::graphical_core::camera::UniformBufferObject;
use crate::graphical_core::shaders::create_shader_module;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::voxel::heightmap_generator::HeightmapVertex;
use crate::voxel::material::MaterialPalette;
use vk::Handle;
use vulkan_rust::{vk, Device};

/// Push constants for the heightmap vertex shader.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct HeightmapPush {
    pub camera_pos: [f32; 3],
    pub morph_factor: f32,
}

/// All GPU resources for the heightmap rendering path.
pub struct HeightmapPipeline {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl HeightmapPipeline {
    pub unsafe fn create(device: &Device, data: &VulkanApplicationData) -> anyhow::Result<Self> {
        let descriptor_set_layout = create_descriptor_layout(device)?;
        let descriptor_pool = create_descriptor_pool(device)?;
        let descriptor_set = allocate_descriptor_set(device, descriptor_pool, descriptor_set_layout)?;
        write_descriptors(device, descriptor_set, data);

        let push_range = *vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(std::mem::size_of::<HeightmapPush>() as u32);

        let set_layouts = [descriptor_set_layout];
        let push_ranges = [push_range];
        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);

        let pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;
        let pipeline = create_graphics_pipeline(device, data, pipeline_layout)?;

        Ok(Self {
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            pipeline_layout,
            pipeline,
        })
    }

    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_descriptor_pool(self.descriptor_pool, None);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
    }
}

unsafe fn create_descriptor_layout(device: &Device) -> anyhow::Result<vk::DescriptorSetLayout> {
    let bindings = [
        // Binding 0: CameraUBO (VERTEX | FRAGMENT)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT),
        // Binding 1: MaterialPalette (FRAGMENT)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // Binding 2: Texture array (FRAGMENT)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];

    let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    Ok(device.create_descriptor_set_layout(&create_info, None)?)
}

unsafe fn create_descriptor_pool(device: &Device) -> anyhow::Result<vk::DescriptorPool> {
    let pool_sizes = [
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(2) // CameraUBO + MaterialPalette
            .r#type(vk::DescriptorType::UNIFORM_BUFFER),
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(1) // texture array
            .r#type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
    ];
    let pool_info = vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&pool_sizes);
    Ok(device.create_descriptor_pool(&pool_info, None)?)
}

unsafe fn allocate_descriptor_set(device: &Device, pool: vk::DescriptorPool, layout: vk::DescriptorSetLayout) -> anyhow::Result<vk::DescriptorSet> {
    let layouts = [layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(pool).set_layouts(&layouts);
    let sets = device.allocate_descriptor_sets(&alloc_info)?;
    Ok(sets[0])
}

unsafe fn write_descriptors(device: &Device, set: vk::DescriptorSet, data: &VulkanApplicationData) {
    // Binding 0: CameraUBO
    let ubo_info = [*vk::DescriptorBufferInfo::builder()
        .buffer(data.uniform_buffer)
        .range(std::mem::size_of::<UniformBufferObject>() as u64)];

    // Binding 1: MaterialPalette
    let palette_info = [*vk::DescriptorBufferInfo::builder()
        .buffer(data.palette_buffer)
        .range(std::mem::size_of::<MaterialPalette>() as u64)];

    // Binding 2: Texture array
    let tex_info = [*vk::DescriptorImageInfo::builder()
        .image_view(data.texture_image_view)
        .sampler(data.texture_sampler)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

    let writes = [
        *vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&ubo_info),
        *vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&palette_info),
        *vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&tex_info),
    ];

    device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
}

unsafe fn create_graphics_pipeline(
    device: &Device,
    data: &VulkanApplicationData,
    pipeline_layout: vk::PipelineLayout,
) -> anyhow::Result<vk::Pipeline> {
    let vert_module = create_shader_module(device, include_bytes!("../shaders/heightmap.vert.spv"))?;
    let frag_module = create_shader_module(device, include_bytes!("../shaders/heightmap.frag.spv"))?;

    let vert_stage = *vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_module)
        .name(c"main");
    let frag_stage = *vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_module)
        .name(c"main");

    // Vertex input: 4 attributes matching HeightmapVertex layout
    let binding = [*vk::VertexInputBindingDescription::builder()
        .binding(0)
        .stride(std::mem::size_of::<HeightmapVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX)];

    let attributes = [
        // location 0: position (vec3, offset 0)
        *vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(0)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0),
        // location 1: normal (vec3, offset 12)
        *vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(12),
        // location 2: material_id (uint, offset 24)
        *vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32_UINT)
            .offset(24),
        // location 3: morph_delta_y (float, offset 28)
        *vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(3)
            .format(vk::Format::R32_SFLOAT)
            .offset(28),
    ];

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&binding)
        .vertex_attribute_descriptions(&attributes);

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder().topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport = *vk::Viewport::builder()
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);
    let scissor = *vk::Rect2D::builder().extent(data.swapchain_extent);

    let viewports = [viewport];
    let scissors = [scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder().viewports(&viewports).scissors(&scissors);

    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::BACK)
        // Heightmap tile vertices live in world cartesian (no cube-to-sphere
        // projection). Triangles `(tl, tr, bl)` in `generate_tile_mesh` have
        // a 3D world-space normal pointing OUTWARD from the planet (right-hand
        // rule on the index order, since face_basis is right-handed). For an
        // outside viewer that normal points toward the camera, which is
        // counter-clockwise in screen space — so CCW must be the front face,
        // unlike the mesh-chunk pipeline which uses CW because the cube
        // projection inverts winding.
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder().rasterization_samples(vk::SampleCountFlags::_1);

    let blend_attachment = *vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(false);

    let blend_attachments = [blend_attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder().attachments(&blend_attachments);

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL);

    let stages = [vert_stage, frag_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .layout(pipeline_layout)
        .render_pass(data.render_pass_load)
        .subpass(0);

    let pipeline = device.create_graphics_pipeline(vk::PipelineCache::null(), &info, None)?;

    device.destroy_shader_module(vert_module, None);
    device.destroy_shader_module(frag_module, None);

    Ok(pipeline)
}
