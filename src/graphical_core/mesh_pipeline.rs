#![allow(dead_code)] // Wired up incrementally during mesh shader migration
use crate::graphical_core::camera::UniformBufferObject;
use crate::graphical_core::compute_cull::CullPushConstants;
use crate::graphical_core::shaders::create_shader_module;
use crate::graphical_core::voxel_pool::VoxelPool;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::voxel::material::MaterialPalette;
use vk::Handle;
use vulkan_rust::{vk, Device};

// Workaround: vulkan-rust 0.10.0 is missing TASK/MESH ShaderStageFlagBits.
// Replace with proper enum variants when vulkan-rust is fixed.
const TASK_STAGE: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(0x40);
const MESH_STAGE: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(0x80);

/// All GPU resources for the mesh shader rendering path.
pub struct MeshShaderPipeline {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl MeshShaderPipeline {
    pub unsafe fn create(device: &Device, data: &VulkanApplicationData, voxel_pool: &VoxelPool) -> anyhow::Result<Self> {
        let descriptor_set_layout = create_descriptor_layout(device)?;
        let descriptor_pool = create_descriptor_pool(device)?;
        let descriptor_set = allocate_descriptor_set(device, descriptor_pool, descriptor_set_layout)?;

        write_descriptors(device, descriptor_set, data, voxel_pool);

        let push_range = *vk::PushConstantRange::builder()
            .stage_flags(TASK_STAGE | MESH_STAGE)
            .offset(0)
            .size(std::mem::size_of::<CullPushConstants>() as u32);

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

    /// Re-write the depth pyramid descriptor after swapchain recreation.
    pub unsafe fn update_depth_pyramid(&self, device: &Device, data: &VulkanApplicationData) {
        let depth_info = [*vk::DescriptorImageInfo::builder()
            .image_view(data.depth_pyramid_full_view)
            .sampler(data.depth_pyramid_sampler)
            .image_layout(vk::ImageLayout::GENERAL)];
        let write = [*vk::WriteDescriptorSet::builder()
            .dst_set(self.descriptor_set)
            .dst_binding(7)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&depth_info)];
        device.update_descriptor_sets(&write, &[] as &[vk::CopyDescriptorSet]);
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
        // Binding 0: texture sampler (FRAGMENT)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // Binding 1: CameraUBO (TASK | MESH | FRAGMENT)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(TASK_STAGE | MESH_STAGE | vk::ShaderStageFlags::FRAGMENT),
        // Binding 2: MaterialPalette (FRAGMENT)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // Binding 3: VoxelDataSSBO (TASK | MESH, read-only)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(TASK_STAGE | MESH_STAGE),
        // Binding 4: BoundaryDataSSBO (MESH, read-only)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(4)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(MESH_STAGE),
        // Binding 5: ChunkInfoSSBO (TASK | MESH, read-only)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(5)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(TASK_STAGE | MESH_STAGE),
        // Binding 6: VisibilitySSBO (TASK, read/write)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(6)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(TASK_STAGE),
        // Binding 7: depth_pyramid sampler (TASK)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(7)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(TASK_STAGE),
    ];

    let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    Ok(device.create_descriptor_set_layout(&create_info, None)?)
}

unsafe fn create_descriptor_pool(device: &Device) -> anyhow::Result<vk::DescriptorPool> {
    let pool_sizes = [
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(2) // texture + depth_pyramid
            .r#type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(2) // CameraUBO + MaterialPalette
            .r#type(vk::DescriptorType::UNIFORM_BUFFER),
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(4) // voxel + boundary + chunk_info + visibility
            .r#type(vk::DescriptorType::STORAGE_BUFFER),
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

unsafe fn write_descriptors(device: &Device, set: vk::DescriptorSet, data: &VulkanApplicationData, pool: &VoxelPool) {
    // Binding 0: texture sampler
    let image_info = [*vk::DescriptorImageInfo::builder()
        .image_view(data.texture_image_view)
        .sampler(data.texture_sampler)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

    // Binding 1: CameraUBO
    let ubo_info = [*vk::DescriptorBufferInfo::builder()
        .buffer(data.uniform_buffer)
        .range(std::mem::size_of::<UniformBufferObject>() as u64)];

    // Binding 2: MaterialPalette
    let palette_info = [*vk::DescriptorBufferInfo::builder()
        .buffer(data.palette_buffer)
        .range(std::mem::size_of::<MaterialPalette>() as u64)];

    // Binding 3: VoxelDataSSBO
    let voxel_info = [*vk::DescriptorBufferInfo::builder().buffer(pool.voxel_buffer).range(vk::WHOLE_SIZE)];

    // Binding 4: BoundaryDataSSBO
    let boundary_info = [*vk::DescriptorBufferInfo::builder().buffer(pool.boundary_buffer).range(vk::WHOLE_SIZE)];

    // Binding 5: ChunkInfoSSBO
    let chunk_info = [*vk::DescriptorBufferInfo::builder().buffer(pool.chunk_info_buffer).range(vk::WHOLE_SIZE)];

    // Binding 6: VisibilitySSBO
    let vis_info = [*vk::DescriptorBufferInfo::builder().buffer(pool.visibility_buffer).range(vk::WHOLE_SIZE)];

    // Binding 7: depth_pyramid
    let depth_info = [*vk::DescriptorImageInfo::builder()
        .image_view(data.depth_pyramid_full_view)
        .sampler(data.depth_pyramid_sampler)
        .image_layout(vk::ImageLayout::GENERAL)];

    let writes = [
        *vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&image_info),
        *vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&ubo_info),
        *vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&palette_info),
        *vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&voxel_info),
        *vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&boundary_info),
        *vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(5)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&chunk_info),
        *vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(6)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&vis_info),
        *vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(7)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&depth_info),
    ];

    device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
}

unsafe fn create_graphics_pipeline(
    device: &Device,
    data: &VulkanApplicationData,
    pipeline_layout: vk::PipelineLayout,
) -> anyhow::Result<vk::Pipeline> {
    let task_module = create_shader_module(device, include_bytes!("../shaders/chunk_cull.task.spv"))?;
    let mesh_module = create_shader_module(device, include_bytes!("../shaders/voxel.mesh.spv"))?;
    let frag_module = create_shader_module(device, include_bytes!("../shaders/shader.frag.spv"))?;

    let task_stage = *vk::PipelineShaderStageCreateInfo::builder()
        .stage(TASK_STAGE)
        .module(task_module)
        .name(c"main");
    let mesh_stage = *vk::PipelineShaderStageCreateInfo::builder()
        .stage(MESH_STAGE)
        .module(mesh_module)
        .name(c"main");
    let frag_stage = *vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_module)
        .name(c"main");

    // Mesh shader pipelines have no vertex input or input assembly
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder();
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
        .front_face(vk::FrontFace::CLOCKWISE);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder().rasterization_samples(vk::SampleCountFlags::_1);

    let blend_attachment = *vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    let blend_attachments = [blend_attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder().attachments(&blend_attachments);

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);

    let stages = [task_stage, mesh_stage, frag_stage];
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
        .render_pass(data.render_pass)
        .subpass(0);

    let pipeline = device.create_graphics_pipeline(vk::PipelineCache::null(), &info, None)?;

    device.destroy_shader_module(task_module, None);
    device.destroy_shader_module(mesh_module, None);
    device.destroy_shader_module(frag_module, None);

    Ok(pipeline)
}
