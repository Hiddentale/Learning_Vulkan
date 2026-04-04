#![allow(dead_code)]

use crate::graphical_core::camera::UniformBufferObject;
use crate::graphical_core::shaders::create_shader_module;
use crate::graphical_core::svdag_pool::SvdagPool;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use vk::Handle;
use vulkan_rust::{vk, Device};

/// Push constants for the SVDAG ray march fragment shader.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SvdagPushConstants {
    pub camera_pos: [f32; 3],
    pub chunk_count: u32,
    pub screen_size: [f32; 2],
}

/// Graphics pipeline for SVDAG ray marching of far-field chunks.
/// Fullscreen triangle + fragment shader ray march, alpha-blended over mesh output.
pub struct SvdagPipeline {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl SvdagPipeline {
    pub unsafe fn create(device: &Device, data: &VulkanApplicationData, svdag_pool: &SvdagPool) -> anyhow::Result<Self> {
        let descriptor_set_layout = create_descriptor_layout(device)?;
        let descriptor_pool = create_descriptor_pool(device)?;
        let descriptor_set = allocate_descriptor_set(device, descriptor_pool, descriptor_set_layout)?;
        write_descriptors(device, descriptor_set, data, svdag_pool);

        let push_range = *vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(std::mem::size_of::<SvdagPushConstants>() as u32);

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
        // Binding 0: Camera UBO
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // Binding 1: SVDAG geometry SSBO (includes embedded materials in leaves)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // Binding 2: SVDAG chunk info SSBO
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];

    let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    Ok(device.create_descriptor_set_layout(&create_info, None)?)
}

unsafe fn create_descriptor_pool(device: &Device) -> anyhow::Result<vk::DescriptorPool> {
    let pool_sizes = [
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(1)
            .r#type(vk::DescriptorType::UNIFORM_BUFFER),
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(2)
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

unsafe fn write_descriptors(device: &Device, set: vk::DescriptorSet, data: &VulkanApplicationData, pool: &SvdagPool) {
    let ubo_info = [*vk::DescriptorBufferInfo::builder()
        .buffer(data.uniform_buffer)
        .range(std::mem::size_of::<UniformBufferObject>() as u64)];
    let geo_info = [*vk::DescriptorBufferInfo::builder().buffer(pool.geometry_buffer).range(vk::WHOLE_SIZE)];
    let chunk_info = [*vk::DescriptorBufferInfo::builder().buffer(pool.chunk_info_buffer).range(vk::WHOLE_SIZE)];

    let writes = [
        *vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(&ubo_info),
        *vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&geo_info),
        *vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&chunk_info),
    ];

    device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
}

unsafe fn create_graphics_pipeline(
    device: &Device,
    data: &VulkanApplicationData,
    pipeline_layout: vk::PipelineLayout,
) -> anyhow::Result<vk::Pipeline> {
    // Reuse sky.vert for fullscreen triangle
    let vert_module = create_shader_module(device, include_bytes!("../shaders/sky.vert.spv"))?;
    let frag_module = create_shader_module(device, include_bytes!("../shaders/svdag_raymarch.frag.spv"))?;

    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_module)
        .name(c"main");
    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_module)
        .name(c"main");

    let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder().topology(vk::PrimitiveTopology::TRIANGLE_LIST);

    let viewport = *vk::Viewport::builder()
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);
    let scissor = *vk::Rect2D::builder().extent(data.swapchain_extent);
    let viewports = [viewport];
    let scissors = [scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder().viewports(&viewports).scissors(&scissors);

    let rasterization = vk::PipelineRasterizationStateCreateInfo::builder()
        .polygon_mode(vk::PolygonMode::FILL)
        .line_width(1.0)
        .cull_mode(vk::CullModeFlags::NONE);

    let multisample = vk::PipelineMultisampleStateCreateInfo::builder().rasterization_samples(vk::SampleCountFlags::_1);

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
    let color_blend = vk::PipelineColorBlendStateCreateInfo::builder().attachments(&blend_attachments);

    // Depth test rejects SVDAG fragments behind mesh shader geometry.
    // Fragment shader writes gl_FragDepth from the ray march hit position.
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);

    let stages = [*vert_stage, *frag_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .layout(pipeline_layout)
        .render_pass(data.render_pass_load)
        .subpass(0);

    let pipeline = device.create_graphics_pipeline(vk::PipelineCache::null(), &info, None)?;

    device.destroy_shader_module(vert_module, None);
    device.destroy_shader_module(frag_module, None);

    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svdag_push_constants_size() {
        assert_eq!(std::mem::size_of::<SvdagPushConstants>(), 24);
    }
}
