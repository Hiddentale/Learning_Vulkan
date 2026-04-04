#![allow(dead_code)] // Wired up when SVDAG chunks exist

use crate::graphical_core::camera::UniformBufferObject;
use crate::graphical_core::shaders::create_shader_module;
use crate::graphical_core::svdag_pool::SvdagPool;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use vk::Handle;
use vulkan_rust::{vk, Device};

/// Push constants for the SVDAG ray march compute shader.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct SvdagPushConstants {
    pub camera_pos: [f32; 3],
    pub chunk_count: u32,
    pub screen_size: [u32; 2],
}

/// Compute pipeline for SVDAG ray marching of far-field chunks.
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
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<SvdagPushConstants>() as u32);

        let set_layouts = [descriptor_set_layout];
        let push_ranges = [push_range];
        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);

        let pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

        let shader_module = create_shader_module(device, include_bytes!("../shaders/svdag_raymarch.comp.spv"))?;
        let stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(shader_module)
            .name(c"main");
        let pipeline_info = vk::ComputePipelineCreateInfo::builder().stage(*stage).layout(pipeline_layout);
        let pipeline = device.create_compute_pipeline(vk::PipelineCache::null(), &pipeline_info, None)?;
        device.destroy_shader_module(shader_module, None);

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
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // Binding 1: SVDAG geometry SSBO
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // Binding 2: SVDAG material SSBO
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // Binding 3: SVDAG chunk info SSBO
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // Binding 4: Color output image
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(4)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // Binding 5: Depth input sampler
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(5)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];

    let create_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    Ok(device.create_descriptor_set_layout(&create_info, None)?)
}

unsafe fn create_descriptor_pool(device: &Device) -> anyhow::Result<vk::DescriptorPool> {
    let pool_sizes = [
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(1) // Camera UBO
            .r#type(vk::DescriptorType::UNIFORM_BUFFER),
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(3) // geometry + material + chunk info
            .r#type(vk::DescriptorType::STORAGE_BUFFER),
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(1) // color output
            .r#type(vk::DescriptorType::STORAGE_IMAGE),
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(1) // depth input
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

unsafe fn write_descriptors(device: &Device, set: vk::DescriptorSet, data: &VulkanApplicationData, pool: &SvdagPool) {
    // Binding 0: Camera UBO
    let ubo_info = [*vk::DescriptorBufferInfo::builder()
        .buffer(data.uniform_buffer)
        .range(std::mem::size_of::<UniformBufferObject>() as u64)];

    // Binding 1: SVDAG geometry
    let geo_info = [*vk::DescriptorBufferInfo::builder().buffer(pool.geometry_buffer).range(vk::WHOLE_SIZE)];

    // Binding 2: SVDAG materials
    let mat_info = [*vk::DescriptorBufferInfo::builder().buffer(pool.material_buffer).range(vk::WHOLE_SIZE)];

    // Binding 3: SVDAG chunk info
    let chunk_info = [*vk::DescriptorBufferInfo::builder().buffer(pool.chunk_info_buffer).range(vk::WHOLE_SIZE)];

    // Binding 4: Color output (dedicated SVDAG storage image)
    let color_info = [*vk::DescriptorImageInfo::builder()
        .image_view(data.svdag_output_view)
        .image_layout(vk::ImageLayout::GENERAL)];

    // Binding 5: Depth input
    let depth_info = [*vk::DescriptorImageInfo::builder()
        .sampler(data.depth_pyramid_sampler)
        .image_view(data.depth_image_view)
        .image_layout(vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)];

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
            .buffer_info(&mat_info),
        *vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(&chunk_info),
        *vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&color_info),
        *vk::WriteDescriptorSet::builder()
            .dst_set(set)
            .dst_binding(5)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&depth_info),
    ];

    device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svdag_push_constants_size() {
        assert_eq!(std::mem::size_of::<SvdagPushConstants>(), 24);
    }
}
