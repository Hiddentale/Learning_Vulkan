use crate::graphical_core::shaders::create_shader_module;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use vk::Handle;
use vulkan_rust::{vk, Device};

/// Push constants shared by the task shader and the legacy compute cull shader.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CullPushConstants {
    pub planes: [[f32; 4]; 6],
    pub camera_pos: [f32; 3],
    pub chunk_count: u32,
    pub screen_size: [f32; 2],
    /// 1 = phase 1 (prev visible, no occlusion), 2 = phase 2 (prev invisible, occlusion test)
    pub phase: u32,
    /// Opaque block mask (mesh shader path) or draw buffer offset (legacy path).
    pub draw_offset: u32,
    /// Planet radius in blocks; used by the horizon culling early-out.
    pub planet_radius: f32,
    /// 1 = stereo (test both eye matrices for occlusion), 0 = mono (eye 0 only).
    pub stereo: u32,
    pub _pad: [f32; 2],
}

/// Resources for the depth pyramid generation compute pipeline.
pub struct DepthPyramidResources {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    /// One descriptor set per mip pass. Set 0: depth buffer → mip 0. Set N: mip N-1 → mip N.
    pub descriptor_sets: Vec<vk::DescriptorSet>,
}

/// Push constants for the depth reduce shader.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct DepthReducePush {
    pub dst_size: [u32; 2],
    /// 1 for mip 0 (1:1 copy from depth buffer), 0 for mip 1+ (2x2 max reduction).
    pub is_copy: u32,
    pub _pad: u32,
}

pub unsafe fn create_depth_pyramid_pipeline(device: &Device, data: &VulkanApplicationData) -> anyhow::Result<DepthPyramidResources> {
    let mip_count = data.depth_pyramid_mip_count;

    // Layout: binding 0 = combined image sampler (source), binding 1 = storage image (dest mip)
    let bindings = [
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    let desc_layout = device.create_descriptor_set_layout(&layout_info, None)?;

    // Pool: one sampler + one storage image per mip pass
    let pool_sizes = [
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(mip_count)
            .r#type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(mip_count)
            .r#type(vk::DescriptorType::STORAGE_IMAGE),
    ];
    let pool_info = vk::DescriptorPoolCreateInfo::builder().max_sets(mip_count).pool_sizes(&pool_sizes);
    let desc_pool = device.create_descriptor_pool(&pool_info, None)?;

    // Allocate one set per mip level
    let layouts: Vec<_> = (0..mip_count).map(|_| desc_layout).collect();
    let alloc_info = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(desc_pool).set_layouts(&layouts);
    let desc_sets = device.allocate_descriptor_sets(&alloc_info)?;

    // Write descriptors: set 0 reads depth buffer, sets 1+ read previous pyramid mip
    for mip in 0..mip_count {
        let (src_view, src_layout) = if mip == 0 {
            (data.depth_image_view, vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        } else {
            (data.depth_pyramid_mip_views[(mip - 1) as usize], vk::ImageLayout::GENERAL)
        };

        let src_info = [*vk::DescriptorImageInfo::builder()
            .sampler(data.depth_pyramid_sampler)
            .image_view(src_view)
            .image_layout(src_layout)];
        let src_write = *vk::WriteDescriptorSet::builder()
            .dst_set(desc_sets[mip as usize])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&src_info);

        let dst_info = [*vk::DescriptorImageInfo::builder()
            .image_view(data.depth_pyramid_mip_views[mip as usize])
            .image_layout(vk::ImageLayout::GENERAL)];
        let dst_write = *vk::WriteDescriptorSet::builder()
            .dst_set(desc_sets[mip as usize])
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&dst_info);

        device.update_descriptor_sets(&[src_write, dst_write], &[] as &[vk::CopyDescriptorSet]);
    }

    // Pipeline layout with push constants for dst_size
    let push_ranges = [*vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<DepthReducePush>() as u32)];
    let pyramid_set_layouts = [desc_layout];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&pyramid_set_layouts)
        .push_constant_ranges(&push_ranges);
    let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;

    // Compute pipeline
    let shader_module = create_shader_module(device, &include_bytes!("../shaders/depth_reduce.comp.spv")[..])?;
    let stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(c"main");
    let pipeline_info = vk::ComputePipelineCreateInfo::builder().stage(*stage).layout(pipeline_layout);
    let pipeline = device.create_compute_pipeline(vk::PipelineCache::null(), &pipeline_info, None)?;
    device.destroy_shader_module(shader_module, None);

    Ok(DepthPyramidResources {
        pipeline,
        pipeline_layout,
        descriptor_set_layout: desc_layout,
        descriptor_pool: desc_pool,
        descriptor_sets: desc_sets,
    })
}

pub unsafe fn destroy_depth_pyramid_pipeline(device: &Device, res: &DepthPyramidResources) {
    device.destroy_pipeline(res.pipeline, None);
    device.destroy_pipeline_layout(res.pipeline_layout, None);
    device.destroy_descriptor_pool(res.descriptor_pool, None);
    device.destroy_descriptor_set_layout(res.descriptor_set_layout, None);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cull_push_constants_size_is_144_bytes() {
        assert_eq!(std::mem::size_of::<CullPushConstants>(), 144);
    }

    #[test]
    fn cull_push_constants_field_offsets_match_glsl() {
        assert_eq!(memoffset::offset_of!(CullPushConstants, planes), 0);
        assert_eq!(memoffset::offset_of!(CullPushConstants, camera_pos), 96);
        assert_eq!(memoffset::offset_of!(CullPushConstants, chunk_count), 108);
        assert_eq!(memoffset::offset_of!(CullPushConstants, screen_size), 112);
        assert_eq!(memoffset::offset_of!(CullPushConstants, phase), 120);
        assert_eq!(memoffset::offset_of!(CullPushConstants, draw_offset), 124);
        assert_eq!(memoffset::offset_of!(CullPushConstants, planet_radius), 128);
        assert_eq!(memoffset::offset_of!(CullPushConstants, stereo), 132);
    }

    #[test]
    fn depth_reduce_push_size_is_16_bytes() {
        assert_eq!(std::mem::size_of::<DepthReducePush>(), 16);
    }
}
