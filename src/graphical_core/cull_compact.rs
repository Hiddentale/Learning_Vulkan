use crate::graphical_core::camera::UniformBufferObject;
use crate::graphical_core::compute_cull::CullPushConstants;
use crate::graphical_core::shaders::create_shader_module;
use crate::graphical_core::voxel_pool::VoxelPool;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use vk::Handle;
use vulkan_rust::{vk, Device};

/// GPU compaction + indirect mesh task dispatch (Niagara-style).
///
/// Runs `chunk_cull_compact.comp` once per phase, atomically appending the
/// indices of surviving chunks to `voxel_pool.visible_chunks_buffer[phase]`
/// and writing the visible count into `voxel_pool.indirect_args_buffer[phase]`
/// at offset 0 (the `groupCountX` field of `VkDrawMeshTasksIndirectCommandEXT`).
/// The indirect args buffer is then bound directly to
/// `cmd_draw_mesh_tasks_indirect_ext`, so the host never has to know how many
/// chunks survived the cull.
pub struct CullCompactPipeline {
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    /// One descriptor set per phase. Differs only in bindings 2 (visible
    /// chunks) and 3 (indirect args).
    pub descriptor_sets: [vk::DescriptorSet; 2],
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
}

impl CullCompactPipeline {
    pub unsafe fn create(device: &Device, data: &VulkanApplicationData, voxel_pool: &VoxelPool) -> anyhow::Result<Self> {
        let descriptor_set_layout = create_layout(device)?;
        let descriptor_pool = create_pool(device)?;
        let descriptor_sets = allocate_sets(device, descriptor_pool, descriptor_set_layout)?;
        write_descriptors(device, descriptor_sets, data, voxel_pool);

        let push_range = *vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(std::mem::size_of::<CullPushConstants>() as u32);
        let set_layouts = [descriptor_set_layout];
        let push_ranges = [push_range];
        let layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(&set_layouts)
            .push_constant_ranges(&push_ranges);
        let pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

        let module = create_shader_module(device, include_bytes!("../shaders/chunk_cull_compact.comp.spv"))?;
        let stage = vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(c"main");
        let pipeline_info = vk::ComputePipelineCreateInfo::builder().stage(*stage).layout(pipeline_layout);
        let pipeline = device.create_compute_pipeline(vk::PipelineCache::null(), &pipeline_info, None)?;
        device.destroy_shader_module(module, None);

        Ok(Self {
            descriptor_set_layout,
            descriptor_pool,
            descriptor_sets,
            pipeline_layout,
            pipeline,
        })
    }

    /// Re-write the depth pyramid descriptor on both sets after swapchain recreation.
    pub unsafe fn update_depth_pyramid(&self, device: &Device, data: &VulkanApplicationData) {
        let depth_info = [*vk::DescriptorImageInfo::builder()
            .image_view(data.depth_pyramid_full_view)
            .sampler(data.depth_pyramid_sampler)
            .image_layout(vk::ImageLayout::GENERAL)];
        let writes: Vec<_> = (0..2)
            .map(|i| {
                *vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_sets[i])
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&depth_info)
            })
            .collect();
        device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
    }

    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_descriptor_pool(self.descriptor_pool, None);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
    }
}

unsafe fn create_layout(device: &Device) -> anyhow::Result<vk::DescriptorSetLayout> {
    let bindings = [
        // 0: chunk info (read)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 1: visibility (read/write)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 2: visible chunks (write)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 3: indirect args / atomic count
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 4: depth pyramid (phase 2 occlusion)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(4)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 5: camera UBO
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(5)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    Ok(device.create_descriptor_set_layout(&info, None)?)
}

unsafe fn create_pool(device: &Device) -> anyhow::Result<vk::DescriptorPool> {
    // 2 sets × bindings: 8 storage buffers, 2 samplers, 2 ubos.
    let sizes = [
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(8)
            .r#type(vk::DescriptorType::STORAGE_BUFFER),
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(2)
            .r#type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(2)
            .r#type(vk::DescriptorType::UNIFORM_BUFFER),
    ];
    let info = vk::DescriptorPoolCreateInfo::builder().max_sets(2).pool_sizes(&sizes);
    Ok(device.create_descriptor_pool(&info, None)?)
}

unsafe fn allocate_sets(device: &Device, pool: vk::DescriptorPool, layout: vk::DescriptorSetLayout) -> anyhow::Result<[vk::DescriptorSet; 2]> {
    let layouts = [layout, layout];
    let info = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(pool).set_layouts(&layouts);
    let sets = device.allocate_descriptor_sets(&info)?;
    Ok([sets[0], sets[1]])
}

unsafe fn write_descriptors(device: &Device, sets: [vk::DescriptorSet; 2], data: &VulkanApplicationData, pool: &VoxelPool) {
    for (phase, &set) in sets.iter().enumerate() {
        let chunk_info = [*vk::DescriptorBufferInfo::builder().buffer(pool.chunk_info_buffer).range(vk::WHOLE_SIZE)];
        let vis = [*vk::DescriptorBufferInfo::builder().buffer(pool.visibility_buffer).range(vk::WHOLE_SIZE)];
        let visible = [*vk::DescriptorBufferInfo::builder()
            .buffer(pool.visible_chunks_buffer[phase])
            .range(vk::WHOLE_SIZE)];
        let args = [*vk::DescriptorBufferInfo::builder()
            .buffer(pool.indirect_args_buffer[phase])
            .range(vk::WHOLE_SIZE)];
        let depth = [*vk::DescriptorImageInfo::builder()
            .image_view(data.depth_pyramid_full_view)
            .sampler(data.depth_pyramid_sampler)
            .image_layout(vk::ImageLayout::GENERAL)];
        let ubo = [*vk::DescriptorBufferInfo::builder()
            .buffer(data.uniform_buffer)
            .range(std::mem::size_of::<UniformBufferObject>() as u64)];

        let writes = [
            *vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&chunk_info),
            *vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(1)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&vis),
            *vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(2)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&visible),
            *vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(3)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(&args),
            *vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&depth),
            *vk::WriteDescriptorSet::builder()
                .dst_set(set)
                .dst_binding(5)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(&ubo),
        ];
        device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
    }
}
