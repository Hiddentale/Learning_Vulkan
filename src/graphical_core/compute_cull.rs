use crate::graphical_core::buffers::allocate_buffer;
use crate::graphical_core::mesh_pool::MeshPool;
use crate::graphical_core::shaders::create_shader_module;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::voxel::chunk::CHUNK_SIZE;
use crate::voxel::meshing::BUCKET_COUNT;
use vulkanalia::vk::{self, DeviceV1_0, Handle, HasBuilder};
use vulkanalia::{Device, Instance};

/// Matches the GLSL FaceBucket struct (std430).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct GpuFaceBucket {
    first_index: u32,
    index_count: u32,
}

/// Matches the GLSL ChunkInfo struct (std430).
/// vec3 + int = 16 bytes, vec3 + uint = 16 bytes, 6 buckets = 48 bytes → total 80.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct GpuChunkInfo {
    aabb_min: [f32; 3],
    vertex_offset: i32,
    aabb_max: [f32; 3],
    transform_index: u32,
    buckets: [GpuFaceBucket; BUCKET_COUNT],
}

/// Push constants passed to the compute shader each frame.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CullPushConstants {
    pub planes: [[f32; 4]; 6],
    pub camera_pos: [f32; 3],
    pub chunk_count: u32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Resources for the GPU culling compute pipeline.
pub struct ComputeCullResources {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub chunk_info_buffer: vk::Buffer,
    pub chunk_info_memory: vk::DeviceMemory,
    pub chunk_info_ptr: *mut GpuChunkInfo,
    pub draw_count_buffer: vk::Buffer,
    pub draw_count_memory: vk::DeviceMemory,
}

impl ComputeCullResources {
    pub fn workgroup_count(chunk_count: u32) -> u32 {
        chunk_count.div_ceil(WORKGROUP_SIZE)
    }
}

pub unsafe fn create_compute_cull(
    device: &Device,
    instance: &Instance,
    data: &mut VulkanApplicationData,
    max_chunks: usize,
    indirect_buffer: vk::Buffer,
    indirect_buffer_size: u64,
) -> anyhow::Result<ComputeCullResources> {
    let host_visible = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    // Chunk info buffer (input to compute)
    let chunk_info_size = (max_chunks * std::mem::size_of::<GpuChunkInfo>()) as u64;
    let (ci_buf, ci_mem, ci_ptr) = allocate_buffer::<GpuChunkInfo>(
        chunk_info_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        device,
        instance,
        data,
        host_visible,
    )?;

    // Draw count buffer (atomic counter, reset each frame via vkCmdFillBuffer)
    let (dc_buf, dc_mem, _dc_ptr) = allocate_buffer::<u32>(
        4,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        device,
        instance,
        data,
        host_visible,
    )?;

    // Descriptor set layout: 3 SSBOs
    let bindings = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build(),
    ];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings).build();
    let desc_layout = device.create_descriptor_set_layout(&layout_info, None)?;

    // Descriptor pool
    let pool_size = vk::DescriptorPoolSize::builder()
        .descriptor_count(3)
        .type_(vk::DescriptorType::STORAGE_BUFFER);
    let pool_info = vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&[pool_size]).build();
    let desc_pool = device.create_descriptor_pool(&pool_info, None)?;

    // Allocate descriptor set
    let alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(desc_pool)
        .set_layouts(&[desc_layout])
        .build();
    let desc_set = device.allocate_descriptor_sets(&alloc_info)?[0];

    // Write descriptors
    write_compute_descriptors(device, desc_set, ci_buf, chunk_info_size, indirect_buffer, indirect_buffer_size, dc_buf);

    // Pipeline layout with push constants
    let push_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<CullPushConstants>() as u32);
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&[desc_layout])
        .push_constant_ranges(&[push_range])
        .build();
    let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;

    // Compute pipeline
    let shader_module = create_shader_module(device, &include_bytes!("../shaders/cull.comp.spv")[..])?;
    let stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(b"main\0");
    let pipeline_info = vk::ComputePipelineCreateInfo::builder().stage(stage).layout(pipeline_layout);
    let pipeline = device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)?.0[0];
    device.destroy_shader_module(shader_module, None);

    Ok(ComputeCullResources {
        pipeline,
        pipeline_layout,
        descriptor_set_layout: desc_layout,
        descriptor_pool: desc_pool,
        descriptor_set: desc_set,
        chunk_info_buffer: ci_buf,
        chunk_info_memory: ci_mem,
        chunk_info_ptr: ci_ptr,
        draw_count_buffer: dc_buf,
        draw_count_memory: dc_mem,
    })
}

fn write_compute_descriptors(
    device: &Device,
    set: vk::DescriptorSet,
    chunk_info_buffer: vk::Buffer,
    chunk_info_size: u64,
    indirect_buffer: vk::Buffer,
    indirect_buffer_size: u64,
    draw_count_buffer: vk::Buffer,
) {
    let ci_info = vk::DescriptorBufferInfo::builder()
        .buffer(chunk_info_buffer)
        .offset(0)
        .range(chunk_info_size);
    let ci_write = vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&[ci_info])
        .build();

    let ind_info = vk::DescriptorBufferInfo::builder()
        .buffer(indirect_buffer)
        .offset(0)
        .range(indirect_buffer_size);
    let ind_write = vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&[ind_info])
        .build();

    let dc_info = vk::DescriptorBufferInfo::builder().buffer(draw_count_buffer).offset(0).range(4);
    let dc_write = vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(2)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&[dc_info])
        .build();

    unsafe {
        device.update_descriptor_sets(&[ci_write, ind_write, dc_write], &[] as &[vk::CopyDescriptorSet]);
    }
}

/// Writes per-chunk GPU info (AABBs, draw params, face buckets) into the mapped buffer.
/// Returns the number of chunks written.
pub fn write_chunk_info(pool: &MeshPool, ptr: *mut GpuChunkInfo) -> u32 {
    let mut count = 0u32;
    for &pos in pool.chunk_positions() {
        if let Some(params) = pool.draw_params(&pos) {
            let aabb_min = [pos[0] as f32 * CHUNK_SIZE as f32, 0.0, pos[1] as f32 * CHUNK_SIZE as f32];
            let aabb_max = [
                aabb_min[0] + CHUNK_SIZE as f32,
                aabb_min[1] + CHUNK_SIZE as f32,
                aabb_min[2] + CHUNK_SIZE as f32,
            ];
            let mut buckets = [GpuFaceBucket::default(); BUCKET_COUNT];
            for (i, b) in params.buckets.iter().enumerate() {
                buckets[i] = GpuFaceBucket {
                    first_index: b.first_index,
                    index_count: b.index_count,
                };
            }
            let info = GpuChunkInfo {
                aabb_min,
                vertex_offset: params.vertex_offset,
                aabb_max,
                transform_index: params.transform_index,
                buckets,
            };
            unsafe {
                std::ptr::write(ptr.add(count as usize), info);
            }
            count += 1;
        }
    }
    count
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
}

pub unsafe fn create_depth_pyramid_pipeline(device: &Device, data: &VulkanApplicationData) -> anyhow::Result<DepthPyramidResources> {
    let mip_count = data.depth_pyramid_mip_count;

    // Layout: binding 0 = combined image sampler (source), binding 1 = storage image (dest mip)
    let bindings = [
        vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build(),
    ];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings).build();
    let desc_layout = device.create_descriptor_set_layout(&layout_info, None)?;

    // Pool: one sampler + one storage image per mip pass
    let pool_sizes = [
        vk::DescriptorPoolSize::builder()
            .descriptor_count(mip_count)
            .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .build(),
        vk::DescriptorPoolSize::builder()
            .descriptor_count(mip_count)
            .type_(vk::DescriptorType::STORAGE_IMAGE)
            .build(),
    ];
    let pool_info = vk::DescriptorPoolCreateInfo::builder()
        .max_sets(mip_count)
        .pool_sizes(&pool_sizes)
        .build();
    let desc_pool = device.create_descriptor_pool(&pool_info, None)?;

    // Allocate one set per mip level
    let layouts: Vec<_> = (0..mip_count).map(|_| desc_layout).collect();
    let alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(desc_pool)
        .set_layouts(&layouts)
        .build();
    let desc_sets = device.allocate_descriptor_sets(&alloc_info)?;

    // Write descriptors: set 0 reads depth buffer, sets 1+ read previous pyramid mip
    for mip in 0..mip_count {
        let (src_view, src_layout) = if mip == 0 {
            // First pass reads the depth buffer
            (data.depth_image_view, vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        } else {
            // Subsequent passes read the previous pyramid mip
            (data.depth_pyramid_mip_views[(mip - 1) as usize], vk::ImageLayout::GENERAL)
        };

        let src_info = vk::DescriptorImageInfo::builder()
            .sampler(data.depth_pyramid_sampler)
            .image_view(src_view)
            .image_layout(src_layout);
        let src_write = vk::WriteDescriptorSet::builder()
            .dst_set(desc_sets[mip as usize])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&[src_info])
            .build();

        let dst_info = vk::DescriptorImageInfo::builder()
            .image_view(data.depth_pyramid_mip_views[mip as usize])
            .image_layout(vk::ImageLayout::GENERAL);
        let dst_write = vk::WriteDescriptorSet::builder()
            .dst_set(desc_sets[mip as usize])
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .image_info(&[dst_info])
            .build();

        device.update_descriptor_sets(&[src_write, dst_write], &[] as &[vk::CopyDescriptorSet]);
    }

    // Pipeline layout with push constants for dst_size
    let push_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<DepthReducePush>() as u32);
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&[desc_layout])
        .push_constant_ranges(&[push_range])
        .build();
    let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;

    // Compute pipeline
    let shader_module = create_shader_module(device, &include_bytes!("../shaders/depth_reduce.comp.spv")[..])?;
    let stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(b"main\0");
    let pipeline_info = vk::ComputePipelineCreateInfo::builder().stage(stage).layout(pipeline_layout);
    let pipeline = device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)?.0[0];
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

pub unsafe fn destroy_compute_cull(device: &Device, res: &ComputeCullResources) {
    device.destroy_pipeline(res.pipeline, None);
    device.destroy_pipeline_layout(res.pipeline_layout, None);
    device.destroy_descriptor_pool(res.descriptor_pool, None);
    device.destroy_descriptor_set_layout(res.descriptor_set_layout, None);

    device.unmap_memory(res.chunk_info_memory);
    device.destroy_buffer(res.chunk_info_buffer, None);
    device.free_memory(res.chunk_info_memory, None);

    device.unmap_memory(res.draw_count_memory);
    device.destroy_buffer(res.draw_count_buffer, None);
    device.free_memory(res.draw_count_memory, None);
}
