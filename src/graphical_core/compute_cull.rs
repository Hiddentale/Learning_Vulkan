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
