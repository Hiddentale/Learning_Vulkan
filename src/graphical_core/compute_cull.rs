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
    pub screen_size: [f32; 2],
    /// 1 = phase 1 (prev visible, no occlusion), 2 = phase 2 (prev invisible, occlusion test)
    pub phase: u32,
    /// Offset into the draw command buffer for this phase's output.
    pub draw_offset: u32,
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
    pub visibility_buffer: vk::Buffer,
    pub visibility_memory: vk::DeviceMemory,
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

    // Draw count buffer (2 atomic counters: phase1_count + phase2_count)
    let (dc_buf, dc_mem, _dc_ptr) = allocate_buffer::<[u32; 2]>(
        8,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        device,
        instance,
        data,
        host_visible,
    )?;

    // Visibility buffer (1 uint per chunk, persists across frames)
    let vis_size = (max_chunks * std::mem::size_of::<u32>()) as u64;
    let (vis_buf, vis_mem, _vis_ptr) = allocate_buffer::<u32>(
        vis_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        device,
        instance,
        data,
        host_visible,
    )?;
    // Initialize all to 1 (treat all chunks as "was visible" on first frame)
    for i in 0..max_chunks {
        std::ptr::write(_vis_ptr.add(i), 1u32);
    }

    // Descriptor set layout: 4 SSBOs + depth pyramid sampler + camera UBO
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
        vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(4)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build(),
        vk::DescriptorSetLayoutBinding::builder()
            .binding(5)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build(),
    ];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings).build();
    let desc_layout = device.create_descriptor_set_layout(&layout_info, None)?;

    // Descriptor pool
    let pool_sizes = [
        vk::DescriptorPoolSize::builder()
            .descriptor_count(4)
            .type_(vk::DescriptorType::STORAGE_BUFFER)
            .build(),
        vk::DescriptorPoolSize::builder()
            .descriptor_count(1)
            .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .build(),
        vk::DescriptorPoolSize::builder()
            .descriptor_count(1)
            .type_(vk::DescriptorType::UNIFORM_BUFFER)
            .build(),
    ];
    let pool_info = vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&pool_sizes).build();
    let desc_pool = device.create_descriptor_pool(&pool_info, None)?;

    // Allocate descriptor set
    let alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(desc_pool)
        .set_layouts(&[desc_layout])
        .build();
    let desc_set = device.allocate_descriptor_sets(&alloc_info)?[0];

    // Write descriptors
    write_compute_descriptors(
        device,
        desc_set,
        ci_buf,
        chunk_info_size,
        indirect_buffer,
        indirect_buffer_size,
        dc_buf,
        vis_buf,
        vis_size,
    );
    write_depth_pyramid_descriptors(device, desc_set, data);

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
        visibility_buffer: vis_buf,
        visibility_memory: vis_mem,
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
    visibility_buffer: vk::Buffer,
    visibility_size: u64,
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

    let dc_info = vk::DescriptorBufferInfo::builder().buffer(draw_count_buffer).offset(0).range(8);
    let dc_write = vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(2)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&[dc_info])
        .build();

    let vis_info = vk::DescriptorBufferInfo::builder()
        .buffer(visibility_buffer)
        .offset(0)
        .range(visibility_size);
    let vis_write = vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(5)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&[vis_info])
        .build();

    unsafe {
        device.update_descriptor_sets(&[ci_write, ind_write, dc_write, vis_write], &[] as &[vk::CopyDescriptorSet]);
    }
}

/// Writes depth pyramid sampler (binding 3) and camera UBO (binding 4) into the cull descriptor set.
fn write_depth_pyramid_descriptors(device: &Device, set: vk::DescriptorSet, data: &VulkanApplicationData) {
    let img_info = vk::DescriptorImageInfo::builder()
        .sampler(data.depth_pyramid_sampler)
        .image_view(data.depth_pyramid_full_view)
        .image_layout(vk::ImageLayout::GENERAL);
    let img_write = vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(3)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&[img_info])
        .build();

    // Binding 4: camera UBO (view-projection matrix)
    let ubo_info = vk::DescriptorBufferInfo::builder()
        .buffer(data.uniform_buffer)
        .offset(0)
        .range(std::mem::size_of::<crate::graphical_core::camera::UniformBufferObject>() as u64);
    let ubo_write = vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(4)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(&[ubo_info])
        .build();

    unsafe {
        device.update_descriptor_sets(&[img_write, ubo_write], &[] as &[vk::CopyDescriptorSet]);
    }
}

/// Updates the depth pyramid descriptor after swapchain recreation.
pub unsafe fn update_cull_depth_pyramid(device: &Device, res: &ComputeCullResources, data: &VulkanApplicationData) {
    write_depth_pyramid_descriptors(device, res.descriptor_set, data);
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
    /// 1 for mip 0 (1:1 copy from depth buffer), 0 for mip 1+ (2x2 max reduction).
    pub is_copy: u32,
    pub _pad: u32,
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

    device.unmap_memory(res.visibility_memory);
    device.destroy_buffer(res.visibility_buffer, None);
    device.free_memory(res.visibility_memory, None);
}

// --- Testable occlusion math (mirrors the GLSL shader logic) ---

/// Result of projecting an AABB to screen space.
pub struct ScreenProjection {
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    /// Nearest depth in NDC space.
    pub ndc_min_z: f32,
    /// Nearest depth converted to viewport [0,1] range.
    pub viewport_min_z: f32,
}

/// Projects an AABB's 8 corners through a VP matrix to screen-space UVs and depth.
/// Returns None if any corner is behind the camera (conservatively visible).
pub fn project_aabb_to_screen(vp: &glam::Mat4, aabb_min: glam::Vec3, aabb_max: glam::Vec3) -> Option<ScreenProjection> {
    let corners = [
        glam::Vec3::new(aabb_min.x, aabb_min.y, aabb_min.z),
        glam::Vec3::new(aabb_max.x, aabb_min.y, aabb_min.z),
        glam::Vec3::new(aabb_min.x, aabb_max.y, aabb_min.z),
        glam::Vec3::new(aabb_max.x, aabb_max.y, aabb_min.z),
        glam::Vec3::new(aabb_min.x, aabb_min.y, aabb_max.z),
        glam::Vec3::new(aabb_max.x, aabb_min.y, aabb_max.z),
        glam::Vec3::new(aabb_min.x, aabb_max.y, aabb_max.z),
        glam::Vec3::new(aabb_max.x, aabb_max.y, aabb_max.z),
    ];

    let mut ndc_min = glam::Vec2::splat(1.0);
    let mut ndc_max = glam::Vec2::splat(-1.0);
    let mut min_z: f32 = 1.0;

    for corner in &corners {
        let clip = *vp * corner.extend(1.0);
        if clip.w <= 0.0 {
            return None;
        }
        let ndc = glam::Vec3::new(clip.x / clip.w, clip.y / clip.w, clip.z / clip.w);
        ndc_min = ndc_min.min(glam::Vec2::new(ndc.x, ndc.y));
        ndc_max = ndc_max.max(glam::Vec2::new(ndc.x, ndc.y));
        min_z = min_z.min(ndc.z);
    }

    ndc_min = ndc_min.clamp(glam::Vec2::splat(-1.0), glam::Vec2::splat(1.0));
    ndc_max = ndc_max.clamp(glam::Vec2::splat(-1.0), glam::Vec2::splat(1.0));

    let uv_min = [ndc_min.x * 0.5 + 0.5, ndc_min.y * 0.5 + 0.5];
    let uv_max = [ndc_max.x * 0.5 + 0.5, ndc_max.y * 0.5 + 0.5];

    Some(ScreenProjection {
        uv_min,
        uv_max,
        ndc_min_z: min_z,
        viewport_min_z: min_z * 0.5 + 0.5,
    })
}

/// Selects the depth pyramid mip level for a given screen-space projection.
pub fn select_mip_level(proj: &ScreenProjection, screen_width: f32, screen_height: f32) -> f32 {
    let size_x = (proj.uv_max[0] - proj.uv_min[0]) * screen_width;
    let size_y = (proj.uv_max[1] - proj.uv_min[1]) * screen_height;
    let max_extent = size_x.max(size_y).max(1.0);
    max_extent.log2().ceil()
}

/// Returns true if the chunk is NOT occluded (visible).
/// `pyramid_depth` is the max depth sampled from the Hi-Z pyramid.
pub fn occlusion_test(viewport_min_z: f32, pyramid_depth: f32) -> bool {
    viewport_min_z <= pyramid_depth
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec3};

    fn test_vp_matrix() -> Mat4 {
        let mut proj = Mat4::perspective_rh(90_f32.to_radians(), 16.0 / 9.0, 0.1, 500.0);
        proj.y_axis.y *= -1.0; // Vulkan Y-flip
        let view = Mat4::look_at_rh(Vec3::new(0.0, 30.0, 0.0), Vec3::new(0.0, 30.0, -1.0), Vec3::Y);
        proj * view
    }

    #[test]
    fn project_aabb_in_front_of_camera_returns_some() {
        let vp = test_vp_matrix();
        let result = project_aabb_to_screen(&vp, Vec3::new(-8.0, 22.0, -20.0), Vec3::new(8.0, 38.0, -4.0));
        assert!(result.is_some());
    }

    #[test]
    fn project_aabb_behind_camera_returns_none() {
        let vp = test_vp_matrix();
        let result = project_aabb_to_screen(&vp, Vec3::new(-8.0, 22.0, 10.0), Vec3::new(8.0, 38.0, 26.0));
        assert!(result.is_none());
    }

    #[test]
    fn projected_uvs_are_in_zero_one_range() {
        let vp = test_vp_matrix();
        let proj = project_aabb_to_screen(&vp, Vec3::new(-8.0, 22.0, -20.0), Vec3::new(8.0, 38.0, -4.0)).unwrap();
        assert!(proj.uv_min[0] >= 0.0 && proj.uv_min[0] <= 1.0, "uv_min.x = {}", proj.uv_min[0]);
        assert!(proj.uv_min[1] >= 0.0 && proj.uv_min[1] <= 1.0, "uv_min.y = {}", proj.uv_min[1]);
        assert!(proj.uv_max[0] >= 0.0 && proj.uv_max[0] <= 1.0, "uv_max.x = {}", proj.uv_max[0]);
        assert!(proj.uv_max[1] >= 0.0 && proj.uv_max[1] <= 1.0, "uv_max.y = {}", proj.uv_max[1]);
    }

    #[test]
    fn viewport_depth_is_in_zero_one_range() {
        let vp = test_vp_matrix();
        let proj = project_aabb_to_screen(&vp, Vec3::new(-8.0, 22.0, -20.0), Vec3::new(8.0, 38.0, -4.0)).unwrap();
        assert!(
            proj.viewport_min_z >= 0.0 && proj.viewport_min_z <= 1.0,
            "viewport_min_z = {} (ndc_min_z = {})",
            proj.viewport_min_z,
            proj.ndc_min_z
        );
    }

    #[test]
    fn near_chunk_has_smaller_viewport_depth_than_far_chunk() {
        let vp = test_vp_matrix();
        let near = project_aabb_to_screen(&vp, Vec3::new(-4.0, 26.0, -10.0), Vec3::new(4.0, 34.0, -2.0)).unwrap();
        let far = project_aabb_to_screen(&vp, Vec3::new(-4.0, 26.0, -100.0), Vec3::new(4.0, 34.0, -92.0)).unwrap();
        assert!(
            near.viewport_min_z < far.viewport_min_z,
            "near={} should be < far={}",
            near.viewport_min_z,
            far.viewport_min_z
        );
    }

    #[test]
    fn mip_level_increases_for_larger_projection() {
        let small = ScreenProjection {
            uv_min: [0.4, 0.4],
            uv_max: [0.6, 0.6],
            ndc_min_z: 0.5,
            viewport_min_z: 0.75,
        };
        let large = ScreenProjection {
            uv_min: [0.1, 0.1],
            uv_max: [0.9, 0.9],
            ndc_min_z: 0.5,
            viewport_min_z: 0.75,
        };
        let mip_small = select_mip_level(&small, 1920.0, 1080.0);
        let mip_large = select_mip_level(&large, 1920.0, 1080.0);
        assert!(mip_large > mip_small, "large mip {} should be > small mip {}", mip_large, mip_small);
    }

    #[test]
    fn chunk_in_front_of_occluder_is_visible() {
        assert!(occlusion_test(0.3, 0.5));
    }

    #[test]
    fn chunk_behind_occluder_is_culled() {
        assert!(!occlusion_test(0.7, 0.5));
    }

    #[test]
    fn chunk_at_far_plane_with_sky_depth_is_visible() {
        assert!(occlusion_test(0.99, 1.0));
    }

    #[test]
    fn face_bucket_visibility() {
        let cam = Vec3::new(10.0, 5.0, 10.0);
        let aabb_min = Vec3::new(0.0, 0.0, 0.0);
        let aabb_max = Vec3::new(16.0, 16.0, 16.0);

        // Camera is inside the AABB on all axes → all faces visible
        assert!(cam.x > aabb_min.x); // +X
        assert!(cam.x < aabb_max.x); // -X
        assert!(cam.y > aabb_min.y); // +Y
        assert!(cam.y < aabb_max.y); // -Y
        assert!(cam.z > aabb_min.z); // +Z
        assert!(cam.z < aabb_max.z); // -Z

        // Camera far to the right → -X faces not visible
        let cam_right = Vec3::new(100.0, 5.0, 10.0);
        assert!(cam_right.x > aabb_max.x); // camera past +X side
        assert!(!(cam_right.x < aabb_max.x)); // -X faces NOT visible (camera.x >= aabb_max.x)
    }
}
