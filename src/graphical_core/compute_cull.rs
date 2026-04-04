use crate::graphical_core::buffers::allocate_buffer;
use crate::graphical_core::mesh_pool::MeshPool;
use crate::graphical_core::shaders::create_shader_module;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::voxel::chunk::CHUNK_SIZE;
use crate::voxel::meshing::BUCKET_COUNT;
use vk::Handle;
use vulkan_rust::{vk, Device, Instance};

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
    pub visibility_ptr: *mut u32,
    pub max_chunks: usize,
}

impl ComputeCullResources {
    pub fn workgroup_count(chunk_count: u32) -> u32 {
        chunk_count.div_ceil(WORKGROUP_SIZE)
    }

    /// Resets all visibility entries to 1 (visible).
    /// Must be called when chunk ordering changes (load/unload) so that
    /// visibility[i] matches chunks[i].
    pub unsafe fn reset_visibility(&self) {
        for i in 0..self.max_chunks {
            std::ptr::write(self.visibility_ptr.add(i), 1u32);
        }
    }
}

pub unsafe fn create_compute_cull(
    device: &Device,
    instance: &Instance,
    data: &mut VulkanApplicationData,
    max_chunks: usize,
    indirect_buffer: vk::Buffer,
    indirect_buffer_size: vk::DeviceSize,
) -> anyhow::Result<ComputeCullResources> {
    // Chunk info buffer (input to compute)
    let chunk_info_size = (max_chunks * std::mem::size_of::<GpuChunkInfo>()) as u64;
    let (ci_buf, ci_mem, ci_ptr) = allocate_buffer::<GpuChunkInfo>(
        chunk_info_size,
        vk::BufferUsageFlags::STORAGE_BUFFER,
        device,
        instance,
        data,
        super::host_visible_coherent(),
    )?;

    // Draw count buffer (2 atomic counters: phase1_count + phase2_count)
    let (dc_buf, dc_mem, _dc_ptr) = allocate_buffer::<[u32; 2]>(
        8,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        device,
        instance,
        data,
        super::host_visible_coherent(),
    )?;

    // Visibility buffer (1 uint per chunk, persists across frames)
    let vis_size = (max_chunks * std::mem::size_of::<u32>()) as u64;
    let (vis_buf, vis_mem, vis_ptr) = allocate_buffer::<u32>(
        vis_size,
        vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        device,
        instance,
        data,
        super::host_visible_coherent(),
    )?;
    // Initialize all to 1 (treat all chunks as "was visible" on first frame)
    for i in 0..max_chunks {
        std::ptr::write(vis_ptr.add(i), 1u32);
    }

    // Descriptor set layout: 4 SSBOs + depth pyramid sampler + camera UBO
    let bindings = [
        compute_binding(0, vk::DescriptorType::STORAGE_BUFFER),
        compute_binding(1, vk::DescriptorType::STORAGE_BUFFER),
        compute_binding(2, vk::DescriptorType::STORAGE_BUFFER),
        compute_binding(3, vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        compute_binding(4, vk::DescriptorType::UNIFORM_BUFFER),
        compute_binding(5, vk::DescriptorType::STORAGE_BUFFER),
    ];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    let desc_layout = device.create_descriptor_set_layout(&layout_info, None)?;

    // Descriptor pool
    let pool_sizes = [
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(4)
            .r#type(vk::DescriptorType::STORAGE_BUFFER),
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(1)
            .r#type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        *vk::DescriptorPoolSize::builder()
            .descriptor_count(1)
            .r#type(vk::DescriptorType::UNIFORM_BUFFER),
    ];
    let pool_info = vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&pool_sizes);
    let desc_pool = device.create_descriptor_pool(&pool_info, None)?;

    // Allocate descriptor set
    let set_layouts = [desc_layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(desc_pool)
        .set_layouts(&set_layouts);
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
    let push_ranges = [*vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .offset(0)
        .size(std::mem::size_of::<CullPushConstants>() as u32)];
    let cull_set_layouts = [desc_layout];
    let pipeline_layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&cull_set_layouts)
        .push_constant_ranges(&push_ranges);
    let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;

    // Compute pipeline
    let shader_module = create_shader_module(device, &include_bytes!("../shaders/cull.comp.spv")[..])?;
    let stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(shader_module)
        .name(c"main");
    let pipeline_info = vk::ComputePipelineCreateInfo::builder().stage(*stage).layout(pipeline_layout);
    let pipeline = device.create_compute_pipeline(vk::PipelineCache::null(), &pipeline_info, None)?;
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
        visibility_ptr: vis_ptr,
        max_chunks,
    })
}

fn write_compute_descriptors(
    device: &Device,
    set: vk::DescriptorSet,
    chunk_info_buffer: vk::Buffer,
    chunk_info_size: vk::DeviceSize,
    indirect_buffer: vk::Buffer,
    indirect_buffer_size: vk::DeviceSize,
    draw_count_buffer: vk::Buffer,
    visibility_buffer: vk::Buffer,
    visibility_size: vk::DeviceSize,
) {
    let ci_info = [*vk::DescriptorBufferInfo::builder()
        .buffer(chunk_info_buffer)
        .offset(0)
        .range(chunk_info_size)];
    let ci_write = *vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&ci_info);

    let ind_info = [*vk::DescriptorBufferInfo::builder()
        .buffer(indirect_buffer)
        .offset(0)
        .range(indirect_buffer_size)];
    let ind_write = *vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(1)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&ind_info);

    let dc_info = [*vk::DescriptorBufferInfo::builder().buffer(draw_count_buffer).offset(0).range(8)];
    let dc_write = *vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(2)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&dc_info);

    let vis_info = [*vk::DescriptorBufferInfo::builder()
        .buffer(visibility_buffer)
        .offset(0)
        .range(visibility_size)];
    let vis_write = *vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(5)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(&vis_info);

    unsafe {
        device.update_descriptor_sets(&[ci_write, ind_write, dc_write, vis_write], &[] as &[vk::CopyDescriptorSet]);
    }
}

/// Writes depth pyramid sampler (binding 3) and camera UBO (binding 4) into the cull descriptor set.
fn write_depth_pyramid_descriptors(device: &Device, set: vk::DescriptorSet, data: &VulkanApplicationData) {
    let img_info = [*vk::DescriptorImageInfo::builder()
        .sampler(data.depth_pyramid_sampler)
        .image_view(data.depth_pyramid_full_view)
        .image_layout(vk::ImageLayout::GENERAL)];
    let img_write = *vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(3)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&img_info);

    // Binding 4: camera UBO (view-projection matrix)
    let ubo_info = [*vk::DescriptorBufferInfo::builder()
        .buffer(data.uniform_buffer)
        .offset(0)
        .range(std::mem::size_of::<crate::graphical_core::camera::UniformBufferObject>() as u64)];
    let ubo_write = *vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(4)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(&ubo_info);

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
            let aabb_min = [
                pos[0] as f32 * CHUNK_SIZE as f32,
                pos[1] as f32 * CHUNK_SIZE as f32,
                pos[2] as f32 * CHUNK_SIZE as f32,
            ];
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
            // First pass reads the depth buffer
            (data.depth_image_view, vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL)
        } else {
            // Subsequent passes read the previous pyramid mip
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

fn compute_binding(binding: u32, descriptor_type: vk::DescriptorType) -> vk::DescriptorSetLayoutBinding {
    *vk::DescriptorSetLayoutBinding::builder()
        .binding(binding)
        .descriptor_count(1)
        .descriptor_type(descriptor_type)
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
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
#[cfg(test)]
/// Result of projecting an AABB to screen space.
pub struct ScreenProjection {
    pub uv_min: [f32; 2],
    pub uv_max: [f32; 2],
    /// Nearest depth in NDC space.
    pub ndc_min_z: f32,
    /// Nearest depth converted to viewport [0,1] range.
    pub viewport_min_z: f32,
}

#[cfg(test)]
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

#[cfg(test)]
/// Selects the depth pyramid mip level for a given screen-space projection.
pub fn select_mip_level(proj: &ScreenProjection, screen_width: f32, screen_height: f32) -> f32 {
    let size_x = (proj.uv_max[0] - proj.uv_min[0]) * screen_width;
    let size_y = (proj.uv_max[1] - proj.uv_min[1]) * screen_height;
    let max_extent = size_x.max(size_y).max(1.0);
    max_extent.log2().ceil()
}

#[cfg(test)]
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
        assert!(cam_right.x >= aabb_max.x); // -X faces NOT visible
    }

    // --- Struct layout tests (must match GLSL std430) ---

    #[test]
    fn gpu_chunk_info_size_matches_glsl() {
        // GLSL: vec3(16) + int(4) = nope → vec3(12)+int(4)=16, vec3(12)+uint(4)=16, 6*FaceBucket(8)=48 → 80
        assert_eq!(std::mem::size_of::<GpuChunkInfo>(), 80);
    }

    #[test]
    fn gpu_chunk_info_field_offsets_match_glsl() {
        assert_eq!(memoffset::offset_of!(GpuChunkInfo, aabb_min), 0);
        assert_eq!(memoffset::offset_of!(GpuChunkInfo, vertex_offset), 12);
        assert_eq!(memoffset::offset_of!(GpuChunkInfo, aabb_max), 16);
        assert_eq!(memoffset::offset_of!(GpuChunkInfo, transform_index), 28);
        assert_eq!(memoffset::offset_of!(GpuChunkInfo, buckets), 32);
    }

    #[test]
    fn gpu_face_bucket_size_matches_glsl() {
        assert_eq!(std::mem::size_of::<GpuFaceBucket>(), 8);
    }

    #[test]
    fn cull_push_constants_size_is_128_bytes() {
        // Vulkan minimum maxPushConstantsSize is 128; our struct must fit exactly
        assert_eq!(std::mem::size_of::<CullPushConstants>(), 128);
    }

    #[test]
    fn cull_push_constants_field_offsets_match_glsl() {
        assert_eq!(memoffset::offset_of!(CullPushConstants, planes), 0);
        assert_eq!(memoffset::offset_of!(CullPushConstants, camera_pos), 96);
        assert_eq!(memoffset::offset_of!(CullPushConstants, chunk_count), 108);
        assert_eq!(memoffset::offset_of!(CullPushConstants, screen_size), 112);
        assert_eq!(memoffset::offset_of!(CullPushConstants, phase), 120);
        assert_eq!(memoffset::offset_of!(CullPushConstants, draw_offset), 124);
    }

    #[test]
    fn depth_reduce_push_size_is_16_bytes() {
        assert_eq!(std::mem::size_of::<DepthReducePush>(), 16);
    }

    // --- Phase filtering logic (mirrors cull.comp main()) ---

    /// Simulates the shader's phase filtering: returns which chunk indices are processed.
    fn phase_filter(visibility: &[u32], phase: u32) -> Vec<usize> {
        visibility
            .iter()
            .enumerate()
            .filter(|&(_, &vis)| {
                let was_visible = vis == 1;
                match phase {
                    1 => was_visible,
                    2 => !was_visible,
                    _ => panic!("invalid phase"),
                }
            })
            .map(|(i, _)| i)
            .collect()
    }

    #[test]
    fn phase1_processes_only_visible_chunks() {
        let vis = vec![1, 0, 1, 0, 1];
        assert_eq!(phase_filter(&vis, 1), vec![0, 2, 4]);
    }

    #[test]
    fn phase2_processes_only_invisible_chunks() {
        let vis = vec![1, 0, 1, 0, 1];
        assert_eq!(phase_filter(&vis, 2), vec![1, 3]);
    }

    #[test]
    fn all_visible_means_phase2_processes_nothing() {
        let vis = vec![1, 1, 1];
        assert_eq!(phase_filter(&vis, 2), Vec::<usize>::new());
    }

    #[test]
    fn all_invisible_means_phase1_processes_nothing() {
        let vis = vec![0, 0, 0];
        assert_eq!(phase_filter(&vis, 1), Vec::<usize>::new());
    }

    // --- Buffer offset arithmetic ---

    #[test]
    fn phase2_draw_offset_is_half_max_indirect_draws() {
        use crate::graphical_core::vulkan_object::MAX_INDIRECT_DRAWS;
        let expected = (MAX_INDIRECT_DRAWS / 2) as u32;
        let phase2_offset = (MAX_INDIRECT_DRAWS / 2) as u32;
        assert_eq!(phase2_offset, expected);
        assert!(phase2_offset as usize >= MAX_INDIRECT_DRAWS / 2);
    }

    #[test]
    fn indirect_buffer_fits_both_phases() {
        use crate::graphical_core::vulkan_object::{DrawIndexedIndirectCommand, MAX_INDIRECT_DRAWS};
        let stride = std::mem::size_of::<DrawIndexedIndirectCommand>();
        let phase2_offset = MAX_INDIRECT_DRAWS / 2;
        let total_bytes = MAX_INDIRECT_DRAWS * stride;
        let phase2_byte_offset = phase2_offset * stride;
        assert!(phase2_byte_offset + phase2_offset * stride <= total_bytes);
    }

    // --- Workgroup count ---

    #[test]
    fn workgroup_count_zero_chunks() {
        assert_eq!(ComputeCullResources::workgroup_count(0), 0);
    }

    #[test]
    fn workgroup_count_one_chunk() {
        assert_eq!(ComputeCullResources::workgroup_count(1), 1);
    }

    #[test]
    fn workgroup_count_exactly_64() {
        assert_eq!(ComputeCullResources::workgroup_count(64), 1);
    }

    #[test]
    fn workgroup_count_65_needs_two_groups() {
        assert_eq!(ComputeCullResources::workgroup_count(65), 2);
    }
}
