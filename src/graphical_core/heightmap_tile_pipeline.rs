//! GPU pipeline for the SSE quadtree heightmap path.
//!
//! Owns four things:
//! 1. **Buffers**: a host-visible `TileDesc` SSBO the CPU writes once per
//!    frame, a device-local `visible_tiles[]` SSBO the cull shader appends
//!    to, and a small `IndirectArgs` SSBO holding a `VkDrawMeshTasksIndirectCommandEXT`.
//! 2. **Cull compute pipeline** running `heightmap_cull.comp`.
//! 3. **Graphics pipeline** chaining `heightmap_tile.task` →
//!    `heightmap_tile.mesh` → `heightmap_tile.frag`.
//! 4. **Descriptor set** binding all of the above + the camera UBO and the
//!    heightmap atlas image.
//!
//! Phase 3: this module compiles and creates valid pipelines but is not yet
//! invoked by the frame loop. Phase 4 wires it into command buffer recording.

#![allow(dead_code)] // Phase 3: created but not yet dispatched.

use crate::graphical_core::camera::UniformBufferObject;
use crate::graphical_core::buffers::allocate_buffer;
use crate::graphical_core::heightmap_atlas::HeightmapAtlas;
use crate::graphical_core::shaders::create_shader_module;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::voxel::heightmap_quadtree::{GpuTileDesc, MAX_RESIDENT_TILES};
use vk::Handle;
use vulkan_rust::{vk, Device, Instance};

// Workaround for missing TASK/MESH stage flags in vulkan-rust 0.10.0.
const TASK_STAGE: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(0x40);
const MESH_STAGE: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(0x80);

/// Push constants for `heightmap_cull.comp`. Frustum planes + camera +
/// tile count + planet radius. Layout matches the shader's std430 block.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CullPush {
    pub planes: [[f32; 4]; 6],
    pub camera_pos: [f32; 3],
    pub tile_count: u32,
    pub planet_radius: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

const _: () = assert!(std::mem::size_of::<CullPush>() == 16 * 6 + 16 + 16);

/// Push constants for the task/mesh stages.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct TilePush {
    pub camera_pos: [f32; 3],
    pub atlas_cols: u32,
    pub face_side_blocks: f32,
    pub _pad0: f32,
    pub _pad1: f32,
    pub _pad2: f32,
}

const _: () = assert!(std::mem::size_of::<TilePush>() == 32);

pub struct HeightmapTileBuffers {
    pub tile_desc_buffer: vk::Buffer,
    tile_desc_memory: vk::DeviceMemory,
    pub tile_desc_ptr: *mut GpuTileDesc,

    pub visible_tiles_buffer: vk::Buffer,
    visible_tiles_memory: vk::DeviceMemory,

    pub indirect_args_buffer: vk::Buffer,
    indirect_args_memory: vk::DeviceMemory,
    indirect_args_ptr: *mut u32,
}

unsafe impl Send for HeightmapTileBuffers {}
unsafe impl Sync for HeightmapTileBuffers {}

impl HeightmapTileBuffers {
    pub unsafe fn new(device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<Self> {
        let tile_desc_size = (MAX_RESIDENT_TILES * std::mem::size_of::<GpuTileDesc>()) as u64;
        let (tile_desc_buffer, tile_desc_memory, tile_desc_ptr) = allocate_buffer::<GpuTileDesc>(
            tile_desc_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            device, instance, data, super::host_visible_coherent(),
        )?;

        let visible_size = (MAX_RESIDENT_TILES * 4) as u64;
        let (visible_tiles_buffer, visible_tiles_memory, _ignored) = allocate_buffer::<u32>(
            visible_size,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            device, instance, data, super::host_visible_coherent(),
        )?;

        // Indirect args buffer: 3 u32s, used as both an SSBO (compute writes
        // group_count_x) and an INDIRECT_BUFFER (cmd_draw_mesh_tasks_indirect
        // reads it). Host-visible so the CPU can pre-init group_count_y/z.
        let args_size = (3 * 4) as u64;
        let (indirect_args_buffer, indirect_args_memory, indirect_args_ptr) = allocate_buffer::<u32>(
            args_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
            device, instance, data, super::host_visible_coherent(),
        )?;
        // groupCountX is cleared each frame via cmd_fill_buffer; Y/Z stay 1.
        std::ptr::write(indirect_args_ptr.add(0), 0);
        std::ptr::write(indirect_args_ptr.add(1), 1);
        std::ptr::write(indirect_args_ptr.add(2), 1);

        Ok(Self {
            tile_desc_buffer, tile_desc_memory, tile_desc_ptr,
            visible_tiles_buffer, visible_tiles_memory,
            indirect_args_buffer, indirect_args_memory, indirect_args_ptr,
        })
    }

    pub unsafe fn destroy(&self, device: &Device) {
        device.unmap_memory(self.tile_desc_memory);
        device.destroy_buffer(self.tile_desc_buffer, None);
        device.free_memory(self.tile_desc_memory, None);

        device.destroy_buffer(self.visible_tiles_buffer, None);
        device.free_memory(self.visible_tiles_memory, None);

        device.unmap_memory(self.indirect_args_memory);
        device.destroy_buffer(self.indirect_args_buffer, None);
        device.free_memory(self.indirect_args_memory, None);
    }
}

pub struct HeightmapTilePipeline {
    pub buffers: HeightmapTileBuffers,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub descriptor_set: vk::DescriptorSet,
    pub cull_pipeline_layout: vk::PipelineLayout,
    pub cull_pipeline: vk::Pipeline,
    pub tile_pipeline_layout: vk::PipelineLayout,
    pub tile_pipeline: vk::Pipeline,
}

impl HeightmapTilePipeline {
    pub unsafe fn create(
        device: &Device,
        instance: &Instance,
        data: &mut VulkanApplicationData,
        atlas: &HeightmapAtlas,
    ) -> anyhow::Result<Self> {
        let buffers = HeightmapTileBuffers::new(device, instance, data)?;
        let descriptor_set_layout = create_descriptor_layout(device)?;
        let descriptor_pool = create_descriptor_pool(device)?;
        let descriptor_set = allocate_descriptor_set(device, descriptor_pool, descriptor_set_layout)?;
        write_descriptors(device, descriptor_set, &buffers, atlas, data);

        let cull_pipeline_layout = {
            let push = *vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(std::mem::size_of::<CullPush>() as u32);
            let set_layouts = [descriptor_set_layout];
            let push_ranges = [push];
            let info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_ranges);
            device.create_pipeline_layout(&info, None)?
        };
        let cull_pipeline = create_cull_pipeline(device, cull_pipeline_layout)?;

        let tile_pipeline_layout = {
            let push = *vk::PushConstantRange::builder()
                .stage_flags(TASK_STAGE | MESH_STAGE)
                .offset(0)
                .size(std::mem::size_of::<TilePush>() as u32);
            let set_layouts = [descriptor_set_layout];
            let push_ranges = [push];
            let info = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(&set_layouts)
                .push_constant_ranges(&push_ranges);
            device.create_pipeline_layout(&info, None)?
        };
        let tile_pipeline = create_tile_graphics_pipeline(device, data, tile_pipeline_layout)?;

        Ok(Self {
            buffers,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            cull_pipeline_layout,
            cull_pipeline,
            tile_pipeline_layout,
            tile_pipeline,
        })
    }

    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_pipeline(self.tile_pipeline, None);
        device.destroy_pipeline_layout(self.tile_pipeline_layout, None);
        device.destroy_pipeline(self.cull_pipeline, None);
        device.destroy_pipeline_layout(self.cull_pipeline_layout, None);
        device.destroy_descriptor_pool(self.descriptor_pool, None);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        self.buffers.destroy(device);
    }
}

unsafe fn create_descriptor_layout(device: &Device) -> anyhow::Result<vk::DescriptorSetLayout> {
    let bindings = [
        // 0: TileDescBuffer (compute + task + mesh, read-only from each)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(0)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE | TASK_STAGE | MESH_STAGE),
        // 1: VisibleTilesBuffer (compute write, task read)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(1)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE | TASK_STAGE),
        // 2: IndirectArgsBuffer (compute write only)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(2)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .stage_flags(vk::ShaderStageFlags::COMPUTE),
        // 3: CameraUBO (mesh + fragment)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(3)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .stage_flags(MESH_STAGE | vk::ShaderStageFlags::FRAGMENT),
        // 4: Height atlas (mesh)
        *vk::DescriptorSetLayoutBinding::builder()
            .binding(4)
            .descriptor_count(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .stage_flags(MESH_STAGE),
    ];
    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    Ok(device.create_descriptor_set_layout(&info, None)?)
}

unsafe fn create_descriptor_pool(device: &Device) -> anyhow::Result<vk::DescriptorPool> {
    let sizes = [
        *vk::DescriptorPoolSize::builder().descriptor_count(3).r#type(vk::DescriptorType::STORAGE_BUFFER),
        *vk::DescriptorPoolSize::builder().descriptor_count(1).r#type(vk::DescriptorType::UNIFORM_BUFFER),
        *vk::DescriptorPoolSize::builder().descriptor_count(1).r#type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
    ];
    let info = vk::DescriptorPoolCreateInfo::builder().max_sets(1).pool_sizes(&sizes);
    Ok(device.create_descriptor_pool(&info, None)?)
}

unsafe fn allocate_descriptor_set(device: &Device, pool: vk::DescriptorPool, layout: vk::DescriptorSetLayout) -> anyhow::Result<vk::DescriptorSet> {
    let layouts = [layout];
    let info = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(pool).set_layouts(&layouts);
    let sets = device.allocate_descriptor_sets(&info)?;
    Ok(sets[0])
}

unsafe fn write_descriptors(
    device: &Device,
    set: vk::DescriptorSet,
    buffers: &HeightmapTileBuffers,
    atlas: &HeightmapAtlas,
    data: &VulkanApplicationData,
) {
    let tile_info = [*vk::DescriptorBufferInfo::builder().buffer(buffers.tile_desc_buffer).range(vk::WHOLE_SIZE)];
    let visible_info = [*vk::DescriptorBufferInfo::builder().buffer(buffers.visible_tiles_buffer).range(vk::WHOLE_SIZE)];
    let args_info = [*vk::DescriptorBufferInfo::builder().buffer(buffers.indirect_args_buffer).range(vk::WHOLE_SIZE)];
    let ubo_info = [*vk::DescriptorBufferInfo::builder()
        .buffer(data.uniform_buffer)
        .range(std::mem::size_of::<UniformBufferObject>() as u64)];
    let atlas_info = [*vk::DescriptorImageInfo::builder()
        .image_view(atlas.image_view)
        .sampler(atlas.sampler)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];

    let writes = [
        *vk::WriteDescriptorSet::builder().dst_set(set).dst_binding(0).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&tile_info),
        *vk::WriteDescriptorSet::builder().dst_set(set).dst_binding(1).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&visible_info),
        *vk::WriteDescriptorSet::builder().dst_set(set).dst_binding(2).descriptor_type(vk::DescriptorType::STORAGE_BUFFER).buffer_info(&args_info),
        *vk::WriteDescriptorSet::builder().dst_set(set).dst_binding(3).descriptor_type(vk::DescriptorType::UNIFORM_BUFFER).buffer_info(&ubo_info),
        *vk::WriteDescriptorSet::builder().dst_set(set).dst_binding(4).descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER).image_info(&atlas_info),
    ];
    device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
}

unsafe fn create_cull_pipeline(device: &Device, layout: vk::PipelineLayout) -> anyhow::Result<vk::Pipeline> {
    let module = create_shader_module(device, include_bytes!("../shaders/heightmap_cull.comp.spv"))?;
    let stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(c"main");
    let info = vk::ComputePipelineCreateInfo::builder().stage(*stage).layout(layout);
    let pipeline = device.create_compute_pipeline(vk::PipelineCache::null(), &info, None)?;
    device.destroy_shader_module(module, None);
    Ok(pipeline)
}

unsafe fn create_tile_graphics_pipeline(
    device: &Device,
    data: &VulkanApplicationData,
    layout: vk::PipelineLayout,
) -> anyhow::Result<vk::Pipeline> {
    let task_module = create_shader_module(device, include_bytes!("../shaders/heightmap_tile.task.spv"))?;
    let mesh_module = create_shader_module(device, include_bytes!("../shaders/heightmap_tile.mesh.spv"))?;
    let frag_module = create_shader_module(device, include_bytes!("../shaders/heightmap_tile.frag.spv"))?;

    let task_stage = *vk::PipelineShaderStageCreateInfo::builder().stage(TASK_STAGE).module(task_module).name(c"main");
    let mesh_stage = *vk::PipelineShaderStageCreateInfo::builder().stage(MESH_STAGE).module(mesh_module).name(c"main");
    let frag_stage = *vk::PipelineShaderStageCreateInfo::builder().stage(vk::ShaderStageFlags::FRAGMENT).module(frag_module).name(c"main");

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
        // The mesh shader projects vertices through cube_to_sphere, so the
        // emitted triangles share the voxel mesh shader's winding inversion:
        // cull FRONT faces, front_face = CW (matches mesh_pipeline.rs).
        .cull_mode(vk::CullModeFlags::FRONT)
        .front_face(vk::FrontFace::CLOCKWISE);

    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder().rasterization_samples(vk::SampleCountFlags::_1);

    let blend_attachment = *vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(false);
    let blend_attachments = [blend_attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder().attachments(&blend_attachments);

    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        // Reverse-Z: same as the legacy heightmap pipeline.
        .depth_compare_op(vk::CompareOp::GREATER_OR_EQUAL);

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
        .layout(layout)
        // Use the load render pass so it composites over what the mesh
        // chunk pipeline drew, like the legacy heightmap pipeline does.
        .render_pass(data.render_pass_load)
        .subpass(0);

    let pipeline = device.create_graphics_pipeline(vk::PipelineCache::null(), &info, None)?;
    device.destroy_shader_module(task_module, None);
    device.destroy_shader_module(mesh_module, None);
    device.destroy_shader_module(frag_module, None);
    Ok(pipeline)
}
