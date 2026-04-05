#![allow(dead_code)]

use crate::graphical_core::buffers::allocate_buffer;
use crate::graphical_core::memory::find_memory_type;
use crate::graphical_core::shaders::create_shader_module;
use crate::graphical_core::svdag_pool::SvdagPool;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use vk::Handle;
use vulkan_rust::{vk, Device, Instance};

const TILE_SIZE: u32 = 8;
const MAX_CHUNKS_PER_TILE: u32 = 64;
const TILE_STRIDE: u32 = MAX_CHUNKS_PER_TILE + 1;
const MAX_VISIBLE_CHUNKS: u32 = 8192;

// --- Push constant structs ---

#[repr(C)]
#[derive(Copy, Clone)]
pub struct CullPush {
    pub planes: [[f32; 4]; 6],
    pub total_chunks: u32,
    pub _padding: [u32; 3],
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct TileAssignPush {
    pub view_projection: [[f32; 4]; 4],
    pub screen_size: [u32; 2],
    pub tile_count: [u32; 2],
    pub camera_pos: [f32; 3],
    pub _padding: u32,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct RaymarchPush {
    pub camera_pos: [f32; 3],
    pub _padding: u32,
    pub screen_size: [u32; 2],
    pub tile_count: [u32; 2],
}

/// The full Aokana-style 3-pass compute pipeline + composite fragment pass.
pub struct SvdagPipeline {
    // Pass 1: frustum cull
    pub cull_pipeline: vk::Pipeline,
    pub cull_layout: vk::PipelineLayout,
    pub cull_desc_layout: vk::DescriptorSetLayout,
    pub cull_desc_set: vk::DescriptorSet,

    // Pass 2: tile assignment
    pub tile_pipeline: vk::Pipeline,
    pub tile_layout: vk::PipelineLayout,
    pub tile_desc_layout: vk::DescriptorSetLayout,
    pub tile_desc_set: vk::DescriptorSet,

    // Pass 3: ray march
    pub march_pipeline: vk::Pipeline,
    pub march_layout: vk::PipelineLayout,
    pub march_desc_layout: vk::DescriptorSetLayout,
    pub march_desc_set: vk::DescriptorSet,

    // Composite: fragment pass
    pub composite_pipeline: vk::Pipeline,
    pub composite_layout: vk::PipelineLayout,
    pub composite_desc_layout: vk::DescriptorSetLayout,
    pub composite_desc_set: vk::DescriptorSet,

    // Shared descriptor pool
    pub descriptor_pool: vk::DescriptorPool,

    // Intermediate SSBOs
    pub visible_buffer: vk::Buffer,
    pub visible_memory: vk::DeviceMemory,
    pub visible_count_buffer: vk::Buffer,
    pub visible_count_memory: vk::DeviceMemory,
    pub visible_count_ptr: *mut u32,
    pub tile_buffer: vk::Buffer,
    pub tile_memory: vk::DeviceMemory,

    // Output images
    pub color_image: vk::Image,
    pub color_memory: vk::DeviceMemory,
    pub color_view: vk::ImageView,
    pub depth_image: vk::Image,
    pub depth_mem: vk::DeviceMemory,
    pub depth_view: vk::ImageView,
    pub sampler: vk::Sampler,

    pub tile_count: [u32; 2],
}

impl SvdagPipeline {
    pub unsafe fn create(device: &Device, instance: &Instance, data: &mut VulkanApplicationData, svdag_pool: &SvdagPool) -> anyhow::Result<Self> {
        let extent = data.swapchain_extent;
        let tile_count = [extent.width.div_ceil(TILE_SIZE), extent.height.div_ceil(TILE_SIZE)];
        let total_tiles = tile_count[0] * tile_count[1];
        let host_visible = super::host_visible_coherent();
        let ssbo = vk::BufferUsageFlags::STORAGE_BUFFER;

        // --- Intermediate SSBOs ---
        let vis_size = (MAX_VISIBLE_CHUNKS as u64) * 4;
        let (visible_buffer, visible_memory, _vis_ptr) = allocate_buffer::<u32>(vis_size, ssbo, device, instance, data, host_visible)?;
        let (visible_count_buffer, visible_count_memory, visible_count_ptr) =
            allocate_buffer::<u32>(4, ssbo | vk::BufferUsageFlags::TRANSFER_DST, device, instance, data, host_visible)?;
        let tile_size = (total_tiles as u64) * (TILE_STRIDE as u64) * 4;
        let (tile_buffer, tile_memory, _tile_ptr) = allocate_buffer::<u32>(tile_size, ssbo, device, instance, data, host_visible)?;

        // --- Output images ---
        let (color_image, color_memory, color_view) = create_output_image(device, instance, data, extent, vk::Format::R8G8B8A8_UNORM)?;
        let (depth_image, depth_mem, depth_view) = create_output_image(device, instance, data, extent, vk::Format::R32_SFLOAT)?;

        let sampler_info = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        let sampler = device.create_sampler(&sampler_info, None)?;

        // --- Descriptor pool (4 sets) ---
        let pool_sizes = [
            *vk::DescriptorPoolSize::builder()
                .descriptor_count(3)
                .r#type(vk::DescriptorType::UNIFORM_BUFFER),
            *vk::DescriptorPoolSize::builder()
                .descriptor_count(12)
                .r#type(vk::DescriptorType::STORAGE_BUFFER),
            *vk::DescriptorPoolSize::builder()
                .descriptor_count(4)
                .r#type(vk::DescriptorType::STORAGE_IMAGE),
            *vk::DescriptorPoolSize::builder()
                .descriptor_count(4)
                .r#type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER),
        ];
        let pool_info = vk::DescriptorPoolCreateInfo::builder().max_sets(4).pool_sizes(&pool_sizes);
        let descriptor_pool = device.create_descriptor_pool(&pool_info, None)?;

        // --- Pass 1: Cull ---
        let cull_desc_layout = create_desc_layout(
            device,
            &[
                (vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE), // chunk info
                (vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE), // visible indices
                (vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE), // visible count
            ],
        )?;
        let cull_desc_set = alloc_desc_set(device, descriptor_pool, cull_desc_layout)?;
        write_ssbo_descriptors(
            device,
            cull_desc_set,
            &[(0, svdag_pool.chunk_info_buffer), (1, visible_buffer), (2, visible_count_buffer)],
        );
        let cull_layout = create_pipeline_layout(device, cull_desc_layout, std::mem::size_of::<CullPush>() as u32)?;
        let cull_pipeline = create_compute_pipeline(device, cull_layout, include_bytes!("../shaders/svdag_cull.comp.spv"))?;

        // --- Pass 2: Tile assign ---
        let tile_desc_layout = create_desc_layout(
            device,
            &[
                (vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE), // chunk info
                (vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE), // visible indices
                (vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE), // visible count
                (vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE), // tile data
                (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::COMPUTE), // depth pyramid
            ],
        )?;
        let tile_desc_set = alloc_desc_set(device, descriptor_pool, tile_desc_layout)?;
        write_ssbo_descriptors(
            device,
            tile_desc_set,
            &[
                (0, svdag_pool.chunk_info_buffer),
                (1, visible_buffer),
                (2, visible_count_buffer),
                (3, tile_buffer),
            ],
        );
        // Binding 4: depth pyramid sampler
        {
            let pyramid_info = [*vk::DescriptorImageInfo::builder()
                .image_view(data.depth_pyramid_full_view)
                .sampler(data.depth_pyramid_sampler)
                .image_layout(vk::ImageLayout::GENERAL)];
            let write = [*vk::WriteDescriptorSet::builder()
                .dst_set(tile_desc_set)
                .dst_binding(4)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&pyramid_info)];
            device.update_descriptor_sets(&write, &[] as &[vk::CopyDescriptorSet]);
        }
        let tile_layout = create_pipeline_layout(device, tile_desc_layout, std::mem::size_of::<TileAssignPush>() as u32)?;
        let tile_pipeline = create_compute_pipeline(device, tile_layout, include_bytes!("../shaders/svdag_tile_assign.comp.spv"))?;

        // --- Pass 3: Ray march ---
        let march_desc_layout = create_desc_layout(
            device,
            &[
                (vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::COMPUTE), // camera UBO
                (vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE), // geometry
                (vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE), // chunk info
                (vk::DescriptorType::STORAGE_BUFFER, vk::ShaderStageFlags::COMPUTE), // tile data
                (vk::DescriptorType::STORAGE_IMAGE, vk::ShaderStageFlags::COMPUTE),  // color output
                (vk::DescriptorType::STORAGE_IMAGE, vk::ShaderStageFlags::COMPUTE),  // depth output
                (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::COMPUTE), // texture array
                (vk::DescriptorType::UNIFORM_BUFFER, vk::ShaderStageFlags::COMPUTE), // material palette
            ],
        )?;
        let march_desc_set = alloc_desc_set(device, descriptor_pool, march_desc_layout)?;
        {
            // Write buffer descriptors (binding 0 is UBO, rest are SSBOs)
            write_buffer_descriptor(device, march_desc_set, 0, data.uniform_buffer, vk::DescriptorType::UNIFORM_BUFFER);
            write_ssbo_descriptors(
                device,
                march_desc_set,
                &[(1, svdag_pool.geometry_buffer), (2, svdag_pool.chunk_info_buffer), (3, tile_buffer)],
            );
            // Write image descriptors
            let color_info = [*vk::DescriptorImageInfo::builder()
                .image_view(color_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let depth_info = [*vk::DescriptorImageInfo::builder()
                .image_view(depth_view)
                .image_layout(vk::ImageLayout::GENERAL)];
            let writes = [
                *vk::WriteDescriptorSet::builder()
                    .dst_set(march_desc_set)
                    .dst_binding(4)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&color_info),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(march_desc_set)
                    .dst_binding(5)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&depth_info),
            ];
            device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);

            // Binding 6: texture array sampler
            let tex_info = [*vk::DescriptorImageInfo::builder()
                .image_view(data.texture_image_view)
                .sampler(data.texture_sampler)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            // Binding 7: material palette UBO
            let palette_info = [*vk::DescriptorBufferInfo::builder()
                .buffer(data.palette_buffer)
                .offset(0)
                .range(vk::WHOLE_SIZE)];
            let writes2 = [
                *vk::WriteDescriptorSet::builder()
                    .dst_set(march_desc_set)
                    .dst_binding(6)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&tex_info),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(march_desc_set)
                    .dst_binding(7)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(&palette_info),
            ];
            device.update_descriptor_sets(&writes2, &[] as &[vk::CopyDescriptorSet]);
        }
        let march_layout = create_pipeline_layout(device, march_desc_layout, std::mem::size_of::<RaymarchPush>() as u32)?;
        let march_pipeline = create_compute_pipeline(device, march_layout, include_bytes!("../shaders/svdag_raymarch.comp.spv"))?;

        // --- Composite: fragment pass ---
        let composite_desc_layout = create_desc_layout(
            device,
            &[
                (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT), // color
                (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, vk::ShaderStageFlags::FRAGMENT), // depth
            ],
        )?;
        let composite_desc_set = alloc_desc_set(device, descriptor_pool, composite_desc_layout)?;
        {
            let color_info = [*vk::DescriptorImageInfo::builder()
                .sampler(sampler)
                .image_view(color_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let depth_info = [*vk::DescriptorImageInfo::builder()
                .sampler(sampler)
                .image_view(depth_view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)];
            let writes = [
                *vk::WriteDescriptorSet::builder()
                    .dst_set(composite_desc_set)
                    .dst_binding(0)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&color_info),
                *vk::WriteDescriptorSet::builder()
                    .dst_set(composite_desc_set)
                    .dst_binding(1)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .image_info(&depth_info),
            ];
            device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);
        }
        let composite_layout = create_pipeline_layout(device, composite_desc_layout, 0)?;
        let composite_pipeline = create_composite_pipeline(device, data, composite_layout)?;

        Ok(Self {
            cull_pipeline,
            cull_layout,
            cull_desc_layout,
            cull_desc_set,
            tile_pipeline,
            tile_layout,
            tile_desc_layout,
            tile_desc_set,
            march_pipeline,
            march_layout,
            march_desc_layout,
            march_desc_set,
            composite_pipeline,
            composite_layout,
            composite_desc_layout,
            composite_desc_set,
            descriptor_pool,
            visible_buffer,
            visible_memory,
            visible_count_buffer,
            visible_count_memory,
            visible_count_ptr,
            tile_buffer,
            tile_memory,
            color_image,
            color_memory,
            color_view,
            depth_image,
            depth_mem,
            depth_view,
            sampler,
            tile_count,
        })
    }

    /// Re-write the depth pyramid descriptor after swapchain recreation.
    pub unsafe fn update_depth_pyramid(&self, device: &Device, data: &VulkanApplicationData) {
        let pyramid_info = [*vk::DescriptorImageInfo::builder()
            .image_view(data.depth_pyramid_full_view)
            .sampler(data.depth_pyramid_sampler)
            .image_layout(vk::ImageLayout::GENERAL)];
        let write = [*vk::WriteDescriptorSet::builder()
            .dst_set(self.tile_desc_set)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(&pyramid_info)];
        device.update_descriptor_sets(&write, &[] as &[vk::CopyDescriptorSet]);
    }

    pub unsafe fn destroy(&self, device: &Device) {
        device.destroy_pipeline(self.cull_pipeline, None);
        device.destroy_pipeline_layout(self.cull_layout, None);
        device.destroy_descriptor_set_layout(self.cull_desc_layout, None);

        device.destroy_pipeline(self.tile_pipeline, None);
        device.destroy_pipeline_layout(self.tile_layout, None);
        device.destroy_descriptor_set_layout(self.tile_desc_layout, None);

        device.destroy_pipeline(self.march_pipeline, None);
        device.destroy_pipeline_layout(self.march_layout, None);
        device.destroy_descriptor_set_layout(self.march_desc_layout, None);

        device.destroy_pipeline(self.composite_pipeline, None);
        device.destroy_pipeline_layout(self.composite_layout, None);
        device.destroy_descriptor_set_layout(self.composite_desc_layout, None);

        device.destroy_descriptor_pool(self.descriptor_pool, None);

        device.unmap_memory(self.visible_memory);
        device.destroy_buffer(self.visible_buffer, None);
        device.free_memory(self.visible_memory, None);

        device.unmap_memory(self.visible_count_memory);
        device.destroy_buffer(self.visible_count_buffer, None);
        device.free_memory(self.visible_count_memory, None);

        device.unmap_memory(self.tile_memory);
        device.destroy_buffer(self.tile_buffer, None);
        device.free_memory(self.tile_memory, None);

        device.destroy_image_view(self.color_view, None);
        device.destroy_image(self.color_image, None);
        device.free_memory(self.color_memory, None);

        device.destroy_image_view(self.depth_view, None);
        device.destroy_image(self.depth_image, None);
        device.free_memory(self.depth_mem, None);

        device.destroy_sampler(self.sampler, None);
    }
}

// --- Helpers ---

unsafe fn create_output_image(
    device: &Device,
    instance: &Instance,
    data: &mut VulkanApplicationData,
    extent: vk::Extent2D,
    format: vk::Format,
) -> anyhow::Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .format(format)
        .extent(vk::Extent3D {
            width: extent.width,
            height: extent.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = device.create_image(&info, None)?;

    let req = device.get_image_memory_requirements(image);
    let props = instance.get_physical_device_memory_properties(data.physical_device);
    let idx = find_memory_type(&props, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    let alloc = vk::MemoryAllocateInfo::builder().allocation_size(req.size).memory_type_index(idx);
    let memory = device.allocate_memory(&alloc, None)?;
    device.bind_image_memory(image, memory, 0)?;

    let view_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(format)
        .subresource_range(super::subresource_range(vk::ImageAspectFlags::COLOR, 1));
    let view = device.create_image_view(&view_info, None)?;

    Ok((image, memory, view))
}

unsafe fn create_desc_layout(device: &Device, bindings: &[(vk::DescriptorType, vk::ShaderStageFlags)]) -> anyhow::Result<vk::DescriptorSetLayout> {
    let layout_bindings: Vec<_> = bindings
        .iter()
        .enumerate()
        .map(|(i, (ty, stage))| {
            *vk::DescriptorSetLayoutBinding::builder()
                .binding(i as u32)
                .descriptor_count(1)
                .descriptor_type(*ty)
                .stage_flags(*stage)
        })
        .collect();
    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&layout_bindings);
    Ok(device.create_descriptor_set_layout(&info, None)?)
}

unsafe fn alloc_desc_set(device: &Device, pool: vk::DescriptorPool, layout: vk::DescriptorSetLayout) -> anyhow::Result<vk::DescriptorSet> {
    let layouts = [layout];
    let info = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(pool).set_layouts(&layouts);
    Ok(device.allocate_descriptor_sets(&info)?[0])
}

unsafe fn write_buffer_descriptor(device: &Device, set: vk::DescriptorSet, binding: u32, buffer: vk::Buffer, desc_type: vk::DescriptorType) {
    let buf_info = [*vk::DescriptorBufferInfo::builder().buffer(buffer).range(vk::WHOLE_SIZE)];
    let write = [*vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(binding)
        .descriptor_type(desc_type)
        .buffer_info(&buf_info)];
    device.update_descriptor_sets(&write, &[] as &[vk::CopyDescriptorSet]);
}

unsafe fn write_ssbo_descriptors(device: &Device, set: vk::DescriptorSet, bindings: &[(u32, vk::Buffer)]) {
    for &(binding, buffer) in bindings {
        write_buffer_descriptor(device, set, binding, buffer, vk::DescriptorType::STORAGE_BUFFER);
    }
}

unsafe fn create_pipeline_layout(device: &Device, desc_layout: vk::DescriptorSetLayout, push_size: u32) -> anyhow::Result<vk::PipelineLayout> {
    let layouts = [desc_layout];
    let mut builder = vk::PipelineLayoutCreateInfo::builder().set_layouts(&layouts);
    let push_range;
    if push_size > 0 {
        push_range = [*vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(push_size)];
        builder = builder.push_constant_ranges(&push_range);
    }
    Ok(device.create_pipeline_layout(&builder, None)?)
}

unsafe fn create_compute_pipeline(device: &Device, layout: vk::PipelineLayout, spv: &[u8]) -> anyhow::Result<vk::Pipeline> {
    let module = create_shader_module(device, spv)?;
    let stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::COMPUTE)
        .module(module)
        .name(c"main");
    let info = vk::ComputePipelineCreateInfo::builder().stage(*stage).layout(layout);
    let pipeline = device.create_compute_pipeline(vk::PipelineCache::null(), &info, None)?;
    device.destroy_shader_module(module, None);
    Ok(pipeline)
}

unsafe fn create_composite_pipeline(device: &Device, data: &VulkanApplicationData, layout: vk::PipelineLayout) -> anyhow::Result<vk::Pipeline> {
    let vert = create_shader_module(device, include_bytes!("../shaders/sky.vert.spv"))?;
    let frag = create_shader_module(device, include_bytes!("../shaders/svdag_composite.frag.spv"))?;

    let stages = [
        *vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert)
            .name(c"main"),
        *vk::PipelineShaderStageCreateInfo::builder()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag)
            .name(c"main"),
    ];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::builder().topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let viewport = *vk::Viewport::builder()
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
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
    let blend_att = *vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);
    let blend_atts = [blend_att];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::builder().attachments(&blend_atts);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS);

    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .layout(layout)
        .render_pass(data.render_pass_load)
        .subpass(0);
    let pipeline = device.create_graphics_pipeline(vk::PipelineCache::null(), &info, None)?;

    device.destroy_shader_module(vert, None);
    device.destroy_shader_module(frag, None);
    Ok(pipeline)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_constant_sizes() {
        assert_eq!(std::mem::size_of::<CullPush>(), 112);
        assert_eq!(std::mem::size_of::<TileAssignPush>(), 96);
        assert_eq!(std::mem::size_of::<RaymarchPush>(), 32);
    }
}
