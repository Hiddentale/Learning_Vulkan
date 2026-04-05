use crate::graphical_core::buffers::allocate_buffer;
use crate::graphical_core::memory::find_memory_type;
use crate::graphical_core::shaders::create_shader_module;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use vk::Handle;
use vulkan_rust::{vk, Device, Instance};

const MAX_VERTICES: u32 = 4096;
const GLYPH_W: u32 = 6;
const GLYPH_H: u32 = 10;
const ATLAS_COLS: u32 = 16;
const ATLAS_ROWS: u32 = 6;
const ATLAS_W: u32 = GLYPH_W * ATLAS_COLS;
const ATLAS_H: u32 = GLYPH_H * ATLAS_ROWS;

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct UiVertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

pub struct UiPipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    font_image: vk::Image,
    font_memory: vk::DeviceMemory,
    font_view: vk::ImageView,
    font_sampler: vk::Sampler,
    vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    vertex_ptr: *mut UiVertex,
    vertex_count: u32,
}

impl UiPipeline {
    pub unsafe fn create(device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<Self> {
        let (font_image, font_memory, font_view, font_sampler) = create_font_atlas(device, instance, data)?;
        let (descriptor_set_layout, descriptor_pool, descriptor_set) = create_descriptors(device, font_view, font_sampler)?;
        let (pipeline_layout, pipeline) = create_pipeline(device, data, descriptor_set_layout)?;
        let (vertex_buffer, vertex_memory, vertex_ptr) = allocate_buffer::<UiVertex>(
            (MAX_VERTICES as u64) * std::mem::size_of::<UiVertex>() as u64,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            device,
            instance,
            data,
            super::host_visible_coherent(),
        )?;

        Ok(Self {
            pipeline,
            pipeline_layout,
            descriptor_set_layout,
            descriptor_pool,
            descriptor_set,
            font_image,
            font_memory,
            font_view,
            font_sampler,
            vertex_buffer,
            vertex_memory,
            vertex_ptr,
            vertex_count: 0,
        })
    }

    pub fn begin_frame(&mut self) {
        self.vertex_count = 0;
    }

    /// Draw a solid colored rectangle in pixel coordinates.
    pub fn draw_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        let uv = [0.0, 0.0];
        self.push_quad(x, y, x + w, y + h, uv, uv, color);
    }

    /// Draw a string of ASCII text. `size` is the pixel height of each character.
    pub fn draw_text(&mut self, text: &str, x: f32, y: f32, size: f32, color: [f32; 4]) {
        let scale = size / GLYPH_H as f32;
        let char_w = GLYPH_W as f32 * scale;
        let mut cx = x;
        for ch in text.chars() {
            let code = ch as u32;
            if !(32..=126).contains(&code) {
                cx += char_w;
                continue;
            }
            let idx = code - 32;
            let col = idx % ATLAS_COLS;
            let row = idx / ATLAS_COLS;
            let u0 = col as f32 * GLYPH_W as f32 / ATLAS_W as f32;
            let v0 = row as f32 * GLYPH_H as f32 / ATLAS_H as f32;
            let u1 = (col + 1) as f32 * GLYPH_W as f32 / ATLAS_W as f32;
            let v1 = (row + 1) as f32 * GLYPH_H as f32 / ATLAS_H as f32;
            self.push_quad(cx, y, cx + char_w, y + size, [u0, v0], [u1, v1], color);
            cx += char_w;
        }
    }

    /// Measure the width of a text string at the given size.
    pub fn text_width(text: &str, size: f32) -> f32 {
        let scale = size / GLYPH_H as f32;
        text.len() as f32 * GLYPH_W as f32 * scale
    }

    /// Record draw commands into the given command buffer.
    /// Must be called inside an active render pass.
    pub unsafe fn record(&self, device: &Device, cmd: vk::CommandBuffer, screen_size: [f32; 2]) {
        if self.vertex_count == 0 {
            return;
        }
        device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
        device.cmd_bind_descriptor_sets(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline_layout, 0, &[self.descriptor_set], &[]);
        let push_bytes: &[u8] = std::slice::from_raw_parts(screen_size.as_ptr() as *const u8, std::mem::size_of::<[f32; 2]>());
        device.cmd_push_constants(cmd, self.pipeline_layout, vk::ShaderStageFlags::VERTEX, 0, push_bytes);
        device.cmd_bind_vertex_buffers(cmd, 0, &[self.vertex_buffer], &[0]);
        device.cmd_draw(cmd, self.vertex_count, 1, 0, 0);
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        device.unmap_memory(self.vertex_memory);
        device.destroy_buffer(self.vertex_buffer, None);
        device.free_memory(self.vertex_memory, None);
        device.destroy_pipeline(self.pipeline, None);
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_descriptor_pool(self.descriptor_pool, None);
        device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        device.destroy_sampler(self.font_sampler, None);
        device.destroy_image_view(self.font_view, None);
        device.destroy_image(self.font_image, None);
        device.free_memory(self.font_memory, None);
    }

    fn push_quad(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, uv0: [f32; 2], uv1: [f32; 2], color: [f32; 4]) {
        if self.vertex_count + 6 > MAX_VERTICES {
            return;
        }
        let verts = [
            UiVertex {
                pos: [x0, y0],
                uv: uv0,
                color,
            },
            UiVertex {
                pos: [x1, y0],
                uv: [uv1[0], uv0[1]],
                color,
            },
            UiVertex {
                pos: [x0, y1],
                uv: [uv0[0], uv1[1]],
                color,
            },
            UiVertex {
                pos: [x1, y0],
                uv: [uv1[0], uv0[1]],
                color,
            },
            UiVertex {
                pos: [x1, y1],
                uv: uv1,
                color,
            },
            UiVertex {
                pos: [x0, y1],
                uv: [uv0[0], uv1[1]],
                color,
            },
        ];
        unsafe {
            let dst = self.vertex_ptr.add(self.vertex_count as usize);
            std::ptr::copy_nonoverlapping(verts.as_ptr(), dst, 6);
        }
        self.vertex_count += 6;
    }
}

// --- Font atlas generation ---

unsafe fn create_font_atlas(
    device: &Device,
    instance: &Instance,
    data: &mut VulkanApplicationData,
) -> anyhow::Result<(vk::Image, vk::DeviceMemory, vk::ImageView, vk::Sampler)> {
    let pixels = generate_font_bitmap();

    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .format(vk::Format::R8_UNORM)
        .extent(vk::Extent3D {
            width: ATLAS_W,
            height: ATLAS_H,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::_1)
        .tiling(vk::ImageTiling::LINEAR)
        .usage(vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::PREINITIALIZED);

    let image = device.create_image(&image_info, None)?;
    let mem_reqs = device.get_image_memory_requirements(image);
    let mem_props = instance.get_physical_device_memory_properties(data.physical_device);
    let mem_type = find_memory_type(&mem_props, mem_reqs.memory_type_bits, super::host_visible_coherent())?;
    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_reqs.size)
        .memory_type_index(mem_type);
    let memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_image_memory(image, memory, 0)?;

    let ptr = device.map_memory(memory, 0, mem_reqs.size, vk::MemoryMapFlags::empty())?;
    // Get row pitch for linear tiling
    let subresource = vk::ImageSubresource::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .array_layer(0);
    let layout = device.get_image_subresource_layout(image, &subresource);
    let row_pitch = layout.row_pitch as usize;
    for y in 0..ATLAS_H as usize {
        let src_offset = y * ATLAS_W as usize;
        let dst_offset = layout.offset as usize + y * row_pitch;
        std::ptr::copy_nonoverlapping(pixels[src_offset..].as_ptr(), (ptr as *mut u8).add(dst_offset), ATLAS_W as usize);
    }
    device.unmap_memory(memory);

    // Transition to SHADER_READ_ONLY_OPTIMAL
    transition_font_image(device, data, image)?;

    let view_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(vk::Format::R8_UNORM)
        .subresource_range(super::subresource_range(vk::ImageAspectFlags::COLOR, 1));
    let view = device.create_image_view(&view_info, None)?;

    let sampler_info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::NEAREST)
        .min_filter(vk::Filter::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE);
    let sampler = device.create_sampler(&sampler_info, None)?;

    Ok((image, memory, view, sampler))
}

unsafe fn transition_font_image(device: &Device, data: &VulkanApplicationData, image: vk::Image) -> anyhow::Result<()> {
    let alloc_info = vk::CommandBufferAllocateInfo::builder()
        .command_pool(data.command_pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = device.allocate_command_buffers(&alloc_info)?[0];
    device.begin_command_buffer(
        cmd,
        &vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
    )?;

    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(vk::ImageLayout::PREINITIALIZED)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_access_mask(vk::AccessFlags::HOST_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .image(image)
        .subresource_range(super::subresource_range(vk::ImageAspectFlags::COLOR, 1));
    device.cmd_pipeline_barrier(
        cmd,
        vk::PipelineStageFlags::HOST,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[],
        &[],
        &[*barrier],
    );

    device.end_command_buffer(cmd)?;
    let cmds = [cmd];
    let submit = vk::SubmitInfo::builder().command_buffers(&cmds);
    device.queue_submit(data.graphics_queue, &[*submit], vk::Fence::null())?;
    device.queue_wait_idle(data.graphics_queue)?;
    device.free_command_buffers(data.command_pool, &[cmd]);
    Ok(())
}

// --- Descriptor set ---

unsafe fn create_descriptors(
    device: &Device,
    font_view: vk::ImageView,
    font_sampler: vk::Sampler,
) -> anyhow::Result<(vk::DescriptorSetLayout, vk::DescriptorPool, vk::DescriptorSet)> {
    let binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);
    let bindings = [*binding];
    let layout_info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(&bindings);
    let layout = device.create_descriptor_set_layout(&layout_info, None)?;

    let pool_size = *vk::DescriptorPoolSize::builder()
        .r#type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1);
    let pool_sizes = [pool_size];
    let pool_info = vk::DescriptorPoolCreateInfo::builder().pool_sizes(&pool_sizes).max_sets(1);
    let pool = device.create_descriptor_pool(&pool_info, None)?;

    let set_layouts = [layout];
    let alloc_info = vk::DescriptorSetAllocateInfo::builder().descriptor_pool(pool).set_layouts(&set_layouts);
    let set = device.allocate_descriptor_sets(&alloc_info)?[0];

    let image_info = *vk::DescriptorImageInfo::builder()
        .sampler(font_sampler)
        .image_view(font_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let image_infos = [image_info];
    let write = *vk::WriteDescriptorSet::builder()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&image_infos);
    let writes = [write];
    device.update_descriptor_sets(&writes, &[] as &[vk::CopyDescriptorSet]);

    Ok((layout, pool, set))
}

// --- Graphics pipeline ---

unsafe fn create_pipeline(
    device: &Device,
    data: &VulkanApplicationData,
    descriptor_set_layout: vk::DescriptorSetLayout,
) -> anyhow::Result<(vk::PipelineLayout, vk::Pipeline)> {
    let vert_module = create_shader_module(device, include_bytes!("../shaders/ui.vert.spv"))?;
    let frag_module = create_shader_module(device, include_bytes!("../shaders/ui.frag.spv"))?;

    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_module)
        .name(c"main");
    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_module)
        .name(c"main");

    let binding = vk::VertexInputBindingDescription::builder()
        .binding(0)
        .stride(std::mem::size_of::<UiVertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let attrs = [
        *vk::VertexInputAttributeDescription::builder()
            .location(0)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(0),
        *vk::VertexInputAttributeDescription::builder()
            .location(1)
            .format(vk::Format::R32G32_SFLOAT)
            .offset(8),
        *vk::VertexInputAttributeDescription::builder()
            .location(2)
            .format(vk::Format::R32G32B32A32_SFLOAT)
            .offset(16),
    ];
    let bindings_arr = [*binding];
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(&bindings_arr)
        .vertex_attribute_descriptions(&attrs);

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

    let blend_attachment = *vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all())
        .blend_enable(true)
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .alpha_blend_op(vk::BlendOp::ADD);
    let blend_attachments = [blend_attachment];
    let color_blend = vk::PipelineColorBlendStateCreateInfo::builder().attachments(&blend_attachments);

    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(false)
        .depth_write_enable(false);

    let push_range = *vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .size(std::mem::size_of::<[f32; 2]>() as u32);
    let push_ranges = [push_range];
    let layouts = [descriptor_set_layout];
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(&layouts)
        .push_constant_ranges(&push_ranges);
    let pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

    let stages = [*vert_stage, *frag_stage];
    let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .layout(pipeline_layout)
        .render_pass(data.render_pass)
        .subpass(0);

    let pipeline = device.create_graphics_pipeline(vk::PipelineCache::null(), &pipeline_info, None)?;

    device.destroy_shader_module(vert_module, None);
    device.destroy_shader_module(frag_module, None);

    Ok((pipeline_layout, pipeline))
}

// --- Procedural bitmap font (5x7 glyphs in 6x10 cells, ASCII 32-126) ---

fn generate_font_bitmap() -> Vec<u8> {
    let mut pixels = vec![0u8; (ATLAS_W * ATLAS_H) as usize];
    for (i, glyph) in FONT_DATA.iter().enumerate() {
        let col = i as u32 % ATLAS_COLS;
        let row = i as u32 / ATLAS_COLS;
        let base_x = col * GLYPH_W;
        let base_y = row * GLYPH_H + 1; // 1px top padding
        for (gy, &bits) in glyph.iter().enumerate() {
            for gx in 0..5u32 {
                if bits & (1 << (4 - gx)) != 0 {
                    let px = base_x + gx;
                    let py = base_y + gy as u32;
                    if px < ATLAS_W && py < ATLAS_H {
                        pixels[(py * ATLAS_W + px) as usize] = 255;
                    }
                }
            }
        }
    }
    pixels
}

/// 5x7 bitmap font data for ASCII 32-126 (95 characters).
/// Each glyph is 7 rows of 5 bits packed into u8 (MSB = leftmost pixel).
#[rustfmt::skip]
const FONT_DATA: &[[u8; 7]] = &[
    [0x00,0x00,0x00,0x00,0x00,0x00,0x00], // 32 ' '
    [0x04,0x04,0x04,0x04,0x04,0x00,0x04], // 33 '!'
    [0x0A,0x0A,0x00,0x00,0x00,0x00,0x00], // 34 '"'
    [0x0A,0x1F,0x0A,0x0A,0x1F,0x0A,0x00], // 35 '#'
    [0x04,0x0F,0x14,0x0E,0x05,0x1E,0x04], // 36 '$'
    [0x19,0x1A,0x02,0x04,0x0B,0x13,0x00], // 37 '%'
    [0x08,0x14,0x14,0x08,0x15,0x12,0x0D], // 38 '&'
    [0x04,0x04,0x00,0x00,0x00,0x00,0x00], // 39 '''
    [0x02,0x04,0x08,0x08,0x08,0x04,0x02], // 40 '('
    [0x08,0x04,0x02,0x02,0x02,0x04,0x08], // 41 ')'
    [0x00,0x04,0x15,0x0E,0x15,0x04,0x00], // 42 '*'
    [0x00,0x04,0x04,0x1F,0x04,0x04,0x00], // 43 '+'
    [0x00,0x00,0x00,0x00,0x00,0x04,0x08], // 44 ','
    [0x00,0x00,0x00,0x1F,0x00,0x00,0x00], // 45 '-'
    [0x00,0x00,0x00,0x00,0x00,0x00,0x04], // 46 '.'
    [0x01,0x02,0x02,0x04,0x08,0x08,0x10], // 47 '/'
    [0x0E,0x11,0x13,0x15,0x19,0x11,0x0E], // 48 '0'
    [0x04,0x0C,0x04,0x04,0x04,0x04,0x0E], // 49 '1'
    [0x0E,0x11,0x01,0x06,0x08,0x10,0x1F], // 50 '2'
    [0x0E,0x11,0x01,0x06,0x01,0x11,0x0E], // 51 '3'
    [0x02,0x06,0x0A,0x12,0x1F,0x02,0x02], // 52 '4'
    [0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E], // 53 '5'
    [0x06,0x08,0x10,0x1E,0x11,0x11,0x0E], // 54 '6'
    [0x1F,0x01,0x02,0x04,0x08,0x08,0x08], // 55 '7'
    [0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E], // 56 '8'
    [0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C], // 57 '9'
    [0x00,0x00,0x04,0x00,0x00,0x04,0x00], // 58 ':'
    [0x00,0x00,0x04,0x00,0x00,0x04,0x08], // 59 ';'
    [0x02,0x04,0x08,0x10,0x08,0x04,0x02], // 60 '<'
    [0x00,0x00,0x1F,0x00,0x1F,0x00,0x00], // 61 '='
    [0x08,0x04,0x02,0x01,0x02,0x04,0x08], // 62 '>'
    [0x0E,0x11,0x01,0x06,0x04,0x00,0x04], // 63 '?'
    [0x0E,0x11,0x17,0x15,0x17,0x10,0x0E], // 64 '@'
    [0x0E,0x11,0x11,0x1F,0x11,0x11,0x11], // 65 'A'
    [0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E], // 66 'B'
    [0x0E,0x11,0x10,0x10,0x10,0x11,0x0E], // 67 'C'
    [0x1E,0x11,0x11,0x11,0x11,0x11,0x1E], // 68 'D'
    [0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F], // 69 'E'
    [0x1F,0x10,0x10,0x1E,0x10,0x10,0x10], // 70 'F'
    [0x0E,0x11,0x10,0x17,0x11,0x11,0x0F], // 71 'G'
    [0x11,0x11,0x11,0x1F,0x11,0x11,0x11], // 72 'H'
    [0x0E,0x04,0x04,0x04,0x04,0x04,0x0E], // 73 'I'
    [0x07,0x02,0x02,0x02,0x02,0x12,0x0C], // 74 'J'
    [0x11,0x12,0x14,0x18,0x14,0x12,0x11], // 75 'K'
    [0x10,0x10,0x10,0x10,0x10,0x10,0x1F], // 76 'L'
    [0x11,0x1B,0x15,0x15,0x11,0x11,0x11], // 77 'M'
    [0x11,0x19,0x15,0x13,0x11,0x11,0x11], // 78 'N'
    [0x0E,0x11,0x11,0x11,0x11,0x11,0x0E], // 79 'O'
    [0x1E,0x11,0x11,0x1E,0x10,0x10,0x10], // 80 'P'
    [0x0E,0x11,0x11,0x11,0x15,0x12,0x0D], // 81 'Q'
    [0x1E,0x11,0x11,0x1E,0x14,0x12,0x11], // 82 'R'
    [0x0E,0x11,0x10,0x0E,0x01,0x11,0x0E], // 83 'S'
    [0x1F,0x04,0x04,0x04,0x04,0x04,0x04], // 84 'T'
    [0x11,0x11,0x11,0x11,0x11,0x11,0x0E], // 85 'U'
    [0x11,0x11,0x11,0x11,0x0A,0x0A,0x04], // 86 'V'
    [0x11,0x11,0x11,0x15,0x15,0x1B,0x11], // 87 'W'
    [0x11,0x11,0x0A,0x04,0x0A,0x11,0x11], // 88 'X'
    [0x11,0x11,0x0A,0x04,0x04,0x04,0x04], // 89 'Y'
    [0x1F,0x01,0x02,0x04,0x08,0x10,0x1F], // 90 'Z'
    [0x0E,0x08,0x08,0x08,0x08,0x08,0x0E], // 91 '['
    [0x10,0x08,0x08,0x04,0x02,0x02,0x01], // 92 '\'
    [0x0E,0x02,0x02,0x02,0x02,0x02,0x0E], // 93 ']'
    [0x04,0x0A,0x11,0x00,0x00,0x00,0x00], // 94 '^'
    [0x00,0x00,0x00,0x00,0x00,0x00,0x1F], // 95 '_'
    [0x08,0x04,0x00,0x00,0x00,0x00,0x00], // 96 '`'
    [0x00,0x00,0x0E,0x01,0x0F,0x11,0x0F], // 97 'a'
    [0x10,0x10,0x1E,0x11,0x11,0x11,0x1E], // 98 'b'
    [0x00,0x00,0x0E,0x11,0x10,0x11,0x0E], // 99 'c'
    [0x01,0x01,0x0F,0x11,0x11,0x11,0x0F], // 100 'd'
    [0x00,0x00,0x0E,0x11,0x1F,0x10,0x0E], // 101 'e'
    [0x06,0x08,0x1E,0x08,0x08,0x08,0x08], // 102 'f'
    [0x00,0x00,0x0F,0x11,0x0F,0x01,0x0E], // 103 'g'
    [0x10,0x10,0x1E,0x11,0x11,0x11,0x11], // 104 'h'
    [0x04,0x00,0x0C,0x04,0x04,0x04,0x0E], // 105 'i'
    [0x02,0x00,0x06,0x02,0x02,0x12,0x0C], // 106 'j'
    [0x10,0x10,0x12,0x14,0x18,0x14,0x12], // 107 'k'
    [0x0C,0x04,0x04,0x04,0x04,0x04,0x0E], // 108 'l'
    [0x00,0x00,0x1A,0x15,0x15,0x11,0x11], // 109 'm'
    [0x00,0x00,0x1E,0x11,0x11,0x11,0x11], // 110 'n'
    [0x00,0x00,0x0E,0x11,0x11,0x11,0x0E], // 111 'o'
    [0x00,0x00,0x1E,0x11,0x1E,0x10,0x10], // 112 'p'
    [0x00,0x00,0x0F,0x11,0x0F,0x01,0x01], // 113 'q'
    [0x00,0x00,0x16,0x19,0x10,0x10,0x10], // 114 'r'
    [0x00,0x00,0x0F,0x10,0x0E,0x01,0x1E], // 115 's'
    [0x08,0x08,0x1E,0x08,0x08,0x09,0x06], // 116 't'
    [0x00,0x00,0x11,0x11,0x11,0x11,0x0F], // 117 'u'
    [0x00,0x00,0x11,0x11,0x11,0x0A,0x04], // 118 'v'
    [0x00,0x00,0x11,0x11,0x15,0x15,0x0A], // 119 'w'
    [0x00,0x00,0x11,0x0A,0x04,0x0A,0x11], // 120 'x'
    [0x00,0x00,0x11,0x11,0x0F,0x01,0x0E], // 121 'y'
    [0x00,0x00,0x1F,0x02,0x04,0x08,0x1F], // 122 'z'
    [0x02,0x04,0x04,0x08,0x04,0x04,0x02], // 123 '{'
    [0x04,0x04,0x04,0x04,0x04,0x04,0x04], // 124 '|'
    [0x08,0x04,0x04,0x02,0x04,0x04,0x08], // 125 '}'
    [0x00,0x00,0x08,0x15,0x02,0x00,0x00], // 126 '~'
];
