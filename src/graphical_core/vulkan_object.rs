use crate::graphical_core::{
    camera::{create_uniform_buffer, destroy_uniform_buffer, update_uniform_buffer, view_projection_matrix, Camera, UniformBufferObject},
    commands::{allocate_command_buffers, create_command_pool, create_frame_buffers, create_sync_objects, record_command_buffer},
    compute_cull::{self, ComputeCullResources, CullPushConstants, DepthPyramidResources},
    depth::{create_depth_image, create_depth_pyramid, destroy_depth_image, destroy_depth_pyramid},
    descriptors,
    frustum::Frustum,
    gpu::choose_gpu,
    instance::{create_instance, create_logical_device},
    mesh_pool::MeshPool,
    palette_buffer::{create_palette_buffer, destroy_palette_buffer},
    pipeline::create_pipeline,
    render_pass::create_render_pass,
    scene::Transform,
    swapchain::{create_swapchain, create_swapchain_image_views},
    texture_mapping::{create_texture_image, destroy_textures},
    MAX_FRAMES_IN_FLIGHT,
};
use crate::voxel::{
    chunk::CHUNK_SIZE,
    meshing::{self, ChunkNeighbors},
    world::World,
};
use crate::VALIDATION_ENABLED;
use anyhow::anyhow;
use vulkanalia::{
    loader::{LibloadingLoader, LIBRARY},
    prelude::v1_0::*,
    vk::{ExtDebugUtilsExtension, KhrSurfaceExtension, KhrSwapchainExtension},
    window as vulkan_window,
};
use winit::window::Window;

#[derive(Clone, Debug, Default)]
pub struct VulkanApplicationData {
    pub surface: vk::SurfaceKHR,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
    pub physical_device: vk::PhysicalDevice,
    pub graphics_queue: vk::Queue,
    pub presentation_queue: vk::Queue,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub render_pass: vk::RenderPass,
    pub render_pass_load: vk::RenderPass,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub(crate) in_flight_fences: Vec<vk::Fence>,
    pub(crate) images_in_flight: Vec<vk::Fence>,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub texture_image: vk::Image,
    pub texture_memory: vk::DeviceMemory,
    pub texture_image_view: vk::ImageView,
    pub texture_sampler: vk::Sampler,
    pub descriptor_set: vk::DescriptorSet,
    pub uniform_buffer: vk::Buffer,
    pub uniform_buffer_memory: vk::DeviceMemory,
    pub uniform_buffer_ptr: *mut UniformBufferObject,
    pub depth_image: vk::Image,
    pub depth_image_memory: vk::DeviceMemory,
    pub depth_image_view: vk::ImageView,
    pub depth_pyramid_image: vk::Image,
    pub depth_pyramid_memory: vk::DeviceMemory,
    pub depth_pyramid_mip_views: Vec<vk::ImageView>,
    pub depth_pyramid_full_view: vk::ImageView,
    pub depth_pyramid_sampler: vk::Sampler,
    pub depth_pyramid_mip_count: u32,
    pub palette_buffer: vk::Buffer,
    pub palette_buffer_memory: vk::DeviceMemory,
    pub transform_buffer: vk::Buffer,
    pub transform_buffer_memory: vk::DeviceMemory,
    pub transform_buffer_ptr: *mut [[f32; 4]; 4],
    pub indirect_buffer: vk::Buffer,
    pub indirect_buffer_memory: vk::DeviceMemory,
    pub indirect_buffer_ptr: *mut DrawIndexedIndirectCommand,
}

const RENDER_DISTANCE: i32 = 5;
const MAX_LOADED_CHUNKS: usize = ((2 * RENDER_DISTANCE + 1) * (2 * RENDER_DISTANCE + 1)) as usize;
/// Each chunk can emit up to 6 indirect draws (one per face-direction bucket).
pub const MAX_INDIRECT_DRAWS: usize = MAX_LOADED_CHUNKS * 6;

/// Mirrors VkDrawIndexedIndirectCommand.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct DrawIndexedIndirectCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub vertex_offset: i32,
    pub first_instance: u32,
}

pub struct VulkanApplication {
    _vulkan_entry_point: Entry,
    vulkan_instance: Instance,
    vulkan_application_data: VulkanApplicationData,
    device: Device,
    frame: usize,
    pub(crate) resized: bool,
    world: World,
    mesh_pool: MeshPool,
    compute_cull: ComputeCullResources,
    depth_pyramid_pipeline: DepthPyramidResources,
    chunk_count: u32,
    depth_pyramid_needs_init: bool,
    last_player_chunk: [i32; 2],
}

impl VulkanApplication {
    pub fn world(&self) -> &World {
        &self.world
    }
}

impl VulkanApplication {
    /// Creates a fully initialized Vulkan renderer for the given window.
    ///
    /// # Safety
    /// Calls unsafe Vulkan APIs. The caller must call [`destroy_vulkan_application`]
    /// before dropping the returned value or closing the window.
    pub unsafe fn create_vulkan_application(user_window: &Window) -> anyhow::Result<Self> {
        let (entry, instance, device, mut data) = create_core_infrastructure(user_window)?;
        create_presentation_pipeline(user_window, &instance, &device, &mut data)?;
        create_resources(&device, &instance, &mut data)?;
        allocate_command_buffers(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;

        let mut world = World::new(RENDER_DISTANCE);
        world.update(0, 0);
        let mut mesh_pool = MeshPool::new();
        build_world_meshes(&world, &mut mesh_pool, &device, &instance, &mut data)?;

        let indirect_buffer_size = (MAX_INDIRECT_DRAWS * std::mem::size_of::<DrawIndexedIndirectCommand>()) as u64;
        let indirect_buffer = data.indirect_buffer;
        let compute_cull =
            compute_cull::create_compute_cull(&device, &instance, &mut data, MAX_LOADED_CHUNKS, indirect_buffer, indirect_buffer_size)?;
        let chunk_count = compute_cull::write_chunk_info(&mesh_pool, compute_cull.chunk_info_ptr);
        let depth_pyramid_pipeline = compute_cull::create_depth_pyramid_pipeline(&device, &data)?;

        Ok(Self {
            _vulkan_entry_point: entry,
            vulkan_instance: instance,
            vulkan_application_data: data,
            device,
            frame: 0,
            resized: false,
            world,
            mesh_pool,
            compute_cull,
            depth_pyramid_pipeline,
            chunk_count,
            depth_pyramid_needs_init: true,
            last_player_chunk: [0, 0],
        })
    }
}

unsafe fn create_core_infrastructure(window: &Window) -> anyhow::Result<(Entry, Instance, Device, VulkanApplicationData)> {
    let loader = LibloadingLoader::new(LIBRARY)?;
    let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
    let mut data = VulkanApplicationData::default();
    let instance = create_instance(window, &entry, &mut data)?;
    data.surface = vulkan_window::create_surface(&instance, &window, &window)?;
    choose_gpu(&instance, &mut data)?;
    let device = create_logical_device(&entry, &instance, &mut data)?;
    Ok((entry, instance, device, data))
}

unsafe fn create_presentation_pipeline(
    window: &Window,
    instance: &Instance,
    device: &Device,
    data: &mut VulkanApplicationData,
) -> anyhow::Result<()> {
    create_swapchain(window, instance, device, data)?;
    create_swapchain_image_views(device, data)?;
    create_depth_image(device, instance, data)?;
    create_depth_pyramid(device, instance, data)?;
    create_render_pass(instance, device, data)?;
    descriptors::create_layout(device, data)?;
    create_pipeline(device, data)?;
    create_frame_buffers(device, data)?;
    Ok(())
}

unsafe fn create_resources(device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    create_command_pool(instance, device, data)?;
    create_uniform_buffer(device, instance, data)?;
    create_palette_buffer(device, instance, data)?;
    create_transform_and_indirect_buffers(device, instance, data)?;

    let (texture_image, texture_memory, texture_image_view, texture_sampler) = create_texture_image(device, instance, data)?;

    let transform_buffer_size = (MAX_LOADED_CHUNKS * std::mem::size_of::<[[f32; 4]; 4]>()) as u64;
    descriptors::create_pool(device, data)?;
    let descriptor_sets = descriptors::allocate_set(device, data.descriptor_pool, data.descriptor_set_layout)?;
    let descriptor_set = descriptor_sets
        .first()
        .copied()
        .ok_or_else(|| anyhow!("Failed to allocate descriptor set"))?;
    descriptors::update_set(
        device,
        descriptor_set,
        texture_image_view,
        texture_sampler,
        data.uniform_buffer,
        data.palette_buffer,
        data.transform_buffer,
        transform_buffer_size,
    );

    data.texture_image = texture_image;
    data.texture_memory = texture_memory;
    data.texture_image_view = texture_image_view;
    data.texture_sampler = texture_sampler;
    data.descriptor_set = descriptor_set;

    Ok(())
}

/// Pre-allocates persistently mapped buffers for per-chunk model matrices (SSBO)
/// and indirect draw commands.
unsafe fn create_transform_and_indirect_buffers(device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    use crate::graphical_core::buffers::allocate_buffer;

    let host_visible = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    let transform_size = (MAX_LOADED_CHUNKS * std::mem::size_of::<[[f32; 4]; 4]>()) as u64;
    let (tb, tm, tp) = allocate_buffer::<[[f32; 4]; 4]>(transform_size, vk::BufferUsageFlags::STORAGE_BUFFER, device, instance, data, host_visible)?;
    data.transform_buffer = tb;
    data.transform_buffer_memory = tm;
    data.transform_buffer_ptr = tp;

    let indirect_size = (MAX_INDIRECT_DRAWS * std::mem::size_of::<DrawIndexedIndirectCommand>()) as u64;
    let (ib, im, ip) = allocate_buffer::<DrawIndexedIndirectCommand>(
        indirect_size,
        vk::BufferUsageFlags::INDIRECT_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        device,
        instance,
        data,
        host_visible,
    )?;
    data.indirect_buffer = ib;
    data.indirect_buffer_memory = im;
    data.indirect_buffer_ptr = ip;

    Ok(())
}

unsafe fn destroy_transform_and_indirect_buffers(device: &Device, data: &mut VulkanApplicationData) {
    device.unmap_memory(data.transform_buffer_memory);
    device.destroy_buffer(data.transform_buffer, None);
    device.free_memory(data.transform_buffer_memory, None);

    device.unmap_memory(data.indirect_buffer_memory);
    device.destroy_buffer(data.indirect_buffer, None);
    device.free_memory(data.indirect_buffer_memory, None);
}

unsafe fn build_world_meshes(
    world: &World,
    pool: &mut MeshPool,
    device: &Device,
    instance: &Instance,
    data: &mut VulkanApplicationData,
) -> anyhow::Result<u32> {
    for [cx, cz] in world.chunk_positions() {
        mesh_chunk_into_pool(world, cx, cz, pool);
    }
    pool.rebuild(device, instance, data)?;
    write_transforms_to_ssbo(pool, data);

    Ok(0) // chunk_count set after compute_cull is created
}

/// Meshes a single chunk and adds its data to the pool (CPU-side only).
fn mesh_chunk_into_pool(world: &World, cx: i32, cz: i32, pool: &mut MeshPool) {
    let chunk = match world.get_chunk(cx, cz) {
        Some(c) => c,
        None => return,
    };
    let neighbors = ChunkNeighbors {
        pos_x: world.get_chunk(cx + 1, cz),
        neg_x: world.get_chunk(cx - 1, cz),
        pos_z: world.get_chunk(cx, cz + 1),
        neg_z: world.get_chunk(cx, cz - 1),
    };

    let (vertices, bucket_indices) = meshing::mesh_chunk(chunk, &neighbors);
    if vertices.is_empty() {
        return;
    }
    pool.add_chunk([cx, cz], vertices, bucket_indices);
}

/// Writes all chunk model matrices into the persistently mapped transform SSBO.
fn write_transforms_to_ssbo(pool: &MeshPool, data: &VulkanApplicationData) {
    for &pos in pool.chunk_positions() {
        if let Some(params) = pool.draw_params(&pos) {
            let matrix = chunk_transform(pos[0], pos[1]).model_matrix().to_cols_array_2d();
            unsafe {
                std::ptr::write(data.transform_buffer_ptr.add(params.transform_index as usize), matrix);
            }
        }
    }
}

fn chunk_transform(cx: i32, cz: i32) -> Transform {
    Transform {
        position: glam::Vec3::new(cx as f32 * CHUNK_SIZE as f32, 0.0, cz as f32 * CHUNK_SIZE as f32),
        ..Default::default()
    }
}

impl VulkanApplication {
    /// Acquires a swapchain image, submits the command buffer, and presents the result.
    ///
    /// # Safety
    /// Calls unsafe Vulkan queue and synchronization APIs.
    pub unsafe fn render_frame(&mut self, window: &Window, camera: &Camera) -> anyhow::Result<()> {
        self.update_chunks(camera)?;

        let image_index = match self.acquire_next_image(window)? {
            Some(index) => index,
            None => return Ok(()), // swapchain was recreated, skip this frame
        };
        update_uniform_buffer(&self.vulkan_application_data, camera)?;

        let vp = view_projection_matrix(camera, self.vulkan_application_data.swapchain_extent);
        let frustum = Frustum::from_view_projection(&vp);
        let cull_push = CullPushConstants {
            planes: [
                frustum.plane(0),
                frustum.plane(1),
                frustum.plane(2),
                frustum.plane(3),
                frustum.plane(4),
                frustum.plane(5),
            ],
            camera_pos: camera.position.to_array(),
            chunk_count: self.chunk_count,
            screen_size: [
                self.vulkan_application_data.swapchain_extent.width as f32,
                self.vulkan_application_data.swapchain_extent.height as f32,
            ],
            phase: 0,
            draw_offset: 0,
        };

        record_command_buffer(
            &self.device,
            &self.vulkan_application_data,
            image_index,
            &self.compute_cull,
            &self.depth_pyramid_pipeline,
            &cull_push,
            self.mesh_pool.vertex_buffer,
            self.mesh_pool.index_buffer,
            self.depth_pyramid_needs_init,
        )?;
        self.depth_pyramid_needs_init = false;
        self.submit_command_buffer(image_index)?;
        self.present_frame(image_index, window)?;
        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }

    /// Checks if the player moved to a new chunk and loads/unloads accordingly.
    unsafe fn update_chunks(&mut self, camera: &Camera) -> anyhow::Result<()> {
        let player_cx = (camera.position.x / CHUNK_SIZE as f32).floor() as i32;
        let player_cz = (camera.position.z / CHUNK_SIZE as f32).floor() as i32;

        if [player_cx, player_cz] == self.last_player_chunk {
            return Ok(());
        }
        self.last_player_chunk = [player_cx, player_cz];

        let delta = self.world.update(player_cx, player_cz);
        if delta.loaded.is_empty() && delta.unloaded.is_empty() {
            return Ok(());
        }

        // Wait for GPU to finish before rebuilding the shared buffer
        self.device.device_wait_idle()?;

        for pos in &delta.unloaded {
            self.mesh_pool.remove_chunk(pos);
        }
        for &[cx, cz] in &delta.loaded {
            mesh_chunk_into_pool(&self.world, cx, cz, &mut self.mesh_pool);
        }

        self.mesh_pool
            .rebuild(&self.device, &self.vulkan_instance, &mut self.vulkan_application_data)?;
        write_transforms_to_ssbo(&self.mesh_pool, &self.vulkan_application_data);
        self.chunk_count = compute_cull::write_chunk_info(&self.mesh_pool, self.compute_cull.chunk_info_ptr);
        self.compute_cull.reset_visibility();

        Ok(())
    }

    /// Waits for the current frame's fence, then acquires the next swapchain image.
    /// Returns `None` if the swapchain was out of date and had to be recreated.
    unsafe fn acquire_next_image(&mut self, window: &Window) -> anyhow::Result<Option<usize>> {
        let data = &self.vulkan_application_data;
        self.device.wait_for_fences(&[data.in_flight_fences[self.frame]], true, u64::MAX)?;

        let result = self
            .device
            .acquire_next_image_khr(data.swapchain, u64::MAX, data.image_available_semaphores[self.frame], vk::Fence::null());
        let image_index = match result {
            Ok((index, _)) => index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => {
                self.recreate_swapchain(window)?;
                return Ok(None);
            }
            Err(e) => return Err(anyhow!(e)),
        };

        if !self.vulkan_application_data.images_in_flight[image_index].is_null() {
            self.device
                .wait_for_fences(&[self.vulkan_application_data.images_in_flight[image_index]], true, u64::MAX)?;
        }
        self.vulkan_application_data.images_in_flight[image_index] = self.vulkan_application_data.in_flight_fences[self.frame];

        Ok(Some(image_index))
    }

    unsafe fn submit_command_buffer(&self, image_index: usize) -> anyhow::Result<()> {
        let data = &self.vulkan_application_data;
        let wait_semaphores = &[data.image_available_semaphores[self.frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[data.command_buffers[image_index]];
        let signal_semaphores = &[data.render_finished_semaphores[self.frame]];
        let submit_info = vk::SubmitInfo::builder()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        self.device.reset_fences(&[data.in_flight_fences[self.frame]])?;
        self.device
            .queue_submit(data.graphics_queue, &[submit_info], data.in_flight_fences[self.frame])?;
        Ok(())
    }

    unsafe fn present_frame(&mut self, image_index: usize, window: &Window) -> anyhow::Result<()> {
        let data = &self.vulkan_application_data;
        let signal_semaphores = &[data.render_finished_semaphores[self.frame]];
        let swapchains = &[data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        self.device.queue_wait_idle(data.presentation_queue)?;
        let result = self.device.queue_present_khr(data.presentation_queue, &present_info);

        if result == Err(vk::ErrorCode::OUT_OF_DATE_KHR) {
            self.recreate_swapchain(window)?;
        }
        Ok(())
    }

    /// Destroys and rebuilds the swapchain and all dependent resources.
    ///
    /// Required when the window resizes or the swapchain becomes suboptimal,
    /// because most pipeline resources reference swapchain dimensions or format.
    ///
    /// # Safety
    /// Calls unsafe Vulkan destruction and creation APIs.
    pub unsafe fn recreate_swapchain(&mut self, user_window: &Window) -> anyhow::Result<()> {
        self.device.device_wait_idle()?;
        compute_cull::destroy_depth_pyramid_pipeline(&self.device, &self.depth_pyramid_pipeline);
        self.destroy_swapchain();
        create_swapchain(user_window, &self.vulkan_instance, &self.device, &mut self.vulkan_application_data)?;
        create_swapchain_image_views(&self.device, &mut self.vulkan_application_data)?;
        create_depth_image(&self.device, &self.vulkan_instance, &mut self.vulkan_application_data)?;
        create_depth_pyramid(&self.device, &self.vulkan_instance, &mut self.vulkan_application_data)?;
        self.depth_pyramid_pipeline = compute_cull::create_depth_pyramid_pipeline(&self.device, &self.vulkan_application_data)?;
        compute_cull::update_cull_depth_pyramid(&self.device, &self.compute_cull, &self.vulkan_application_data);
        self.depth_pyramid_needs_init = true;
        create_render_pass(&self.vulkan_instance, &self.device, &mut self.vulkan_application_data)?;
        create_pipeline(&self.device, &mut self.vulkan_application_data)?;
        create_frame_buffers(&self.device, &mut self.vulkan_application_data)?;
        allocate_command_buffers(&self.device, &mut self.vulkan_application_data)?;
        self.vulkan_application_data
            .images_in_flight
            .resize(self.vulkan_application_data.swapchain_images.len(), vk::Fence::null());
        Ok(())
    }

    /// Destroys the swapchain and all resources that depend on it.
    ///
    /// # Safety
    /// Calls unsafe Vulkan destruction APIs. The GPU must be idle before calling.
    pub unsafe fn destroy_swapchain(&mut self) {
        self.vulkan_application_data
            .framebuffers
            .iter()
            .for_each(|framebuffer| self.device.destroy_framebuffer(*framebuffer, None));
        self.device
            .free_command_buffers(self.vulkan_application_data.command_pool, &self.vulkan_application_data.command_buffers);
        self.device.destroy_pipeline(self.vulkan_application_data.pipeline, None);
        self.device.destroy_pipeline_layout(self.vulkan_application_data.pipeline_layout, None);
        self.device.destroy_render_pass(self.vulkan_application_data.render_pass, None);
        self.device.destroy_render_pass(self.vulkan_application_data.render_pass_load, None);
        destroy_depth_pyramid(&self.device, &mut self.vulkan_application_data);
        destroy_depth_image(&self.device, &mut self.vulkan_application_data);
        self.vulkan_application_data
            .swapchain_image_views
            .iter()
            .for_each(|image_view| self.device.destroy_image_view(*image_view, None));
        self.device.destroy_swapchain_khr(self.vulkan_application_data.swapchain, None);
    }

    /// Destroys all Vulkan resources. Must be called exactly once before the
    /// window closes, because `Drop` cannot guarantee the required destruction order.
    ///
    /// # Safety
    /// Calls unsafe Vulkan destruction APIs. No rendering is possible after this call.
    pub unsafe fn destroy_vulkan_application(&mut self) {
        self.device.device_wait_idle().unwrap();
        self.destroy_resources();
        self.destroy_swapchain();
        self.destroy_sync_objects();
        self.destroy_core_infrastructure();
    }

    unsafe fn destroy_resources(&mut self) {
        compute_cull::destroy_depth_pyramid_pipeline(&self.device, &self.depth_pyramid_pipeline);
        compute_cull::destroy_compute_cull(&self.device, &self.compute_cull);
        destroy_textures(&self.device, &mut self.vulkan_application_data);
        destroy_palette_buffer(&self.device, &mut self.vulkan_application_data);
        destroy_uniform_buffer(&self.device, &mut self.vulkan_application_data);
        destroy_transform_and_indirect_buffers(&self.device, &mut self.vulkan_application_data);
        self.device.destroy_descriptor_pool(self.vulkan_application_data.descriptor_pool, None);
        self.device
            .destroy_descriptor_set_layout(self.vulkan_application_data.descriptor_set_layout, None);
        self.mesh_pool.destroy(&self.device);
    }

    unsafe fn destroy_sync_objects(&self) {
        self.vulkan_application_data
            .in_flight_fences
            .iter()
            .for_each(|f| self.device.destroy_fence(*f, None));
        self.vulkan_application_data
            .render_finished_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));
        self.vulkan_application_data
            .image_available_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));
        self.device.destroy_command_pool(self.vulkan_application_data.command_pool, None);
    }

    unsafe fn destroy_core_infrastructure(&mut self) {
        self.device.destroy_device(None);
        self.vulkan_instance.destroy_surface_khr(self.vulkan_application_data.surface, None);
        if VALIDATION_ENABLED {
            self.vulkan_instance
                .destroy_debug_utils_messenger_ext(self.vulkan_application_data.debug_messenger, None);
        }
        self.vulkan_instance.destroy_instance(None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draw_indirect_command_size_matches_vulkan_spec() {
        // VkDrawIndexedIndirectCommand is 5 × u32 = 20 bytes
        assert_eq!(std::mem::size_of::<DrawIndexedIndirectCommand>(), 20);
    }

    #[test]
    fn draw_indirect_command_field_offsets_match_vulkan_spec() {
        assert_eq!(memoffset::offset_of!(DrawIndexedIndirectCommand, index_count), 0);
        assert_eq!(memoffset::offset_of!(DrawIndexedIndirectCommand, instance_count), 4);
        assert_eq!(memoffset::offset_of!(DrawIndexedIndirectCommand, first_index), 8);
        assert_eq!(memoffset::offset_of!(DrawIndexedIndirectCommand, vertex_offset), 12);
        assert_eq!(memoffset::offset_of!(DrawIndexedIndirectCommand, first_instance), 16);
    }

    #[test]
    fn max_indirect_draws_accommodates_all_chunks() {
        let max_chunks = ((2 * 5 + 1) * (2 * 5 + 1)) as usize; // RENDER_DISTANCE = 5
        assert_eq!(MAX_INDIRECT_DRAWS, max_chunks * 6);
    }
}
