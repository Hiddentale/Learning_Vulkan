use crate::graphical_core::{
    camera::{create_uniform_buffer, destroy_uniform_buffer, update_uniform_buffer, Camera, EyeMatrices, UniformBufferObject},
    commands::{allocate_command_buffers, create_command_pool, create_frame_buffers, create_sync_objects, record_mesh_shader_command_buffer},
    compute_cull::{CullPushConstants, DepthPyramidResources},
    depth::{create_depth_image, create_depth_pyramid, destroy_depth_image, destroy_depth_pyramid},
    descriptors,
    frustum::Frustum,
    gpu::choose_gpu,
    instance::{create_instance, create_logical_device},
    mesh_pipeline::MeshShaderPipeline,
    palette_buffer::{create_palette_buffer, destroy_palette_buffer},
    pipeline::create_sky_pipeline,
    render_pass::create_render_pass,
    svdag_pipeline::{CullPush, RaymarchPush, SvdagPipeline, TileAssignPush},
    svdag_pool::SvdagPool,
    swapchain::{create_swapchain, create_swapchain_image_views},
    texture_mapping::{create_texture_image, destroy_textures},
    voxel_pool::VoxelPool,
    MAX_FRAMES_IN_FLIGHT,
};
use crate::voxel::chunk::CHUNK_SIZE;
use crate::voxel::svdag_compressor::SvdagCompressor;
use crate::voxel::world::World;
use crate::vr::{VrContext, VrSession, VrSwapchain};
use crate::VALIDATION_ENABLED;
use anyhow::anyhow;
use log::info;
use vk::Handle;
use vulkan_rust::{vk, Device, Entry, Instance, LibloadingLoader};
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
    pub sky_pipeline: vk::Pipeline,
    pub sky_pipeline_layout: vk::PipelineLayout,
}

use crate::voxel::world::{MAX_CHUNK_Y, MIN_CHUNK_Y};

/// Mesh shader renders chunks within this distance. Full detail, editable, textured.
const MESH_DISTANCE: i32 = 8;
/// SVDAG ray march renders chunks from MESH_DISTANCE to SVDAG_DISTANCE.
const SVDAG_DISTANCE: i32 = 24;
/// World generates terrain out to this distance.
const WORLD_DISTANCE: i32 = SVDAG_DISTANCE;
const CHUNK_LAYERS: usize = (MAX_CHUNK_Y - MIN_CHUNK_Y + 1) as usize;
const MAX_MESH_CHUNKS: usize = ((2 * MESH_DISTANCE + 1) * (2 * MESH_DISTANCE + 1)) as usize * CHUNK_LAYERS;
const MAX_SVDAG_CHUNKS: u32 = 16384;

pub struct VulkanApplication {
    _vulkan_entry_point: Entry,
    vulkan_instance: Instance,
    vulkan_application_data: VulkanApplicationData,
    device: Device,
    frame: usize,
    pub(crate) resized: bool,
    world: World,
    depth_pyramid_pipeline: DepthPyramidResources,
    depth_pyramid_needs_init: bool,
    last_player_chunk: [i32; 2],
    _vr_session: Option<VrSession>,
    _vr_swapchain: Option<VrSwapchain>,
    voxel_pool: VoxelPool,
    mesh_shader_pipeline: MeshShaderPipeline,
    svdag_pool: SvdagPool,
    svdag_pipeline: SvdagPipeline,
    svdag_compressor: SvdagCompressor,
    /// Far chunks awaiting SVDAG compression. Drained in distance order each frame.
    svdag_pending: std::collections::HashSet<[i32; 3]>,
}

impl VulkanApplication {
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Sets a block in the world and re-uploads the affected chunk to the GPU.
    /// Sets a block in the world and re-uploads to the mesh shader pool.
    /// Only near-field chunks (in VoxelPool) are editable.
    pub unsafe fn set_block(&mut self, wx: i32, wy: i32, wz: i32, block: crate::voxel::block::BlockType) {
        if !self.world.set_block(wx, wy, wz, block) {
            return;
        }
        let chunk_pos = World::block_to_chunk(wx, wy, wz);
        let [cx, cy, cz] = chunk_pos;
        if let Some(chunk) = self.world.get_chunk(cx, cy, cz) {
            self.voxel_pool.reupload_chunk(chunk_pos, chunk, &self.world);
            self.voxel_pool.invalidate_neighbor_boundaries(chunk_pos, &self.world);
        }
    }

    pub fn swapchain_extent(&self) -> vk::Extent2D {
        self.vulkan_application_data.swapchain_extent
    }

    pub fn has_vr(&self) -> bool {
        self._vr_session.is_some()
    }

    /// Poll OpenXR events and handle session state transitions.
    /// Returns `false` if the VR session should be abandoned.
    pub fn poll_vr_events(&mut self) -> anyhow::Result<bool> {
        match &mut self._vr_session {
            Some(session) => session.poll_events(),
            None => Ok(true),
        }
    }

    /// Run one VR frame. Returns eye matrices if the headset rendered.
    ///
    /// # Safety
    /// Calls unsafe Vulkan and OpenXR APIs.
    pub unsafe fn render_vr_frame(&mut self) -> anyhow::Result<Option<EyeMatrices>> {
        let (Some(session), Some(swapchain)) = (&mut self._vr_session, &mut self._vr_swapchain) else {
            return Ok(None);
        };
        crate::vr::frame::render_vr_frame(
            &self.device,
            &self.vulkan_application_data,
            session,
            swapchain,
            &self.mesh_shader_pipeline,
            &self.voxel_pool,
        )
    }
}

impl VulkanApplication {
    /// Creates a fully initialized Vulkan renderer for the given window.
    ///
    /// # Safety
    /// Calls unsafe Vulkan APIs. The caller must call [`destroy_vulkan_application`]
    /// before dropping the returned value or closing the window.
    pub unsafe fn create_vulkan_application(user_window: &Window, vr_context: Option<VrContext>) -> anyhow::Result<Self> {
        let CoreInfrastructure {
            entry,
            instance,
            device,
            mut data,
            vr_session,
            vr_swapchain,
        } = create_core_infrastructure(user_window, vr_context)?;
        create_presentation_pipeline(user_window, &instance, &device, &mut data)?;
        create_resources(&device, &instance, &mut data)?;
        allocate_command_buffers(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;

        let mut world = World::new(WORLD_DISTANCE);
        world.update(0, 0);

        let depth_pyramid_pipeline = crate::graphical_core::compute_cull::create_depth_pyramid_pipeline(&device, &data)?;

        let mut voxel_pool = VoxelPool::new(MAX_MESH_CHUNKS as u32, &device, &instance, &mut data)?;
        for pos in world.chunk_positions() {
            let [cx, cy, cz] = pos;
            if crate::voxel::world::chunk_distance(cx, cz, 0, 0) <= MESH_DISTANCE {
                if let Some(chunk) = world.get_chunk(cx, cy, cz) {
                    voxel_pool.upload_chunk(pos, chunk, &world);
                }
            }
        }
        let mesh_shader_pipeline = MeshShaderPipeline::create(&device, &data, &voxel_pool)?;

        let svdag_pool = SvdagPool::new(MAX_SVDAG_CHUNKS, &device, &instance, &mut data)?;
        let svdag_pipeline = SvdagPipeline::create(&device, &instance, &mut data, &svdag_pool)?;
        let svdag_compressor = SvdagCompressor::new();

        Ok(Self {
            _vulkan_entry_point: entry,
            vulkan_instance: instance,
            vulkan_application_data: data,
            device,
            frame: 0,
            resized: false,
            world,
            depth_pyramid_pipeline,
            depth_pyramid_needs_init: true,
            last_player_chunk: [0, 0],
            _vr_session: vr_session,
            _vr_swapchain: vr_swapchain,
            voxel_pool,
            mesh_shader_pipeline,
            svdag_pool,
            svdag_pipeline,
            svdag_compressor,
            svdag_pending: std::collections::HashSet::new(),
        })
    }
}

struct CoreInfrastructure {
    entry: Entry,
    instance: Instance,
    device: Device,
    data: VulkanApplicationData,
    vr_session: Option<VrSession>,
    vr_swapchain: Option<VrSwapchain>,
}

unsafe fn create_core_infrastructure(window: &Window, vr_context: Option<VrContext>) -> anyhow::Result<CoreInfrastructure> {
    let loader = LibloadingLoader::new().map_err(|e| anyhow!("{}", e))?;
    let entry = unsafe { Entry::new(loader) }.map_err(|b| anyhow!("{}", b))?;
    let mut data = VulkanApplicationData::default();
    let instance = create_instance(window, &entry, &mut data, vr_context.as_ref())?;
    data.surface = instance.create_surface(&window, &window, None).map_err(|e| anyhow!("{}", e))?;

    let vr_preferred_gpu = match &vr_context {
        Some(vr) => match vr.preferred_gpu(instance.handle()) {
            Ok(gpu) => {
                info!("OpenXR preferred GPU identified");
                Some(gpu)
            }
            Err(e) => {
                log::warn!("Could not query VR preferred GPU: {e:#} — falling back");
                None
            }
        },
        None => None,
    };

    choose_gpu(&instance, &mut data, vr_preferred_gpu)?;
    let device = create_logical_device(&entry, &instance, &mut data, vr_context.as_ref())?;

    let vr_session = match vr_context {
        Some(vr) => {
            let indices = crate::graphical_core::queue_families::RequiredQueueFamilies::get(&instance, &data, data.physical_device)?;
            let session = vr.create_session(instance.handle(), data.physical_device, device.handle(), indices.graphics_queue_index)?;
            Some(session)
        }
        None => None,
    };

    let vr_swapchain = match &vr_session {
        Some(session) => Some(VrSwapchain::create(session, &device, &instance, data.physical_device, data.command_pool)?),
        None => None,
    };

    Ok(CoreInfrastructure {
        entry,
        instance,
        device,
        data,
        vr_session,
        vr_swapchain,
    })
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
    create_sky_pipeline(device, data)?;
    create_frame_buffers(device, data)?;
    Ok(())
}

unsafe fn create_resources(device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    create_command_pool(instance, device, data)?;
    create_uniform_buffer(device, instance, data)?;
    create_palette_buffer(device, instance, data)?;

    let (texture_image, texture_memory, texture_image_view, texture_sampler) = create_texture_image(device, instance, data)?;

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
    );

    data.texture_image = texture_image;
    data.texture_memory = texture_memory;
    data.texture_image_view = texture_image_view;
    data.texture_sampler = texture_sampler;
    data.descriptor_set = descriptor_set;

    Ok(())
}

impl VulkanApplication {
    /// Acquires a swapchain image, submits the command buffer, and presents the result.
    ///
    /// # Safety
    /// Calls unsafe Vulkan queue and synchronization APIs.
    pub unsafe fn render_frame(&mut self, window: &Window, camera: &Camera, eyes: &EyeMatrices) -> anyhow::Result<()> {
        self.update_chunks(camera)?;

        let image_index = match self.acquire_next_image(window)? {
            Some(index) => index,
            None => return Ok(()), // swapchain was recreated, skip this frame
        };
        update_uniform_buffer(&self.vulkan_application_data, eyes)?;

        let frustum = if eyes.is_stereo() {
            Frustum::combined_stereo(&eyes.view_projection[0], &eyes.view_projection[1])
        } else {
            Frustum::from_view_projection(&eyes.primary_vp())
        };
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
            chunk_count: self.voxel_pool.chunk_count(),
            screen_size: [
                self.vulkan_application_data.swapchain_extent.width as f32,
                self.vulkan_application_data.swapchain_extent.height as f32,
            ],
            phase: 1,
            draw_offset: crate::voxel::block::BlockType::opaque_mask(),
        };
        let svdag_chunk_count = self.svdag_pool.chunk_count();
        let extent = self.vulkan_application_data.swapchain_extent;
        let svdag_args = if svdag_chunk_count > 0 {
            let svdag_cull = CullPush {
                planes: [
                    frustum.plane(0),
                    frustum.plane(1),
                    frustum.plane(2),
                    frustum.plane(3),
                    frustum.plane(4),
                    frustum.plane(5),
                ],
                total_chunks: svdag_chunk_count,
                _pad: [0; 3],
            };
            let svdag_tile = TileAssignPush {
                view_projection: eyes.primary_vp().to_cols_array_2d(),
                screen_size: [extent.width, extent.height],
                tile_count: self.svdag_pipeline.tile_count,
            };
            let svdag_march = RaymarchPush {
                camera_pos: camera.position.to_array(),
                _pad0: 0,
                screen_size: [extent.width, extent.height],
                tile_count: self.svdag_pipeline.tile_count,
            };
            Some((&self.svdag_pipeline, svdag_cull, svdag_tile, svdag_march))
        } else {
            None
        };
        record_mesh_shader_command_buffer(
            &self.device,
            &self.vulkan_application_data,
            image_index,
            &self.mesh_shader_pipeline,
            &self.depth_pyramid_pipeline,
            &cull_push,
            self.depth_pyramid_needs_init,
            svdag_args.as_ref().map(|(sp, c, t, m)| (*sp, c, t, m)),
        )?;
        self.depth_pyramid_needs_init = false;
        self.submit_command_buffer(image_index)?;
        self.present_frame(image_index, window)?;
        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }

    /// Loads/unloads chunks. Mesh shader gets 0..MESH_DISTANCE, SVDAG gets MESH_DISTANCE..WORLD_DISTANCE.
    unsafe fn update_chunks(&mut self, camera: &Camera) -> anyhow::Result<()> {
        let player_cx = (camera.position.x / CHUNK_SIZE as f32).floor() as i32;
        let player_cz = (camera.position.z / CHUNK_SIZE as f32).floor() as i32;

        let delta = self.world.update(player_cx, player_cz);

        // Process completed SVDAG compressions
        for result in self.svdag_compressor.receive() {
            if self.world.get_chunk(result.pos[0], result.pos[1], result.pos[2]).is_none() {
                continue;
            }
            if self.svdag_pool.chunk_count() >= MAX_SVDAG_CHUNKS - 1 {
                self.svdag_pool.evict_farthest(64, player_cx, player_cz);
            }
            if self.svdag_pool.is_near_budget() {
                self.svdag_pool.evict_farthest(64, player_cx, player_cz);
            }
            self.svdag_pool.upload_chunk(result.pos, &result.dag_data, result.lod_level);
        }

        // Handle unloaded chunks — remove from whichever pool they're in
        for pos in &delta.unloaded {
            self.voxel_pool.invalidate_neighbor_boundaries(*pos, &self.world);
            self.voxel_pool.remove_chunk(pos);
            self.svdag_pool.remove_chunk(pos);
            self.svdag_pending.remove(pos);
        }

        // Evict mesh chunks that drifted beyond MESH_DISTANCE (player moved)
        for pos in self.voxel_pool.chunk_positions() {
            let dist = crate::voxel::world::chunk_distance(pos[0], pos[2], player_cx, player_cz);
            if dist > MESH_DISTANCE {
                self.voxel_pool.invalidate_neighbor_boundaries(pos, &self.world);
                self.voxel_pool.remove_chunk(&pos);
                if dist <= SVDAG_DISTANCE && self.world.get_chunk(pos[0], pos[1], pos[2]).is_some() {
                    self.svdag_pending.insert(pos);
                }
            }
        }

        // Ensure all chunks within MESH_DISTANCE are in VoxelPool (promote from SVDAG on approach)
        for cz in (player_cz - MESH_DISTANCE)..=(player_cz + MESH_DISTANCE) {
            for cx in (player_cx - MESH_DISTANCE)..=(player_cx + MESH_DISTANCE) {
                for cy in crate::voxel::world::MIN_CHUNK_Y..=crate::voxel::world::MAX_CHUNK_Y {
                    let pos = [cx, cy, cz];
                    if self.voxel_pool.has_chunk(&pos) {
                        continue;
                    }
                    if let Some(chunk) = self.world.get_chunk(cx, cy, cz) {
                        self.voxel_pool.upload_chunk(pos, chunk, &self.world);
                        self.voxel_pool.invalidate_neighbor_boundaries(pos, &self.world);
                        self.svdag_pool.remove_chunk(&pos);
                        self.svdag_pending.remove(&pos);
                    }
                }
            }
        }

        // Route newly loaded far chunks to SVDAG pending (within SVDAG_DISTANCE)
        for pos in &delta.loaded {
            let [cx, _cy, cz] = *pos;
            let dist = crate::voxel::world::chunk_distance(cx, cz, player_cx, player_cz);
            if dist > MESH_DISTANCE && dist <= SVDAG_DISTANCE {
                self.svdag_pending.insert(*pos);
            }
        }

        // Drain pending SVDAG compressions, closest first, budgeted
        const COMPRESSIONS_PER_FRAME: usize = 16;
        if !self.svdag_pending.is_empty() {
            let mut candidates: Vec<[i32; 3]> = self.svdag_pending.iter().copied().collect();
            candidates.sort_by_key(|pos| {
                let dx = pos[0] - player_cx;
                let dz = pos[2] - player_cz;
                dx * dx + dz * dz
            });
            for pos in candidates.into_iter().take(COMPRESSIONS_PER_FRAME) {
                let [cx, cy, cz] = pos;
                if let Some(chunk) = self.world.get_chunk(cx, cy, cz) {
                    self.svdag_compressor.request(pos, chunk.clone());
                    self.svdag_pending.remove(&pos);
                }
            }
        }

        self.last_player_chunk = [player_cx, player_cz];
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
            Ok(index) => index as usize,
            Err(e) if e == vk::Result::ERROR_OUT_OF_DATE => {
                self.recreate_swapchain(window)?;
                return Ok(None);
            }
            Err(e) => return Err(anyhow!("{e:?}")),
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
            .queue_submit(data.graphics_queue, &[*submit_info], data.in_flight_fences[self.frame])?;
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

        let result = self.device.queue_present_khr(data.presentation_queue, &present_info);

        if result == Err(vk::Result::ERROR_OUT_OF_DATE) {
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
        use crate::graphical_core::compute_cull;
        self.device.device_wait_idle()?;
        compute_cull::destroy_depth_pyramid_pipeline(&self.device, &self.depth_pyramid_pipeline);
        self.destroy_swapchain();
        create_swapchain(user_window, &self.vulkan_instance, &self.device, &mut self.vulkan_application_data)?;
        create_swapchain_image_views(&self.device, &mut self.vulkan_application_data)?;
        create_depth_image(&self.device, &self.vulkan_instance, &mut self.vulkan_application_data)?;
        create_depth_pyramid(&self.device, &self.vulkan_instance, &mut self.vulkan_application_data)?;
        self.depth_pyramid_pipeline = compute_cull::create_depth_pyramid_pipeline(&self.device, &self.vulkan_application_data)?;
        self.depth_pyramid_needs_init = true;
        create_render_pass(&self.vulkan_instance, &self.device, &mut self.vulkan_application_data)?;
        create_sky_pipeline(&self.device, &mut self.vulkan_application_data)?;
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
        self.device.destroy_pipeline(self.vulkan_application_data.sky_pipeline, None);
        self.device
            .destroy_pipeline_layout(self.vulkan_application_data.sky_pipeline_layout, None);
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
        use crate::graphical_core::compute_cull;
        if let Some(vr_sc) = self._vr_swapchain.take() {
            vr_sc.destroy(&self.device);
        }
        self.svdag_pipeline.destroy(&self.device);
        self.svdag_pool.destroy(&self.device);
        self.mesh_shader_pipeline.destroy(&self.device);
        self.voxel_pool.destroy(&self.device);
        compute_cull::destroy_depth_pyramid_pipeline(&self.device, &self.depth_pyramid_pipeline);
        destroy_textures(&self.device, &mut self.vulkan_application_data);
        destroy_palette_buffer(&self.device, &mut self.vulkan_application_data);
        destroy_uniform_buffer(&self.device, &mut self.vulkan_application_data);
        self.device.destroy_descriptor_pool(self.vulkan_application_data.descriptor_pool, None);
        self.device
            .destroy_descriptor_set_layout(self.vulkan_application_data.descriptor_set_layout, None);
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
        self.vulkan_instance.destroy_surface(self.vulkan_application_data.surface, None);
        if VALIDATION_ENABLED {
            self.vulkan_instance
                .destroy_debug_utils_messenger_ext(self.vulkan_application_data.debug_messenger, None);
        }
        self.vulkan_instance.destroy_instance(None);
    }
}
