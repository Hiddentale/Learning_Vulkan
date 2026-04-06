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
    ui_pipeline::UiPipeline,
    voxel_pool::VoxelPool,
    MAX_FRAMES_IN_FLIGHT,
};
use crate::storage::region::RegionStore;
use crate::voxel::chunk::{Chunk, CHUNK_SIZE};
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

/// Mesh shader renders chunks within this distance. Full detail, editable, textured.
const MESH_DISTANCE: i32 = 8;
/// Mesh shader renders chunks within this distance.
const LOD0_DISTANCE: i32 = 24; // 384m
const LOD1_DISTANCE: i32 = 48; // 768m
const LOD2_DISTANCE: i32 = 96; // 1536m
const LOD3_DISTANCE: i32 = 192; // 3072m
const LOD4_DISTANCE: i32 = 384; // 6144m (>5km)
/// World generates terrain out to this distance (raw chunks for LOD-0 super-chunk building).
const WORLD_DISTANCE: i32 = LOD0_DISTANCE + 8 + 4;
/// Maximum SVDAG render distance across all LOD levels.
const SVDAG_DISTANCE: i32 = LOD4_DISTANCE;
const MAX_MESH_CHUNKS: usize = ((2 * MESH_DISTANCE + 1) * (2 * MESH_DISTANCE + 1) * (2 * MESH_DISTANCE + 1)) as usize;
const MAX_SVDAG_CHUNKS: u32 = 32768;
const SUPER_CHUNK_VOXELS: u32 = 64;

/// World-specific resources created when entering a world, destroyed when returning to menu.
pub struct WorldResources {
    pub world: World,
    voxel_pool: VoxelPool,
    mesh_shader_pipeline: MeshShaderPipeline,
    svdag_pool: SvdagPool,
    svdag_pipeline: SvdagPipeline,
    svdag_compressor: SvdagCompressor,
    svdag_pending: std::collections::HashSet<[i32; 3]>,
    svdag_in_flight: std::collections::HashSet<[i32; 3]>,
    /// Per-LOD disk caches. Index 0 = LOD-0, 1 = LOD-1, 2 = LOD-2.
    svdag_caches: Vec<RegionStore>,
    last_player_chunk: [i32; 3],
    seed: u32,
    /// In-flight LOD generation requests.
    lod_in_flight: std::collections::HashSet<[i32; 3]>,
    /// Chunks that generated as empty (all air). Never retry these.
    lod_empty: std::collections::HashSet<[i32; 3]>,
    /// Consecutive frames where LOD scheduling had no work.
    lod_idle_frames: u32,
}

pub struct VulkanApplication {
    _vulkan_entry_point: Entry,
    vulkan_instance: Instance,
    vulkan_application_data: VulkanApplicationData,
    device: Device,
    frame: usize,
    pub(crate) resized: bool,
    depth_pyramid_pipeline: DepthPyramidResources,
    depth_pyramid_needs_init: bool,
    _vr_session: Option<VrSession>,
    _vr_swapchain: Option<VrSwapchain>,
    /// None when in the menu, Some when a world is loaded.
    wr: Option<WorldResources>,
    pub ui: UiPipeline,
}

impl VulkanApplication {
    /// Returns the world if one is loaded.
    pub fn world(&self) -> Option<&World> {
        self.wr.as_ref().map(|wr| &wr.world)
    }

    /// Returns true if a world is currently loaded.
    pub fn has_world(&self) -> bool {
        self.wr.is_some()
    }

    /// True if LOD generation has settled (nothing to submit, nothing in flight).
    pub fn lod_settled(&self) -> bool {
        self.wr.as_ref().is_none_or(|wr| wr.lod_idle_frames >= 3)
    }

    /// Sets a block in the world and re-uploads to the mesh shader pool.
    /// Only near-field chunks (in VoxelPool) are editable.
    pub unsafe fn set_block(&mut self, wx: i32, wy: i32, wz: i32, block: crate::voxel::block::BlockType) {
        let wr = match self.wr.as_mut() {
            Some(wr) => wr,
            None => return,
        };
        if !wr.world.set_block(wx, wy, wz, block) {
            return;
        }
        let chunk_pos = World::block_to_chunk(wx, wy, wz);
        let [cx, cy, cz] = chunk_pos;
        if let Some(chunk) = wr.world.get_chunk(cx, cy, cz) {
            wr.voxel_pool.reupload_chunk(chunk_pos, chunk, &wr.world);
            wr.voxel_pool.invalidate_neighbor_boundaries(chunk_pos, &wr.world);
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
        let wr = match &self.wr {
            Some(wr) => wr,
            None => return Ok(None),
        };
        crate::vr::frame::render_vr_frame(
            &self.device,
            &self.vulkan_application_data,
            session,
            swapchain,
            &wr.mesh_shader_pipeline,
            &wr.voxel_pool,
        )
    }
}

impl VulkanApplication {
    /// Creates the core Vulkan renderer without loading a world.
    /// Call `enter_world()` to load a world before rendering game frames.
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
        let depth_pyramid_pipeline = crate::graphical_core::compute_cull::create_depth_pyramid_pipeline(&device, &data)?;
        let ui = UiPipeline::create(&device, &instance, &mut data)?;

        Ok(Self {
            _vulkan_entry_point: entry,
            vulkan_instance: instance,
            vulkan_application_data: data,
            device,
            frame: 0,
            resized: false,
            depth_pyramid_pipeline,
            depth_pyramid_needs_init: true,
            _vr_session: vr_session,
            _vr_swapchain: vr_swapchain,
            wr: None,
            ui,
        })
    }

    /// Load a world and create all GPU resources for rendering it.
    ///
    /// # Safety
    /// Calls unsafe Vulkan APIs.
    pub unsafe fn enter_world(&mut self, world_dir: &std::path::Path, seed: u32) -> anyhow::Result<()> {
        let mut world = World::new(WORLD_DISTANCE, seed);
        world.update(0, 5, 0); // Initial camera Y ≈ 90 blocks → chunk 5

        let mut voxel_pool = VoxelPool::new(
            MAX_MESH_CHUNKS as u32,
            &self.device,
            &self.vulkan_instance,
            &mut self.vulkan_application_data,
        )?;
        for pos in world.chunk_positions() {
            let [cx, cy, cz] = pos;
            if crate::voxel::world::chunk_distance(cx, cy, cz, 0, 5, 0) <= MESH_DISTANCE {
                if let Some(chunk) = world.get_chunk(cx, cy, cz) {
                    voxel_pool.upload_chunk(pos, chunk, &world);
                }
            }
        }
        let mesh_shader_pipeline = MeshShaderPipeline::create(&self.device, &self.vulkan_application_data, &voxel_pool)?;

        let svdag_pool = SvdagPool::new(MAX_SVDAG_CHUNKS, &self.device, &self.vulkan_instance, &mut self.vulkan_application_data)?;
        let svdag_pipeline = SvdagPipeline::create(&self.device, &self.vulkan_instance, &mut self.vulkan_application_data, &svdag_pool)?;
        let svdag_compressor = SvdagCompressor::new();
        let mut svdag_caches = Vec::new();
        for lod in 0..=LOD_BANDS.len() as u32 {
            svdag_caches.push(RegionStore::new(&crate::storage::world_meta::svdag_lod_dir(world_dir, lod))?);
        }

        self.wr = Some(WorldResources {
            world,
            voxel_pool,
            mesh_shader_pipeline,
            svdag_pool,
            svdag_pipeline,
            svdag_compressor,
            svdag_pending: std::collections::HashSet::new(),
            svdag_in_flight: std::collections::HashSet::new(),
            svdag_caches,
            last_player_chunk: [0, 0, 0],
            lod_in_flight: std::collections::HashSet::new(),
            lod_empty: std::collections::HashSet::new(),
            lod_idle_frames: 0,
            seed,
        });
        self.depth_pyramid_needs_init = true;
        Ok(())
    }

    /// Unload the current world and return to menu state.
    ///
    /// # Safety
    /// Calls unsafe Vulkan APIs.
    pub unsafe fn exit_world(&mut self) {
        self.device.device_wait_idle().unwrap();
        if let Some(mut wr) = self.wr.take() {
            wr.svdag_pipeline.destroy(&self.device);
            wr.svdag_pool.destroy(&self.device);
            wr.mesh_shader_pipeline.destroy(&self.device);
            wr.voxel_pool.destroy(&self.device);
        }
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
        let wr = self.wr.as_mut().expect("render_frame called without a loaded world");
        Self::update_chunks_inner(wr, camera)?;

        let image_index = match self.acquire_next_image(window)? {
            Some(index) => index,
            None => return Ok(()),
        };
        update_uniform_buffer(&self.vulkan_application_data, eyes)?;

        let wr = self.wr.as_ref().unwrap();
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
            chunk_count: wr.voxel_pool.chunk_count(),
            screen_size: [
                self.vulkan_application_data.swapchain_extent.width as f32,
                self.vulkan_application_data.swapchain_extent.height as f32,
            ],
            phase: 1,
            draw_offset: crate::voxel::block::BlockType::opaque_mask(),
        };
        let svdag_chunk_count = wr.svdag_pool.chunk_count();
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
                _padding: [0; 3],
            };
            let svdag_tile = TileAssignPush {
                view_projection: eyes.primary_vp().to_cols_array_2d(),
                screen_size: [extent.width, extent.height],
                tile_count: wr.svdag_pipeline.tile_count,
                camera_pos: camera.position.to_array(),
                _padding: 0,
            };
            let svdag_march = RaymarchPush {
                camera_pos: camera.position.to_array(),
                _padding: 0,
                screen_size: [extent.width, extent.height],
                tile_count: wr.svdag_pipeline.tile_count,
            };
            Some((&wr.svdag_pipeline, svdag_cull, svdag_tile, svdag_march))
        } else {
            None
        };
        record_mesh_shader_command_buffer(
            &self.device,
            &self.vulkan_application_data,
            image_index,
            &wr.mesh_shader_pipeline,
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

    /// Render a menu frame (sky background + UI overlay).
    pub unsafe fn render_menu_frame(&mut self, window: &Window, eyes: &EyeMatrices) -> anyhow::Result<()> {
        let image_index = match self.acquire_next_image(window)? {
            Some(index) => index,
            None => return Ok(()),
        };
        update_uniform_buffer(&self.vulkan_application_data, eyes)?;

        let cmd = self.vulkan_application_data.command_buffers[image_index];
        self.device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;
        self.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo::builder())?;

        crate::graphical_core::commands::begin_render_pass(&self.device, cmd, &self.vulkan_application_data, image_index);
        crate::graphical_core::commands::draw_sky(&self.device, cmd, &self.vulkan_application_data);
        let screen = [
            self.vulkan_application_data.swapchain_extent.width as f32,
            self.vulkan_application_data.swapchain_extent.height as f32,
        ];
        self.ui.record(&self.device, cmd, screen);
        self.device.cmd_end_render_pass(cmd);
        self.device.end_command_buffer(cmd)?;

        self.submit_command_buffer(image_index)?;
        self.present_frame(image_index, window)?;
        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }

    /// Pump chunk loading and SVDAG compression without rendering.
    /// Call during pre-generation to route chunks to the correct pools.
    pub unsafe fn update_world(&mut self, camera: &Camera) -> anyhow::Result<()> {
        if let Some(wr) = self.wr.as_mut() {
            Self::update_chunks_inner(wr, camera)?;
        }
        Ok(())
    }

    /// Loads/unloads chunks. Routes to mesh pool, SVDAG pool, and LOD merging.
    unsafe fn update_chunks_inner(wr: &mut WorldResources, camera: &Camera) -> anyhow::Result<()> {
        let player_cx = (camera.position.x / CHUNK_SIZE as f32).floor() as i32;
        let player_cy = (camera.position.y / CHUNK_SIZE as f32).floor() as i32;
        let player_cz = (camera.position.z / CHUNK_SIZE as f32).floor() as i32;

        let delta = wr.world.update(player_cx, player_cy, player_cz);

        // Process completed SVDAG compressions — drop results that don't fit (no eviction thrashing)
        for result in wr.svdag_compressor.receive() {
            wr.svdag_in_flight.remove(&result.pos);
            wr.lod_in_flight.remove(&result.pos);
            if result.dag_data.len() <= 68 {
                wr.lod_empty.insert(result.pos);
                continue;
            }
            if wr.svdag_pool.chunk_count() >= MAX_SVDAG_CHUNKS - 1 || wr.svdag_pool.is_near_budget() {
                continue;
            }
            wr.svdag_pool
                .upload_chunk(result.pos, &result.dag_data, result.lod_level, SUPER_CHUNK_VOXELS);
            let compressed = lz4_flex::compress_prepend_size(&result.dag_data);
            let cache_idx = result.lod_level.saturating_sub(2) as usize;
            if cache_idx < wr.svdag_caches.len() {
                let _ = wr.svdag_caches[cache_idx].write(result.pos, &compressed);
            }
        }

        // Handle unloaded chunks — remove from whichever pool they're in
        for pos in &delta.unloaded {
            wr.voxel_pool.invalidate_neighbor_boundaries(*pos, &wr.world);
            wr.voxel_pool.remove_chunk(pos);
            wr.svdag_pool.remove_chunk(pos);
            wr.svdag_pending.remove(pos);
        }

        // Evict mesh chunks that drifted beyond MESH_DISTANCE (player moved)
        for pos in wr.voxel_pool.chunk_positions() {
            let dist = crate::voxel::world::chunk_distance(pos[0], pos[1], pos[2], player_cx, player_cy, player_cz);
            if dist > MESH_DISTANCE {
                wr.voxel_pool.invalidate_neighbor_boundaries(pos, &wr.world);
                wr.voxel_pool.remove_chunk(&pos);
                if dist <= SVDAG_DISTANCE && wr.world.get_chunk(pos[0], pos[1], pos[2]).is_some() {
                    wr.svdag_pending.insert(pos);
                }
            }
        }

        // Ensure all chunks within MESH_DISTANCE are in VoxelPool (promote from SVDAG on approach)
        for cz in (player_cz - MESH_DISTANCE)..=(player_cz + MESH_DISTANCE) {
            for cx in (player_cx - MESH_DISTANCE)..=(player_cx + MESH_DISTANCE) {
                for cy in (player_cy - MESH_DISTANCE)..=(player_cy + MESH_DISTANCE) {
                    let pos = [cx, cy, cz];
                    if wr.voxel_pool.has_chunk(&pos) {
                        continue;
                    }
                    if let Some(chunk) = wr.world.get_chunk(cx, cy, cz) {
                        wr.voxel_pool.upload_chunk(pos, chunk, &wr.world);
                        wr.voxel_pool.invalidate_neighbor_boundaries(pos, &wr.world);
                        wr.svdag_pool.remove_chunk(&pos);
                        wr.svdag_pending.remove(&pos);
                    }
                }
            }
        }

        // Route newly loaded far chunks to SVDAG pending (within SVDAG_DISTANCE)
        for pos in &delta.loaded {
            let [cx, cy, cz] = *pos;
            let dist = crate::voxel::world::chunk_distance(cx, cy, cz, player_cx, player_cy, player_cz);
            if dist > MESH_DISTANCE && dist <= SVDAG_DISTANCE {
                wr.svdag_pending.insert(*pos);
            }
        }

        const COMPRESSIONS_PER_FRAME: usize = 64;
        if !wr.svdag_pending.is_empty() {
            let mut candidates: Vec<[i32; 3]> = wr.svdag_pending.iter().copied().collect();
            candidates.sort_by_key(|pos| {
                let dx = pos[0] - player_cx;
                let dy = pos[1] - player_cy;
                let dz = pos[2] - player_cz;
                dx * dx + dy * dy + dz * dz
            });
            let mut groups: Vec<[i32; 3]> = candidates.iter().map(|&[cx, cy, cz]| [cx & !3, cy & !3, cz & !3]).collect();
            groups.sort_unstable();
            groups.dedup();
            groups.sort_by_key(|pos| {
                let dx = pos[0] - player_cx;
                let dy = pos[1] - player_cy;
                let dz = pos[2] - player_cz;
                dx * dx + dy * dy + dz * dz
            });

            let mut cache_loaded = Vec::new();
            for &group_pos in &groups {
                if wr.svdag_pool.has_chunk(&group_pos) || wr.svdag_in_flight.contains(&group_pos) || wr.lod_empty.contains(&group_pos) {
                    cache_loaded.push(group_pos);
                    continue;
                }
                if let Some(compressed) = wr.svdag_caches[0].read(group_pos) {
                    if let Ok(dag_data) = lz4_flex::decompress_size_prepended(compressed) {
                        if dag_data.len() > 68 && wr.svdag_pool.chunk_count() < MAX_SVDAG_CHUNKS - 1 && !wr.svdag_pool.is_near_budget() {
                            wr.svdag_pool.upload_chunk(group_pos, &dag_data, 2, SUPER_CHUNK_VOXELS);
                        }
                        cache_loaded.push(group_pos);
                    }
                }
            }
            for gp in &cache_loaded {
                for dy in 0..4i32 {
                    for dz in 0..4i32 {
                        for dx in 0..4i32 {
                            wr.svdag_pending.remove(&[gp[0] + dx, gp[1] + dy, gp[2] + dz]);
                        }
                    }
                }
            }

            let mut submitted = 0;
            for &group_pos in &groups {
                if submitted >= COMPRESSIONS_PER_FRAME {
                    break;
                }
                let [gx, gy, gz] = group_pos;
                if cache_loaded.contains(&group_pos) || wr.svdag_in_flight.contains(&group_pos) || wr.svdag_pool.has_chunk(&group_pos) {
                    continue;
                }
                let mut chunks: Vec<Option<Chunk>> = Vec::with_capacity(64);
                let mut all_present = true;
                for dy in 0..4i32 {
                    for dz in 0..4i32 {
                        for dx in 0..4i32 {
                            let c = wr.world.get_chunk(gx + dx, gy + dy, gz + dz);
                            if c.is_none() {
                                all_present = false;
                            }
                            chunks.push(c.cloned());
                        }
                    }
                }
                if !all_present {
                    continue;
                }
                let boxed: Box<[Option<Chunk>; 64]> = match chunks.into_boxed_slice().try_into() {
                    Ok(b) => b,
                    Err(_) => unreachable!("collected exactly 64 entries"),
                };
                wr.svdag_compressor.request_super_chunk(group_pos, boxed);
                wr.svdag_in_flight.insert(group_pos);
                for dy in 0..4i32 {
                    for dz in 0..4i32 {
                        for dx in 0..4i32 {
                            wr.svdag_pending.remove(&[gx + dx, gy + dy, gz + dz]);
                        }
                    }
                }
                submitted += 1;
            }
        }

        // --- LOD super-chunk generation beyond LOD-0 range ---
        let lod_submitted = schedule_lod_generation(wr, player_cx, player_cy, player_cz);
        if lod_submitted || !wr.lod_in_flight.is_empty() {
            wr.lod_idle_frames = 0;
        } else {
            wr.lod_idle_frames = wr.lod_idle_frames.saturating_add(1);
        }

        wr.last_player_chunk = [player_cx, player_cy, player_cz];
        Ok(())
    }
}

/// LOD level definitions: (chunk alignment, voxel size in blocks, lod_level for pool, distance band).
const LOD_BANDS: &[(i32, u32, u32, i32, i32)] = &[
    // (align, voxel_size, lod_level, min_dist, max_dist)
    // Each band extends +align beyond its nominal boundary so the inner LOD's chunks
    // overlap into the outer band. This prevents gaps where shallow-angle rays miss
    // terrain in the outer LOD's boundary chunks.
    // LOD-1 extends inward to MESH_DISTANCE as a fallback while LOD-0 super-chunks
    // assemble (LOD-0 requires all 64 child chunks; LOD-1 generates directly from
    // noise). Once LOD-0 completes, its higher-resolution chunks win in the ray march.
    // Each band extends inward to the previous band's nominal start as a fallback
    // while finer LODs assemble. Coarser LODs generate from noise (no child dependency)
    // so they're available immediately. Finer LODs replace them once ready.
    (8, 2, 3, MESH_DISTANCE, LOD1_DISTANCE + 8),   // LOD-1: covers LOD-0 area (8-24) as fallback
    (16, 4, 4, LOD0_DISTANCE, LOD2_DISTANCE + 16), // LOD-2: covers LOD-1 area (24-48) as fallback
    (32, 8, 5, LOD1_DISTANCE, LOD3_DISTANCE + 32), // LOD-3: covers LOD-2 area (48-96) as fallback
    (64, 16, 6, LOD2_DISTANCE, LOD4_DISTANCE),     // LOD-4: covers LOD-3 area (96-192) as fallback
];
const MAX_LOD_IN_FLIGHT: usize = 32;
const LOD_SUBMISSIONS_PER_FRAME: usize = 32;

/// Schedule direct LOD terrain generation for positions beyond LOD-0 range.
/// Returns true if any LOD work was submitted this frame.
unsafe fn schedule_lod_generation(wr: &mut WorldResources, player_cx: i32, player_cy: i32, player_cz: i32) -> bool {
    if wr.lod_in_flight.len() >= MAX_LOD_IN_FLIGHT {
        return true;
    }
    let mut total_submitted = 0usize;
    for &(align, voxel_size, lod_level, min_dist, max_dist) in LOD_BANDS {
        let half = max_dist / align + 1;
        let pcx = player_cx.div_euclid(align) * align;
        let pcy = player_cy.div_euclid(align) * align;
        let pcz = player_cz.div_euclid(align) * align;
        let mut submitted = 0;

        for ring in 0..=half {
            if submitted >= LOD_SUBMISSIONS_PER_FRAME {
                break;
            }
            for dz in -ring..=ring {
                for dy in -ring..=ring {
                    for dx in -ring..=ring {
                        // Only process the shell border of this ring
                        if dx.abs() != ring && dy.abs() != ring && dz.abs() != ring {
                            continue;
                        }
                        let gx = pcx + dx * align;
                        let gy = pcy + dy * align;
                        let gz = pcz + dz * align;
                        if !in_lod_band(gx, gy, gz, player_cx, player_cy, player_cz, min_dist, max_dist) {
                            continue;
                        }
                        // Skip LOD chunks whose AABB overlaps the mesh shader cube.
                        // The mesh is authoritative in its area; LOD fallbacks are
                        // only needed beyond it.
                        let mesh_lo = [player_cx - MESH_DISTANCE, player_cy - MESH_DISTANCE, player_cz - MESH_DISTANCE];
                        let mesh_hi = [player_cx + MESH_DISTANCE, player_cy + MESH_DISTANCE, player_cz + MESH_DISTANCE];
                        let overlaps_mesh = gx <= mesh_hi[0]
                            && gx + align > mesh_lo[0]
                            && gy <= mesh_hi[1]
                            && gy + align > mesh_lo[1]
                            && gz <= mesh_hi[2]
                            && gz + align > mesh_lo[2];
                        if overlaps_mesh {
                            continue;
                        }
                        let pos = [gx, gy, gz];
                        if wr.svdag_pool.has_chunk(&pos) || wr.lod_in_flight.contains(&pos) || wr.lod_empty.contains(&pos) {
                            continue;
                        }
                        if try_load_cached_lod(wr, pos, lod_level) {
                            continue;
                        }
                        if submitted >= LOD_SUBMISSIONS_PER_FRAME {
                            break;
                        }
                        let origin = [gx * CHUNK_SIZE as i32, gy * CHUNK_SIZE as i32, gz * CHUNK_SIZE as i32];
                        wr.svdag_compressor.request_lod_generate(pos, origin, voxel_size, lod_level, wr.seed);
                        wr.lod_in_flight.insert(pos);
                        submitted += 1;
                    }
                }
            }
        }
        total_submitted += submitted;
    }
    total_submitted > 0
}

fn in_lod_band(gx: i32, gy: i32, gz: i32, px: i32, py: i32, pz: i32, min_dist: i32, max_dist: i32) -> bool {
    let dist = crate::voxel::world::chunk_distance(gx, gy, gz, px, py, pz);
    dist >= min_dist && dist <= max_dist
}

unsafe fn try_load_cached_lod(wr: &mut WorldResources, pos: [i32; 3], lod_level: u32) -> bool {
    let cache_idx = lod_level.saturating_sub(2) as usize;
    if cache_idx >= wr.svdag_caches.len() {
        return false;
    }
    let Some(compressed) = wr.svdag_caches[cache_idx].read(pos) else {
        return false;
    };
    let Ok(dag_data) = lz4_flex::decompress_size_prepended(compressed) else {
        return false;
    };
    if dag_data.len() > 68 && wr.svdag_pool.chunk_count() < MAX_SVDAG_CHUNKS - 1 && !wr.svdag_pool.is_near_budget() {
        wr.svdag_pool.upload_chunk(pos, &dag_data, lod_level, SUPER_CHUNK_VOXELS);
    }
    true
}

impl VulkanApplication {
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
        // Re-bind depth pyramid descriptors for world pipelines (handles were invalidated)
        if let Some(wr) = &self.wr {
            wr.mesh_shader_pipeline.update_depth_pyramid(&self.device, &self.vulkan_application_data);
            wr.svdag_pipeline.update_depth_pyramid(&self.device, &self.vulkan_application_data);
        }
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
        self.ui.destroy(&self.device);
        if let Some(vr_sc) = self._vr_swapchain.take() {
            vr_sc.destroy(&self.device);
        }
        if let Some(mut wr) = self.wr.take() {
            wr.svdag_pipeline.destroy(&self.device);
            wr.svdag_pool.destroy(&self.device);
            wr.mesh_shader_pipeline.destroy(&self.device);
            wr.voxel_pool.destroy(&self.device);
        }
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
