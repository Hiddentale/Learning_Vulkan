use crate::graphical_core::{
    camera::{create_uniform_buffer, destroy_uniform_buffer, update_uniform_buffer, Camera, EyeMatrices, UniformBufferObject},
    commands::{allocate_command_buffers, create_command_pool, create_frame_buffers, create_sync_objects, record_mesh_shader_command_buffer},
    compute_cull::{CullPushConstants, DepthPyramidResources},
    depth::{create_depth_image, create_depth_pyramid, destroy_depth_image, destroy_depth_pyramid},
    descriptors,
    frustum::Frustum,
    gpu::choose_gpu,
    heightmap_pipeline::HeightmapPipeline,
    heightmap_pool::HeightmapPool,
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
use crate::voxel::erosion::ErosionMap;
use crate::voxel::heightmap_generator::{HeightmapGenerator, HeightmapRequest};
use crate::voxel::svdag_compressor::SvdagCompressor;
use crate::voxel::world::World;
use crate::vr::{VrContext, VrSession, VrSwapchain};
use crate::VALIDATION_ENABLED;
use anyhow::anyhow;
use log::info;
use std::sync::Arc;
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
pub const WORLD_DISTANCE: i32 = LOD0_DISTANCE + 8 + 4;
/// Angular margin (radians) by which the heightmap reaches *inside* the
/// mesh-chunk arc. Lets the heightmap cover the load-in frontier while the
/// outermost mesh chunks are still being generated. Sized at ~4 chunks of
/// arc at the current planet radius — large enough to absorb generator
/// latency, small enough that the resulting overlap ring is sub-pixel
/// when the player is high enough to notice it at all.
const HEIGHTMAP_OVERLAP_ARC: f32 =
    (4.0 * crate::voxel::chunk::CHUNK_SIZE as f32) / crate::voxel::sphere::PLANET_RADIUS_BLOCKS as f32;
/// Maximum SVDAG render distance across all LOD levels.
const SVDAG_DISTANCE: i32 = LOD4_DISTANCE;
/// Sized to hold the working set: a `(2·WORLD_DISTANCE+1)²` column window
/// around the player times the radial column height, with generous slack.
/// Independent of `FACE_SIDE_CHUNKS` — the planet can be arbitrarily large.
/// At `WORLD_DISTANCE=36` and 48-chunk-tall columns the bound is
/// `73² × 48 ≈ 256k` worst case; we round to a power of two for slot
/// arithmetic. Pinned by `world_resident_set_is_bounded_by_render_distance`.
const MAX_MESH_CHUNKS: usize = 262144;
const MAX_SVDAG_CHUNKS: u32 = 32768;
const SUPER_CHUNK_VOXELS: u32 = 64;

/// World-specific resources created when entering a world, destroyed when returning to menu.
pub struct WorldResources {
    pub world: World,
    voxel_pool: VoxelPool,
    mesh_shader_pipeline: MeshShaderPipeline,
    cull_compact_pipeline: crate::graphical_core::cull_compact::CullCompactPipeline,
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
    heightmap_pool: HeightmapPool,
    heightmap_pipeline: HeightmapPipeline,
    heightmap_generator: HeightmapGenerator,
    heightmap_in_flight: std::collections::HashSet<crate::voxel::heightmap_generator::TileKey>,
    #[allow(dead_code)] // retained for future access (shadow rays, re-erosion)
    erosion_map: Option<Arc<ErosionMap>>,
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
    /// Single-slot timestamp query pool for per-stage GPU timing.
    /// Size = `TIMING_QUERY_COUNT`. Read back synchronously each frame.
    timing_query_pool: vk::QueryPool,
    timing_period_ns: f64,
}

/// Number of timestamp queries written per frame in `record_mesh_shader_command_buffer`.
/// Slot meanings (set by the recording code):
/// 0 = start, 1 = after sky, 2 = after phase1 mesh, 3 = after depth pyramid,
/// 4 = after phase2 mesh + heightmap, 5 = after svdag, 6 = after ui (= end).
pub const TIMING_QUERY_COUNT: u32 = 7;

/// True iff every one of `cp`'s six axis-neighbors is uniform-opaque (or out
/// of generated range, which the world treats as solid below the terrain
/// layer). A uniform-opaque chunk with this property is buried — none of its
/// faces touch air, so it emits no geometry and can be skipped at upload.
fn neighbors_all_opaque(world: &World, cp: crate::voxel::sphere::ChunkPos) -> bool {
    use crate::voxel::sphere::{cross_face_neighbor, ChunkPos, FACE_SIDE_CHUNKS};
    let neighbor_solid = |nb: ChunkPos| -> bool {
        // Out of (cx, cz) face range → cross-face neighbor; treat as solid if
        // we can't resolve (the world cap above the terrain band is air, not
        // solid, so we explicitly only count radial neighbors as solid when
        // they're inside the band).
        let resolved = if nb.cx < 0 || nb.cx >= FACE_SIDE_CHUNKS || nb.cz < 0 || nb.cz >= FACE_SIDE_CHUNKS {
            cross_face_neighbor(nb).unwrap_or(nb)
        } else {
            nb
        };
        match world.get_chunk_at(resolved) {
            Some(c) => c.is_uniform_opaque(),
            None => {
                // Missing chunk → solid only if it's inside the radial terrain
                // band. Above the band is sky.
                (crate::voxel::world::TERRAIN_MIN_CY..=crate::voxel::world::TERRAIN_MAX_CY).contains(&resolved.cy)
            }
        }
    };
    let neighbors = [
        ChunkPos { face: cp.face, cx: cp.cx + 1, cy: cp.cy, cz: cp.cz },
        ChunkPos { face: cp.face, cx: cp.cx - 1, cy: cp.cy, cz: cp.cz },
        ChunkPos { face: cp.face, cx: cp.cx, cy: cp.cy + 1, cz: cp.cz },
        ChunkPos { face: cp.face, cx: cp.cx, cy: cp.cy - 1, cz: cp.cz },
        ChunkPos { face: cp.face, cx: cp.cx, cy: cp.cy, cz: cp.cz + 1 },
        ChunkPos { face: cp.face, cx: cp.cx, cy: cp.cy, cz: cp.cz - 1 },
    ];
    neighbors.iter().all(|&n| neighbor_solid(n))
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

    /// Set a single block in cube space and re-upload its chunk (and any
    /// neighbor chunks whose boundary slice depends on it). Used by the
    /// raycast-driven place / break inputs.
    pub unsafe fn set_block_at(
        &mut self,
        cp: crate::voxel::sphere::ChunkPos,
        lx: usize,
        ly: usize,
        lz: usize,
        block: crate::voxel::block::BlockType,
    ) {
        let Some(wr) = self.wr.as_mut() else { return };
        if !wr.world.set_block_at(cp, lx, ly, lz, block) {
            return;
        }
        if let Some(chunk) = wr.world.get_chunk_at(cp) {
            let chunk_ptr: *const crate::voxel::chunk::Chunk = chunk;
            wr.voxel_pool.reupload_chunk(cp, &*chunk_ptr, &wr.world);
            wr.voxel_pool.invalidate_neighbor_boundaries(cp, &wr.world);
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
            &wr.cull_compact_pipeline,
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

        // Per-stage GPU timestamp query pool. One pool with TIMING_QUERY_COUNT
        // slots — we read it back synchronously after each frame, so no
        // double-buffering is needed.
        let qp_info = vk::QueryPoolCreateInfo::builder()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(TIMING_QUERY_COUNT);
        let timing_query_pool = device.create_query_pool(&qp_info, None)?;
        let props = instance.get_physical_device_properties(data.physical_device);
        let timing_period_ns = props.limits.timestamp_period as f64;
        // Clear debug.log on startup so the user can see whether the new
        // perf line ever gets written this run.
        let _ = std::fs::write("debug.log", format!("[startup] timestamp_period={} ns\n", timing_period_ns));

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
            timing_query_pool,
            timing_period_ns,
        })
    }

    /// Load a world and create all GPU resources for rendering it.
    ///
    /// # Safety
    /// Calls unsafe Vulkan APIs.
    pub unsafe fn enter_world(&mut self, world_dir: &std::path::Path, seed: u32, erosion_map: Option<Arc<ErosionMap>>) -> anyhow::Result<()> {
        let mut world = World::new(WORLD_DISTANCE, seed, erosion_map.clone());
        // Spawn point is just above the +Y pole; the streamer needs a real
        // world position so it knows which face neighborhood to load.
        let spawn = glam::DVec3::new(0.0, crate::voxel::sphere::SURFACE_RADIUS_BLOCKS as f64, 0.0);
        world.update(spawn);

        let mut voxel_pool = VoxelPool::new(
            MAX_MESH_CHUNKS as u32,
            &self.device,
            &self.vulkan_instance,
            &mut self.vulkan_application_data,
        )?;
        // Note: chunks are uploaded per-frame in `update_chunks_inner` as the
        // chunk generator threads finish — not here. Here the world is empty.
        let mesh_shader_pipeline = MeshShaderPipeline::create(&self.device, &self.vulkan_application_data, &voxel_pool)?;
        let cull_compact_pipeline =
            crate::graphical_core::cull_compact::CullCompactPipeline::create(&self.device, &self.vulkan_application_data, &voxel_pool)?;

        let svdag_pool = SvdagPool::new(MAX_SVDAG_CHUNKS, &self.device, &self.vulkan_instance, &mut self.vulkan_application_data)?;
        let svdag_pipeline = SvdagPipeline::create(&self.device, &self.vulkan_instance, &mut self.vulkan_application_data, &svdag_pool)?;
        let svdag_compressor = SvdagCompressor::new(erosion_map.clone());
        let mut svdag_caches = Vec::new();
        for lod in 0..=LOD_BANDS.len() as u32 {
            svdag_caches.push(RegionStore::new(&crate::storage::world_meta::svdag_lod_dir(world_dir, lod))?);
        }

        let heightmap_pool = HeightmapPool::new(&self.device, &self.vulkan_instance, &mut self.vulkan_application_data)?;
        let heightmap_pipeline = HeightmapPipeline::create(&self.device, &self.vulkan_application_data)?;
        let heightmap_generator = HeightmapGenerator::new(erosion_map.clone());

        self.wr = Some(WorldResources {
            world,
            voxel_pool,
            mesh_shader_pipeline,
            cull_compact_pipeline,
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
            heightmap_pool,
            heightmap_pipeline,
            heightmap_generator,
            heightmap_in_flight: std::collections::HashSet::new(),
            erosion_map,
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
            wr.heightmap_pipeline.destroy(&self.device);
            wr.heightmap_pool.destroy(&self.device);
            wr.svdag_pipeline.destroy(&self.device);
            wr.svdag_pool.destroy(&self.device);
            wr.cull_compact_pipeline.destroy(&self.device);
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
        let t_total = std::time::Instant::now();

        let t0 = std::time::Instant::now();
        let wr = self.wr.as_mut().expect("render_frame called without a loaded world");
        Self::update_chunks_inner(wr, camera)?;
        let dt_update_chunks = t0.elapsed();
        let resident_count = wr.voxel_pool.chunk_count();

        let t1 = std::time::Instant::now();
        let image_index = match self.acquire_next_image(window)? {
            Some(index) => index,
            None => return Ok(()),
        };
        let dt_acquire = t1.elapsed();
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
            planet_radius: crate::voxel::sphere::PLANET_RADIUS_BLOCKS as f32,
            stereo: if eyes.is_stereo() { 1 } else { 0 },
            _pad: [0.0; 2],
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
        let chunked_arc = chunked_arc_radians();
        let hm_bands: Vec<(f32, f32)> = HEIGHTMAP_BANDS.iter().map(|b| (chunked_arc + b.3, chunked_arc + b.4)).collect();
        let player_world_d = glam::DVec3::new(camera.position.x as f64, camera.position.y as f64, camera.position.z as f64);
        let heightmap_visible = wr.heightmap_pool.visible_tiles(&frustum, player_world_d, &hm_bands);
        let heightmap_push = crate::graphical_core::heightmap_pipeline::HeightmapPush {
            camera_pos: camera.position.to_array(),
            morph_factor: 0.0,
        };
        let t2 = std::time::Instant::now();
        record_mesh_shader_command_buffer(
            &self.device,
            &self.vulkan_application_data,
            image_index,
            &wr.mesh_shader_pipeline,
            &wr.cull_compact_pipeline,
            &wr.voxel_pool,
            &self.depth_pyramid_pipeline,
            &cull_push,
            self.depth_pyramid_needs_init,
            svdag_args.as_ref().map(|(sp, c, t, m)| (*sp, c, t, m)),
            &self.ui,
            Some((&wr.heightmap_pipeline, &wr.heightmap_pool, &heightmap_visible, &heightmap_push)),
            self.timing_query_pool,
        )?;
        self.depth_pyramid_needs_init = false;
        let dt_record = t2.elapsed();

        let t3 = std::time::Instant::now();
        self.submit_command_buffer(image_index)?;
        let dt_submit = t3.elapsed();

        let t4 = std::time::Instant::now();
        self.present_frame(image_index, window)?;
        let dt_present = t4.elapsed();

        // Read back GPU timestamps from THIS frame (blocks until done; the
        // CPU is normally idle during this window because acquire blocks
        // anyway). Convert tick deltas → ms via timestamp_period.
        let mut ts = [0u64; TIMING_QUERY_COUNT as usize];
        let _ = self.device.get_query_pool_results(
            self.timing_query_pool,
            0,
            TIMING_QUERY_COUNT,
            std::mem::size_of_val(&ts),
            ts.as_mut_ptr() as *mut std::ffi::c_void,
            std::mem::size_of::<u64>() as u64,
            vk::QueryResultFlags::_64 | vk::QueryResultFlags::WAIT,
        );
        let to_ms = |ticks: u64| -> f64 { (ticks as f64) * self.timing_period_ns / 1_000_000.0 };
        let gpu_sky = to_ms(ts[1] - ts[0]);
        let gpu_phase1 = to_ms(ts[2] - ts[1]);
        let gpu_pyramid = to_ms(ts[3] - ts[2]);
        let gpu_phase2 = to_ms(ts[4] - ts[3]);
        let gpu_svdag = to_ms(ts[5] - ts[4]);
        let gpu_ui = to_ms(ts[6] - ts[5]);
        let gpu_total = to_ms(ts[6] - ts[0]);

        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;
        let dt_total = t_total.elapsed();

        // Rolling 60-frame average; write to debug.log once per second.
        struct PerfAccum {
            update_chunks: u128,
            acquire: u128,
            record: u128,
            submit: u128,
            present: u128,
            total: u128,
            gpu_sky: f64,
            gpu_phase1: f64,
            gpu_pyramid: f64,
            gpu_phase2: f64,
            gpu_svdag: f64,
            gpu_ui: f64,
            gpu_total: f64,
            n: u32,
        }
        static PERF: std::sync::Mutex<PerfAccum> = std::sync::Mutex::new(PerfAccum {
            update_chunks: 0, acquire: 0, record: 0, submit: 0, present: 0, total: 0,
            gpu_sky: 0.0, gpu_phase1: 0.0, gpu_pyramid: 0.0, gpu_phase2: 0.0,
            gpu_svdag: 0.0, gpu_ui: 0.0, gpu_total: 0.0, n: 0,
        });
        let mut p = PERF.lock().unwrap();
        p.update_chunks += dt_update_chunks.as_micros();
        p.acquire += dt_acquire.as_micros();
        p.record += dt_record.as_micros();
        p.submit += dt_submit.as_micros();
        p.present += dt_present.as_micros();
        p.total += dt_total.as_micros();
        p.gpu_sky += gpu_sky;
        p.gpu_phase1 += gpu_phase1;
        p.gpu_pyramid += gpu_pyramid;
        p.gpu_phase2 += gpu_phase2;
        p.gpu_svdag += gpu_svdag;
        p.gpu_ui += gpu_ui;
        p.gpu_total += gpu_total;
        p.n += 1;
        if p.n >= 60 {
            let n = p.n as f64;
            let resident = resident_count;
            let msg = format!(
                "[perf avg over {}f] cpu_total={:.1}ms acquire={:.1} record={:.1} submit={:.1} present={:.1} update={:.1} | gpu_total={:.1}ms sky={:.1} phase1={:.1} pyramid={:.1} phase2={:.1} svdag={:.1} ui={:.1} | resident={}\n",
                p.n,
                p.total as f64 / n / 1000.0,
                p.acquire as f64 / n / 1000.0,
                p.record as f64 / n / 1000.0,
                p.submit as f64 / n / 1000.0,
                p.present as f64 / n / 1000.0,
                p.update_chunks as f64 / n / 1000.0,
                p.gpu_total / n,
                p.gpu_sky / n,
                p.gpu_phase1 / n,
                p.gpu_pyramid / n,
                p.gpu_phase2 / n,
                p.gpu_svdag / n,
                p.gpu_ui / n,
                resident,
            );
            let _ = std::fs::write("debug.log", &msg);
            *p = PerfAccum {
                update_chunks: 0, acquire: 0, record: 0, submit: 0, present: 0, total: 0,
                gpu_sky: 0.0, gpu_phase1: 0.0, gpu_pyramid: 0.0, gpu_phase2: 0.0,
                gpu_svdag: 0.0, gpu_ui: 0.0, gpu_total: 0.0, n: 0,
            };
        }
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
        let player_world = camera.position.as_dvec3();
        let player_cx = (camera.position.x / CHUNK_SIZE as f32).floor() as i32;
        let player_cy = (camera.position.y / CHUNK_SIZE as f32).floor() as i32;
        let player_cz = (camera.position.z / CHUNK_SIZE as f32).floor() as i32;
        let delta = wr.world.update(player_world);

        wr.svdag_pool.tick();

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

        // Working-set streaming: World::update bounds delta.loaded /
        // delta.unloaded to a (2·WORLD_DISTANCE+1)² × radial column window
        // around the player. Everything outside that window is rendered by
        // the heightmap LOD path, not the mesh pool.
        for pos in &delta.unloaded {
            wr.voxel_pool.invalidate_neighbor_boundaries(*pos, &wr.world);
            wr.voxel_pool.remove_chunk(pos);
        }
        // Track which newly-loaded chunks were uniform-opaque so we can
        // re-check their neighbors for "now buried" after this batch is done.
        let mut newly_opaque: Vec<crate::voxel::sphere::ChunkPos> = Vec::new();
        for pos in &delta.loaded {
            if wr.voxel_pool.has_chunk(pos) {
                continue;
            }
            if let Some(chunk) = wr.world.get_chunk_at(*pos) {
                if chunk.is_uniform_air() {
                    continue;
                }
                if chunk.is_uniform_opaque() {
                    newly_opaque.push(*pos);
                    if neighbors_all_opaque(&wr.world, *pos) {
                        continue;
                    }
                }
                let chunk_ptr = chunk as *const _;
                wr.voxel_pool.upload_chunk(*pos, &*chunk_ptr, &wr.world);
                wr.voxel_pool.invalidate_neighbor_boundaries(*pos, &wr.world);
            }
        }
        // Second sweep: each newly-opaque chunk may have just turned a
        // previously-uploaded neighbor into a buried chunk. Re-check the
        // 6 axis-neighbors of every newly-opaque chunk and evict any that
        // are now buried.
        use crate::voxel::sphere::ChunkPos;
        let mut to_evict: std::collections::HashSet<ChunkPos> = Default::default();
        for &cp in &newly_opaque {
            let neighbors = [
                ChunkPos { face: cp.face, cx: cp.cx + 1, cy: cp.cy, cz: cp.cz },
                ChunkPos { face: cp.face, cx: cp.cx - 1, cy: cp.cy, cz: cp.cz },
                ChunkPos { face: cp.face, cx: cp.cx, cy: cp.cy + 1, cz: cp.cz },
                ChunkPos { face: cp.face, cx: cp.cx, cy: cp.cy - 1, cz: cp.cz },
                ChunkPos { face: cp.face, cx: cp.cx, cy: cp.cy, cz: cp.cz + 1 },
                ChunkPos { face: cp.face, cx: cp.cx, cy: cp.cy, cz: cp.cz - 1 },
            ];
            for n in neighbors {
                if !wr.voxel_pool.has_chunk(&n) {
                    continue;
                }
                let Some(nc) = wr.world.get_chunk_at(n) else { continue };
                if nc.is_uniform_opaque() && neighbors_all_opaque(&wr.world, n) {
                    to_evict.insert(n);
                }
            }
        }
        for cp in to_evict {
            wr.voxel_pool.invalidate_neighbor_boundaries(cp, &wr.world);
            wr.voxel_pool.remove_chunk(&cp);
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

        // --- Heightmap tile generation (replaces LOD-3 and LOD-4) ---
        wr.heightmap_pool.tick();
        let player_world_d = glam::DVec3::new(camera.position.x as f64, camera.position.y as f64, camera.position.z as f64);
        for mesh in wr.heightmap_generator.receive() {
            wr.heightmap_in_flight.remove(&mesh.key);
            wr.heightmap_pool.upload_tile(&mesh, player_world_d);
        }
        // The heightmap covers regions the mesh-chunk working set does not
        // cover. When the player is in the terrain altitude band, mesh
        // chunks reach `chunked_arc` radians around the player; the
        // heightmap stays outside that. When the player is above terrain,
        // mesh chunks vanish (3D Chebyshev cube above the band) and the
        // heightmap fills from angular distance 0.
        let chunked_arc = chunked_arc_radians();
        // Heightmap shadow uses the same altitude cutoff as the mesh
        // streamer: when the player is below `ORBITAL_CUTOFF_BLOCKS`, mesh
        // chunks cover the inner area around the player and the heightmap
        // is excluded from that arc. Above the cutoff there are no mesh
        // chunks, so the heightmap fills from angle 0.
        //
        // The shadow is shrunk by `HEIGHTMAP_OVERLAP_ARC` so the heightmap
        // keeps rendering in a thin ring just inside the mesh frontier.
        // This hides the load-in strip during descent: while the outermost
        // mesh chunks are still being generated, the heightmap is already
        // there covering the same angular region. At any altitude where
        // the strip would be visible, the two layers agree on the surface
        // to within sub-pixel, so the overlap doesn't z-fight noticeably.
        let altitude_blocks =
            player_world_d.length() - crate::voxel::sphere::PLANET_RADIUS_BLOCKS as f64;
        let mesh_shadow_arc = if altitude_blocks <= crate::voxel::world::ORBITAL_CUTOFF_BLOCKS {
            (chunked_arc - HEIGHTMAP_OVERLAP_ARC).max(0.0)
        } else {
            0.0
        };
        let hm_max_angle = chunked_arc + HEIGHTMAP_BANDS.last().map(|b| b.4).unwrap_or(0.0) + 0.05;
        wr.heightmap_pool
            .evict_outside_band(player_world_d, mesh_shadow_arc, hm_max_angle);
        schedule_heightmap_generation(wr, player_world_d, mesh_shadow_arc);

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
    (8, 2, 3, MESH_DISTANCE, LOD1_DISTANCE + 8), // LOD-1: covers LOD-0 area (8-24) as fallback
    (16, 4, 4, LOD0_DISTANCE, LOD2_DISTANCE + 16), // LOD-2: covers LOD-1 area (24-48) as fallback
                                                 // LOD-3 and LOD-4 replaced by rasterized heightmap tiles (HEIGHTMAP_BANDS)
];
const MAX_LOD_IN_FLIGHT: usize = 32;
const LOD_SUBMISSIONS_PER_FRAME: usize = 32;

/// Schedule direct LOD terrain generation for positions beyond LOD-0 range.
/// Returns true if any LOD work was submitted this frame.
unsafe fn schedule_lod_generation(wr: &mut WorldResources, player_cx: i32, player_cy: i32, player_cz: i32) -> bool {
    // Phase B2c: LOD super-chunks generate from noise into the SVDAG
    // ray-march pipeline, which still operates in flat world space and
    // would render slabs through the sphere-projected mesh. The tiny
    // planet fits entirely in the mesh range, so disable until the SVDAG
    // path is rebuilt for sphere geometry.
    let _ = (wr, player_cx, player_cy, player_cz);
    return false;
    #[allow(unreachable_code)]
    {
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

/// Heightmap tile bands: (tile_chunks_per_side, grid_posts, coarse_grid_posts,
/// extra_min_angle, extra_max_angle). The actual angular range is computed
/// at runtime as `chunked_arc + extra_*`, where `chunked_arc = WORLD_DISTANCE
/// * CHUNK_SIZE / PLANET_RADIUS` is the angular reach of the mesh-shader
/// chunk renderer. Tiles never overlap chunk geometry: the inner band starts
/// exactly where chunks stop. On a small planet where the chunk reach
/// already covers > π radians, no tile renders at all (correct — the planet
/// is fully chunked).
/// Bands are interpreted as `(tile_chunks, grid_posts, coarse_grid_posts,
/// extra_min, extra_max)`. The angular range of a band is
/// `[max(0, chunked_arc + extra_min), chunked_arc + extra_max]`. Negative
/// `extra_min` allows a band to start before mesh-shader reach so the
/// heightmap covers directly-below the player even when the mesh chunks have
/// been evicted (e.g. at altitude where the 3D Chebyshev cube no longer
/// intersects the surface band). Mesh chunks and the inner heightmap tiles
/// are coincident in world position (both derive from the same canonical
/// `surface_radius_at_world` after the parity fix), so where mesh chunks do
/// exist they win the depth test cleanly without z-fighting in practice.
const HEIGHTMAP_BANDS: &[(i32, usize, usize, f32, f32)] = &[
    (8, 65, 17, -10.0, 0.50),
    (16, 65, 17, 0.50, 1.50),
];

fn chunked_arc_radians() -> f32 {
    (WORLD_DISTANCE as f32 * CHUNK_SIZE as f32) / crate::voxel::sphere::PLANET_RADIUS_BLOCKS as f32
}
const MAX_HEIGHTMAP_IN_FLIGHT: usize = 16;
const HEIGHTMAP_SUBMISSIONS_PER_FRAME: usize = 8;

/// Walk every (face, cx, cz) tile in the world, classify by angular distance
/// to the player, and submit any that fall inside a heightmap band and are
/// not already loaded or in flight. The tile-aligned (cx, cz) is the
/// chunk-grid coordinate of the tile's lower corner; tiles on a face never
/// straddle the seam — neighboring face tiles cover the seam from the other
/// side, and the curved meshes meet at the projected cube edge.
fn schedule_heightmap_generation(
    wr: &mut WorldResources,
    player_world: glam::DVec3,
    mesh_shadow_arc: f32,
) {
    use crate::voxel::heightmap_generator::{schedule_candidates, HeightmapRequest};
    use crate::voxel::sphere::FACE_SIDE_CHUNKS;
    if wr.heightmap_in_flight.len() >= MAX_HEIGHTMAP_IN_FLIGHT {
        return;
    }
    let chunked_arc = chunked_arc_radians();
    let pool = &wr.heightmap_pool;
    let in_flight = &wr.heightmap_in_flight;
    let is_loaded_or_in_flight = |key| pool.has_tile(&key) || in_flight.contains(&key);

    let picks = schedule_candidates(
        player_world,
        HEIGHTMAP_BANDS,
        chunked_arc,
        mesh_shadow_arc,
        FACE_SIDE_CHUNKS,
        &is_loaded_or_in_flight,
        HEIGHTMAP_SUBMISSIONS_PER_FRAME,
    );

    for cand in picks {
        // Pull the per-band geometry config from HEIGHTMAP_BANDS.
        let (_, grid_posts, coarse_grid_posts, _, _) = HEIGHTMAP_BANDS[cand.band_idx as usize];
        wr.heightmap_generator.request(HeightmapRequest {
            key: cand.key,
            lod: cand.band_idx,
            tile_chunks_per_side: cand.tile_chunks,
            grid_posts,
            coarse_grid_posts,
            seed: wr.seed,
        });
        wr.heightmap_in_flight.insert(cand.key);
    }
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
            wr.cull_compact_pipeline.update_depth_pyramid(&self.device, &self.vulkan_application_data);
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
            wr.cull_compact_pipeline.destroy(&self.device);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graphical_core::heightmap_pool;

    /// Estimate the steady-state working set of heightmap tiles for the
    /// current band layout. Pure geometry — no GPU, no allocation.
    ///
    /// Each band covers an angular annulus `[θ_min, θ_max]` on the sphere.
    /// The annulus area in steradians is `2π · ∫ sin(θ) dθ` from θ_min to
    /// θ_max, which equals `2π · (cos(θ_min) − cos(θ_max))`. Each tile
    /// occupies a square patch of side `tile_arc = tile_chunks · CHUNK_SIZE
    /// / PLANET_RADIUS` radians, so its area is `tile_arc²`. Tile count is
    /// the ratio, ceilinged.
    fn working_set_estimate() -> usize {
        let r = crate::voxel::sphere::PLANET_RADIUS_BLOCKS as f32;
        let chunked_arc = chunked_arc_radians();
        let mut total = 0usize;
        for &(tile_chunks, _grid, _coarse, extra_min, extra_max) in HEIGHTMAP_BANDS {
            // Must mirror the runtime clamp in `schedule_heightmap_generation`
            // exactly: bands may have negative `extra_min` to start the inner
            // ring at the player's directly-below position.
            let theta_min = (chunked_arc + extra_min).max(0.0).min(std::f32::consts::PI);
            let theta_max = (chunked_arc + extra_max).max(0.0).min(std::f32::consts::PI);
            if theta_max <= theta_min {
                continue;
            }
            let ring_area =
                2.0 * std::f32::consts::PI * (theta_min.cos() - theta_max.cos());
            let tile_arc = (tile_chunks as f32 * CHUNK_SIZE as f32) / r;
            let tile_area = tile_arc * tile_arc;
            total += (ring_area / tile_area).ceil() as usize;
        }
        total
    }

    /// The heightmap tile pool must be large enough to hold the steady-state
    /// working set of every band, with headroom for eviction lag during
    /// player movement. If a future band-config change pushes the bound
    /// above `MAX_TILES`, this test fails — bump `MAX_TILES` (or coarsen a
    /// band) and update its derivation comment.
    #[test]
    fn heightmap_band_working_set_fits_pool() {
        let bound = working_set_estimate();
        // 1.4× headroom for transient overshoot during fast player motion.
        let with_headroom = (bound as f32 * 1.4) as usize;
        assert!(
            with_headroom <= heightmap_pool::MAX_TILES as usize,
            "heightmap working set {} (×1.4 = {}) exceeds MAX_TILES={} \
             at PLANET_RADIUS_BLOCKS={}, chunked_arc={:.3} rad. \
             Bands need to be coarsened or MAX_TILES bumped (with derivation).",
            bound,
            with_headroom,
            heightmap_pool::MAX_TILES,
            crate::voxel::sphere::PLANET_RADIUS_BLOCKS,
            chunked_arc_radians(),
        );
    }
}
