use crate::graphical_core::{
    buffers::allocate_and_fill_buffer,
    extra::{create_command_buffers, create_command_pool, create_frame_buffers, create_instance, create_logical_device, create_sync_objects},
    gpu::choose_gpu,
    pipeline::create_pipeline,
    render_pass::create_render_pass,
    swapchain::{create_swapchain, create_swapchain_image_views},
    texture_mapping::{allocate_descriptor_set, create_descriptor_pool, create_descriptor_set_layout, create_texture_image, update_descriptor_set},
    MAX_FRAMES_IN_FLIGHT,
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
    pub swapchain_accepted_images_width_and_height: vk::Extent2D,
    pub swapchain: vk::SwapchainKHR,
    pub swapchain_images: Vec<vk::Image>,
    pub swapchain_image_views: Vec<vk::ImageView>,
    pub render_pass: vk::RenderPass,
    pub pipeline_layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    pub framebuffers: Vec<vk::Framebuffer>,
    pub command_pool: vk::CommandPool,
    pub command_buffers: Vec<vk::CommandBuffer>,
    pub image_available_semaphore: vk::Semaphore,
    pub render_finished_semaphore: vk::Semaphore,
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub(crate) in_flight_fences: Vec<vk::Fence>,
    pub(crate) images_in_flight: Vec<vk::Fence>,
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_buffer_memory: vk::DeviceMemory,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub descriptor_pool: vk::DescriptorPool,
    pub texture_image: vk::Image,
    pub texture_memory: vk::DeviceMemory,
    pub texture_image_view: vk::ImageView,
    pub texture_sampler: vk::Sampler,
    pub descriptor_set: vk::DescriptorSet,
}

/// Represents a single vertex with position and color data.
///
/// # Memory Layout
/// `#[repr(C)]` ensures the struct has a predictable memory layout matching C/Vulkan expectations.
/// This is critical because the GPU needs to know exactly where each field is in memory.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Vertex {
    position: [f32; 3],
    uv_coordinate: [f32; 2],
}
const VERTICES: [Vertex; 8] = [
    Vertex {
        position: [-1.0, -1.0, 0.3],
        uv_coordinate: [0.0, 1.0],
    },
    Vertex {
        position: [1.0, -1.0, 0.3],
        uv_coordinate: [1.0, 1.0],
    },
    Vertex {
        position: [1.0, 1.0, 0.3],
        uv_coordinate: [1.0, 0.0],
    },
    Vertex {
        position: [-1.0, 1.0, 0.3],
        uv_coordinate: [0.0, 0.0],
    },
    Vertex {
        position: [-1.0, -1.0, 0.7],
        uv_coordinate: [0.0, 1.0],
    },
    Vertex {
        position: [1.0, -1.0, 0.7],
        uv_coordinate: [1.0, 1.0],
    },
    Vertex {
        position: [1.0, 1.0, 0.7],
        uv_coordinate: [1.0, 0.0],
    },
    Vertex {
        position: [-1.0, 1.0, 0.7],
        uv_coordinate: [0.0, 0.0],
    },
];

const INDICES: [u16; 36] = [
    0, 1, 2, 0, 2, 3, // Front face
    5, 4, 7, 5, 7, 6, // Back Face
    1, 5, 6, 1, 6, 2, // Right face
    4, 0, 3, 4, 3, 7, // Left face
    3, 2, 6, 3, 6, 7, // Top face
    4, 5, 1, 4, 1, 0, // Bottom face
];

#[derive(Clone, Debug)]
pub struct VulkanApplication {
    vulkan_entry_point: Entry,
    vulkan_instance: Instance,
    vulkan_application_data: VulkanApplicationData,
    device: Device,
    frame: usize,
    pub(crate) resized: bool,
}
impl VulkanApplication {
    /// Initializes the complete Vulkan rendering pipeline and all required resources.
    ///
    /// # Process Overview - Initialization Phases
    ///
    /// **Phase 1: Core Infrastructure**
    /// 1. Load Vulkan library and create entry point
    /// 2. Create Vulkan instance (validation layers if debug build)
    /// 3. Create window surface (OS-specific rendering target)
    /// 4. Choose physical GPU
    /// 5. Create logical device and retrieve queue handles
    ///
    /// **Phase 2: Presentation Setup**
    ///
    /// 6. Create swapchain (double/triple buffering for smooth presentation)
    /// 7. Create image views for swapchain images
    ///
    /// **Phase 3: Rendering Pipeline**
    ///
    /// 8. Create render pass (defines rendering structure)
    /// 9. Create descriptor set layout (shader resource interface)
    /// 10. Create graphics pipeline (vertex processing + fragment shading)
    ///
    /// **Phase 4: Command Infrastructure**
    ///
    /// 11. Create framebuffers (attachments for rendering)
    /// 12. Create command pool (allocator for command buffers)
    ///
    /// **Phase 5: Texture Resources**
    ///
    /// 13. Load texture from disk
    /// 14. Create staging buffer, GPU image, transfer data
    /// 15. Create image view and sampler
    /// 16. Create descriptor pool
    /// 17. Allocate and update descriptor set with texture
    ///
    /// **Phase 6: Geometry Buffers**
    ///
    /// 18. Create and fill vertex buffer (cube vertices)
    /// 19. Create and fill index buffer (cube triangles)
    ///
    /// **Phase 7: Command Recording & Synchronization**
    ///
    /// 20. Record command buffers (rendering commands for each swapchain image)
    /// 21. Create synchronization objects (semaphores and fences)
    ///
    /// # Parameters
    /// - `user_window`: The OS window to render into
    ///
    /// # Returns
    /// A fully initialized `VulkanApplication` ready to render frames
    ///
    /// # Safety
    /// This function is marked `unsafe` because it:
    /// - Calls many unsafe Vulkan API functions
    /// - Creates resources that must be properly cleaned up
    /// - Initializes GPU state that could crash if misused
    ///
    /// # Errors
    /// Returns an error if any initialization step fails:
    /// - Vulkan library cannot be loaded
    /// - No suitable GPU found
    /// - Out of GPU memory
    /// - Shader compilation fails
    /// - Texture file not found
    ///
    /// # Note
    /// All created resources are stored in `VulkanApplicationData` and must be
    /// destroyed via `destroy_vulkan_application()` before the application exits.
    pub unsafe fn create_vulkan_application(user_window: &Window) -> anyhow::Result<Self> {
        let platform_specific_vulkan_api = LibloadingLoader::new(LIBRARY)?;
        let vulkan_api_entry_point = Entry::new(platform_specific_vulkan_api).map_err(|b| anyhow!("{}", b))?;
        let mut vulkan_application_data = VulkanApplicationData::default();
        let instance = create_instance(user_window, &vulkan_api_entry_point, &mut vulkan_application_data)?;
        vulkan_application_data.surface = vulkan_window::create_surface(&instance, &user_window, &user_window)?;

        choose_gpu(&instance, &mut vulkan_application_data)?;
        let device = create_logical_device(&vulkan_api_entry_point, &instance, &mut vulkan_application_data)?;
        create_swapchain(user_window, &instance, &device, &mut vulkan_application_data)?;
        create_swapchain_image_views(&device, &mut vulkan_application_data)?;
        create_render_pass(&instance, &device, &mut vulkan_application_data)?;
        create_descriptor_set_layout(&device, &mut vulkan_application_data)?;
        create_pipeline(&device, &mut vulkan_application_data)?;

        create_frame_buffers(&device, &mut vulkan_application_data)?;
        create_command_pool(&instance, &device, &mut vulkan_application_data)?;

        let (texture_image, texture_memory, texture_image_view, texture_sampler) =
            create_texture_image(&device, &instance, &mut vulkan_application_data)?;

        create_descriptor_pool(&device, &mut vulkan_application_data)?;
        let descriptor_sets = allocate_descriptor_set(
            &device,
            vulkan_application_data.descriptor_pool,
            vulkan_application_data.descriptor_set_layout,
        )?;
        let descriptor_set = descriptor_sets
            .first()
            .copied()
            .ok_or_else(|| anyhow::anyhow!("Failed to allocate descriptor set"))?;

        update_descriptor_set(&device, descriptor_set, texture_image_view, texture_sampler);

        vulkan_application_data.texture_image = texture_image;
        vulkan_application_data.texture_memory = texture_memory;
        vulkan_application_data.texture_image_view = texture_image_view;
        vulkan_application_data.texture_sampler = texture_sampler;
        vulkan_application_data.descriptor_set = descriptor_set;

        let vertex_buffer_size_in_bytes = (VERTICES.len() * size_of::<Vertex>()) as u64;
        let (vertex_buffer, vertex_buffer_memory) = allocate_and_fill_buffer(
            &VERTICES,
            vertex_buffer_size_in_bytes,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            &device,
            &instance,
            &mut vulkan_application_data,
        )?;

        let index_buffer_size_in_bytes = INDICES.len() as u64;
        let (index_buffer, index_buffer_memory) = allocate_and_fill_buffer(
            &INDICES,
            index_buffer_size_in_bytes,
            vk::BufferUsageFlags::INDEX_BUFFER,
            &device,
            &instance,
            &mut vulkan_application_data,
        )?;

        vulkan_application_data.vertex_buffer = vertex_buffer;
        vulkan_application_data.index_buffer = index_buffer;
        vulkan_application_data.vertex_buffer_memory = vertex_buffer_memory;
        vulkan_application_data.index_buffer_memory = index_buffer_memory;

        create_command_buffers(&device, &mut vulkan_application_data)?;
        create_sync_objects(&device, &mut vulkan_application_data)?;

        Ok(Self {
            vulkan_entry_point: vulkan_api_entry_point,
            vulkan_instance: instance,
            vulkan_application_data,
            device,
            frame: 0,
            resized: false,
        })
    }

    /// Renders a single frame to the screen using the Vulkan rendering pipeline.
    ///
    /// # Process Overview - Frame Rendering Cycle
    ///
    /// **Phase 1: Synchronization**
    /// 1. Wait for the current frame's fence (ensure GPU isn't still using resources)
    /// 2. Acquire next swapchain image (which backbuffer to render to)
    /// 3. Wait if that image is still being presented
    ///
    /// **Phase 2: Command Submission**
    ///
    /// 4. Reset the fence for this frame
    /// 5. Submit pre-recorded command buffer to graphics queue:
    ///    - Wait for image available semaphore
    ///    - Execute: begin render pass → bind pipeline → bind buffers → draw → end
    ///    - Signal render finished semaphore
    /// 6. Wait for fence to track completion
    ///
    /// **Phase 3: Presentation**
    ///
    /// 7. Present rendered image to screen:
    ///    - Wait for render finished semaphore
    ///    - Display image in window
    /// 8. Handle swapchain recreation if needed (window resized or suboptimal)
    ///
    /// **Phase 4: Frame Pacing**
    ///
    /// 9. Advance frame counter (for double/triple buffering)
    ///
    /// # Parameters
    /// - `window`: The window to render into (needed for swapchain recreation)
    ///
    /// # Safety
    /// This function is marked `unsafe` because it calls many unsafe Vulkan functions
    /// for queue operations and synchronization.
    ///
    /// # Errors
    /// Returns an error if:
    /// - Fence wait times out
    /// - Queue submission fails
    /// - Swapchain recreation fails
    /// - Presentation fails (except OUT_OF_DATE, which triggers recreation)
    ///
    /// # Performance Note
    /// This function blocks on fences, which stalls the CPU until GPU work completes.
    /// For optimal performance, we'd want triple buffering with async submission.
    pub unsafe fn render_frame(&mut self, window: &Window) -> anyhow::Result<()> {
        self.device
            .wait_for_fences(&[self.vulkan_application_data.in_flight_fences[self.frame]], true, u64::MAX)?;

        let result = self.device.acquire_next_image_khr(
            self.vulkan_application_data.swapchain,
            u64::MAX,
            self.vulkan_application_data.image_available_semaphores[self.frame],
            vk::Fence::null(),
        );
        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e)),
        };

        if !self.vulkan_application_data.images_in_flight[image_index].is_null() {
            self.device
                .wait_for_fences(&[self.vulkan_application_data.images_in_flight[image_index]], true, u64::MAX)?;
        }

        self.vulkan_application_data.images_in_flight[image_index] = self.vulkan_application_data.in_flight_fences[self.frame];

        let semaphore_to_wait_on_before_execution = &[self.vulkan_application_data.image_available_semaphores[self.frame]];
        let stage_of_pipeline_to_wait_on_before_execution = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffer_to_use_at_execution = &[self.vulkan_application_data.command_buffers[image_index]];
        let semaphores_to_signal_after_command_buffer_finished_executing = &[self.vulkan_application_data.render_finished_semaphores[self.frame]];
        let info_to_submit_to_queue = vk::SubmitInfo::builder()
            .wait_semaphores(semaphore_to_wait_on_before_execution)
            .wait_dst_stage_mask(stage_of_pipeline_to_wait_on_before_execution)
            .command_buffers(command_buffer_to_use_at_execution)
            .signal_semaphores(semaphores_to_signal_after_command_buffer_finished_executing);

        self.device.reset_fences(&[self.vulkan_application_data.in_flight_fences[self.frame]])?;

        self.device.queue_submit(
            self.vulkan_application_data.graphics_queue,
            &[info_to_submit_to_queue],
            self.vulkan_application_data.in_flight_fences[self.frame],
        )?;

        let swapchains_to_present_images_to = &[self.vulkan_application_data.swapchain];
        let image_index_in_swapchain = &[image_index as u32];
        let image_presentation_configuration = vk::PresentInfoKHR::builder()
            .wait_semaphores(semaphores_to_signal_after_command_buffer_finished_executing)
            .swapchains(swapchains_to_present_images_to)
            .image_indices(image_index_in_swapchain);

        self.device.queue_wait_idle(self.vulkan_application_data.presentation_queue)?;
        let result = self
            .device
            .queue_present_khr(self.vulkan_application_data.presentation_queue, &image_presentation_configuration);

        let changed = result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);

        if changed {
            self.recreate_swapchain(window)?;
        }
        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }

    /// Recreates the swapchain and all dependent resources after window resize or invalidation.
    ///
    /// # Why Swapchain Recreation is Needed
    /// The swapchain is tied to window dimensions. When the window resizes, minimizes,
    /// or becomes suboptimal (e.g., moved to different monitor), the swapchain must be
    /// recreated with new dimensions.
    ///
    /// # Process Overview
    /// 1. Wait for all GPU work to finish (`device_wait_idle`)
    /// 2. Destroy old swapchain and dependent resources:
    ///    - Framebuffers
    ///    - Command buffers
    ///    - Graphics pipeline
    ///    - Pipeline layout
    ///    - Render pass
    ///    - Swapchain image views
    ///    - Swapchain
    /// 3. Recreate everything with new window dimensions:
    ///    - New swapchain matching current window size
    ///    - New image views for new swapchain images
    ///    - New render pass (references new format/dimensions)
    ///    - New pipeline (references new render pass)
    ///    - New framebuffers (references new image views)
    ///    - New command buffers (references new framebuffers)
    /// 4. Resize in-flight fence tracking
    ///
    /// # Parameters
    /// - `user_window`: The window (queried for new dimensions)
    ///
    /// # Safety
    /// This function is marked `unsafe` because it calls many unsafe Vulkan destruction
    /// and creation functions.
    ///
    /// # Why Destroy So Much?
    /// Many Vulkan resources reference each other. The pipeline references the render
    /// pass, which references the swapchain format. Framebuffers reference image views.
    /// Rather than track dependencies, it's simpler to destroy and recreate the entire
    /// chain.
    ///
    /// # Errors
    /// Returns an error if any recreation step fails (usually out of memory).
    pub unsafe fn recreate_swapchain(&mut self, user_window: &Window) -> anyhow::Result<()> {
        self.device.device_wait_idle()?;
        self.destroy_swapchain();
        create_swapchain(user_window, &self.vulkan_instance, &self.device, &mut self.vulkan_application_data)?;
        create_swapchain_image_views(&self.device, &mut self.vulkan_application_data)?;
        create_render_pass(&self.vulkan_instance, &self.device, &mut self.vulkan_application_data)?;
        create_pipeline(&self.device, &mut self.vulkan_application_data)?;
        create_frame_buffers(&self.device, &mut self.vulkan_application_data)?;
        create_command_buffers(&self.device, &mut self.vulkan_application_data)?;
        self.vulkan_application_data
            .images_in_flight
            .resize(self.vulkan_application_data.swapchain_images.len(), vk::Fence::null());
        Ok(())
    }

    /// Destroys the swapchain and all resources that depend on it.
    ///
    /// # Resources Destroyed
    /// In reverse order of creation (children before parents):
    /// 1. Framebuffers (reference image views)
    /// 2. Command buffers (freed back to pool)
    /// 3. Graphics pipeline (references pipeline layout and render pass)
    /// 4. Pipeline layout
    /// 5. Render pass (references swapchain format)
    /// 6. Swapchain image views (reference swapchain images)
    /// 7. Swapchain itself
    ///
    /// # Safety
    /// This function is marked `unsafe` because it calls unsafe Vulkan destruction functions.
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
        self.vulkan_application_data
            .swapchain_image_views
            .iter()
            .for_each(|image_view| self.device.destroy_image_view(*image_view, None));
        self.device.destroy_swapchain_khr(self.vulkan_application_data.swapchain, None);
    }

    /// Destroys all Vulkan resources and shuts down the rendering system.
    ///
    /// # Cleanup Order
    /// Resources must be destroyed in reverse order of creation to satisfy Vulkan's
    /// dependency requirements:
    ///
    /// 1. **Swapchain resources** (via `destroy_swapchain()`)
    /// 2. **Synchronization objects**: Fences, semaphores
    /// 3. **Command pool** (automatically frees command buffers)
    /// 4. **Geometry buffers**: Vertex buffer, index buffer, and their memory
    /// 5. **Texture resources**: Sampler, image view, image, memory
    /// 6. **Descriptor resources**: Pool (auto-frees sets), layout
    /// 7. **Logical device** (must be destroyed before instance)
    /// 8. **Surface** (OS window integration)
    /// 9. **Debug messenger** (if validation layers enabled)
    /// 10. **Instance** (last - owns all handles)
    ///
    /// # Safety
    /// This function is marked `unsafe` because it calls many unsafe Vulkan destruction
    /// functions and must be called exactly once before application exit.
    ///
    /// # Critical Requirements
    /// - Must be called before the window closes
    /// - Must be called before VulkanApplication is dropped
    /// - Cannot render after calling this (all resources are gone)
    /// - Missing any cleanup causes Vulkan validation errors and potential driver leaks
    ///
    /// # Why Manual Cleanup?
    /// Rust's Drop trait doesn't work well with Vulkan because:
    /// - Destruction order matters (Drop order isn't guaranteed)
    /// - Many operations need &mut self (Drop only gives self)
    /// - GPU operations may still be in flight (need explicit wait)
    ///
    /// Manual cleanup with `destroy_vulkan_application()` gives us full control.
    pub unsafe fn destroy_vulkan_application(&mut self) {
        self.device.destroy_sampler(self.vulkan_application_data.texture_sampler, None);
        self.device.destroy_image_view(self.vulkan_application_data.texture_image_view, None);
        self.device.destroy_image(self.vulkan_application_data.texture_image, None);
        self.device.free_memory(self.vulkan_application_data.texture_memory, None);

        self.device.destroy_descriptor_pool(self.vulkan_application_data.descriptor_pool, None);
        self.device
            .destroy_descriptor_set_layout(self.vulkan_application_data.descriptor_set_layout, None);

        self.destroy_swapchain();
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
        self.device.destroy_buffer(self.vulkan_application_data.vertex_buffer, None);
        self.device.free_memory(self.vulkan_application_data.vertex_buffer_memory, None);
        self.device.destroy_buffer(self.vulkan_application_data.index_buffer, None);
        self.device.free_memory(self.vulkan_application_data.index_buffer_memory, None);
        self.device.destroy_device(None);
        self.vulkan_instance.destroy_surface_khr(self.vulkan_application_data.surface, None);
        if VALIDATION_ENABLED {
            self.vulkan_instance
                .destroy_debug_utils_messenger_ext(self.vulkan_application_data.debug_messenger, None);
        }
        self.vulkan_instance.destroy_instance(None);
    }

    /// Presents a rendered image to the swapchain for display on screen.
    ///
    /// # Purpose
    /// After the GPU finishes rendering to a swapchain image, this function tells
    /// the presentation engine: "Display this image in the window now."
    ///
    /// # Parameters
    /// - `present_info`: Configuration specifying:
    ///   - Which swapchain to present to
    ///   - Which image index to present
    ///   - Which semaphores to wait on (ensures rendering finished)
    ///
    /// # Safety
    /// This function is marked `unsafe` because it calls the unsafe Vulkan function
    /// `queue_present_khr`.
    ///
    /// # Errors
    /// Panics if presentation fails. This function expects the error to be handled
    /// by the caller.
    ///
    /// # Note
    /// This function is currently unused
    unsafe fn present_image_to_swapchain(&mut self, present_info: vk::PresentInfoKHRBuilder) {
        self.device
            .queue_present_khr(self.vulkan_application_data.presentation_queue, &present_info)
            .expect("Presenting the image to the swapchain resulted in an error!");
    }
}
