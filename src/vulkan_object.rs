use anyhow::anyhow;
use vulkanalia::{Device, Entry, Instance, vk};
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use winit::window::Window;
use crate::{create_command_buffers, create_command_pool, create_framebuffers, create_instance, create_logical_device, create_pipeline, create_render_pass, create_swapchain, create_swapchain_image_views, create_sync_objects, pick_physical_device, VALIDATION_ENABLED};
use vulkanalia::{
    vk::{KhrSurfaceExtension, KhrSwapchainExtension},
    window as vk_window,
    prelude::v1_0::*,
};
use vulkanalia::vk::ExtDebugUtilsExtension;

#[derive(Clone, Debug, Default)]
pub struct ApplicationData {
    pub surface: vk::SurfaceKHR,
    pub messenger: vk::DebugUtilsMessengerEXT,
    pub physical_device: vk::PhysicalDevice,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue,
    pub swapchain_format: vk::Format,
    pub swapchain_extent: vk::Extent2D,
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
    pub render_finished_semaphore: vk::Semaphore
}

#[derive(Clone, Debug)]
pub struct VulkanApplication {
    entry: Entry,
    instance: Instance,
    data: ApplicationData,
    device: Device
}

impl VulkanApplication {
    /// Creates our Vulkan app.
    pub unsafe fn create(window: &Window) -> anyhow::Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let mut data = ApplicationData::default();
        let instance = create_instance(window, &entry, &mut data)?;
        data.surface = vk_window::create_surface(&instance, &window, &window)?;
        pick_physical_device(&instance, &mut data)?;
        let device = create_logical_device(&entry,&instance, &mut data)?;
        create_swapchain(window, &instance, &device, &mut data)?;
        create_swapchain_image_views(&device, &mut data)?;
        create_render_pass(&instance, &device, &mut data)?;
        create_pipeline(&device, &mut data)?;
        create_framebuffers(&device, &mut data)?;
        create_command_pool(&instance, &device, &mut data)?;
        create_command_buffers(&device, &mut data)?;
        create_sync_objects(&device, &mut data)?;
        Ok(Self {entry, instance, data, device})
    }

    /// Renders a frame for our Vulkan app.
    pub unsafe fn render(&mut self, window: &Window) -> anyhow::Result<()> {
        let image_index = self.device.acquire_next_image_khr(self.data.swapchain, u64::MAX, self.data.image_available_semaphore, vk::Fence::null())?.0 as usize;
        let wait_semaphores = &[self.data.image_available_semaphore];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finished_semaphore];
        let info_to_submit_to_queue = vk::SubmitInfo::builder().wait_semaphores(wait_semaphores).wait_dst_stage_mask(wait_stages).command_buffers(command_buffers).signal_semaphores(signal_semaphores);
        self.device.queue_submit(self.data.graphics_queue, &[info_to_submit_to_queue], vk::Fence::null())?;
        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);
        self.device.queue_present_khr(self.data.present_queue, &present_info)?;

        Ok(())
    }

    /// Destroys our Vulkan app.
    pub unsafe fn destroy(&mut self) {
        self.device.destroy_semaphore(self.data.render_finished_semaphore, None);
        self.device.destroy_semaphore(self.data.image_available_semaphore, None);
        self.device.destroy_command_pool(self.data.command_pool, None);
        self.data.framebuffers.iter().for_each(|f| self.device.destroy_framebuffer(*f, None));
        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device.destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);
        self.data.swapchain_image_views.iter().for_each(|v| self.device.destroy_image_view(*v, None));
        self.device.destroy_swapchain_khr(self.data.swapchain, None);
        self.device.destroy_device(None);
        if VALIDATION_ENABLED {
            self.instance.destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }
        self.instance.destroy_surface_khr(self.data.surface, None);
        self.instance.destroy_instance(None);
    }
}