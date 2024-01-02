use anyhow::anyhow;
use winit::window::Window;
use vulkanalia::{
    loader::{LibloadingLoader, LIBRARY},
    vk::{KhrSurfaceExtension, KhrSwapchainExtension, ExtDebugUtilsExtension},
    window as vulkan_window,
    prelude::v1_0::*,
};
use crate::graphical_core::{
    gpu::pick_physical_device,
    swapchain::{create_swapchain, create_swapchain_image_views},
    render_pass::create_render_pass,
    pipeline::create_pipeline,
    extra::{create_command_buffers, create_command_pool, create_frame_buffers, create_instance, create_logical_device, create_sync_objects}
};
use crate::VALIDATION_ENABLED;

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
    pub render_finished_semaphore: vk::Semaphore
}
#[derive(Clone, Debug)]
pub struct VulkanApplication {
    vulkan_entry_point: Entry,
    vulkan_instance: Instance,
    data: VulkanApplicationData,
    vulkan_device: Device
}
impl VulkanApplication {
    pub unsafe fn create_vulkan_application(user_window: &Window) -> anyhow::Result<Self> {
        let platform_specific_vulkan_api = LibloadingLoader::new(LIBRARY)?;
        let vulkan_api_entry_point = Entry::new(platform_specific_vulkan_api).map_err(|b| anyhow!("{}", b))?;
        let mut vulkan_application_data = VulkanApplicationData::default();
        let vulkan_instance = create_instance(user_window, &vulkan_api_entry_point, &mut vulkan_application_data)?;
        vulkan_application_data.surface = vulkan_window::create_surface(&vulkan_instance, &user_window, &user_window)?;
        pick_physical_device(&vulkan_instance, &mut vulkan_application_data)?;
        let vulkan_logical_device = create_logical_device(&vulkan_api_entry_point, &vulkan_instance, &mut vulkan_application_data)?;
        create_swapchain(user_window, &vulkan_instance, &vulkan_logical_device, &mut vulkan_application_data)?;
        create_swapchain_image_views(&vulkan_logical_device, &mut vulkan_application_data)?;
        create_render_pass(&vulkan_instance, &vulkan_logical_device, &mut vulkan_application_data)?;
        create_pipeline(&vulkan_logical_device, &mut vulkan_application_data)?;
        create_frame_buffers(&vulkan_logical_device, &mut vulkan_application_data)?;
        create_command_pool(&vulkan_instance, &vulkan_logical_device, &mut vulkan_application_data)?;
        create_command_buffers(&vulkan_logical_device, &mut vulkan_application_data)?;
        create_sync_objects(&vulkan_logical_device, &mut vulkan_application_data)?;
        Ok(Self{vulkan_entry_point: vulkan_api_entry_point, vulkan_instance, data: vulkan_application_data, vulkan_device: vulkan_logical_device})
    }
    pub unsafe fn render_frame(&mut self, window: &Window) -> anyhow::Result<()> {
        let image_index = self.vulkan_device.acquire_next_image_khr(self.data.swapchain, u64::MAX, self.data.image_available_semaphore, vk::Fence::null())?.0 as usize;
        let wait_semaphores = &[self.data.image_available_semaphore];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.data.command_buffers[image_index]];
        let signal_semaphores = &[self.data.render_finished_semaphore];
        let info_to_submit_to_queue = vk::SubmitInfo::builder().wait_semaphores(wait_semaphores).wait_dst_stage_mask(wait_stages).command_buffers(command_buffers).signal_semaphores(signal_semaphores);
        self.vulkan_device.queue_submit(self.data.graphics_queue, &[info_to_submit_to_queue], vk::Fence::null())?;
        let swapchains = &[self.data.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);
        self.vulkan_device.queue_present_khr(self.data.presentation_queue, &present_info)?;

        Ok(())
    }
    pub unsafe fn destroy_vulkan_application(&mut self) {
        self.vulkan_device.destroy_semaphore(self.data.render_finished_semaphore, None);
        self.vulkan_device.destroy_semaphore(self.data.image_available_semaphore, None);
        self.vulkan_device.destroy_command_pool(self.data.command_pool, None);
        self.data.framebuffers.iter().for_each(|framebuffer| self.vulkan_device.destroy_framebuffer(*framebuffer, None));
        self.vulkan_device.destroy_pipeline(self.data.pipeline, None);
        self.vulkan_device.destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.vulkan_device.destroy_render_pass(self.data.render_pass, None);
        self.data.swapchain_image_views.iter().for_each(|image_view| self.vulkan_device.destroy_image_view(*image_view, None));
        self.vulkan_device.destroy_swapchain_khr(self.data.swapchain, None);
        self.vulkan_device.destroy_device(None);
        if VALIDATION_ENABLED {
            self.vulkan_instance.destroy_debug_utils_messenger_ext(self.data.debug_messenger, None);
        }
        self.vulkan_instance.destroy_surface_khr(self.data.surface, None);
        self.vulkan_instance.destroy_instance(None);
    }
}