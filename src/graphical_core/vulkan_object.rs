use anyhow::anyhow;
use winit::window::Window;
use vulkanalia::{
    loader::{LibloadingLoader, LIBRARY},
    vk::{KhrSurfaceExtension, KhrSwapchainExtension, ExtDebugUtilsExtension},
    window as vulkan_window,
    prelude::v1_0::*,
};
use crate::graphical_core::{
    gpu::choose_gpu,
    swapchain::{create_swapchain, create_swapchain_image_views},
    render_pass::create_render_pass,
    pipeline::create_pipeline,
    extra::{create_command_buffers, create_command_pool, create_frame_buffers, create_instance, create_logical_device, create_sync_objects},
    MAX_FRAMES_IN_FLIGHT
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
    pub render_finished_semaphore: vk::Semaphore,
    pub image_available_semaphores: Vec<vk::Semaphore>,
    pub render_finished_semaphores: Vec<vk::Semaphore>,
    pub(crate) in_flight_fences: Vec<vk::Fence>,
    pub(crate) images_in_flight: Vec<vk::Fence>
}
#[derive(Clone, Debug)]
pub struct VulkanApplication {
    vulkan_entry_point: Entry,
    vulkan_instance: Instance,
    vulkan_application_data: VulkanApplicationData,
    vulkan_logical_device: Device,
    frame: usize,
    pub(crate) resized: bool
}
impl VulkanApplication {
    pub unsafe fn create_vulkan_application(user_window: &Window) -> anyhow::Result<Self> {
        let platform_specific_vulkan_api = LibloadingLoader::new(LIBRARY)?;
        let vulkan_api_entry_point = Entry::new(platform_specific_vulkan_api).map_err(|b| anyhow!("{}", b))?;
        let mut vulkan_application_data = VulkanApplicationData::default();
        let vulkan_instance = create_instance(user_window, &vulkan_api_entry_point, &mut vulkan_application_data)?;
        vulkan_application_data.surface = vulkan_window::create_surface(&vulkan_instance, &user_window, &user_window)?;
        choose_gpu(&vulkan_instance, &mut vulkan_application_data)?;
        let vulkan_logical_device = create_logical_device(&vulkan_api_entry_point, &vulkan_instance, &mut vulkan_application_data)?;
        create_swapchain(user_window, &vulkan_instance, &vulkan_logical_device, &mut vulkan_application_data)?;
        create_swapchain_image_views(&vulkan_logical_device, &mut vulkan_application_data)?;
        create_render_pass(&vulkan_instance, &vulkan_logical_device, &mut vulkan_application_data)?;
        create_pipeline(&vulkan_logical_device, &mut vulkan_application_data)?;
        create_frame_buffers(&vulkan_logical_device, &mut vulkan_application_data)?;
        create_command_pool(&vulkan_instance, &vulkan_logical_device, &mut vulkan_application_data)?;
        create_command_buffers(&vulkan_logical_device, &mut vulkan_application_data)?;
        create_sync_objects(&vulkan_logical_device, &mut vulkan_application_data)?;
        Ok(Self{vulkan_entry_point: vulkan_api_entry_point, vulkan_instance, vulkan_application_data, vulkan_logical_device, frame: 0, resized: false})
    }
    pub unsafe fn render_frame(&mut self, window: &Window) -> anyhow::Result<()> {

        self.vulkan_logical_device.wait_for_fences(&[self.vulkan_application_data.in_flight_fences[self.frame]], true, u64::MAX, )?;

        let result = self.vulkan_logical_device.acquire_next_image_khr(self.vulkan_application_data.swapchain, u64::MAX, self.vulkan_application_data.image_available_semaphores[self.frame], vk::Fence::null());
        let image_index = match result {
            Ok((image_index, _)) => image_index as usize,
            Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
            Err(e) => return Err(anyhow!(e))
        };

        if !self.vulkan_application_data.images_in_flight[image_index].is_null() {
            self.vulkan_logical_device.wait_for_fences(&[self.vulkan_application_data.images_in_flight[image_index]], true, u64::MAX, )?;
        }

        self.vulkan_application_data.images_in_flight[image_index] = self.vulkan_application_data.in_flight_fences[self.frame];

        let semaphore_to_wait_on_before_execution = &[self.vulkan_application_data.image_available_semaphores[self.frame]];
        let stage_of_pipeline_to_wait_on_before_execution = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffer_to_use_at_execution = &[self.vulkan_application_data.command_buffers[image_index]];
        let semaphores_to_signal_after_command_buffer_finished_executing = &[self.vulkan_application_data.render_finished_semaphores[self.frame]];
        let info_to_submit_to_queue = vk::SubmitInfo::builder().wait_semaphores(semaphore_to_wait_on_before_execution).wait_dst_stage_mask(stage_of_pipeline_to_wait_on_before_execution)
            .command_buffers(command_buffer_to_use_at_execution).signal_semaphores(semaphores_to_signal_after_command_buffer_finished_executing);

        self.vulkan_logical_device.reset_fences(&[self.vulkan_application_data.in_flight_fences[self.frame]])?;

        self.vulkan_logical_device.queue_submit(self.vulkan_application_data.graphics_queue, &[info_to_submit_to_queue], self.vulkan_application_data.in_flight_fences[self.frame])?;

        let swapchains_to_present_images_to = &[self.vulkan_application_data.swapchain];
        let image_index_in_swapchain = &[image_index as u32];
        let image_presentation_configuration = vk::PresentInfoKHR::builder()
            .wait_semaphores(semaphores_to_signal_after_command_buffer_finished_executing)
            .swapchains(swapchains_to_present_images_to)
            .image_indices(image_index_in_swapchain);

        self.vulkan_logical_device.queue_wait_idle(self.vulkan_application_data.presentation_queue)?;
        let result = self.vulkan_logical_device.queue_present_khr(self.vulkan_application_data.presentation_queue, &image_presentation_configuration);

        let changed = result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);
        println!("Changed it has, or not, {}", changed);

        //let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR) || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);
        //println!("{:?}", result?);

        if changed {
            self.recreate_swapchain(window)?;
        }

        //if self.resized || changed {
            //self.resized = false;
            //self.recreate_swapchain(window)?;
        //} else if let Err(e) = result {
            //return Err(anyhow!(e));
        //}
        self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;

        Ok(())
    }
    pub unsafe fn recreate_swapchain(&mut self, user_window: &Window) -> anyhow::Result<()> {
        self.vulkan_logical_device.device_wait_idle()?;
        println!("Recreating the swapchain!");
        self.destroy_swapchain();
        create_swapchain(user_window, &self.vulkan_instance, &self.vulkan_logical_device, &mut self.vulkan_application_data)?;
        create_swapchain_image_views(&self.vulkan_logical_device, &mut self.vulkan_application_data)?;
        create_render_pass(&self.vulkan_instance, &self.vulkan_logical_device, &mut self.vulkan_application_data)?;
        create_pipeline(&self.vulkan_logical_device, &mut self.vulkan_application_data)?;
        create_frame_buffers(&self.vulkan_logical_device, &mut self.vulkan_application_data)?;
        create_command_buffers(&self.vulkan_logical_device, &mut self.vulkan_application_data)?;
        self.vulkan_application_data.images_in_flight.resize(self.vulkan_application_data.swapchain_images.len(), vk::Fence::null());
        Ok(())
    }
    pub unsafe fn destroy_swapchain(&mut self) {
        self.vulkan_application_data.framebuffers.iter().for_each(|framebuffer| self.vulkan_logical_device.destroy_framebuffer(*framebuffer, None));
        self.vulkan_logical_device.free_command_buffers(self.vulkan_application_data.command_pool, &self.vulkan_application_data.command_buffers);
        self.vulkan_logical_device.destroy_pipeline(self.vulkan_application_data.pipeline, None);
        self.vulkan_logical_device.destroy_pipeline_layout(self.vulkan_application_data.pipeline_layout, None);
        self.vulkan_logical_device.destroy_render_pass(self.vulkan_application_data.render_pass, None);
        self.vulkan_application_data.swapchain_image_views.iter().for_each(|image_view| self.vulkan_logical_device.destroy_image_view(*image_view, None));
        self.vulkan_logical_device.destroy_swapchain_khr(self.vulkan_application_data.swapchain, None);
    }

    pub unsafe fn destroy_vulkan_application(&mut self) {
        self.destroy_swapchain();
        self.vulkan_application_data.in_flight_fences.iter().for_each(|f| self.vulkan_logical_device.destroy_fence(*f, None));
        self.vulkan_application_data.render_finished_semaphores.iter().for_each(|s| self.vulkan_logical_device.destroy_semaphore(*s, None));
        self.vulkan_application_data.image_available_semaphores.iter().for_each(|s| self.vulkan_logical_device.destroy_semaphore(*s, None));
        //self.vulkan_logical_device.destroy_semaphore(self.vulkan_application_data.render_finished_semaphore, None);
        //self.vulkan_logical_device.destroy_semaphore(self.vulkan_application_data.image_available_semaphore, None);
        self.vulkan_logical_device.destroy_command_pool(self.vulkan_application_data.command_pool, None);
        self.vulkan_logical_device.destroy_device(None);
        self.vulkan_instance.destroy_surface_khr(self.vulkan_application_data.surface, None);
        if VALIDATION_ENABLED {
            self.vulkan_instance.destroy_debug_utils_messenger_ext(self.vulkan_application_data.debug_messenger, None);
         }
        self.vulkan_instance.destroy_instance(None);
    }
    unsafe fn present_image_to_swapchain(&mut self, present_info: vk::PresentInfoKHRBuilder) {
        self.vulkan_logical_device.queue_present_khr(self.vulkan_application_data.presentation_queue, &present_info).expect("Presenting the image to the swapchain resulted in an error!");
    }
    //unsafe fn index_of_next_available_presentable_image(&mut self) {
       // self.vulkan_logical_device.acquire_next_image_khr(self.vulkan_application_data.swapchain, u64::MAX, self.vulkan_application_data.image_available_semaphores[self.frame], vk::Fence::null()).expect("Retrieving the next presentable image index resulted in an error!");
    //}
}
