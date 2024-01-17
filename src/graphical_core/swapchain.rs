use vulkanalia::{Device, Instance, vk};
use vulkanalia::vk::{DeviceV1_0, Handle, HasBuilder, KhrSurfaceExtension, KhrSwapchainExtension};
use winit::window::Window;
use crate::graphical_core::queue_families::RequiredQueueFamilies;
use crate::graphical_core::vulkan_object::VulkanApplicationData;

pub unsafe fn create_swapchain(user_window: &Window, current_system: &Instance, vulkan_logical_device: &Device, vulkan_application_data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    let indices = RequiredQueueFamilies::get(current_system, vulkan_application_data, vulkan_application_data.physical_device)?;
    let current_swapchain_capabilities = SwapchainSupport::get(current_system, vulkan_application_data, vulkan_application_data.physical_device)?;

    let surface_format = get_swapchain_surface_format(&current_swapchain_capabilities.formats);
    let presentation_mode = get_swapchain_presentation_mode(&current_swapchain_capabilities.present_modes);
    let swapchain_image_resolution = get_swapchain_extent(user_window, current_swapchain_capabilities.capabilities);

    vulkan_application_data.swapchain_format = surface_format.format;
    vulkan_application_data.swapchain_accepted_images_width_and_height = swapchain_image_resolution;

    let mut image_count = current_swapchain_capabilities.capabilities.min_image_count + 1; //We add one more image to the image count to make sure we never have to wait for an image.
    if current_swapchain_capabilities.capabilities.max_image_count != 0 && image_count > current_swapchain_capabilities.capabilities.max_image_count {
        image_count = current_swapchain_capabilities.capabilities.max_image_count;
    }

    let mut queue_family_indices = vec![];
    let image_sharing_mode = if indices.graphics_queue_index != indices.presentation_queue_index {
        queue_family_indices.push(indices.graphics_queue_index);
        queue_family_indices.push(indices.presentation_queue_index);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(vulkan_application_data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(swapchain_image_resolution)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        .pre_transform(current_swapchain_capabilities.capabilities.current_transform)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(presentation_mode)
        .clipped(true)
        .old_swapchain(vk::SwapchainKHR::null());

    vulkan_application_data.swapchain = vulkan_logical_device.create_swapchain_khr(&info, None)?;
    vulkan_application_data.swapchain_images = vulkan_logical_device.get_swapchain_images_khr(vulkan_application_data.swapchain)?;

    Ok(())
}

fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats.iter().cloned().find(|f| f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR).unwrap_or_else(|| formats[0])
}

fn get_swapchain_presentation_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes.iter().cloned().find(|m| *m == vk::PresentModeKHR::MAILBOX).unwrap_or(vk::PresentModeKHR::FIFO)
}

fn get_swapchain_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    }
    else {
        let size = window.inner_size();
        let clamp = |min: u32, max: u32, v: u32| min.max(max.min(v));
        vk::Extent2D::builder().width(clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width, size.width))
            .height(clamp(capabilities.min_image_extent.height, capabilities.max_image_extent.height, size.height)).build()
    }
}
pub unsafe fn create_swapchain_image_views(device: &Device, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    data.swapchain_image_views = data.swapchain_images.iter().map(|i|{
        let components = vk::ComponentMapping::builder().r(vk::ComponentSwizzle::IDENTITY).g(vk::ComponentSwizzle::IDENTITY).b(vk::ComponentSwizzle::IDENTITY)
            .a(vk::ComponentSwizzle::IDENTITY);
        let subresource_range = vk::ImageSubresourceRange::builder().aspect_mask(vk::ImageAspectFlags::COLOR).base_mip_level(0).level_count(1).base_array_layer(0)
            .layer_count(1);
        let info = vk::ImageViewCreateInfo::builder().image(*i).view_type(vk::ImageViewType::_2D).format(data.swapchain_format).components(components)
            .subresource_range(subresource_range);
        device.create_image_view(&info, None)
    }).collect::<anyhow::Result<Vec<_>, _>>()?;
    Ok(())
}

#[derive(Clone, Debug)]
pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>,
}
impl SwapchainSupport {
    pub unsafe fn get(instance: &Instance, data: &VulkanApplicationData, physical_device: vk::PhysicalDevice) -> anyhow::Result<Self> {
        Ok(Self {
            capabilities: instance.get_physical_device_surface_capabilities_khr(physical_device, data.surface)?,
            formats: instance.get_physical_device_surface_formats_khr(physical_device, data.surface)?,
            present_modes: instance.get_physical_device_surface_present_modes_khr(physical_device, data.surface)?,
        })
    }
}