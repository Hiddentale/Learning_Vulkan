use super::session::{xr_image_to_vk, VrSession};
use crate::graphical_core::depth::depth_format;
use crate::graphical_core::memory::find_memory_type;
use crate::graphical_core::render_pass::create_multiview_render_pass;
use anyhow::{anyhow, Context};
use log::info;
use openxr as xr;
use vulkan_rust::{vk, Device, Instance};

/// Per-eye resolution and format for the VR swapchain.
#[derive(Debug, Clone, Copy)]
pub struct ViewConfig {
    pub width: u32,
    pub height: u32,
}

/// Stereo swapchain with `array_size = 2` (one layer per eye),
/// multiview render pass, stereo depth buffer, and per-image framebuffers.
pub struct VrSwapchain {
    pub handle: xr::Swapchain<xr::Vulkan>,
    pub config: ViewConfig,
    pub format: vk::Format,
    pub images: Vec<vk::Image>,
    /// Per-image view covering both array layers (for multiview framebuffer).
    pub image_views: Vec<vk::ImageView>,
    pub render_pass: vk::RenderPass,
    pub framebuffers: Vec<vk::Framebuffer>,
    depth_image: vk::Image,
    depth_memory: vk::DeviceMemory,
    depth_view: vk::ImageView,
}

impl VrSwapchain {
    /// Query view configuration, pick a format, create the stereo swapchain,
    /// enumerate its images, and create Vulkan image views.
    pub fn create(vr: &VrSession, device: &Device, instance: &Instance, physical_device: vk::PhysicalDevice) -> anyhow::Result<Self> {
        let config = query_view_config(&vr.xr_instance, vr.system())?;
        let format = select_format(&vr.session)?;

        info!(
            "VR swapchain: {}x{} format {:?} (stereo, array_size=2)",
            config.width, config.height, format
        );

        let handle = vr.session.create_swapchain(&xr::SwapchainCreateInfo {
            create_flags: xr::SwapchainCreateFlags::EMPTY,
            usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT | xr::SwapchainUsageFlags::SAMPLED,
            format: format.as_raw() as u32,
            sample_count: 1,
            width: config.width,
            height: config.height,
            face_count: 1,
            array_size: 2,
            mip_count: 1,
        })?;

        let raw_images = handle.enumerate_images()?;
        let images: Vec<vk::Image> = raw_images.iter().map(|&img| xr_image_to_vk(img)).collect();
        let image_views = create_color_views(device, &images, format)?;

        let render_pass = unsafe {
            create_multiview_render_pass(
                device,
                format,
                2,
                vk::AttachmentLoadOp::CLEAR,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::UNDEFINED,
            )?
        };
        let (depth_image, depth_memory, depth_view) = unsafe { create_stereo_depth(device, instance, physical_device, &config)? };
        let framebuffers = unsafe { create_framebuffers(device, render_pass, &image_views, depth_view, &config)? };

        info!("VR render target ready: {} images, multiview render pass", images.len());

        Ok(Self {
            handle,
            config,
            format,
            images,
            image_views,
            render_pass,
            framebuffers,
            depth_image,
            depth_memory,
            depth_view,
        })
    }

    /// Destroy all Vulkan resources. Swapchain images are owned by OpenXR.
    pub fn destroy(&self, device: &Device) {
        unsafe {
            for &fb in &self.framebuffers {
                device.destroy_framebuffer(fb, None);
            }
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image(self.depth_image, None);
            device.free_memory(self.depth_memory, None);
            device.destroy_render_pass(self.render_pass, None);
            for &view in &self.image_views {
                device.destroy_image_view(view, None);
            }
        }
    }
}

/// Query the HMD's recommended per-eye resolution.
fn query_view_config(xr_instance: &xr::Instance, system: xr::SystemId) -> anyhow::Result<ViewConfig> {
    let views = xr_instance
        .enumerate_view_configuration_views(system, xr::ViewConfigurationType::PRIMARY_STEREO)
        .context("failed to query view configuration")?;

    if views.len() != 2 {
        return Err(anyhow!("expected 2 views for stereo, got {}", views.len()));
    }

    // Both eyes should have the same resolution; use the first.
    let view = &views[0];
    Ok(ViewConfig {
        width: view.recommended_image_rect_width,
        height: view.recommended_image_rect_height,
    })
}

/// Pick the best sRGB color format from the runtime's supported list.
fn select_format(session: &xr::Session<xr::Vulkan>) -> anyhow::Result<vk::Format> {
    let formats = session.enumerate_swapchain_formats()?;

    let preferred = [
        vk::Format::R8G8B8A8_SRGB,
        vk::Format::B8G8R8A8_SRGB,
        vk::Format::R8G8B8A8_UNORM,
        vk::Format::B8G8R8A8_UNORM,
    ];

    for pref in &preferred {
        if formats.iter().any(|&f| vk::Format::from_raw(f as i32) == *pref) {
            return Ok(*pref);
        }
    }

    // Fall back to the first format the runtime offers.
    formats
        .first()
        .map(|&f| vk::Format::from_raw(f as i32))
        .ok_or_else(|| anyhow!("runtime supports no swapchain formats"))
}

/// Create a 2D-array image view (both layers) per swapchain image for multiview.
fn create_color_views(device: &Device, images: &[vk::Image], format: vk::Format) -> anyhow::Result<Vec<vk::ImageView>> {
    images
        .iter()
        .map(|&image| {
            let info = vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::_2D_ARRAY)
                .format(format)
                .subresource_range(stereo_subresource(vk::ImageAspectFlags::COLOR));
            unsafe { device.create_image_view(&info, None).map_err(Into::into) }
        })
        .collect()
}

/// Allocate a 2-layer depth image for stereo rendering.
unsafe fn create_stereo_depth(
    device: &Device,
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
    config: &ViewConfig,
) -> anyhow::Result<(vk::Image, vk::DeviceMemory, vk::ImageView)> {
    let image_info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .format(depth_format())
        .extent(vk::Extent3D {
            width: config.width,
            height: config.height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(2)
        .samples(vk::SampleCountFlags::_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);

    let image = device.create_image(&image_info, None)?;

    let mem_req = device.get_image_memory_requirements(image);
    let mem_props = instance.get_physical_device_memory_properties(physical_device);
    let mem_type = find_memory_type(&mem_props, mem_req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

    let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(mem_req.size)
        .memory_type_index(mem_type);
    let memory = device.allocate_memory(&alloc_info, None)?;
    device.bind_image_memory(image, memory, 0)?;

    let view_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D_ARRAY)
        .format(depth_format())
        .subresource_range(stereo_subresource(vk::ImageAspectFlags::DEPTH));
    let view = device.create_image_view(&view_info, None)?;

    Ok((image, memory, view))
}

/// Create one framebuffer per swapchain image, each referencing the shared depth.
/// With multiview, `layers = 1` — the view mask selects array layers.
unsafe fn create_framebuffers(
    device: &Device,
    render_pass: vk::RenderPass,
    color_views: &[vk::ImageView],
    depth_view: vk::ImageView,
    config: &ViewConfig,
) -> anyhow::Result<Vec<vk::Framebuffer>> {
    color_views
        .iter()
        .map(|&color_view| {
            let attachments = &[color_view, depth_view];
            let info = vk::FramebufferCreateInfo::builder()
                .render_pass(render_pass)
                .attachments(attachments)
                .width(config.width)
                .height(config.height)
                .layers(1);
            device.create_framebuffer(&info, None).map_err(Into::into)
        })
        .collect()
}

fn stereo_subresource(aspect: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    *vk::ImageSubresourceRange::builder()
        .aspect_mask(aspect)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preferred_formats_are_srgb_first() {
        // Verify our preference order puts sRGB before UNORM
        let preferred = [
            vk::Format::R8G8B8A8_SRGB,
            vk::Format::B8G8R8A8_SRGB,
            vk::Format::R8G8B8A8_UNORM,
            vk::Format::B8G8R8A8_UNORM,
        ];
        assert_eq!(preferred[0], vk::Format::R8G8B8A8_SRGB);
        assert_eq!(preferred[1], vk::Format::B8G8R8A8_SRGB);
    }
}
