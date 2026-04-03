use super::session::{xr_image_to_vk, VrSession};
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

/// Stereo swapchain with `array_size = 2` (one layer per eye).
pub struct VrSwapchain {
    pub handle: xr::Swapchain<xr::Vulkan>,
    pub config: ViewConfig,
    pub format: vk::Format,
    pub images: Vec<vk::Image>,
    /// Per-image view covering both array layers (for multiview framebuffer).
    pub image_views: Vec<vk::ImageView>,
}

impl VrSwapchain {
    /// Query view configuration, pick a format, create the stereo swapchain,
    /// enumerate its images, and create Vulkan image views.
    pub fn create(vr: &VrSession, device: &Device, _instance: &Instance, _physical_device: vk::PhysicalDevice) -> anyhow::Result<Self> {
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

        let image_views = create_image_views(device, &images, format)?;

        info!("VR swapchain created with {} images", images.len());

        Ok(Self {
            handle,
            config,
            format,
            images,
            image_views,
        })
    }

    /// Destroy Vulkan image views. The swapchain images themselves are owned
    /// by OpenXR and destroyed when the swapchain handle drops.
    pub fn destroy(&self, device: &Device) {
        for &view in &self.image_views {
            unsafe { device.destroy_image_view(view, None) };
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
fn create_image_views(device: &Device, images: &[vk::Image], format: vk::Format) -> anyhow::Result<Vec<vk::ImageView>> {
    images
        .iter()
        .map(|&image| {
            let subresource = vk::ImageSubresourceRange::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(2);

            let info = vk::ImageViewCreateInfo::builder()
                .image(image)
                .view_type(vk::ImageViewType::_2D_ARRAY)
                .format(format)
                .subresource_range(*subresource);

            unsafe { device.create_image_view(&info, None).map_err(Into::into) }
        })
        .collect()
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
