use crate::graphical_core::{queue_families::RequiredQueueFamilies, vulkan_object::VulkanApplicationData};
use crate::DEVICE_EXTENSIONS;
use anyhow::anyhow;
use log::{info, warn};
use std::collections::HashSet;
use thiserror::Error;
use vulkan_rust::{vk, Instance};

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

/// Selects a suitable physical device. When VR is active, prefers the GPU
/// that OpenXR requests; otherwise picks the first suitable one.
pub unsafe fn choose_gpu(instance: &Instance, data: &mut VulkanApplicationData, vr_preferred: Option<vk::PhysicalDevice>) -> anyhow::Result<()> {
    if let Some(preferred) = vr_preferred {
        let properties = instance.get_physical_device_properties(preferred);
        if is_gpu_suitable(instance, data, preferred) {
            info!("Selected VR-preferred GPU (`{}`).", properties.device_name);
            data.physical_device = preferred;
            return Ok(());
        }
        warn!(
            "VR-preferred GPU (`{}`) is not suitable — falling back to enumeration.",
            properties.device_name
        );
    }

    for gpu in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(gpu);
        if is_gpu_suitable(instance, data, gpu) {
            info!("Selected GPU (`{}`).", properties.device_name);
            data.physical_device = gpu;
            return Ok(());
        } else {
            warn!("Skipping GPU (`{}.`)", properties.device_name);
        }
    }
    Err(anyhow!("Failed to find suitable GPU."))
}

/// Validates that a GPU has the required queue families, extensions, and swapchain support.
pub unsafe fn check_gpu(instance: &Instance, data: &VulkanApplicationData, gpu: vk::PhysicalDevice) -> anyhow::Result<()> {
    RequiredQueueFamilies::get(instance, data, gpu)?;
    check_gpu_extensions(instance, gpu)?;
    let support = crate::graphical_core::swapchain::SwapchainSupport::get(instance, data, gpu)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
    }
    Ok(())
}

unsafe fn check_gpu_extensions(instance: &Instance, gpu: vk::PhysicalDevice) -> anyhow::Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(gpu, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.iter().any(|ext| ext == e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError("Missing required GPU extensions.")))
    }
}

unsafe fn is_gpu_suitable(instance: &Instance, data: &VulkanApplicationData, gpu: vk::PhysicalDevice) -> bool {
    check_gpu(instance, data, gpu).is_ok()
}
