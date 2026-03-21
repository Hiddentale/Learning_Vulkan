use crate::graphical_core::{queue_families::RequiredQueueFamilies, vulkan_object::VulkanApplicationData};
use crate::DEVICE_EXTENSIONS;
use anyhow::anyhow;
use log::{info, warn};
use std::collections::HashSet;
use thiserror::Error;
use vulkanalia::vk::{InstanceV1_0, PhysicalDevice};
use vulkanalia::Instance;

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

/// Selects the first suitable physical device that supports required queues, extensions, and swapchain.
pub unsafe fn choose_gpu(instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
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
pub unsafe fn check_gpu(instance: &Instance, data: &VulkanApplicationData, gpu: PhysicalDevice) -> anyhow::Result<()> {
    RequiredQueueFamilies::get(instance, data, gpu)?;
    check_gpu_extensions(instance, gpu)?;
    let support = crate::graphical_core::swapchain::SwapchainSupport::get(instance, data, gpu)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
    }
    Ok(())
}

unsafe fn check_gpu_extensions(instance: &Instance, gpu: PhysicalDevice) -> anyhow::Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(gpu, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError("Missing required GPU extensions.")))
    }
}

unsafe fn is_gpu_suitable(instance: &Instance, data: &VulkanApplicationData, gpu: PhysicalDevice) -> bool {
    check_gpu(instance, data, gpu).is_ok()
}
