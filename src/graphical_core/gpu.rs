use std::collections::HashSet;
use anyhow::anyhow;
use log::{info, warn};
use thiserror::Error;
use vulkanalia::{Instance, vk};
use vulkanalia::vk::{InstanceV1_0, PhysicalDevice, PhysicalDeviceProperties};
use crate::DEVICE_EXTENSIONS;
use crate::graphical_core::{vulkan_object::VulkanApplicationData, queue_families::RequiredQueueFamilies};

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

pub unsafe fn choose_gpu(current_system: &Instance, vulkan_application_data: &mut VulkanApplicationData) -> anyhow::Result<()> {
    for gpu in all_available_gpus(current_system)? {
        let gpu_properties = get_gpu_properties(current_system, gpu);
        if gpu_not_have_required_properties(current_system, vulkan_application_data, gpu) {
            warn!("Skipping GPU (`{}.`)", gpu_properties.device_name);
        } else {
            info!("Selected GPU (`{}`).", gpu_properties.device_name);
            vulkan_application_data.physical_device = gpu;
            return Ok(());
        }
    }
    Err(anyhow!("Failed to find suitable GPU."))
}
pub unsafe fn check_gpu(current_system: &Instance, vulkan_application_data: &VulkanApplicationData, gpu: PhysicalDevice) -> anyhow::Result<()> {
    RequiredQueueFamilies::get(current_system, vulkan_application_data, gpu)?;
    check_gpu_extensions(current_system, gpu)?;
    let support = crate::graphical_core::swapchain::SwapchainSupport::get(current_system, vulkan_application_data, gpu)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
    }
    Ok(())
}
pub unsafe fn check_gpu_extensions(instance: &Instance, physical_device: vk::PhysicalDevice) -> anyhow::Result<()> {
    let extensions = instance.enumerate_device_extension_properties(physical_device, None)?.iter().map(|e| e.extension_name).collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError("Missing required GPU extensions.")))
    }
}
unsafe fn all_available_gpus(current_system: &Instance) -> crate::VkResult<Vec<PhysicalDevice>> {
    current_system.enumerate_physical_devices()
}
unsafe fn get_gpu_properties(current_system: &Instance, gpu: PhysicalDevice) -> PhysicalDeviceProperties {
    current_system.get_physical_device_properties(gpu)
}
unsafe fn gpu_not_have_required_properties(current_system: &Instance, vulkan_application_data: &VulkanApplicationData, gpu: PhysicalDevice) -> bool  {
    if let Err(_) = check_gpu(current_system, vulkan_application_data, gpu) {
        true
    }
    else {
        false
    }
}