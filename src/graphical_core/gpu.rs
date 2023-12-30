use std::collections::HashSet;
use anyhow::anyhow;
use log::{info, warn};
use thiserror::Error;
use vulkanalia::{Instance, vk};
use vulkanalia::vk::InstanceV1_0;
use crate::DEVICE_EXTENSIONS;
use crate::graphical_core::vulkan_object::ApplicationData;

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

pub unsafe fn pick_physical_device(instance: &Instance, data: &mut ApplicationData) -> anyhow::Result<()> {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);
        if let Err(error) = check_physical_device(instance, data, physical_device) {
            warn!("Skipping physical device (`{}`): {}", properties.device_name, error);
        } else {
            info!("Selected physical device (`{}`).", properties.device_name);
            data.physical_device = physical_device;
            return Ok(());
        }
    }
    Err(anyhow!("Failed to find suitable physical device."))
}
pub unsafe fn check_physical_device(instance: &Instance, data: &ApplicationData, physical_device: vk::PhysicalDevice) -> anyhow::Result<()> {
    crate::graphical_core::queue_families::QueueFamilyIndices::get(instance, data, physical_device)?;
    check_physical_device_extensions(instance, physical_device)?;
    let support = crate::graphical_core::swap_chain::SwapchainSupport::get(instance, data, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
    }
    Ok(())
}
pub unsafe fn check_physical_device_extensions(instance: &Instance, physical_device: vk::PhysicalDevice) -> anyhow::Result<()> {
    let extensions = instance.enumerate_device_extension_properties(physical_device, None)?.iter().map(|e| e.extension_name).collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError("Missing required device extensions.")))
    }
}