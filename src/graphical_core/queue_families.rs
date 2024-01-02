use anyhow::anyhow;
use vulkanalia::{Instance, vk};
use vulkanalia::vk::{InstanceV1_0, KhrSurfaceExtension};
use crate::graphical_core::extra::SuitabilityError;
use crate::graphical_core::vulkan_object::VulkanApplicationData;

#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub present: u32
}
impl QueueFamilyIndices {
    pub unsafe fn get(instance: &Instance, data: &VulkanApplicationData, physical_device: vk::PhysicalDevice) -> anyhow::Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);

        let graphics = properties.iter().position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS)).map(|i| i as u32);

        let mut present = None;
        for (index, properties) in properties.iter().enumerate() {
            if instance.get_physical_device_surface_support_khr(physical_device, index as u32, data.surface)? {
                present = Some(index as u32);
                break;
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(anyhow!(SuitabilityError("Missing required queue families.")))
        }
    }
}