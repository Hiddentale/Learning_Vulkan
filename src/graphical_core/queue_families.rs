use crate::graphical_core::gpu::SuitabilityError;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use anyhow::anyhow;
use vulkanalia::vk::{InstanceV1_0, KhrSurfaceExtension};
use vulkanalia::{vk, Instance, VkResult};

#[derive(Copy, Clone, Debug)]
pub struct RequiredQueueFamilies {
    pub graphics_queue_index: u32,
    pub presentation_queue_index: u32,
}

impl RequiredQueueFamilies {
    /// Returns the graphics and presentation queue family indices for the given GPU.
    pub unsafe fn get(instance: &Instance, data: &VulkanApplicationData, gpu: vk::PhysicalDevice) -> anyhow::Result<Self> {
        let queue_families = instance.get_physical_device_queue_family_properties(gpu);

        let graphics_queue_index = queue_families
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut presentation_queue_index = None;
        for (index, _) in queue_families.iter().enumerate() {
            if supports_presentation(instance, data, gpu, index)? {
                presentation_queue_index = Some(index as u32);
                break;
            }
        }

        match (graphics_queue_index, presentation_queue_index) {
            (Some(graphics), Some(presentation)) => Ok(Self {
                graphics_queue_index: graphics,
                presentation_queue_index: presentation,
            }),
            _ => Err(anyhow!(SuitabilityError("Missing required queue families."))),
        }
    }
}

unsafe fn supports_presentation(
    instance: &Instance,
    data: &VulkanApplicationData,
    gpu: vk::PhysicalDevice,
    queue_family_index: usize,
) -> VkResult<bool> {
    instance.get_physical_device_surface_support_khr(gpu, queue_family_index as u32, data.surface)
}
