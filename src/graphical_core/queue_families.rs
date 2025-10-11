use crate::graphical_core::extra::SuitabilityError;
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
    pub unsafe fn get(current_system: &Instance, vulkan_application_data: &VulkanApplicationData, gpu: vk::PhysicalDevice) -> anyhow::Result<Self> {
        let required_properties = current_system.get_physical_device_queue_family_properties(gpu);

        let graphics_queue_index = required_properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut presentation_queue_index = None;
        for (index, properties) in required_properties.iter().enumerate() {
            if queue_family_has_capability_of_presenting_to_our_window_surface(current_system, vulkan_application_data, gpu, index)? {
                presentation_queue_index = Some(index as u32);
                break;
            }
        }

        if queue_family_indexes_not_empty(graphics_queue_index, presentation_queue_index) {
            let graphics_queue_index = graphics_queue_index.unwrap();
            let presentation_queue_index = presentation_queue_index.unwrap();
            Ok(Self {
                graphics_queue_index,
                presentation_queue_index,
            })
        } else {
            Err(anyhow!(SuitabilityError("Missing required queue families.")))
        }
    }
}
unsafe fn queue_family_has_capability_of_presenting_to_our_window_surface(
    current_system: &Instance,
    vulkan_application_data: &VulkanApplicationData,
    gpu: vk::PhysicalDevice,
    index: usize,
) -> VkResult<bool> {
    current_system.get_physical_device_surface_support_khr(gpu, index as u32, vulkan_application_data.surface)
}
fn queue_family_indexes_not_empty(graphics_queue_index: Option<u32>, presentation_queue_index: Option<u32>) -> bool {
    if let (Some(graphics_queue_index), Some(presentation_queue_index)) = (graphics_queue_index, presentation_queue_index) {
        true
    } else {
        false
    }
}
