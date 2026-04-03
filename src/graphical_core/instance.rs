use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::vr::VrContext;
use crate::{graphical_core, DEVICE_EXTENSIONS, PORTABILITY_MACOS_VERSION, VALIDATION_ENABLED, VALIDATION_LAYER};
use anyhow::anyhow;
use log::{debug, error, info, trace, warn};
use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_void;
use vulkan_rust::{required_extensions, vk, Device, Entry, Instance, Version};
use winit::window::Window;

/// Creates a Vulkan instance with validation layers and debug messaging if enabled.
/// When VR is available, adds OpenXR-required instance extensions.
pub unsafe fn create_instance(_window: &Window, entry: &Entry, data: &mut VulkanApplicationData, vr: Option<&VrContext>) -> anyhow::Result<Instance> {
    let application_info = vk::ApplicationInfo::builder()
        .application_name(c"Vulkan Tutorial")
        .application_version(Version::new(1, 0, 0).to_raw())
        .engine_name(c"No Engine")
        .engine_version(Version::new(1, 0, 0).to_raw())
        .api_version(Version::new(1, 2, 0).to_raw());

    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<Vec<_>>();

    if VALIDATION_ENABLED && !available_layers.iter().any(|l| l == &VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    let mut extensions = required_extensions().iter().map(|e| e.as_ptr()).collect::<Vec<_>>();

    let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        info!("Enabling extensions for macOS portability.");
        extensions.push(vk::extension_names::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION_NAME.as_ptr());
        extensions.push(vk::extension_names::KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME.as_ptr());
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY
    } else {
        vk::InstanceCreateFlags::empty()
    };
    if VALIDATION_ENABLED {
        extensions.push(vk::extension_names::EXT_DEBUG_UTILS_EXTENSION_NAME.as_ptr());
    }

    let vr_instance_extensions = vr.map(|v| v.required_instance_extensions()).unwrap_or_default();
    for ext in &vr_instance_extensions {
        info!("VR requires instance extension: {}", ext.to_string_lossy());
        extensions.push(ext.as_ptr());
    }

    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
        )
        .pfn_user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        info = info.push_next(&mut *debug_info);
    }
    let instance = entry.create_instance(&info, None)?;

    if VALIDATION_ENABLED {
        data.debug_messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
    }

    Ok(instance)
}

/// Creates a logical device with graphics and presentation queues.
/// When VR is available, adds OpenXR-required device extensions.
pub unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut VulkanApplicationData,
    vr: Option<&VrContext>,
) -> anyhow::Result<Device> {
    let indices = graphical_core::queue_families::RequiredQueueFamilies::get(instance, data, data.physical_device)?;
    let mut unique_indices = HashSet::new();

    unique_indices.insert(indices.graphics_queue_index);
    unique_indices.insert(indices.presentation_queue_index);

    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            *vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    let layers = if VALIDATION_ENABLED { vec![VALIDATION_LAYER.as_ptr()] } else { vec![] };
    let mut extensions = DEVICE_EXTENSIONS.iter().map(|n| n.as_ptr()).collect::<Vec<_>>();

    if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        extensions.push(vk::extension_names::KHR_PORTABILITY_SUBSET_EXTENSION_NAME.as_ptr());
    }

    let vr_device_extensions = vr.map(|v| v.required_device_extensions()).unwrap_or_default();
    for ext in &vr_device_extensions {
        info!("VR requires device extension: {}", ext.to_string_lossy());
        extensions.push(ext.as_ptr());
    }

    let features = vk::PhysicalDeviceFeatures::builder().multi_draw_indirect(true);
    let mut features_1_1 = vk::PhysicalDeviceVulkan11Features::builder();
    if vr.is_some() {
        features_1_1 = features_1_1.multiview(true);
    }
    let mut features_1_2 = vk::PhysicalDeviceVulkan12Features::builder();
    if !cfg!(target_os = "macos") {
        features_1_2 = features_1_2.draw_indirect_count(true);
    }
    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features)
        .push_next(&mut *features_1_1)
        .push_next(&mut *features_1_2);
    let device = instance.create_device(data.physical_device, &info, None)?;

    data.graphics_queue = device.get_device_queue(indices.graphics_queue_index, 0);
    data.presentation_queue = device.get_device_queue(indices.presentation_queue_index, 0);

    Ok(device)
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagBitsEXT,
    _type: vk::DebugUtilsMessageTypeFlagBitsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> u32 {
    let data = *data;
    let message = CStr::from_ptr(data.p_message).to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::ERROR {
        error!("(VALIDATION) {}", message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::WARNING {
        warn!("(VALIDATION) {}", message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::INFO {
        debug!("(VALIDATION) {}", message);
    } else {
        trace!("(VALIDATION) {}", message);
    }

    vk::FALSE
}
