#![allow(
dead_code,
unused_variables,
clippy::too_many_arguments,
clippy::unnecessary_wraps
)]

mod add_layers;

use anyhow::{anyhow, Result};
use log::*;

use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{WindowBuilder, Window}
};

use vulkanalia::{
    Entry,
    Instance,
    vk,
    loader::{LibloadingLoader, LIBRARY},
    window as vk_window,
    prelude::v1_0::*,
    Version
};
use std::collections::HashSet;
use std::ffi::CStr;
use std::os::raw::c_void;

use vulkanalia::vk::ExtDebugUtilsExtension;

const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

fn main() -> Result<()> {
    pretty_env_logger::init();

    //Setting up the Window

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Vulkan Tutorial (Rust)")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    let mut application = unsafe { VulkanApplication::create(&window) };
    let mut destroy_application = false;

    event_loop.run(move |event, event_loop_window_target| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                destroy_application = true;
                event_loop_window_target.exit();
            },
            Event::AboutToWait => { window.request_redraw(); },
            Event::WindowEvent { event: WindowEvent::RedrawRequested, .. } => {},
            _ => ()
        }
    }).expect("TODO: panic message");
    Ok(())
}

#[derive(Clone, Debug)]
struct VulkanApplication {
    entry: Entry,
    instance: Instance,
}

impl VulkanApplication {
    /// Creates our Vulkan app.
    unsafe fn create(window: &Window) -> Result<Self> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let instance = create_instance(window, &entry)?;
        Ok(Self {entry, instance})
    }

    /// Renders a frame for our Vulkan app.
    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        Ok(())
    }

    /// Destroys our Vulkan app.
    unsafe fn destroy(&mut self) {
        self.instance.destroy_instance(None);
    }
}

/// The Vulkan handles and associated properties used by our Vulkan app.
#[derive(Clone, Debug, Default)]
struct AppData {}

unsafe fn create_instance(window: &Window, entry: &Entry) -> Result<Instance> {
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Vulkan Tutorial\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    // adding layers

    let layer_creation_result = add_layers::create_layers(entry: &Entry);

    // Adding extensions
    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

// Required by Vulkan SDK on macOS since 1.3.216.
    let flags = if
    cfg!(target_os = "macos") &&
        entry.version()? >= PORTABILITY_MACOS_VERSION
    {
        info!("Enabling extensions for macOS portability.");
        extensions.push(vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION.name.as_ptr());
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    let info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_extension_names(&extensions)
        .flags(flags);

    Ok(entry.create_instance(&info, None)?)
}
const VALIDATION_ENABLED: bool =
    cfg!(debug_assertions);

const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");
