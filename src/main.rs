#![allow(
dead_code,
unused_variables,
clippy::too_many_arguments,
clippy::unnecessary_wraps
)]
mod graphical_core;
use anyhow::Result;
use winit::{
    dpi::LogicalSize,
    event::{Event, WindowEvent},
    event_loop::{EventLoop, EventLoopWindowTarget},
    window::WindowBuilder
};
use vulkanalia::{
    prelude::v1_0::*,
    Version,
};
use graphical_core::vulkan_object::VulkanApplication;

const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);
const VALIDATION_ENABLED: bool =
    cfg!(debug_assertions);

const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

fn main() -> Result<()>
{
    initialize_error_handler();

    let event_handler = EventLoop::new()?;
    let user_window = WindowBuilder::new().with_title("Vulkan Tutorial (Rust)").with_inner_size(LogicalSize::new(1024, 768)).build(&event_handler)?;

    let mut application = unsafe {VulkanApplication::create_vulkan_application(&user_window)}?;
    let mut destroy_application = false;
    let mut minimized = false;

    event_handler.run(move |event, current_window| {
        match event
        {
            Event::WindowEvent {event: WindowEvent::CloseRequested, .. } => {exit_program(&mut destroy_application, current_window, &mut application);},
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {if size.width == 0 || size.height == 0 {minimized = true} else {minimized = false; application.resized = true}},
            Event::AboutToWait => { user_window.request_redraw()},
            Event::WindowEvent {event: WindowEvent::RedrawRequested, .. } => {if !destroy_application && !minimized {unsafe {application.render_frame(&user_window)}.unwrap()}},
            _ => ()
        }
    }).expect("Main function crashed!");
    Ok(())
}
fn exit_program(destroy_application: &mut bool, current_window: &EventLoopWindowTarget<()>, application: &mut VulkanApplication)
{
    *destroy_application = true;
    current_window.exit();
    unsafe {application.destroy_vulkan_application()}
}
fn initialize_error_handler() { pretty_env_logger::init(); }