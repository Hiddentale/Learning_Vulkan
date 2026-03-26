#![allow(clippy::too_many_arguments)]
mod graphical_core;
mod voxel;
use anyhow::Result;
use graphical_core::camera::Camera;
use graphical_core::input::InputState;
use graphical_core::vulkan_object::VulkanApplication;
use std::time::Instant;
use voxel::player::Player;
use vulkanalia::{prelude::v1_0::*, Version};
use winit::{
    dpi::LogicalSize,
    event::{DeviceEvent, ElementState, Event, WindowEvent},
    event_loop::{EventLoop, EventLoopWindowTarget},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, WindowBuilder},
};

const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

const VALIDATION_LAYER: vk::ExtensionName = vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

fn main() -> Result<()> {
    initialize_error_handler();

    let event_handler = EventLoop::new()?;
    let user_window = WindowBuilder::new()
        .with_title("Vulkan Tutorial (Rust)")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_handler)?;

    grab_cursor(&user_window);

    let mut application = unsafe { VulkanApplication::create_vulkan_application(&user_window) }?;
    let mut destroy_application = false;
    let mut minimized = false;
    let mut camera = Camera::default();
    let mut input = InputState::new();
    let mut player = Player::new();
    let mut last_frame = Instant::now();

    event_handler
        .run(move |event, current_window| match event {
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                exit_program(&mut destroy_application, current_window, &mut application);
            }
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                if size.width == 0 || size.height == 0 {
                    minimized = true
                } else {
                    minimized = false;
                    application.resized = true
                }
            }
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { event: key_event, .. },
                ..
            } => {
                if let PhysicalKey::Code(key_code) = key_event.physical_key {
                    match key_event.state {
                        ElementState::Pressed => {
                            if key_code == KeyCode::Escape {
                                exit_program(&mut destroy_application, current_window, &mut application);
                            }
                            input.key_pressed(key_code);
                        }
                        ElementState::Released => input.key_released(key_code),
                    }
                }
            }
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta: (dx, dy) },
                ..
            } => {
                input.accumulate_mouse_delta(dx, dy);
            }
            Event::AboutToWait => {
                let now = Instant::now();
                let delta_time = (now - last_frame).as_secs_f32();
                last_frame = now;

                let old_position = camera.position;
                input.update_camera(&mut camera, delta_time);
                let world = application.world();
                player.apply_physics(&mut camera.position, delta_time, world);
                player.resolve_horizontal(&mut camera.position, old_position, world);
                user_window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                if !destroy_application && !minimized {
                    if let Err(e) = unsafe { application.render_frame(&user_window, &camera) } {
                        eprintln!("Render error: {e}");
                        exit_program(&mut destroy_application, current_window, &mut application);
                    }
                }
            }
            _ => (),
        })
        .expect("Main function crashed!");
    Ok(())
}

fn grab_cursor(window: &winit::window::Window) {
    let _ = window.set_cursor_grab(CursorGrabMode::Confined);
    window.set_cursor_visible(false);
}

fn exit_program(destroy_application: &mut bool, current_window: &EventLoopWindowTarget<()>, application: &mut VulkanApplication) {
    *destroy_application = true;
    current_window.exit();
    unsafe { application.destroy_vulkan_application() }
}
fn initialize_error_handler() {
    pretty_env_logger::init();
}
