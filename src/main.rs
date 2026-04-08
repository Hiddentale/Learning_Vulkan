#![allow(clippy::too_many_arguments)]
mod game_state;
mod graphical_core;
mod storage;
mod voxel;
mod vr;
use anyhow::Result;
use game_state::GameState;
use graphical_core::camera::{Camera, EyeMatrices};
use graphical_core::input::InputState;
use graphical_core::ui_pipeline::UiPipeline;
use graphical_core::vulkan_object::VulkanApplication;
use log::info;
use std::time::Instant;
use voxel::block::BlockType;
use voxel::player::Player;
use voxel::raycast;
use vr::{VrContext, VrSupport};
use vulkan_rust::{vk, Version};
use winit::{
    dpi::LogicalSize,
    event::{DeviceEvent, ElementState, Event, MouseButton, WindowEvent},
    event_loop::{EventLoop, EventLoopWindowTarget},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, WindowBuilder},
};

const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);
const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

const VALIDATION_LAYER: &std::ffi::CStr = c"VK_LAYER_KHRONOS_validation";
const DEVICE_EXTENSIONS: &[&std::ffi::CStr] = &[
    vk::extension_names::KHR_SWAPCHAIN_EXTENSION_NAME,
    vk::extension_names::EXT_MESH_SHADER_EXTENSION_NAME,
];

const BUTTON_W: f32 = 300.0;
const BUTTON_H: f32 = 40.0;
const TEXT_SIZE: f32 = 20.0;
const TITLE_SIZE: f32 = 48.0;
const BUTTON_COLOR: [f32; 4] = [0.2, 0.2, 0.3, 0.85];
const BUTTON_HOVER: [f32; 4] = [0.3, 0.3, 0.5, 0.9];
const TEXT_COLOR: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const DIM_TEXT: [f32; 4] = [0.7, 0.7, 0.7, 1.0];

fn main() -> Result<()> {
    initialize_error_handler();
    let vr_context = probe_vr();

    let event_handler = EventLoop::new()?;
    let user_window = WindowBuilder::new()
        .with_title("Manifold")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_handler)?;

    let mut application = unsafe { VulkanApplication::create_vulkan_application(&user_window, vr_context) }?;
    let mut destroy_application = false;
    let mut minimized = false;
    let mut camera = Camera::default();
    let mut input = InputState::new();
    let mut player = Player::new();
    // Phase D': fixed-step physics at 60 Hz, decoupled from rendering.
    // Mouse look stays per-frame (smoothness), movement + gravity tick at
    // a constant rate so collision is deterministic regardless of FPS.
    const PHYSICS_TICK: f32 = 1.0 / 60.0;
    const MAX_PHYSICS_CATCHUP: f32 = 0.25;
    let mut physics_accumulator: f32 = 0.0;
    let mut last_frame = Instant::now();
    let mut fps_counter = FpsCounter::new();
    let mut game_state = GameState::TitleScreen;
    let mut cursor_pos: [f32; 2] = [0.0; 2];
    let mut menu_click = false;

    release_cursor(&user_window);

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
                    minimized = true;
                } else {
                    minimized = false;
                    application.resized = true;
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CursorMoved { position, .. },
                ..
            } => {
                cursor_pos = [position.x as f32, position.y as f32];
            }
            Event::WindowEvent {
                event: WindowEvent::KeyboardInput { event: key_event, .. },
                ..
            } => {
                if let PhysicalKey::Code(key_code) = key_event.physical_key {
                    match &mut game_state {
                        GameState::Playing => match key_event.state {
                            ElementState::Pressed => {
                                if key_code == KeyCode::Escape {
                                    unsafe { application.exit_world() };
                                    game_state = GameState::TitleScreen;
                                    release_cursor(&user_window);
                                    return;
                                }
                                if key_code == KeyCode::KeyF && !key_event.repeat {
                                    player.toggle_fly_mode();
                                }
                                if key_code == KeyCode::Space {
                                    player.jump();
                                }
                                input.key_pressed(key_code);
                            }
                            ElementState::Released => input.key_released(key_code),
                        },
                        GameState::CreateWorld { name, seed_text } => {
                            if key_event.state == ElementState::Pressed {
                                if key_code == KeyCode::Escape {
                                    game_state = GameState::TitleScreen;
                                    return;
                                }
                                if key_code == KeyCode::Backspace {
                                    if !seed_text.is_empty() {
                                        seed_text.pop();
                                    } else {
                                        name.pop();
                                    }
                                    return;
                                }
                                if key_code == KeyCode::Enter && !name.is_empty() {
                                    let seed: u32 = seed_text.parse().unwrap_or_else(|_| rand_seed());
                                    match storage::world_meta::create_world(name, seed) {
                                        Ok(dir) => {
                                            // Pregen is "ready to play" when the player's working
                                            // set is filled — not when the entire planet is in RAM.
                                            // Working set = (2·WORLD_DISTANCE+1)² columns on the
                                            // player's spawn face. The heightmap LOD covers everything
                                            // outside this neighborhood from the moment world entry.
                                            let stream_rd = graphical_core::vulkan_object::WORLD_DISTANCE;
                                            let side = (2 * stream_rd + 1) as usize;
                                            let total = side * side;
                                            let erosion_path = dir.join("erosion_map.bin");
                                            let ew = voxel::erosion_worker::ErosionWorker::start(seed, erosion_path);
                                            game_state = GameState::PreGenerating {
                                                world_dir: dir,
                                                seed,
                                                loaded: 0,
                                                total,
                                                erosion_worker: Some(ew),
                                                erosion_map: None,
                                            };
                                        }
                                        Err(e) => eprintln!("Failed to create world: {e}"),
                                    }
                                    return;
                                }
                                // Text input: letters go to name, digits go to seed
                                if let Some(text) = &key_event.text {
                                    for ch in text.chars() {
                                        if ch.is_ascii_digit() {
                                            if seed_text.len() < 10 {
                                                seed_text.push(ch);
                                            }
                                        } else if (ch.is_ascii_alphanumeric() || ch == ' ' || ch == '-' || ch == '_') && name.len() < 24 {
                                            name.push(ch);
                                        }
                                    }
                                }
                            }
                        }
                        _ => {
                            if key_event.state == ElementState::Pressed && key_code == KeyCode::Escape {
                                match &game_state {
                                    GameState::WorldSelect { .. } => game_state = GameState::TitleScreen,
                                    GameState::TitleScreen => {
                                        exit_program(&mut destroy_application, current_window, &mut application);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
            }
            Event::WindowEvent {
                event:
                    WindowEvent::MouseInput {
                        state: ElementState::Pressed,
                        button: MouseButton::Left,
                        ..
                    },
                ..
            } => {
                if game_state.is_menu() {
                    menu_click = true;
                } else {
                    input.mouse_pressed(MouseButton::Left);
                }
            }
            Event::WindowEvent {
                event:
                    WindowEvent::MouseInput {
                        state: ElementState::Pressed,
                        button,
                        ..
                    },
                ..
            } => {
                if !game_state.is_menu() {
                    input.mouse_pressed(button);
                }
            }
            Event::DeviceEvent {
                event: DeviceEvent::MouseMotion { delta: (dx, dy) },
                ..
            } => {
                if matches!(game_state, GameState::Playing) {
                    input.accumulate_mouse_delta(dx, dy);
                }
            }
            Event::AboutToWait => {
                let now = Instant::now();
                let delta_time = (now - last_frame).as_secs_f32();
                last_frame = now;

                fps_counter.tick(delta_time);

                match &mut game_state {
                    GameState::Playing => {
                        // Mouse look every frame.
                        input.apply_mouse_look(&mut player);
                        // Fixed-step physics + movement.
                        physics_accumulator = (physics_accumulator + delta_time).min(MAX_PHYSICS_CATCHUP);
                        while physics_accumulator >= PHYSICS_TICK {
                            if let Some(world) = application.world() {
                                let local_p = world.metric.sample(player.world_pos()).p;
                                input.tick_movement(&mut player, world, PHYSICS_TICK, local_p);
                            }
                            physics_accumulator -= PHYSICS_TICK;
                        }
                        camera.sync_from_player(&player);
                        let left = input.take_left_click();
                        let right = input.take_right_click();
                        if left || right {
                            if let Some(world) = application.world() {
                                if let Some(hit) = raycast::raycast(player.eye_pos(), player.forward, world) {
                                    let target = if left { hit.hit } else { hit.adjacent };
                                    let allowed = left || !player.overlaps_block(target.chunk, target.lx, target.ly, target.lz);
                                    if allowed {
                                        let new_block = if left {
                                            voxel::block::BlockType::Air
                                        } else {
                                            voxel::block::BlockType::Stone
                                        };
                                        unsafe {
                                            application.set_block_at(target.chunk, target.lx, target.ly, target.lz, new_block);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    GameState::PreGenerating { .. } => {
                        unsafe { application.update_world(&camera).ok() };
                        tick_pregen(&mut game_state, &mut application, &user_window, &mut camera, &mut player);
                    }
                    GameState::EnteringWorld { .. } => {
                        if let GameState::EnteringWorld { world_dir, seed } = std::mem::replace(&mut game_state, GameState::Playing) {
                            let erosion_map = voxel::erosion::ErosionMap::load(&world_dir.join("erosion_map.bin"))
                                .ok()
                                .map(std::sync::Arc::new);
                            if let Err(e) = unsafe { application.enter_world(&world_dir, seed, erosion_map) } {
                                eprintln!("Failed to enter world: {e}");
                                game_state = GameState::TitleScreen;
                            } else {
                                grab_cursor(&user_window);
                                camera = Camera::default();
                                player = Player::new();
                            }
                        }
                    }
                    _ => {}
                }
                user_window.request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                if destroy_application || minimized {
                    return;
                }
                if application.has_vr() {
                    if let Err(e) = application.poll_vr_events() {
                        eprintln!("VR event error: {e}");
                    }
                }
                let vr_eyes = if application.has_vr() {
                    unsafe { application.render_vr_frame() }.unwrap_or(None)
                } else {
                    None
                };
                let eyes = vr_eyes.unwrap_or_else(|| EyeMatrices::from_camera(&camera, application.swapchain_extent()));

                let result = match &game_state {
                    GameState::Playing => {
                        application.ui.begin_frame();
                        application.ui.draw_text(&fps_counter.display(), 4.0, 4.0, 16.0, [1.0, 1.0, 1.0, 0.8]);
                        let pos_text = format!(
                            "face={:?} chunk=({},{},{}) local=({:.1},{:.1},{:.1})",
                            player.face, player.cx, player.cy, player.cz, player.lx, player.ly, player.lz
                        );
                        application.ui.draw_text(&pos_text, 4.0, 22.0, 14.0, [1.0, 1.0, 1.0, 0.8]);
                        unsafe { application.render_frame(&user_window, &camera, &eyes) }
                    }
                    _ => {
                        let extent = application.swapchain_extent();
                        let sw = extent.width as f32;
                        let sh = extent.height as f32;
                        let clicked = menu_click;
                        menu_click = false;

                        application.ui.begin_frame();
                        draw_menu(&mut application.ui, &mut game_state, sw, sh, cursor_pos, clicked);
                        unsafe { application.render_menu_frame(&user_window, &eyes) }
                    }
                };
                if let Err(e) = result {
                    eprintln!("Render error: {e}");
                    exit_program(&mut destroy_application, current_window, &mut application);
                }
            }
            _ => (),
        })
        .expect("Main function crashed!");
    Ok(())
}

fn draw_menu(ui: &mut UiPipeline, state: &mut GameState, sw: f32, sh: f32, cursor: [f32; 2], clicked: bool) {
    match state {
        GameState::TitleScreen => {
            let title = "MANIFOLD";
            let tw = UiPipeline::text_width(title, TITLE_SIZE);
            ui.draw_text(title, (sw - tw) / 2.0, sh * 0.2, TITLE_SIZE, TEXT_COLOR);

            let cx = (sw - BUTTON_W) / 2.0;
            if draw_button(ui, "Singleplayer", cx, sh * 0.45, cursor, clicked) {
                *state = GameState::WorldSelect {
                    worlds: storage::world_meta::list_worlds(),
                };
            }
            if draw_button(ui, "Quit", cx, sh * 0.55, cursor, clicked) {
                std::process::exit(0);
            }
        }
        GameState::WorldSelect { worlds } => {
            let title = "Select World";
            let tw = UiPipeline::text_width(title, TEXT_SIZE * 1.5);
            ui.draw_text(title, (sw - tw) / 2.0, 40.0, TEXT_SIZE * 1.5, TEXT_COLOR);

            let cx = (sw - BUTTON_W) / 2.0;
            let mut y = 100.0;
            let mut selected = None;
            for (i, (_, meta)) in worlds.iter().enumerate() {
                let label = format!("{} (seed: {})", meta.name, meta.seed);
                if draw_button(ui, &label, cx, y, cursor, clicked) {
                    selected = Some(i);
                }
                y += BUTTON_H + 10.0;
            }
            if let Some(i) = selected {
                let (dir, meta) = &worlds[i];
                // Existing world — enter directly, terrain streams in progressively
                *state = GameState::EnteringWorld {
                    world_dir: dir.clone(),
                    seed: meta.seed,
                };
                return;
            }

            y += 20.0;
            if draw_button(ui, "Create New World", cx, y, cursor, clicked) {
                *state = GameState::CreateWorld {
                    name: String::new(),
                    seed_text: String::new(),
                };
                return;
            }
            if draw_button(ui, "Back", cx, y + BUTTON_H + 10.0, cursor, clicked) {
                *state = GameState::TitleScreen;
            }
        }
        GameState::CreateWorld { name, seed_text } => {
            let title = "Create New World";
            let tw = UiPipeline::text_width(title, TEXT_SIZE * 1.5);
            ui.draw_text(title, (sw - tw) / 2.0, 40.0, TEXT_SIZE * 1.5, TEXT_COLOR);

            let cx = (sw - BUTTON_W) / 2.0;
            let mut y = 120.0;

            ui.draw_text("Name:", cx, y, TEXT_SIZE, DIM_TEXT);
            y += 25.0;
            ui.draw_rect(cx, y, BUTTON_W, BUTTON_H, [0.15, 0.15, 0.2, 0.9]);
            let name_empty = name.is_empty();
            let display_name: &str = if name_empty { "type a name..." } else { name };
            let name_color = if name_empty { DIM_TEXT } else { TEXT_COLOR };
            ui.draw_text(display_name, cx + 10.0, y + 10.0, TEXT_SIZE, name_color);
            y += BUTTON_H + 20.0;

            ui.draw_text("Seed (digits, optional):", cx, y, TEXT_SIZE, DIM_TEXT);
            y += 25.0;
            ui.draw_rect(cx, y, BUTTON_W, BUTTON_H, [0.15, 0.15, 0.2, 0.9]);
            let seed_empty = seed_text.is_empty();
            let display_seed: &str = if seed_empty { "random" } else { seed_text };
            let seed_color = if seed_empty { DIM_TEXT } else { TEXT_COLOR };
            ui.draw_text(display_seed, cx + 10.0, y + 10.0, TEXT_SIZE, seed_color);
            y += BUTTON_H + 30.0;

            if !name.is_empty() {
                if draw_button(ui, "Create & Play", cx, y, cursor, clicked) {
                    let seed: u32 = seed_text.parse().unwrap_or_else(|_| rand_seed());
                    match storage::world_meta::create_world(name, seed) {
                        Ok(dir) => {
                            let stream_rd = graphical_core::vulkan_object::WORLD_DISTANCE;
                            let side = (2 * stream_rd + 1) as usize;
                            let total = side * side;
                            let erosion_path = dir.join("erosion_map.bin");
                            let ew = voxel::erosion_worker::ErosionWorker::start(seed, erosion_path);
                            *state = GameState::PreGenerating {
                                world_dir: dir,
                                seed,
                                loaded: 0,
                                total,
                                erosion_worker: Some(ew),
                                erosion_map: None,
                            };
                        }
                        Err(e) => eprintln!("Failed to create world: {e}"),
                    }
                    return;
                }
                y += BUTTON_H + 10.0;
            }
            if draw_button(ui, "Back", cx, y, cursor, clicked) {
                *state = GameState::TitleScreen;
            }
        }
        GameState::PreGenerating {
            loaded, total, erosion_map, ..
        } => {
            let title = if erosion_map.is_none() {
                "Running hydraulic erosion..."
            } else if *loaded >= *total {
                "Building LOD terrain..."
            } else {
                "Generating terrain..."
            };
            let tw = UiPipeline::text_width(title, TEXT_SIZE * 1.5);
            ui.draw_text(title, (sw - tw) / 2.0, sh * 0.35, TEXT_SIZE * 1.5, TEXT_COLOR);

            let bar_w = 400.0;
            let bar_h = 30.0;
            let bx = (sw - bar_w) / 2.0;
            let by = sh * 0.5;
            let progress = if *total > 0 { *loaded as f32 / *total as f32 } else { 0.0 };
            ui.draw_rect(bx, by, bar_w, bar_h, [0.15, 0.15, 0.2, 0.9]);
            ui.draw_rect(bx + 2.0, by + 2.0, (bar_w - 4.0) * progress.min(1.0), bar_h - 4.0, [0.3, 0.7, 0.3, 1.0]);

            let pct = format!("{}%", (progress * 100.0).min(100.0) as u32);
            let pct_w = UiPipeline::text_width(&pct, TEXT_SIZE);
            ui.draw_text(&pct, (sw - pct_w) / 2.0, by + 5.0, TEXT_SIZE, TEXT_COLOR);
        }
        GameState::EnteringWorld { .. } | GameState::Playing => {}
    }
}

/// Draw a button, returns true if clicked.
fn draw_button(ui: &mut UiPipeline, label: &str, x: f32, y: f32, cursor: [f32; 2], clicked: bool) -> bool {
    let hovered = cursor[0] >= x && cursor[0] <= x + BUTTON_W && cursor[1] >= y && cursor[1] <= y + BUTTON_H;
    let color = if hovered { BUTTON_HOVER } else { BUTTON_COLOR };
    ui.draw_rect(x, y, BUTTON_W, BUTTON_H, color);
    let tw = UiPipeline::text_width(label, TEXT_SIZE);
    let tx = x + (BUTTON_W - tw) / 2.0;
    let ty = y + (BUTTON_H - TEXT_SIZE) / 2.0;
    ui.draw_text(label, tx, ty, TEXT_SIZE, TEXT_COLOR);
    hovered && clicked
}

fn tick_pregen(state: &mut GameState, application: &mut VulkanApplication, window: &winit::window::Window, camera: &mut Camera, player: &mut Player) {
    let GameState::PreGenerating {
        world_dir,
        seed,
        loaded,
        total,
        erosion_worker,
        erosion_map,
    } = state
    else {
        return;
    };

    // Poll erosion worker
    if erosion_map.is_none() {
        if let Some(worker) = erosion_worker.as_ref() {
            if let Some(map) = worker.try_receive() {
                *erosion_map = Some(std::sync::Arc::new(map));
                *erosion_worker = None; // Worker done, drop it
            }
        }
    }

    // Don't enter world until erosion map is ready
    if erosion_map.is_none() {
        return;
    }

    if !application.has_world() {
        let dir = world_dir.clone();
        let s = *seed;
        let emap = erosion_map.clone();
        if let Err(e) = unsafe { application.enter_world(&dir, s, emap) } {
            eprintln!("Failed to enter world: {e}");
            *state = GameState::TitleScreen;
            return;
        }
    }
    let col_count = if let Some(world) = application.world() {
        let mut seen = std::collections::HashSet::new();
        for cp in world.chunk_positions() {
            seen.insert((cp.face, cp.cx, cp.cz));
        }
        seen.len()
    } else {
        0
    };
    *loaded = col_count;
    // Done when raw terrain loaded AND LOD settled (no submissions, no in-flight, stable)
    if col_count >= *total && application.lod_settled() {
        *state = GameState::Playing;
        grab_cursor(window);
        *camera = Camera::default();
        *player = Player::new();
    }
}

fn rand_seed() -> u32 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u32
}

fn grab_cursor(window: &winit::window::Window) {
    let _ = window.set_cursor_grab(CursorGrabMode::Confined);
    window.set_cursor_visible(false);
}

fn release_cursor(window: &winit::window::Window) {
    let _ = window.set_cursor_grab(CursorGrabMode::None);
    window.set_cursor_visible(true);
}

fn exit_program(destroy_application: &mut bool, current_window: &EventLoopWindowTarget<()>, application: &mut VulkanApplication) {
    *destroy_application = true;
    current_window.exit();
    unsafe { application.destroy_vulkan_application() }
}

fn probe_vr() -> Option<VrContext> {
    match VrContext::probe() {
        Ok(VrSupport::Available(ctx)) => {
            info!("VR available — OpenXR session ready for creation");
            Some(ctx)
        }
        Ok(VrSupport::Unavailable(reason)) => {
            info!("VR unavailable: {reason} — running in desktop mode");
            None
        }
        Err(e) => {
            info!("VR probe failed: {e:#} — running in desktop mode");
            None
        }
    }
}

fn initialize_error_handler() {
    pretty_env_logger::init();
}

struct FpsCounter {
    frames: u32,
    elapsed: f32,
    display_fps: u32,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            frames: 0,
            elapsed: 0.0,
            display_fps: 0,
        }
    }

    fn tick(&mut self, dt: f32) {
        self.frames += 1;
        self.elapsed += dt;
        if self.elapsed >= 0.5 {
            self.display_fps = (self.frames as f32 / self.elapsed) as u32;
            self.frames = 0;
            self.elapsed = 0.0;
        }
    }

    fn display(&self) -> String {
        format!("{} fps", self.display_fps)
    }
}
