use crate::graphical_core::camera::Camera;
use std::collections::HashSet;
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

const MOVE_SPEED: f32 = 3.0;
const SPRINT_MULTIPLIER: f32 = 10.0;
const MOUSE_SENSITIVITY: f32 = 0.003;
const MAX_PITCH: f32 = 89.0_f32 * (std::f32::consts::PI / 180.0);

pub struct InputState {
    pressed_keys: HashSet<KeyCode>,
    mouse_delta: (f64, f64),
    left_click: bool,
    right_click: bool,
}

impl InputState {
    pub fn new() -> Self {
        Self {
            pressed_keys: HashSet::new(),
            mouse_delta: (0.0, 0.0),
            left_click: false,
            right_click: false,
        }
    }

    pub fn key_pressed(&mut self, key: KeyCode) {
        self.pressed_keys.insert(key);
    }

    pub fn key_released(&mut self, key: KeyCode) {
        self.pressed_keys.remove(&key);
    }

    pub fn accumulate_mouse_delta(&mut self, dx: f64, dy: f64) {
        self.mouse_delta.0 += dx;
        self.mouse_delta.1 += dy;
    }

    /// Records a mouse button press as a one-shot event (consumed on read).
    pub fn mouse_pressed(&mut self, button: MouseButton) {
        match button {
            MouseButton::Left => self.left_click = true,
            MouseButton::Right => self.right_click = true,
            _ => {}
        }
    }

    /// Returns and clears the left-click flag.
    #[allow(dead_code)] // wired up in block interaction commit
    pub fn take_left_click(&mut self) -> bool {
        std::mem::take(&mut self.left_click)
    }

    /// Returns and clears the right-click flag.
    #[allow(dead_code)] // wired up in block interaction commit
    pub fn take_right_click(&mut self) -> bool {
        std::mem::take(&mut self.right_click)
    }

    pub fn is_pressed(&self, key: KeyCode) -> bool {
        self.pressed_keys.contains(&key)
    }

    pub fn update_camera(&mut self, camera: &mut Camera, delta_time: f32, fly_mode: bool) {
        self.apply_mouse_look(camera);
        if fly_mode {
            self.apply_fly_movement(camera, delta_time);
        } else {
            self.apply_walk_movement(camera, delta_time);
        }
    }

    fn apply_mouse_look(&mut self, camera: &mut Camera) {
        let (dx, dy) = self.mouse_delta;
        self.mouse_delta = (0.0, 0.0);

        camera.yaw += dx as f32 * MOUSE_SENSITIVITY;
        camera.pitch -= dy as f32 * MOUSE_SENSITIVITY;
        camera.pitch = camera.pitch.clamp(-MAX_PITCH, MAX_PITCH);
    }

    fn apply_fly_movement(&self, camera: &mut Camera, delta_time: f32) {
        let multiplier = if self.is_pressed(KeyCode::ShiftLeft) { SPRINT_MULTIPLIER } else { 1.0 };
        let speed = MOVE_SPEED * multiplier * delta_time;
        let front = camera.front();
        let right = camera.right();

        if self.is_pressed(KeyCode::KeyW) {
            camera.position += front * speed;
        }
        if self.is_pressed(KeyCode::KeyS) {
            camera.position -= front * speed;
        }
        if self.is_pressed(KeyCode::KeyD) {
            camera.position += right * speed;
        }
        if self.is_pressed(KeyCode::KeyA) {
            camera.position -= right * speed;
        }
        if self.is_pressed(KeyCode::KeyQ) {
            camera.position -= glam::Vec3::Y * speed;
        }
        if self.is_pressed(KeyCode::KeyE) {
            camera.position += glam::Vec3::Y * speed;
        }
    }

    /// Walk movement: WASD moves horizontally (ignoring pitch), no Q/E vertical.
    fn apply_walk_movement(&self, camera: &mut Camera, delta_time: f32) {
        let speed = MOVE_SPEED * delta_time;
        let front = camera.front();
        let right = camera.right();

        // Flatten front vector to horizontal plane
        let forward = glam::Vec3::new(front.x, 0.0, front.z).normalize_or_zero();

        if self.is_pressed(KeyCode::KeyW) {
            camera.position += forward * speed;
        }
        if self.is_pressed(KeyCode::KeyS) {
            camera.position -= forward * speed;
        }
        if self.is_pressed(KeyCode::KeyD) {
            camera.position += right * speed;
        }
        if self.is_pressed(KeyCode::KeyA) {
            camera.position -= right * speed;
        }
    }
}
