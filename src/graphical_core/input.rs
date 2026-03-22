use crate::graphical_core::camera::Camera;
use std::collections::HashSet;
use winit::keyboard::KeyCode;

const MOVE_SPEED: f32 = 3.0;
const MOUSE_SENSITIVITY: f32 = 0.003;
const MAX_PITCH: f32 = 89.0_f32 * (std::f32::consts::PI / 180.0);

pub struct InputState {
    pressed_keys: HashSet<KeyCode>,
    mouse_delta: (f64, f64),
}

impl InputState {
    pub fn new() -> Self {
        Self {
            pressed_keys: HashSet::new(),
            mouse_delta: (0.0, 0.0),
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

    pub fn is_pressed(&self, key: KeyCode) -> bool {
        self.pressed_keys.contains(&key)
    }

    pub fn update_camera(&mut self, camera: &mut Camera, delta_time: f32) {
        self.apply_mouse_look(camera);
        self.apply_movement(camera, delta_time);
    }

    fn apply_mouse_look(&mut self, camera: &mut Camera) {
        let (dx, dy) = self.mouse_delta;
        self.mouse_delta = (0.0, 0.0);

        camera.yaw += dx as f32 * MOUSE_SENSITIVITY;
        camera.pitch += dy as f32 * MOUSE_SENSITIVITY;
        camera.pitch = camera.pitch.clamp(-MAX_PITCH, MAX_PITCH);
    }

    fn apply_movement(&self, camera: &mut Camera, delta_time: f32) {
        let speed = MOVE_SPEED * delta_time;
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
}
