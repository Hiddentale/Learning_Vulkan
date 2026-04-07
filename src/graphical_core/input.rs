use crate::voxel::metric;
use crate::voxel::player::Player;
use crate::voxel::world::World;
use std::collections::HashSet;
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

const MOVE_SPEED: f32 = 3.0;
const SPRINT_MULTIPLIER: f32 = 100.0;
const MOUSE_SENSITIVITY: f32 = 0.003;

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
    pub fn take_left_click(&mut self) -> bool {
        std::mem::take(&mut self.left_click)
    }

    /// Returns and clears the right-click flag.
    pub fn take_right_click(&mut self) -> bool {
        std::mem::take(&mut self.right_click)
    }

    pub fn is_pressed(&self, key: KeyCode) -> bool {
        self.pressed_keys.contains(&key)
    }

    pub fn update_player(&mut self, player: &mut Player, world: &World, delta_time: f32, local_p: f32) {
        self.apply_mouse_look(player);
        if player.fly_mode {
            self.apply_fly_movement(player, delta_time, local_p);
        } else {
            self.apply_walk_movement(player, world, delta_time, local_p);
        }
    }

    fn apply_mouse_look(&mut self, player: &mut Player) {
        let (dx, dy) = self.mouse_delta;
        self.mouse_delta = (0.0, 0.0);
        player.rotate_yaw(-dx as f32 * MOUSE_SENSITIVITY);
        player.rotate_pitch(-dy as f32 * MOUSE_SENSITIVITY);
    }

    fn apply_fly_movement(&self, player: &mut Player, delta_time: f32, local_p: f32) {
        let multiplier = if self.is_pressed(KeyCode::ShiftLeft) { SPRINT_MULTIPLIER } else { 1.0 };
        let speed = MOVE_SPEED * multiplier * delta_time;
        let front = player.forward;
        let right = player.right();
        let up = player.up();

        let mut move_dir = glam::Vec3::ZERO;
        if self.is_pressed(KeyCode::KeyW) { move_dir += front; }
        if self.is_pressed(KeyCode::KeyS) { move_dir -= front; }
        if self.is_pressed(KeyCode::KeyD) { move_dir += right; }
        if self.is_pressed(KeyCode::KeyA) { move_dir -= right; }
        if self.is_pressed(KeyCode::KeyE) { move_dir += up; }
        if self.is_pressed(KeyCode::KeyQ) { move_dir -= up; }

        if move_dir != glam::Vec3::ZERO {
            let scale = metric::metric_speed_scale(move_dir, local_p);
            player.fly_move(move_dir.normalize() * speed * scale);
        }
    }

    /// Walk movement: WASD moves in the local tangent plane (ignoring pitch).
    fn apply_walk_movement(&self, player: &mut Player, world: &World, delta_time: f32, local_p: f32) {
        let speed = MOVE_SPEED * delta_time;
        let up = player.up();
        let front = player.forward;
        let forward = (front - up * front.dot(up)).normalize_or(player.right());
        let right = forward.cross(up).normalize_or(player.right());

        let mut move_dir = glam::Vec3::ZERO;
        if self.is_pressed(KeyCode::KeyW) { move_dir += forward; }
        if self.is_pressed(KeyCode::KeyS) { move_dir -= forward; }
        if self.is_pressed(KeyCode::KeyD) { move_dir += right; }
        if self.is_pressed(KeyCode::KeyA) { move_dir -= right; }

        if move_dir != glam::Vec3::ZERO {
            let scale = metric::metric_speed_scale(move_dir, local_p);
            player.walk(move_dir.normalize() * speed * scale, world);
        }
    }
}
