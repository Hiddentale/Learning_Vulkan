use super::world::World;
use glam::Vec3;

const GRAVITY: f32 = 20.0;
const JUMP_VELOCITY: f32 = 8.0;
const PLAYER_HEIGHT: f32 = 1.7;

/// Phase D: velocity is stored as a scalar along the radial direction.
/// Horizontal motion is handled by the input system in the tangent plane;
/// only vertical (radial) integration lives in the player physics.
pub struct Player {
    pub radial_velocity: f32,
    pub on_ground: bool,
    pub fly_mode: bool,
}

impl Player {
    pub fn new() -> Self {
        Self {
            radial_velocity: 0.0,
            on_ground: false,
            fly_mode: true,
        }
    }

    pub fn jump(&mut self) {
        if self.on_ground && !self.fly_mode {
            self.radial_velocity = JUMP_VELOCITY;
            self.on_ground = false;
        }
    }

    pub fn toggle_fly_mode(&mut self) {
        self.fly_mode = !self.fly_mode;
        self.radial_velocity = 0.0;
    }

    /// Apply radial gravity. The "down" vector is `-position.normalize()`
    /// (toward planet centre); the "feet" sample is the position offset
    /// inward by PLAYER_HEIGHT along that direction.
    pub fn apply_physics(&mut self, position: &mut Vec3, delta_time: f32, world: &World) {
        if self.fly_mode {
            return;
        }
        let up = position.normalize_or(Vec3::Y);
        self.radial_velocity -= GRAVITY * delta_time;
        *position += up * self.radial_velocity * delta_time;

        // Ground check: sample one block below the feet along -up.
        let feet = *position - up * PLAYER_HEIGHT;
        if world.is_solid(feet.x, feet.y, feet.z) {
            // Push back to the surface: lift along +up until feet are clear.
            for _ in 0..16 {
                let p = *position + up * 0.05;
                let f = p - up * PLAYER_HEIGHT;
                if !world.is_solid(f.x, f.y, f.z) {
                    *position = p;
                    break;
                }
                *position = p;
            }
            self.radial_velocity = 0.0;
            self.on_ground = true;
        } else {
            self.on_ground = false;
        }
    }

    /// Stop horizontal motion if the new position would intersect terrain.
    /// "Horizontal" means the displacement in the tangent plane.
    pub fn resolve_horizontal(&self, position: &mut Vec3, old_position: Vec3, world: &World) {
        if self.fly_mode {
            return;
        }
        let up = position.normalize_or(Vec3::Y);
        let head = *position;
        let feet = *position - up * (PLAYER_HEIGHT - 0.1);
        if world.is_solid(head.x, head.y, head.z) || world.is_solid(feet.x, feet.y, feet.z) {
            *position = old_position;
        }
    }
}
