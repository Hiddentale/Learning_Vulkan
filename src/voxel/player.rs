use super::world::World;
use glam::Vec3;

const GRAVITY: f32 = 20.0;
const JUMP_VELOCITY: f32 = 8.0;
const PLAYER_HEIGHT: f32 = 1.7;
const PLAYER_HALF_WIDTH: f32 = 0.3;

pub struct Player {
    pub velocity: Vec3,
    pub on_ground: bool,
    pub fly_mode: bool,
}

impl Player {
    pub fn new() -> Self {
        Self {
            velocity: Vec3::ZERO,
            on_ground: false,
            fly_mode: true,
        }
    }

    pub fn jump(&mut self) {
        if self.on_ground && !self.fly_mode {
            self.velocity.y = JUMP_VELOCITY;
            self.on_ground = false;
        }
    }

    pub fn toggle_fly_mode(&mut self) {
        self.fly_mode = !self.fly_mode;
        self.velocity = Vec3::ZERO;
    }

    /// Applies gravity and resolves vertical collision against the world.
    pub fn apply_physics(&mut self, position: &mut Vec3, delta_time: f32, world: &World) {
        if self.fly_mode {
            return;
        }

        self.velocity.y -= GRAVITY * delta_time;
        position.y += self.velocity.y * delta_time;

        // Check ground collision (feet position)
        let feet_y = position.y - PLAYER_HEIGHT;
        if is_colliding(position.x, feet_y, position.z, world) {
            let block_top = feet_y.floor() + 1.0;
            position.y = block_top + PLAYER_HEIGHT;
            self.velocity.y = 0.0;
            self.on_ground = true;
        } else {
            self.on_ground = false;
        }

        // Check head collision (ceiling)
        if is_colliding(position.x, position.y, position.z, world) {
            position.y = position.y.floor();
            self.velocity.y = self.velocity.y.min(0.0);
        }
    }

    /// Resolves horizontal collision by checking each axis independently.
    pub fn resolve_horizontal(&self, position: &mut Vec3, old_position: Vec3, world: &World) {
        if self.fly_mode {
            return;
        }

        let feet_y = position.y - PLAYER_HEIGHT;
        let head_y = position.y;
        let check_heights = [feet_y, feet_y + 0.5, head_y - 0.5, head_y];

        // Check X axis
        let test_x = position.x + PLAYER_HALF_WIDTH * position.x.partial_cmp(&old_position.x).map_or(0.0, sign);
        for &y in &check_heights {
            if world.is_solid(test_x, y, position.z) {
                position.x = old_position.x;
                break;
            }
        }

        // Check Z axis
        let test_z = position.z + PLAYER_HALF_WIDTH * position.z.partial_cmp(&old_position.z).map_or(0.0, sign);
        for &y in &check_heights {
            if world.is_solid(position.x, y, test_z) {
                position.z = old_position.z;
                break;
            }
        }
    }
}

fn sign(ord: std::cmp::Ordering) -> f32 {
    match ord {
        std::cmp::Ordering::Greater => 1.0,
        std::cmp::Ordering::Less => -1.0,
        std::cmp::Ordering::Equal => 0.0,
    }
}

fn is_colliding(x: f32, y: f32, z: f32, world: &World) -> bool {
    // Check the 4 corners of the player's horizontal footprint
    let hw = PLAYER_HALF_WIDTH;
    world.is_solid(x - hw, y, z - hw) || world.is_solid(x + hw, y, z - hw) || world.is_solid(x - hw, y, z + hw) || world.is_solid(x + hw, y, z + hw)
}
