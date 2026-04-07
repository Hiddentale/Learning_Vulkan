use super::world::World;
use glam::Vec3;

const GRAVITY: f32 = 20.0;
const JUMP_VELOCITY: f32 = 8.0;
const PLAYER_HEIGHT: f32 = 1.7;
const PLAYER_HALF_WIDTH: f32 = 0.3;

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
    /// inward by PLAYER_HEIGHT along that direction. `on_ground` is sticky
    /// — gravity is suppressed while standing on a surface to avoid
    /// per-frame bobbing.
    pub fn apply_physics(&mut self, position: &mut Vec3, delta_time: f32, world: &World) {
        if self.fly_mode {
            return;
        }
        let feet_solid = |p: Vec3| -> bool {
            let f = p - p.normalize_or(Vec3::Y) * PLAYER_HEIGHT;
            world.is_solid(f.x, f.y, f.z)
        };

        if self.on_ground {
            // Walked off the ledge? Sample a block below the feet.
            let up = position.normalize_or(Vec3::Y);
            let probe = *position - up * (PLAYER_HEIGHT + 0.2);
            if !world.is_solid(probe.x, probe.y, probe.z) {
                self.on_ground = false;
            } else {
                // Standing — no integration this frame.
                return;
            }
        }

        let up = position.normalize_or(Vec3::Y);
        self.radial_velocity -= GRAVITY * delta_time;
        *position += up * self.radial_velocity * delta_time;

        if feet_solid(*position) {
            // Lift radially in 0.1-block steps until clear (max ~3 blocks).
            for _ in 0..32 {
                let new_up = position.normalize_or(Vec3::Y);
                *position += new_up * 0.1;
                if !feet_solid(*position) {
                    break;
                }
            }
            self.radial_velocity = 0.0;
            self.on_ground = true;
        }
    }

    /// Stop horizontal motion if the new position would intersect terrain.
    /// Samples a 4-point capsule footprint at head and feet height in the
    /// local tangent plane around `position`.
    pub fn resolve_horizontal(&self, position: &mut Vec3, old_position: Vec3, world: &World) {
        if self.fly_mode {
            return;
        }
        if capsule_intersects(*position, world) {
            *position = old_position;
        }
    }
}

/// True if any point of the player's capsule footprint at `pos` overlaps a
/// solid block. The footprint is 4 corners offset by PLAYER_HALF_WIDTH in
/// two perpendicular tangent directions, sampled at head, mid, and feet.
fn capsule_intersects(pos: Vec3, world: &World) -> bool {
    let up = pos.normalize_or(Vec3::Y);
    // Pick any two perpendicular tangent axes — they only need to be
    // orthogonal to up; their absolute orientation does not matter for a
    // 4-corner test.
    let helper = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Z };
    let t1 = (helper - up * helper.dot(up)).normalize_or(Vec3::X);
    let t2 = up.cross(t1).normalize_or(Vec3::Z);
    let hw = PLAYER_HALF_WIDTH;
    let offsets = [
        t1 * hw + t2 * hw,
        t1 * hw - t2 * hw,
        -t1 * hw + t2 * hw,
        -t1 * hw - t2 * hw,
    ];
    let heights = [0.0_f32, -0.5 * PLAYER_HEIGHT, -(PLAYER_HEIGHT - 0.1)];
    for h in heights {
        let center = pos + up * h;
        for off in offsets {
            let p = center + off;
            if world.is_solid(p.x, p.y, p.z) {
                return true;
            }
        }
    }
    false
}
