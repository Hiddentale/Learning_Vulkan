use super::block::BlockType;
use super::metric::minkowski_distance;
use super::world::World;
use glam::Vec3;

const MAX_REACH: f32 = 8.0;
const MAX_STEPS: usize = 200;

/// Result of a successful raycast: the hit block and the empty block adjacent to it.
pub struct RaycastHit {
    /// World-space integer coordinates of the block that was hit.
    pub block: [i32; 3],
    /// World-space integer coordinates of the empty block adjacent to the hit face.
    /// Used for block placement.
    pub adjacent: [i32; 3],
}

/// Casts a ray from `origin` in `direction` through the voxel world.
/// Returns the first non-air block hit within reach, plus the adjacent empty position.
/// Reach distance is computed using the local metric at the origin.
pub fn raycast(origin: Vec3, direction: Vec3, world: &World) -> Option<RaycastHit> {
    let dir = direction.normalize();
    let local_p = world.metric.sample(origin).p;

    // Current voxel position
    let mut x = origin.x.floor() as i32;
    let mut y = origin.y.floor() as i32;
    let mut z = origin.z.floor() as i32;

    // Step direction (+1 or -1)
    let step_x = if dir.x >= 0.0 { 1 } else { -1 };
    let step_y = if dir.y >= 0.0 { 1 } else { -1 };
    let step_z = if dir.z >= 0.0 { 1 } else { -1 };

    // Distance along ray to cross one full voxel in each axis
    let t_delta_x = if dir.x != 0.0 { (1.0 / dir.x).abs() } else { f32::INFINITY };
    let t_delta_y = if dir.y != 0.0 { (1.0 / dir.y).abs() } else { f32::INFINITY };
    let t_delta_z = if dir.z != 0.0 { (1.0 / dir.z).abs() } else { f32::INFINITY };

    // Distance along ray to the next voxel boundary in each axis
    let mut t_max_x = if dir.x != 0.0 {
        let boundary = if dir.x > 0.0 { (x + 1) as f32 } else { x as f32 };
        (boundary - origin.x) / dir.x
    } else {
        f32::INFINITY
    };
    let mut t_max_y = if dir.y != 0.0 {
        let boundary = if dir.y > 0.0 { (y + 1) as f32 } else { y as f32 };
        (boundary - origin.y) / dir.y
    } else {
        f32::INFINITY
    };
    let mut t_max_z = if dir.z != 0.0 {
        let boundary = if dir.z > 0.0 { (z + 1) as f32 } else { z as f32 };
        (boundary - origin.z) / dir.z
    } else {
        f32::INFINITY
    };

    let mut prev = [x, y, z];

    for _ in 0..MAX_STEPS {
        let block = world.get_block(x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5);
        if block != BlockType::Air && block != BlockType::Water {
            return Some(RaycastHit {
                block: [x, y, z],
                adjacent: prev,
            });
        }

        prev = [x, y, z];

        // Check reach using the local metric
        let t_min = t_max_x.min(t_max_y).min(t_max_z);
        let current_pos = origin + dir * t_min;
        if minkowski_distance(origin, current_pos, local_p) > MAX_REACH {
            break;
        }

        // Advance along the axis with the smallest t_max
        if t_max_x <= t_max_y && t_max_x <= t_max_z {
            x += step_x;
            t_max_x += t_delta_x;
        } else if t_max_y <= t_max_z {
            y += step_y;
            t_max_y += t_delta_y;
        } else {
            z += step_z;
            t_max_z += t_delta_z;
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::metric::Anomaly;

    fn test_world_with_block(bx: i32, by: i32, bz: i32) -> World {
        let mut world = World::new(2, 42);
        let [cx, cy, cz] = World::block_to_chunk(bx, by, bz);
        world.insert_empty_chunk(cx, cy, cz);
        world.set_block(bx, by, bz, BlockType::Stone);
        world
    }

    #[test]
    fn raycast_hits_block_directly_ahead() {
        let world = test_world_with_block(0, 5, -3);
        let hit = raycast(Vec3::new(0.5, 5.5, 0.5), Vec3::new(0.0, 0.0, -1.0), &world);
        let hit = hit.expect("should hit");
        assert_eq!(hit.block, [0, 5, -3]);
        assert_eq!(hit.adjacent, [0, 5, -2]);
    }

    #[test]
    fn raycast_misses_when_no_block_in_range() {
        let world = test_world_with_block(0, 5, -20);
        let hit = raycast(Vec3::new(0.5, 5.5, 0.5), Vec3::new(0.0, 0.0, -1.0), &world);
        assert!(hit.is_none());
    }

    #[test]
    fn raycast_returns_correct_adjacent_for_top_face() {
        let world = test_world_with_block(0, 3, 0);
        let hit = raycast(Vec3::new(0.5, 7.5, 0.5), Vec3::new(0.0, -1.0, 0.0), &world);
        let hit = hit.expect("should hit");
        assert_eq!(hit.block, [0, 3, 0]);
        assert_eq!(hit.adjacent, [0, 4, 0]);
    }

    #[test]
    fn manhattan_metric_reduces_diagonal_reach() {
        // Place a block 6 blocks away diagonally
        // Euclidean distance: sqrt(36+36) ~ 8.49 (out of reach)
        // Manhattan distance: 6+6 = 12 (way out of reach)
        // With normal Euclidean, a block at 5 diagonal would be sqrt(50) ~ 7.07 (in reach)
        // With Manhattan, 5+5 = 10 (out of reach at MAX_REACH=8)
        let mut world = World::new(2, 42);
        world.insert_empty_chunk(0, 0, 0);
        world.set_block(5, 0, 5, BlockType::Stone);

        // Add Manhattan anomaly at origin
        world.metric.add(Anomaly {
            center: Vec3::ZERO,
            inner_radius: 100.0,
            outer_radius: 200.0,
            target_p: 1.0,
            active: true,
        });

        let hit = raycast(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 0.0, 1.0).normalize(), &world);
        assert!(hit.is_none(), "Manhattan should reduce diagonal reach");
    }
}
