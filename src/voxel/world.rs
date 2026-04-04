use super::block::BlockType;
use super::chunk::{Chunk, CHUNK_SIZE};
use super::chunk_generator::ChunkGenerator;
use super::metric::MetricField;
use std::collections::HashMap;

pub const MIN_CHUNK_Y: i32 = 0;
pub const MAX_CHUNK_Y: i32 = 15;

/// Chunks within this Chebyshev distance (in chunk coords) are near-field (mesh shader).
pub const NEAR_RADIUS: i32 = 6;
/// Width of the transition band around NEAR_RADIUS where chunks exist in both systems.
pub const TRANSITION_WIDTH: i32 = 2;

/// Select LOD level based on Chebyshev distance from player (in chunks).
#[allow(dead_code)] // Used when LOD streaming is fully wired
pub fn lod_for_distance(dist: i32) -> u32 {
    if dist <= NEAR_RADIUS + TRANSITION_WIDTH {
        0 // LOD-0 (full res SVDAG)
    } else if dist <= 24 {
        1
    } else if dist <= 48 {
        2
    } else {
        3
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChunkTier {
    Near,
    /// Overlap zone: chunk exists in both mesh shader and SVDAG pipelines.
    Transition,
    Far,
}

pub struct World {
    chunks: HashMap<[i32; 3], Chunk>,
    tiers: HashMap<[i32; 3], ChunkTier>,
    render_distance: i32,
    generator: ChunkGenerator,
    pub metric: MetricField,
}

/// Result of a world update: which chunks were added/removed or changed tier.
pub struct WorldDelta {
    pub loaded: Vec<[i32; 3]>,
    pub unloaded: Vec<[i32; 3]>,
    /// Chunks that moved closer to player (far→transition or transition→near).
    pub promoted: Vec<[i32; 3]>,
    /// Chunks that moved farther from player (near→transition or transition→far).
    pub demoted: Vec<[i32; 3]>,
    /// Chunks that entered the transition band (need SVDAG pre-compression).
    pub entered_transition: Vec<[i32; 3]>,
}

impl World {
    pub fn new(render_distance: i32) -> Self {
        Self {
            chunks: HashMap::new(),
            tiers: HashMap::new(),
            render_distance,
            generator: ChunkGenerator::new(),
            metric: MetricField::new(),
        }
    }

    /// Non-blocking world update. Unloads out-of-range chunks immediately,
    /// requests missing columns from background threads, and receives any
    /// completed columns. Never blocks the calling thread.
    pub fn update(&mut self, player_cx: i32, player_cz: i32) -> WorldDelta {
        let mut loaded = Vec::new();
        let mut unloaded = Vec::new();
        let mut promoted = Vec::new();
        let mut demoted = Vec::new();
        let mut entered_transition = Vec::new();

        // Unload columns outside render distance (always immediate)
        let keys: Vec<[i32; 3]> = self.chunks.keys().copied().collect();
        for pos in keys {
            if !in_range(pos[0], pos[2], player_cx, player_cz, self.render_distance) {
                self.chunks.remove(&pos);
                self.tiers.remove(&pos);
                unloaded.push(pos);
                self.generator.cancel(pos[0], pos[2]);
            }
        }

        // Reclassify existing chunks that changed tier
        let tier_keys: Vec<[i32; 3]> = self.tiers.keys().copied().collect();
        for pos in tier_keys {
            let new_tier = classify_tier(pos[0], pos[2], player_cx, player_cz);
            let old_tier = self.tiers[&pos];
            if old_tier != new_tier {
                self.tiers.insert(pos, new_tier);
                match (old_tier, new_tier) {
                    (ChunkTier::Far, ChunkTier::Transition) | (ChunkTier::Far, ChunkTier::Near) => {
                        promoted.push(pos);
                    }
                    (ChunkTier::Transition, ChunkTier::Near) => promoted.push(pos),
                    (ChunkTier::Near, ChunkTier::Transition) => {
                        entered_transition.push(pos);
                    }
                    (ChunkTier::Near, ChunkTier::Far) | (ChunkTier::Transition, ChunkTier::Far) => {
                        demoted.push(pos);
                    }
                    _ => {}
                }
            }
        }

        // Request generation for missing columns (non-blocking, just enqueues)
        for cz in (player_cz - self.render_distance)..=(player_cz + self.render_distance) {
            for cx in (player_cx - self.render_distance)..=(player_cx + self.render_distance) {
                if !self.chunks.contains_key(&[cx, MIN_CHUNK_Y, cz]) && !self.generator.is_pending(cx, cz) {
                    self.generator.request(cx, cz);
                }
            }
        }

        // Receive completed columns from workers (non-blocking drain)
        for col in self.generator.receive() {
            if !in_range(col.cx, col.cz, player_cx, player_cz, self.render_distance) {
                continue;
            }
            let tier = classify_tier(col.cx, col.cz, player_cx, player_cz);
            for (i, chunk) in col.chunks.into_iter().enumerate() {
                let cy = MIN_CHUNK_Y + i as i32;
                let key = [col.cx, cy, col.cz];
                loaded.push(key);
                self.chunks.insert(key, chunk);
                self.tiers.insert(key, tier);
            }
        }

        WorldDelta {
            loaded,
            unloaded,
            promoted,
            demoted,
            entered_transition,
        }
    }

    /// Returns the block at a world-space position, or Air if the chunk isn't loaded.
    pub fn get_block(&self, wx: f32, wy: f32, wz: f32) -> BlockType {
        let (cx, cy, cz, lx, ly, lz) = world_to_chunk(wx, wy, wz);
        if !(MIN_CHUNK_Y..=MAX_CHUNK_Y).contains(&cy) {
            return BlockType::Air;
        }
        match self.chunks.get(&[cx, cy, cz]) {
            Some(chunk) => chunk.get(lx, ly, lz),
            None => BlockType::Air,
        }
    }

    /// Returns true if the block at a world-space position is solid.
    pub fn is_solid(&self, wx: f32, wy: f32, wz: f32) -> bool {
        self.get_block(wx, wy, wz).is_opaque()
    }

    /// Sets a block at integer world coordinates. Returns false if the chunk isn't loaded.
    pub fn set_block(&mut self, wx: i32, wy: i32, wz: i32, block: BlockType) -> bool {
        let size = CHUNK_SIZE as i32;
        let cx = wx.div_euclid(size);
        let cy = wy.div_euclid(size);
        let cz = wz.div_euclid(size);
        let lx = wx.rem_euclid(size) as usize;
        let ly = wy.rem_euclid(size) as usize;
        let lz = wz.rem_euclid(size) as usize;
        match self.chunks.get_mut(&[cx, cy, cz]) {
            Some(chunk) => {
                chunk.set(lx, ly, lz, block);
                true
            }
            None => false,
        }
    }

    /// Returns the chunk coordinates [cx, cy, cz] for a given world-space integer position.
    pub fn block_to_chunk(wx: i32, wy: i32, wz: i32) -> [i32; 3] {
        let size = CHUNK_SIZE as i32;
        [wx.div_euclid(size), wy.div_euclid(size), wz.div_euclid(size)]
    }

    pub fn get_chunk(&self, cx: i32, cy: i32, cz: i32) -> Option<&Chunk> {
        self.chunks.get(&[cx, cy, cz])
    }

    pub fn chunk_positions(&self) -> impl Iterator<Item = [i32; 3]> + '_ {
        self.chunks.keys().copied()
    }

    pub fn chunk_tier(&self, pos: &[i32; 3]) -> Option<ChunkTier> {
        self.tiers.get(pos).copied()
    }

    /// Inserts an empty chunk at the given position. Used for testing.
    #[cfg(test)]
    pub fn insert_empty_chunk(&mut self, cx: i32, cy: i32, cz: i32) {
        self.chunks.insert([cx, cy, cz], Chunk::new(BlockType::Air));
    }
}

/// Converts world-space float coordinates to (chunk_x, chunk_y, chunk_z, local_x, local_y, local_z).
fn world_to_chunk(wx: f32, wy: f32, wz: f32) -> (i32, i32, i32, usize, usize, usize) {
    let size = CHUNK_SIZE as f32;
    let cx = wx.div_euclid(size) as i32;
    let cy = wy.div_euclid(size) as i32;
    let cz = wz.div_euclid(size) as i32;
    let lx = wx.rem_euclid(size) as usize;
    let ly = wy.rem_euclid(size) as usize;
    let lz = wz.rem_euclid(size) as usize;
    (cx, cy, cz, lx, ly, lz)
}

fn in_range(cx: i32, cz: i32, player_cx: i32, player_cz: i32, radius: i32) -> bool {
    (cx - player_cx).abs() <= radius && (cz - player_cz).abs() <= radius
}

fn classify_tier(cx: i32, cz: i32, player_cx: i32, player_cz: i32) -> ChunkTier {
    let dist = (cx - player_cx).abs().max((cz - player_cz).abs());
    if dist <= NEAR_RADIUS - TRANSITION_WIDTH {
        ChunkTier::Near
    } else if dist <= NEAR_RADIUS + TRANSITION_WIDTH {
        ChunkTier::Transition
    } else {
        ChunkTier::Far
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_tier_near_at_origin() {
        // Near: dist <= NEAR_RADIUS - TRANSITION_WIDTH = 4
        assert_eq!(classify_tier(0, 0, 0, 0), ChunkTier::Near);
        assert_eq!(classify_tier(4, 0, 0, 0), ChunkTier::Near);
        assert_eq!(classify_tier(0, 4, 0, 0), ChunkTier::Near);
    }

    #[test]
    fn classify_tier_transition_zone() {
        // Transition: dist 5..=8
        assert_eq!(classify_tier(5, 0, 0, 0), ChunkTier::Transition);
        assert_eq!(classify_tier(8, 0, 0, 0), ChunkTier::Transition);
        assert_eq!(classify_tier(0, 6, 0, 0), ChunkTier::Transition);
    }

    #[test]
    fn classify_tier_far_beyond_radius() {
        // Far: dist > NEAR_RADIUS + TRANSITION_WIDTH = 8
        assert_eq!(classify_tier(9, 0, 0, 0), ChunkTier::Far);
        assert_eq!(classify_tier(0, 9, 0, 0), ChunkTier::Far);
        assert_eq!(classify_tier(10, 10, 0, 0), ChunkTier::Far);
    }

    #[test]
    fn classify_tier_with_offset_player() {
        assert_eq!(classify_tier(10, 10, 10, 10), ChunkTier::Near);
        assert_eq!(classify_tier(19, 10, 10, 10), ChunkTier::Far);
    }

    fn drain_world(world: &mut World, player_cx: i32, player_cz: i32) {
        let rd = world.render_distance;
        let expected_columns = ((2 * rd + 1) * (2 * rd + 1)) as usize;
        let expected_chunks = expected_columns * (MAX_CHUNK_Y - MIN_CHUNK_Y + 1) as usize;
        for _ in 0..500 {
            world.update(player_cx, player_cz);
            if world.chunks.len() >= expected_chunks {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("drain_world timed out");
    }

    #[test]
    fn world_delta_classifies_loaded_chunks() {
        let mut world = World::new(10);
        drain_world(&mut world, 0, 0);

        let mut near_count = 0;
        let mut transition_count = 0;
        let mut far_count = 0;
        for pos in world.chunk_positions() {
            match world.chunk_tier(&pos) {
                Some(ChunkTier::Near) => near_count += 1,
                Some(ChunkTier::Transition) => transition_count += 1,
                Some(ChunkTier::Far) => far_count += 1,
                None => panic!("loaded chunk has no tier"),
            }
        }
        assert!(near_count > 0, "should have near-field chunks");
        assert!(transition_count > 0, "should have transition chunks");
        assert!(far_count > 0, "should have far-field chunks");
    }

    #[test]
    fn world_delta_promotes_and_demotes_on_move() {
        let mut world = World::new(12);
        drain_world(&mut world, 0, 0);

        // Move player so tiers shift
        let delta = world.update(4, 0);
        let has_tier_changes = !delta.promoted.is_empty() || !delta.demoted.is_empty() || !delta.entered_transition.is_empty();
        assert!(has_tier_changes, "should have tier changes on move");
    }
}
