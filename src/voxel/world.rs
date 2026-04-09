use super::block::BlockType;
use super::chunk::{Chunk, CHUNK_SIZE};
use super::chunk_generator::ChunkGenerator;
use super::erosion::ErosionMap;
use super::metric::MetricField;
use super::sphere::{self, ChunkPos, PLANET_RADIUS_BLOCKS};
use glam::DVec3;
use std::collections::HashMap;
use std::sync::Arc;

pub const TERRAIN_MIN_CY: i32 = 0;
pub const TERRAIN_MAX_CY: i32 = 47; // 768 blocks tall (48 × 16)

/// Altitude (in blocks above the planet's nominal surface radius) above
/// which the mesh-shader streamer goes silent: the player is "in space",
/// the heightmap LOD covers the planet, and no mesh chunks are kept
/// resident. Below this threshold the streamer always loads the full
/// 2D column working set around the player's ground projection.
pub const ORBITAL_CUTOFF_BLOCKS: f64 = 3200.0;

pub struct World {
    chunks: HashMap<ChunkPos, Chunk>,
    render_distance: i32,
    generator: ChunkGenerator,
    pub metric: MetricField,
}

/// Result of a world update: which chunks were added/removed.
pub struct WorldDelta {
    pub loaded: Vec<ChunkPos>,
    pub unloaded: Vec<ChunkPos>,
}

impl World {
    pub fn new(render_distance: i32, seed: u32, erosion_map: Option<Arc<ErosionMap>>) -> Self {
        Self {
            chunks: HashMap::new(),
            render_distance,
            generator: ChunkGenerator::new(seed, erosion_map),
            metric: MetricField::new(),
        }
    }

    /// Non-blocking world update. Streams a 2D Chebyshev-bounded set of
    /// **columns** around the player's ground projection `(face, cx, cz)`.
    /// A column is in the working set iff `max(|cx-px|, |cz-pz|) <= rd`
    /// on the same face. The full vertical terrain band always loads for
    /// every in-set column — the streamer never reasons about `cy`.
    ///
    /// At altitude (`|world| - PLANET_RADIUS > ORBITAL_CUTOFF_BLOCKS`) the
    /// effective `rd` collapses to `-1`, the working set becomes empty, and
    /// the same eviction loop unloads everything. The heightmap LOD covers
    /// the planet from orbit. Crucially, the generator is still pumped on
    /// every frame: requests don't issue, but `receive()` continues to
    /// drain the channel and clear stale `pending` entries. There is no
    /// special-case early return.
    ///
    /// Working set bound: `(2·rd+1)² × (TERRAIN_MAX_CY-TERRAIN_MIN_CY+1)`
    /// chunks, independent of `FACE_SIDE_CHUNKS`. This is the property that
    /// lets the planet scale arbitrarily without overflowing the GPU mesh
    /// pool.
    ///
    pub fn update(&mut self, player_world: DVec3) -> WorldDelta {
        let mut loaded = Vec::new();
        let mut unloaded = Vec::new();

        let Some((player_pos, _, _, _)) = sphere::world_to_chunk_local(player_world) else {
            return WorldDelta { loaded, unloaded };
        };
        let player_face = player_pos.face;
        let player_cx = player_pos.cx;
        let player_cz = player_pos.cz;

        // Orbital cutoff: when the player is far above the surface, the
        // working set collapses to nothing. `rd = -1` makes `in_set` always
        // false for non-empty working sets, so the existing eviction loop
        // unloads everything via the same code path that handles walking.
        let altitude_blocks = player_world.length() - PLANET_RADIUS_BLOCKS as f64;
        let rd = if altitude_blocks > ORBITAL_CUTOFF_BLOCKS {
            -1
        } else {
            self.render_distance
        };

        // Build the set of valid (face, cx, cz) columns once. Iterate the
        // 2D Chebyshev square around the player; out-of-range columns are
        // remapped to the neighboring face via `cross_face_neighbor`.
        let mut target_columns: std::collections::HashSet<(sphere::Face, i32, i32)> = std::collections::HashSet::new();
        if rd >= 0 {
            for dz in -rd..=rd {
                for dx in -rd..=rd {
                    let cx = player_cx + dx;
                    let cz = player_cz + dz;
                    let col = if !(0..sphere::FACE_SIDE_CHUNKS).contains(&cx) || !(0..sphere::FACE_SIDE_CHUNKS).contains(&cz) {
                        let oob = ChunkPos {
                            face: player_face,
                            cx,
                            cy: 0,
                            cz,
                        };
                        sphere::cross_face_neighbor(oob).map(|p| (p.face, p.cx, p.cz))
                    } else {
                        Some((player_face, cx, cz))
                    };
                    if let Some(c) = col {
                        target_columns.insert(c);
                    }
                }
            }
        }

        let in_set = |pos: ChunkPos| -> bool { target_columns.contains(&(pos.face, pos.cx, pos.cz)) };

        // Evict anything outside the 2D column working set.
        let keys: Vec<ChunkPos> = self.chunks.keys().copied().collect();
        for pos in keys {
            if !in_set(pos) {
                self.chunks.remove(&pos);
                unloaded.push(pos);
            }
        }

        // Request every in-set column not yet loaded or pending.
        for &(face, cx, cz) in &target_columns {
            let any_loaded = (TERRAIN_MIN_CY..=TERRAIN_MAX_CY).any(|cy| self.chunks.contains_key(&ChunkPos { face, cx, cy, cz }));
            if !any_loaded && !self.generator.is_pending(face, cx, cz) {
                self.generator.request(face, cx, cz);
            }
        }

        // Always pump the generator. At orbit this drops stale results on
        // the floor (in_set rejects them) AND clears the pending set so the
        // next descent can re-request cleanly. This is the property the
        // unit test `round_trip_through_orbit_restores_resident_set` pins.
        for col in self.generator.receive() {
            for (i, chunk) in col.chunks.into_iter().enumerate() {
                let cy = TERRAIN_MIN_CY + i as i32;
                let key = ChunkPos {
                    face: col.face,
                    cx: col.cx,
                    cy,
                    cz: col.cz,
                };
                if !in_set(key) {
                    continue;
                }
                loaded.push(key);
                self.chunks.insert(key, chunk);
            }
        }

        WorldDelta { loaded, unloaded }
    }

    /// Phase D': direct cube-space block lookup. The caller already knows
    /// which chunk and integer block index to query — no projection inverse
    /// is involved. Out-of-range chunks are treated as solid below the
    /// terrain layer (so the player cannot fall through ungenerated space).
    pub fn block_solid(&self, cp: ChunkPos, lx: usize, ly: usize, lz: usize) -> bool {
        let lxi = lx.min(CHUNK_SIZE - 1);
        let lyi = ly.min(CHUNK_SIZE - 1);
        let lzi = lz.min(CHUNK_SIZE - 1);
        match self.chunks.get(&cp) {
            Some(chunk) => chunk.get(lxi, lyi, lzi).is_opaque(),
            None => (TERRAIN_MIN_CY..=TERRAIN_MAX_CY).contains(&cp.cy),
        }
    }

    /// Direct cube-space block read.
    pub fn block_at(&self, cp: ChunkPos, lx: usize, ly: usize, lz: usize) -> BlockType {
        match self.chunks.get(&cp) {
            Some(chunk) => chunk.get(lx.min(CHUNK_SIZE - 1), ly.min(CHUNK_SIZE - 1), lz.min(CHUNK_SIZE - 1)),
            None => BlockType::Air,
        }
    }

    /// Direct cube-space block write.
    pub fn set_block_at(&mut self, cp: ChunkPos, lx: usize, ly: usize, lz: usize, block: BlockType) -> bool {
        match self.chunks.get_mut(&cp) {
            Some(chunk) => {
                chunk.set(lx.min(CHUNK_SIZE - 1), ly.min(CHUNK_SIZE - 1), lz.min(CHUNK_SIZE - 1), block);
                true
            }
            None => false,
        }
    }

    pub fn get_chunk(&self, cx: i32, cy: i32, cz: i32) -> Option<&Chunk> {
        self.chunks.get(&ChunkPos::posy(cx, cy, cz))
    }

    pub fn get_chunk_at(&self, cp: ChunkPos) -> Option<&Chunk> {
        self.chunks.get(&cp)
    }

    pub fn chunk_positions(&self) -> impl Iterator<Item = ChunkPos> + '_ {
        self.chunks.keys().copied()
    }

    /// Test-only: insert an empty chunk at any face.
    #[cfg(test)]
    pub fn insert_empty_chunk_at(&mut self, cp: ChunkPos) {
        self.chunks.insert(cp, Chunk::new(BlockType::Air));
    }

    /// Test-only: count of currently loaded chunks.
    #[cfg(test)]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Test-only: direct mutable access to the chunk map for fixture setup.
    #[cfg(test)]
    pub fn chunks_mut_for_test(&mut self) -> &mut HashMap<ChunkPos, Chunk> {
        &mut self.chunks
    }
}

/// 3D Chebyshev distance from a chunk to the player position.
pub fn chunk_distance(cx: i32, cy: i32, cz: i32, px: i32, py: i32, pz: i32) -> i32 {
    (cx - px).abs().max((cy - py).abs()).max((cz - pz).abs())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::voxel::sphere::{self, ChunkPos};

    fn surface_world_at_face_center(face: sphere::Face) -> DVec3 {
        sphere::chunk_to_world(
            ChunkPos {
                face,
                cx: sphere::FACE_SIDE_CHUNKS / 2,
                cy: TERRAIN_MAX_CY / 2,
                cz: sphere::FACE_SIDE_CHUNKS / 2,
            },
            glam::Vec3::splat(8.0),
        )
    }

    fn drain_until_stable(world: &mut World, player_world: DVec3) -> usize {
        // Generation is async on worker threads; under heavy parallel test
        // load the per-poll stability window can race. We need (a) a count
        // > 0 before declaring stability, and (b) several consecutive
        // unchanged polls so an ongoing burst of completions doesn't return
        // a partial set.
        let mut last = 0usize;
        let mut stable = 0usize;
        for _ in 0..2000 {
            world.update(player_world);
            let now = world.chunk_count();
            if now == last && now > 0 {
                stable += 1;
            } else {
                stable = 0;
                last = now;
            }
            if stable >= 12 {
                return now;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        last
    }

    /// Working-set bound: at any planet scale, after streaming has settled,
    /// the resident chunk count is bounded by the 3D Chebyshev cube
    /// `(2·rd+1)² × min(2·rd+1, terrain_height_chunks)` — independent of
    /// `FACE_SIDE_CHUNKS`. This is the structural property that prevents
    /// pool overflow at large planet scales. Future scale-ups must keep
    /// this test passing without bumping `MAX_MESH_CHUNKS`.
    #[test]
    fn world_resident_set_is_bounded_by_render_distance() {
        let render_distance = 4;
        let mut world = World::new(render_distance, 0, None);
        let player_world = surface_world_at_face_center(sphere::Face::PosY);
        let count = drain_until_stable(&mut world, player_world);

        let terrain_h = (TERRAIN_MAX_CY - TERRAIN_MIN_CY + 1) as usize;
        let upper_bound = ((2 * render_distance + 1) as usize).pow(2) * terrain_h;

        assert!(count > 0, "resident set is empty after pregen");
        assert!(
            count <= upper_bound,
            "resident set {} exceeds working-set bound {} (render_distance={}, FACE_SIDE_CHUNKS={})",
            count,
            upper_bound,
            render_distance,
            sphere::FACE_SIDE_CHUNKS,
        );
    }

    /// After a player moves far enough on a face, chunks that fell out of
    /// the working set are evicted (not leaked). Catches the regression
    /// where `update` fails to remove out-of-range chunks.
    #[test]
    fn chunks_unload_when_player_moves_out_of_range() {
        let render_distance = 3;
        let mut world = World::new(render_distance, 0, None);

        let center = surface_world_at_face_center(sphere::Face::PosY);
        drain_until_stable(&mut world, center);
        let initial = world.chunk_count();
        assert!(initial > 0);

        // Move the player far enough on the +Y face that none of the
        // initial chunks remain in the working set.
        let shift_chunks = 2 * render_distance + 2;
        let cx = sphere::FACE_SIDE_CHUNKS / 2 + shift_chunks;
        let cz = sphere::FACE_SIDE_CHUNKS / 2;
        let moved = sphere::chunk_to_world(
            ChunkPos {
                face: sphere::Face::PosY,
                cx,
                cy: TERRAIN_MAX_CY / 2,
                cz,
            },
            glam::Vec3::splat(8.0),
        );
        let delta = world.update(moved);
        assert!(
            !delta.unloaded.is_empty(),
            "no chunks were unloaded after moving {} chunks across the face",
            shift_chunks,
        );
    }

    /// `block_solid` for an unloaded chunk falls back to "solid" inside the
    /// terrain layer band and "air" above it. Catches regressions where the
    /// fall-through guard breaks (player would fall to core through
    /// ungenerated terrain).
    #[test]
    fn unloaded_terrain_chunks_are_solid() {
        let world = World::new(2, 0, None);
        let in_band = ChunkPos {
            face: sphere::Face::PosY,
            cx: 0,
            cy: 5,
            cz: 0,
        };
        assert!(world.block_solid(in_band, 0, 0, 0));
        let above_band = ChunkPos {
            face: sphere::Face::PosY,
            cx: 0,
            cy: TERRAIN_MAX_CY + 5,
            cz: 0,
        };
        assert!(!world.block_solid(above_band, 0, 0, 0));
    }

    /// Round-trip through orbit must restore the resident set bit-for-bit.
    /// Catches the class of bug where the orbital eviction path leaves
    /// generator state (channel, pending set) inconsistent so the streamer
    /// can't recover when the player descends.
    #[test]
    fn round_trip_through_orbit_restores_resident_set() {
        let render_distance = 3;
        let mut world = World::new(render_distance, 0, None);
        let surface = surface_world_at_face_center(sphere::Face::PosY);
        let baseline = drain_until_stable(&mut world, surface);
        assert!(baseline > 0);

        // Fly to orbit (well above any plausible cutoff). Pump for many
        // frames so workers complete in-flight columns while we're "away".
        let orbit = surface * 100.0;
        for _ in 0..200 {
            world.update(orbit);
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        assert_eq!(world.chunk_count(), 0, "orbit should evict everything");

        // Descend. Resident set must come back to the same count.
        let after = drain_until_stable(&mut world, surface);
        assert_eq!(after, baseline, "resident set should be restored after orbital round-trip");
    }

    /// Block writes are visible to subsequent reads.
    #[test]
    fn set_then_read_block() {
        let mut world = World::new(2, 0, None);
        let cp = ChunkPos {
            face: sphere::Face::PosY,
            cx: 0,
            cy: 5,
            cz: 0,
        };
        world.insert_empty_chunk_at(cp);
        assert!(!world.block_solid(cp, 1, 1, 1));
        world.set_block_at(cp, 1, 1, 1, BlockType::Stone);
        assert!(world.block_solid(cp, 1, 1, 1));
        assert_eq!(world.block_at(cp, 1, 1, 1), BlockType::Stone);
    }
}

// Disabled stale tests retained below for reference.
#[cfg(all(test, any()))]
mod stale_tests {
    use super::*;

    const TEST_PLAYER_CY: i32 = 5;
    const TERRAIN_LAYERS: usize = (TERRAIN_MAX_CY - TERRAIN_MIN_CY + 1) as usize;

    fn drain_world(world: &mut World, player_cx: i32, player_cz: i32) {
        let rd = world.render_distance;
        let expected_columns = ((2 * rd + 1) * (2 * rd + 1)) as usize;
        let expected_chunks = expected_columns * TERRAIN_LAYERS;
        for _ in 0..500 {
            world.update(player_cx, TEST_PLAYER_CY, player_cz);
            if world.chunks.len() >= expected_chunks {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("drain_world timed out: have {}, expected {}", world.chunks.len(), expected_chunks);
    }

    #[test]
    fn chunks_load_within_render_distance() {
        let mut world = World::new(4, 42, None);
        drain_world(&mut world, 0, 0);
        let rd = 4;
        let expected = ((2 * rd + 1) * (2 * rd + 1)) as usize * TERRAIN_LAYERS;
        assert_eq!(world.chunks.len(), expected);
    }

    #[test]
    fn chunks_unload_on_move() {
        let mut world = World::new(4, 42, None);
        drain_world(&mut world, 0, 0);
        let delta = world.update(10, 5, 0);
        assert!(!delta.unloaded.is_empty());
    }

    #[test]
    fn chunk_distance_3d_chebyshev() {
        assert_eq!(chunk_distance(0, 0, 0, 0, 0, 0), 0);
        assert_eq!(chunk_distance(5, 3, 2, 0, 0, 0), 5);
        assert_eq!(chunk_distance(3, 10, 7, 0, 0, 0), 10);
        // Y dominates when player is high above terrain
        assert_eq!(chunk_distance(1, 2, 1, 1, 20, 1), 18);
        // XZ only (same as old 2D when Y is equal)
        assert_eq!(chunk_distance(5, 0, 3, 0, 0, 0), 5);
        assert_eq!(chunk_distance(3, 0, 7, 0, 0, 0), 7);
    }
}
