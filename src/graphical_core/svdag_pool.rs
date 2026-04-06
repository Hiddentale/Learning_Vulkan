#![allow(dead_code)]

use crate::graphical_core::buffers::allocate_buffer;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::voxel::chunk::CHUNK_SIZE;
use std::collections::HashMap;
use vulkan_rust::{vk, Device, Instance};

/// Maximum SVDAG geometry budget in bytes (128 MB).
/// Materials are embedded in leaf nodes, so this covers both geometry and materials.
const SVDAG_GEOMETRY_BUDGET: u64 = 128 * 1024 * 1024;

/// Per-chunk metadata for the ray marcher. Must match GLSL layout (std430).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct GpuSvdagChunkInfo {
    pub aabb_min: [f32; 3],
    pub dag_offset: u32,
    pub aabb_max: [f32; 3],
    pub dag_size: u32,
}

/// Allocation handle for freeing geometry on removal.
#[derive(Copy, Clone, Debug)]
struct SvdagHandle {
    dag_offset: u32,
    dag_size: u32,
}

/// Manages GPU SSBOs for SVDAG geometry and chunk info.
/// Materials are embedded in SVDAG leaf nodes — no separate material buffer.
pub struct SvdagPool {
    // Geometry SSBO (contains SVDAG nodes with embedded materials in leaves)
    pub geometry_buffer: vk::Buffer,
    geometry_memory: vk::DeviceMemory,
    geometry_ptr: *mut u8,
    geometry_next: u32,
    geometry_free: Vec<(u32, u32)>,

    // Chunk info SSBO
    pub chunk_info_buffer: vk::Buffer,
    chunk_info_memory: vk::DeviceMemory,
    chunk_info_ptr: *mut GpuSvdagChunkInfo,

    // Slot tracking
    max_chunks: u32,
    chunk_slots: HashMap<[i32; 3], u32>,
    chunk_handles: HashMap<[i32; 3], SvdagHandle>,
    chunk_count: u32,
}

impl SvdagPool {
    pub unsafe fn new(max_chunks: u32, device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<Self> {
        let host_visible = super::host_visible_coherent();
        let ssbo_flags = vk::BufferUsageFlags::STORAGE_BUFFER;

        let (geometry_buffer, geometry_memory, geometry_ptr) =
            allocate_buffer::<u8>(SVDAG_GEOMETRY_BUDGET, ssbo_flags, device, instance, data, host_visible)?;

        let chunk_info_size = max_chunks as u64 * std::mem::size_of::<GpuSvdagChunkInfo>() as u64;
        let (chunk_info_buffer, chunk_info_memory, chunk_info_ptr) =
            allocate_buffer::<GpuSvdagChunkInfo>(chunk_info_size, ssbo_flags, device, instance, data, host_visible)?;

        Ok(Self {
            geometry_buffer,
            geometry_memory,
            geometry_ptr,
            geometry_next: 0,
            geometry_free: Vec::new(),
            chunk_info_buffer,
            chunk_info_memory,
            chunk_info_ptr,
            max_chunks,
            chunk_slots: HashMap::new(),
            chunk_handles: HashMap::new(),
            chunk_count: 0,
        })
    }

    /// Upload compressed SVDAG data for a chunk. Materials are embedded in leaf nodes.
    /// `dag_size` = voxel grid dimension (always 64 for super-chunks).
    pub unsafe fn upload_chunk(&mut self, pos: [i32; 3], dag_data: &[u8], lod_level: u32, dag_size: u32) {
        let dag_offset = self.alloc_geometry(dag_data.len() as u32);
        std::ptr::copy_nonoverlapping(dag_data.as_ptr(), self.geometry_ptr.add(dag_offset as usize), dag_data.len());

        let handle = SvdagHandle {
            dag_offset,
            dag_size: dag_data.len() as u32,
        };

        let [cx, cy, cz] = pos;
        let cs = CHUNK_SIZE as f32;
        let span = 1 << lod_level; // LOD-0=1, LOD-1=2, LOD-2=4 (super-chunk), LOD-3=8, etc.
        let info = GpuSvdagChunkInfo {
            aabb_min: [cx as f32 * cs, cy as f32 * cs, cz as f32 * cs],
            dag_offset,
            aabb_max: [(cx + span) as f32 * cs, (cy + span) as f32 * cs, (cz + span) as f32 * cs],
            dag_size,
        };

        let info_index = self.chunk_count;
        assert!(info_index < self.max_chunks, "SvdagPool: exceeded max chunk count");
        std::ptr::write(self.chunk_info_ptr.add(info_index as usize), info);
        self.chunk_slots.insert(pos, info_index);
        self.chunk_handles.insert(pos, handle);
        self.chunk_count += 1;
    }

    pub unsafe fn remove_chunk(&mut self, pos: &[i32; 3]) {
        let info_index = match self.chunk_slots.remove(pos) {
            Some(i) => i,
            None => return,
        };
        if let Some(handle) = self.chunk_handles.remove(pos) {
            self.geometry_free.push((handle.dag_offset, handle.dag_size));
        }
        let last_index = self.chunk_count - 1;
        if info_index != last_index {
            let last_info = std::ptr::read(self.chunk_info_ptr.add(last_index as usize));
            std::ptr::write(self.chunk_info_ptr.add(info_index as usize), last_info);
            if let Some((&moved_pos, _)) = self.chunk_slots.iter().find(|(_, &v)| v == last_index) {
                self.chunk_slots.insert(moved_pos, info_index);
            }
        }
        self.chunk_count -= 1;
    }

    pub fn chunk_count(&self) -> u32 {
        self.chunk_count
    }

    pub fn has_chunk(&self, pos: &[i32; 3]) -> bool {
        self.chunk_slots.contains_key(pos)
    }

    pub fn is_near_budget(&self) -> bool {
        (self.geometry_next as u64) > SVDAG_GEOMETRY_BUDGET * 9 / 10
    }

    /// Evict the farthest chunks from the player to free space.
    pub unsafe fn evict_farthest(&mut self, count: usize, player_cx: i32, player_cy: i32, player_cz: i32) {
        let mut positions: Vec<[i32; 3]> = self.chunk_slots.keys().copied().collect();
        positions.sort_by_key(|pos| std::cmp::Reverse((pos[0] - player_cx).abs().max((pos[1] - player_cy).abs()).max((pos[2] - player_cz).abs())));
        for pos in positions.into_iter().take(count) {
            self.remove_chunk(&pos);
        }
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        device.unmap_memory(self.geometry_memory);
        device.destroy_buffer(self.geometry_buffer, None);
        device.free_memory(self.geometry_memory, None);

        device.unmap_memory(self.chunk_info_memory);
        device.destroy_buffer(self.chunk_info_buffer, None);
        device.free_memory(self.chunk_info_memory, None);
    }

    fn alloc_geometry(&mut self, size: u32) -> u32 {
        for i in 0..self.geometry_free.len() {
            let (offset, free_size) = self.geometry_free[i];
            if free_size >= size {
                if free_size == size {
                    self.geometry_free.swap_remove(i);
                } else {
                    self.geometry_free[i] = (offset + size, free_size - size);
                }
                return offset;
            }
        }
        let offset = self.geometry_next;
        self.geometry_next += size;
        assert!(
            (self.geometry_next as u64) <= SVDAG_GEOMETRY_BUDGET,
            "SvdagPool: geometry budget exceeded"
        );
        offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CPU-only simulation of SvdagPool's swap-remove pattern.
    /// Tests the invariant: after any sequence of uploads and removals,
    /// every remaining chunk has a unique slot in [0..count) and its
    /// AABB data matches its position.
    struct MockPool {
        infos: Vec<GpuSvdagChunkInfo>,
        slots: HashMap<[i32; 3], u32>,
        count: u32,
    }

    impl MockPool {
        fn new() -> Self {
            Self {
                infos: vec![GpuSvdagChunkInfo::default(); 64],
                slots: HashMap::new(),
                count: 0,
            }
        }

        fn upload(&mut self, pos: [i32; 3]) {
            let cs = CHUNK_SIZE as f32;
            let info = GpuSvdagChunkInfo {
                aabb_min: [pos[0] as f32 * cs, pos[1] as f32 * cs, pos[2] as f32 * cs],
                dag_offset: 0,
                aabb_max: [(pos[0] + 1) as f32 * cs, (pos[1] + 1) as f32 * cs, (pos[2] + 1) as f32 * cs],
                dag_size: 0,
            };
            let idx = self.count;
            self.infos[idx as usize] = info;
            self.slots.insert(pos, idx);
            self.count += 1;
        }

        fn remove(&mut self, pos: &[i32; 3]) {
            let info_index = match self.slots.remove(pos) {
                Some(i) => i,
                None => return,
            };
            let last_index = self.count - 1;
            if info_index != last_index {
                self.infos[info_index as usize] = self.infos[last_index as usize];
                if let Some((&moved_pos, _)) = self.slots.iter().find(|(_, &v)| v == last_index) {
                    self.slots.insert(moved_pos, info_index);
                }
            }
            self.count -= 1;
        }

        /// Verify all invariants: slots are valid, AABBs match positions.
        fn assert_consistent(&self) {
            assert_eq!(self.slots.len() as u32, self.count);
            let mut used_slots: Vec<u32> = self.slots.values().copied().collect();
            used_slots.sort();
            used_slots.dedup();
            assert_eq!(used_slots.len() as u32, self.count, "duplicate slot indices");

            for (&pos, &slot) in &self.slots {
                assert!(slot < self.count, "slot {slot} >= count {}", self.count);
                let info = &self.infos[slot as usize];
                let cs = CHUNK_SIZE as f32;
                let expected_min = [pos[0] as f32 * cs, pos[1] as f32 * cs, pos[2] as f32 * cs];
                assert_eq!(
                    info.aabb_min, expected_min,
                    "AABB mismatch at slot {slot} for pos {pos:?}: \
                     got {:?}, expected {expected_min:?}",
                    info.aabb_min
                );
            }
        }
    }

    #[test]
    fn swap_remove_preserves_all_remaining() {
        let mut pool = MockPool::new();
        pool.upload([0, 0, 0]);
        pool.upload([1, 0, 0]);
        pool.upload([2, 0, 0]);
        pool.upload([3, 0, 0]);

        // Remove middle chunk — last chunk should swap into its slot
        pool.remove(&[1, 0, 0]);
        pool.assert_consistent();
        assert_eq!(pool.count, 3);
        assert!(!pool.slots.contains_key(&[1, 0, 0]));
    }

    #[test]
    fn swap_remove_last_element() {
        let mut pool = MockPool::new();
        pool.upload([0, 0, 0]);
        pool.upload([1, 0, 0]);
        pool.upload([2, 0, 0]);

        // Remove last — no swap needed
        pool.remove(&[2, 0, 0]);
        pool.assert_consistent();
        assert_eq!(pool.count, 2);
    }

    #[test]
    fn swap_remove_first_element() {
        let mut pool = MockPool::new();
        pool.upload([0, 0, 0]);
        pool.upload([1, 0, 0]);
        pool.upload([2, 0, 0]);

        pool.remove(&[0, 0, 0]);
        pool.assert_consistent();
        assert_eq!(pool.count, 2);
    }

    #[test]
    fn interleaved_upload_remove_stays_consistent() {
        let mut pool = MockPool::new();
        // Simulate streaming: upload, remove farthest, upload more
        for i in 0..10 {
            pool.upload([i, 0, 0]);
        }
        pool.assert_consistent();

        // Evict "farthest" (highest index positions)
        pool.remove(&[9, 0, 0]);
        pool.remove(&[8, 0, 0]);
        pool.remove(&[7, 0, 0]);
        pool.assert_consistent();

        // Upload new chunks into the freed space
        pool.upload([10, 0, 0]);
        pool.upload([11, 0, 0]);
        pool.assert_consistent();
        assert_eq!(pool.count, 9);

        // Remove from the middle
        pool.remove(&[3, 0, 0]);
        pool.remove(&[5, 0, 0]);
        pool.assert_consistent();
        assert_eq!(pool.count, 7);
    }

    #[test]
    fn remove_nonexistent_is_noop() {
        let mut pool = MockPool::new();
        pool.upload([0, 0, 0]);
        pool.remove(&[99, 99, 99]); // doesn't exist
        pool.assert_consistent();
        assert_eq!(pool.count, 1);
    }
}
