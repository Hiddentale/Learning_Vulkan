#![allow(dead_code)] // VoxelPool is wired up in Phase 1 step 7
use crate::graphical_core::buffers::allocate_buffer;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::voxel::chunk::{Chunk, CHUNK_SIZE};
use crate::voxel::world::World;
use std::collections::HashMap;
use vulkan_rust::{vk, Device, Instance};

const VOXEL_CHUNK_BYTES: usize = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE; // 4096
const BOUNDARY_FACES: usize = 6;
const BOUNDARY_FACE_BYTES: usize = CHUNK_SIZE * CHUNK_SIZE; // 256
const BOUNDARY_CHUNK_BYTES: usize = BOUNDARY_FACES * BOUNDARY_FACE_BYTES; // 1536

/// GPU-side chunk info for the task shader. Must match GLSL layout (std430).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct GpuMeshChunkInfo {
    pub aabb_min: [f32; 3],
    pub voxel_slot: u32,
    pub aabb_max: [f32; 3],
    pub boundary_slot: u32,
}

/// Manages GPU SSBOs for raw voxel data, boundary slices, and chunk info.
/// Uses slot-based allocation so chunks can be added/removed without rebuilding.
pub struct VoxelPool {
    // Voxel data SSBO
    pub voxel_buffer: vk::Buffer,
    voxel_memory: vk::DeviceMemory,
    voxel_ptr: *mut u8,

    // Boundary data SSBO
    pub boundary_buffer: vk::Buffer,
    boundary_memory: vk::DeviceMemory,
    boundary_ptr: *mut u8,

    // Chunk info SSBO
    pub chunk_info_buffer: vk::Buffer,
    chunk_info_memory: vk::DeviceMemory,
    chunk_info_ptr: *mut GpuMeshChunkInfo,

    // Visibility SSBO
    pub visibility_buffer: vk::Buffer,
    visibility_memory: vk::DeviceMemory,
    visibility_ptr: *mut u32,

    // Slot management
    free_slots: Vec<u32>,
    next_slot: u32,
    max_slots: u32,
    chunk_slots: HashMap<[i32; 3], u32>,

    // Chunk info is packed contiguously for GPU dispatch
    chunk_info_count: u32,
    slot_to_info_index: HashMap<u32, u32>,
    info_index_to_slot: Vec<u32>,
}

impl VoxelPool {
    pub unsafe fn new(max_slots: u32, device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<Self> {
        let host_visible = super::host_visible_coherent();
        let ssbo_flags = vk::BufferUsageFlags::STORAGE_BUFFER;

        let voxel_size = (max_slots as usize * VOXEL_CHUNK_BYTES) as u64;
        let (voxel_buffer, voxel_memory, voxel_ptr) = allocate_buffer::<u8>(voxel_size, ssbo_flags, device, instance, data, host_visible)?;

        let boundary_size = (max_slots as usize * BOUNDARY_CHUNK_BYTES) as u64;
        let (boundary_buffer, boundary_memory, boundary_ptr) =
            allocate_buffer::<u8>(boundary_size, ssbo_flags, device, instance, data, host_visible)?;

        let chunk_info_size = (max_slots as usize * std::mem::size_of::<GpuMeshChunkInfo>()) as u64;
        let (chunk_info_buffer, chunk_info_memory, chunk_info_ptr) =
            allocate_buffer::<GpuMeshChunkInfo>(chunk_info_size, ssbo_flags, device, instance, data, host_visible)?;

        let visibility_size = (max_slots as u64) * 4;
        let (visibility_buffer, visibility_memory, visibility_ptr) =
            allocate_buffer::<u32>(visibility_size, ssbo_flags, device, instance, data, host_visible)?;

        // Zero visibility buffer
        std::ptr::write_bytes(visibility_ptr, 0, max_slots as usize);

        Ok(Self {
            voxel_buffer,
            voxel_memory,
            voxel_ptr,
            boundary_buffer,
            boundary_memory,
            boundary_ptr,
            chunk_info_buffer,
            chunk_info_memory,
            chunk_info_ptr,
            visibility_buffer,
            visibility_memory,
            visibility_ptr,
            free_slots: Vec::new(),
            next_slot: 0,
            max_slots,
            chunk_slots: HashMap::new(),
            chunk_info_count: 0,
            slot_to_info_index: HashMap::new(),
            info_index_to_slot: Vec::new(),
        })
    }

    /// Uploads a chunk's voxel data and boundary slices to GPU.
    pub unsafe fn upload_chunk(&mut self, pos: [i32; 3], chunk: &Chunk, world: &World) {
        let slot = self.allocate_slot(pos);

        // Write voxel data
        let voxel_offset = slot as usize * VOXEL_CHUNK_BYTES;
        std::ptr::copy_nonoverlapping(chunk.as_bytes().as_ptr(), self.voxel_ptr.add(voxel_offset), VOXEL_CHUNK_BYTES);

        // Write boundary slices
        self.write_boundary(slot, pos, world);

        // Write chunk info
        let [cx, cy, cz] = pos;
        let cs = CHUNK_SIZE as f32;
        let info = GpuMeshChunkInfo {
            aabb_min: [cx as f32 * cs, cy as f32 * cs, cz as f32 * cs],
            voxel_slot: slot,
            aabb_max: [(cx + 1) as f32 * cs, (cy + 1) as f32 * cs, (cz + 1) as f32 * cs],
            boundary_slot: slot,
        };
        let info_index = self.chunk_info_count;
        std::ptr::write(self.chunk_info_ptr.add(info_index as usize), info);
        self.slot_to_info_index.insert(slot, info_index);
        self.info_index_to_slot.push(slot);
        self.chunk_info_count += 1;
    }

    /// Re-uploads voxel data for a chunk that is already in the pool.
    /// Used after in-place block edits. Does not allocate a new slot.
    pub unsafe fn reupload_chunk(&mut self, pos: [i32; 3], chunk: &Chunk, world: &World) {
        let slot = match self.chunk_slots.get(&pos) {
            Some(&s) => s,
            None => return,
        };
        let voxel_offset = slot as usize * VOXEL_CHUNK_BYTES;
        std::ptr::copy_nonoverlapping(chunk.as_bytes().as_ptr(), self.voxel_ptr.add(voxel_offset), VOXEL_CHUNK_BYTES);
        self.write_boundary(slot, pos, world);
    }

    /// Removes a chunk from the pool, returning its slot for reuse.
    pub unsafe fn remove_chunk(&mut self, pos: &[i32; 3]) {
        let slot = match self.chunk_slots.remove(pos) {
            Some(s) => s,
            None => return,
        };
        self.free_slots.push(slot);

        // Swap-remove from chunk info array
        if let Some(&info_index) = self.slot_to_info_index.get(&slot) {
            let last_index = self.chunk_info_count - 1;
            if info_index != last_index {
                // Copy last entry into the removed slot
                let last_info = std::ptr::read(self.chunk_info_ptr.add(last_index as usize));
                std::ptr::write(self.chunk_info_ptr.add(info_index as usize), last_info);

                // Update tracking for the moved entry
                let moved_slot = self.info_index_to_slot[last_index as usize];
                self.slot_to_info_index.insert(moved_slot, info_index);
                self.info_index_to_slot[info_index as usize] = moved_slot;
            }
            self.slot_to_info_index.remove(&slot);
            self.info_index_to_slot.pop();
            self.chunk_info_count -= 1;

            // Reset visibility for the swapped index
            std::ptr::write(self.visibility_ptr.add(info_index as usize), 0);
        }
    }

    /// Updates boundary data for a chunk's neighbors (call when a chunk is loaded/unloaded).
    pub unsafe fn invalidate_neighbor_boundaries(&mut self, pos: [i32; 3], world: &World) {
        let [cx, cy, cz] = pos;
        let neighbors = [
            [cx + 1, cy, cz],
            [cx - 1, cy, cz],
            [cx, cy + 1, cz],
            [cx, cy - 1, cz],
            [cx, cy, cz + 1],
            [cx, cy, cz - 1],
        ];
        for neighbor_pos in &neighbors {
            if let Some(&slot) = self.chunk_slots.get(neighbor_pos) {
                self.write_boundary(slot, *neighbor_pos, world);
            }
        }
    }

    pub fn chunk_count(&self) -> u32 {
        self.chunk_info_count
    }

    pub fn has_chunk(&self, pos: &[i32; 3]) -> bool {
        self.chunk_slots.contains_key(pos)
    }

    pub fn chunk_positions(&self) -> Vec<[i32; 3]> {
        self.chunk_slots.keys().copied().collect()
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        device.unmap_memory(self.voxel_memory);
        device.destroy_buffer(self.voxel_buffer, None);
        device.free_memory(self.voxel_memory, None);

        device.unmap_memory(self.boundary_memory);
        device.destroy_buffer(self.boundary_buffer, None);
        device.free_memory(self.boundary_memory, None);

        device.unmap_memory(self.chunk_info_memory);
        device.destroy_buffer(self.chunk_info_buffer, None);
        device.free_memory(self.chunk_info_memory, None);

        device.unmap_memory(self.visibility_memory);
        device.destroy_buffer(self.visibility_buffer, None);
        device.free_memory(self.visibility_memory, None);
    }

    fn allocate_slot(&mut self, pos: [i32; 3]) -> u32 {
        let slot = self.free_slots.pop().unwrap_or_else(|| {
            let s = self.next_slot;
            self.next_slot += 1;
            assert!(s < self.max_slots, "VoxelPool: exceeded max slot count");
            s
        });
        self.chunk_slots.insert(pos, slot);
        slot
    }

    unsafe fn write_boundary(&self, slot: u32, pos: [i32; 3], world: &World) {
        let [cx, cy, cz] = pos;
        let base = slot as usize * BOUNDARY_CHUNK_BYTES;

        // Face 0: +X neighbor's x=0 slice
        self.write_boundary_face(base, 0, world.get_chunk(cx + 1, cy, cz), |c, u, v| c.get(0, v, u));
        // Face 1: -X neighbor's x=15 slice
        self.write_boundary_face(base, 1, world.get_chunk(cx - 1, cy, cz), |c, u, v| c.get(CHUNK_SIZE - 1, v, u));
        // Face 2: +Y neighbor's y=0 slice
        self.write_boundary_face(base, 2, world.get_chunk(cx, cy + 1, cz), |c, u, v| c.get(u, 0, v));
        // Face 3: -Y neighbor's y=15 slice
        self.write_boundary_face(base, 3, world.get_chunk(cx, cy - 1, cz), |c, u, v| c.get(u, CHUNK_SIZE - 1, v));
        // Face 4: +Z neighbor's z=0 slice
        self.write_boundary_face(base, 4, world.get_chunk(cx, cy, cz + 1), |c, u, v| c.get(u, v, 0));
        // Face 5: -Z neighbor's z=15 slice
        self.write_boundary_face(base, 5, world.get_chunk(cx, cy, cz - 1), |c, u, v| c.get(u, v, CHUNK_SIZE - 1));
    }

    unsafe fn write_boundary_face(
        &self,
        base_offset: usize,
        face: usize,
        neighbor: Option<&Chunk>,
        read_block: impl Fn(&Chunk, usize, usize) -> crate::voxel::block::BlockType,
    ) {
        let offset = base_offset + face * BOUNDARY_FACE_BYTES;
        match neighbor {
            Some(chunk) => {
                for v in 0..CHUNK_SIZE {
                    for u in 0..CHUNK_SIZE {
                        let block = read_block(chunk, u, v);
                        *self.boundary_ptr.add(offset + u + v * CHUNK_SIZE) = block as u8;
                    }
                }
            }
            None => {
                // No neighbor loaded — fill with Air (0) so boundary faces are emitted
                std::ptr::write_bytes(self.boundary_ptr.add(offset), 0, BOUNDARY_FACE_BYTES);
            }
        }
    }
}
