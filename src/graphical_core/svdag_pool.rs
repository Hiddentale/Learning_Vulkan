#![allow(dead_code)] // SvdagPool is wired up in Phase 3

use crate::graphical_core::buffers::allocate_buffer;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::voxel::chunk::CHUNK_SIZE;
use std::collections::HashMap;
use vulkan_rust::{vk, Device, Instance};

/// Maximum SVDAG geometry budget in bytes (64 MB).
const SVDAG_GEOMETRY_BUDGET: u64 = 64 * 1024 * 1024;
/// Maximum SVDAG material budget in bytes (16 MB).
const SVDAG_MATERIAL_BUDGET: u64 = 16 * 1024 * 1024;

/// Per-chunk metadata for the ray marcher. Must match GLSL layout (std430).
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct GpuSvdagChunkInfo {
    pub aabb_min: [f32; 3],
    pub dag_offset: u32,
    pub aabb_max: [f32; 3],
    pub material_offset: u32,
    pub dag_size: u32,
    pub lod_level: u32,
    pub _pad: [u32; 2],
}

/// Allocation handle returned after uploading SVDAG data.
#[derive(Copy, Clone, Debug)]
pub struct SvdagHandle {
    pub dag_offset: u32,
    pub dag_size: u32,
    pub material_offset: u32,
    pub material_size: u32,
}

/// Manages GPU SSBOs for SVDAG geometry, materials, and chunk info.
/// Uses a bump allocator with free list for variable-size data.
pub struct SvdagPool {
    // Geometry SSBO
    pub geometry_buffer: vk::Buffer,
    geometry_memory: vk::DeviceMemory,
    geometry_ptr: *mut u8,
    geometry_next: u32,
    geometry_free: Vec<(u32, u32)>, // (offset, size)

    // Material SSBO
    pub material_buffer: vk::Buffer,
    material_memory: vk::DeviceMemory,
    material_ptr: *mut u8,
    material_next: u32,

    // Chunk info SSBO
    pub chunk_info_buffer: vk::Buffer,
    chunk_info_memory: vk::DeviceMemory,
    chunk_info_ptr: *mut GpuSvdagChunkInfo,

    // Slot tracking
    max_chunks: u32,
    chunk_slots: HashMap<[i32; 3], u32>, // pos → info_index
    chunk_handles: HashMap<[i32; 3], SvdagHandle>,
    chunk_count: u32,
}

impl SvdagPool {
    pub unsafe fn new(max_chunks: u32, device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<Self> {
        let host_visible = super::host_visible_coherent();
        let ssbo_flags = vk::BufferUsageFlags::STORAGE_BUFFER;

        let (geometry_buffer, geometry_memory, geometry_ptr) =
            allocate_buffer::<u8>(SVDAG_GEOMETRY_BUDGET, ssbo_flags, device, instance, data, host_visible)?;

        let (material_buffer, material_memory, material_ptr) =
            allocate_buffer::<u8>(SVDAG_MATERIAL_BUDGET, ssbo_flags, device, instance, data, host_visible)?;

        let chunk_info_size = max_chunks as u64 * std::mem::size_of::<GpuSvdagChunkInfo>() as u64;
        let (chunk_info_buffer, chunk_info_memory, chunk_info_ptr) =
            allocate_buffer::<GpuSvdagChunkInfo>(chunk_info_size, ssbo_flags, device, instance, data, host_visible)?;

        Ok(Self {
            geometry_buffer,
            geometry_memory,
            geometry_ptr,
            geometry_next: 0,
            geometry_free: Vec::new(),
            material_buffer,
            material_memory,
            material_ptr,
            material_next: 0,
            chunk_info_buffer,
            chunk_info_memory,
            chunk_info_ptr,
            max_chunks,
            chunk_slots: HashMap::new(),
            chunk_handles: HashMap::new(),
            chunk_count: 0,
        })
    }

    /// Upload compressed SVDAG data and materials for a chunk.
    pub unsafe fn upload_chunk(&mut self, pos: [i32; 3], dag_data: &[u8], material_data: &[u8], lod_level: u32) -> SvdagHandle {
        let dag_offset = self.alloc_geometry(dag_data.len() as u32);
        std::ptr::copy_nonoverlapping(dag_data.as_ptr(), self.geometry_ptr.add(dag_offset as usize), dag_data.len());

        let material_offset = self.alloc_material(material_data.len() as u32);
        std::ptr::copy_nonoverlapping(
            material_data.as_ptr(),
            self.material_ptr.add(material_offset as usize),
            material_data.len(),
        );

        let handle = SvdagHandle {
            dag_offset,
            dag_size: dag_data.len() as u32,
            material_offset,
            material_size: material_data.len() as u32,
        };

        let [cx, cy, cz] = pos;
        let cs = CHUNK_SIZE as f32;
        let info = GpuSvdagChunkInfo {
            aabb_min: [cx as f32 * cs, cy as f32 * cs, cz as f32 * cs],
            dag_offset,
            aabb_max: [(cx + 1) as f32 * cs, (cy + 1) as f32 * cs, (cz + 1) as f32 * cs],
            material_offset,
            dag_size: dag_data.len() as u32,
            lod_level,
            _pad: [0; 2],
        };

        let info_index = self.chunk_count;
        assert!(info_index < self.max_chunks, "SvdagPool: exceeded max chunk count");
        std::ptr::write(self.chunk_info_ptr.add(info_index as usize), info);
        self.chunk_slots.insert(pos, info_index);
        self.chunk_handles.insert(pos, handle);
        self.chunk_count += 1;

        handle
    }

    /// Remove a chunk's SVDAG data, freeing its geometry allocation.
    pub unsafe fn remove_chunk(&mut self, pos: &[i32; 3]) {
        let info_index = match self.chunk_slots.remove(pos) {
            Some(i) => i,
            None => return,
        };

        if let Some(handle) = self.chunk_handles.remove(pos) {
            self.geometry_free.push((handle.dag_offset, handle.dag_size));
        }

        // Swap-remove from chunk info array
        let last_index = self.chunk_count - 1;
        if info_index != last_index {
            let last_info = std::ptr::read(self.chunk_info_ptr.add(last_index as usize));
            std::ptr::write(self.chunk_info_ptr.add(info_index as usize), last_info);

            // Find and update the moved entry's slot mapping
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

    pub fn geometry_used(&self) -> u32 {
        self.geometry_next
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        device.unmap_memory(self.geometry_memory);
        device.destroy_buffer(self.geometry_buffer, None);
        device.free_memory(self.geometry_memory, None);

        device.unmap_memory(self.material_memory);
        device.destroy_buffer(self.material_buffer, None);
        device.free_memory(self.material_memory, None);

        device.unmap_memory(self.chunk_info_memory);
        device.destroy_buffer(self.chunk_info_buffer, None);
        device.free_memory(self.chunk_info_memory, None);
    }

    fn alloc_geometry(&mut self, size: u32) -> u32 {
        // Try free list first (first-fit)
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
        // Bump allocate
        let offset = self.geometry_next;
        self.geometry_next += size;
        assert!(
            (self.geometry_next as u64) <= SVDAG_GEOMETRY_BUDGET,
            "SvdagPool: geometry budget exceeded"
        );
        offset
    }

    fn alloc_material(&mut self, size: u32) -> u32 {
        let offset = self.material_next;
        self.material_next += size;
        assert!(
            (self.material_next as u64) <= SVDAG_MATERIAL_BUDGET,
            "SvdagPool: material budget exceeded"
        );
        offset
    }
}
