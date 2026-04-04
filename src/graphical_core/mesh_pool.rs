use crate::graphical_core::buffers::allocate_and_fill_buffer;
use crate::graphical_core::mesh::Vertex;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::voxel::meshing::BUCKET_COUNT;
use std::collections::HashMap;
use vk::Handle;
use vulkan_rust::{vk, Device, Instance};

/// Index range for one face-direction bucket within the shared index buffer.
#[derive(Copy, Clone, Debug, Default)]
pub struct FaceBucket {
    pub first_index: u32,
    pub index_count: u32,
}

/// Draw parameters for a single chunk within the shared buffer.
#[derive(Copy, Clone, Debug)]
pub struct ChunkDrawParams {
    pub buckets: [FaceBucket; BUCKET_COUNT],
    pub vertex_offset: i32,
    /// Index into the transform SSBO (used as firstInstance in indirect draws).
    pub transform_index: u32,
}

/// CPU-side cached mesh data for one chunk.
struct ChunkMeshData {
    vertices: Vec<Vertex>,
    bucket_indices: [Vec<u32>; BUCKET_COUNT],
}

/// A single shared VBO + IBO containing all chunk meshes.
/// CPU-side mesh data is cached per chunk so the GPU buffers can be rebuilt
/// when chunks are added or removed.
pub struct MeshPool {
    pub vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    index_memory: vk::DeviceMemory,
    chunk_data: HashMap<[i32; 3], ChunkMeshData>,
    draw_params: HashMap<[i32; 3], ChunkDrawParams>,
    has_gpu_buffers: bool,
}

impl MeshPool {
    pub fn new() -> Self {
        Self {
            vertex_buffer: vk::Buffer::null(),
            vertex_memory: vk::DeviceMemory::null(),
            index_buffer: vk::Buffer::null(),
            index_memory: vk::DeviceMemory::null(),
            chunk_data: HashMap::new(),
            draw_params: HashMap::new(),
            has_gpu_buffers: false,
        }
    }

    /// Caches mesh data for a chunk. Call `rebuild` afterwards to upload to GPU.
    pub fn add_chunk(&mut self, pos: [i32; 3], vertices: Vec<Vertex>, bucket_indices: [Vec<u32>; BUCKET_COUNT]) {
        self.chunk_data.insert(pos, ChunkMeshData { vertices, bucket_indices });
    }

    /// Removes cached mesh data for a chunk. Call `rebuild` afterwards to update GPU.
    pub fn remove_chunk(&mut self, pos: &[i32; 3]) {
        self.chunk_data.remove(pos);
        self.draw_params.remove(pos);
    }

    pub fn draw_params(&self, pos: &[i32; 3]) -> Option<&ChunkDrawParams> {
        self.draw_params.get(pos)
    }

    /// Returns all chunk positions that have mesh data.
    pub fn chunk_positions(&self) -> impl Iterator<Item = &[i32; 3]> {
        self.chunk_data.keys()
    }

    /// Concatenates all cached chunk meshes into one VBO + IBO and uploads to GPU.
    /// Destroys any previous GPU buffers first.
    pub unsafe fn rebuild(&mut self, device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<()> {
        self.destroy_gpu_buffers(device);
        self.draw_params.clear();

        if self.chunk_data.is_empty() {
            return Ok(());
        }

        let mut all_vertices: Vec<Vertex> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();

        for (transform_index, (pos, mesh)) in (0u32..).zip(self.chunk_data.iter()) {
            let vertex_offset = all_vertices.len() as i32;
            all_vertices.extend_from_slice(&mesh.vertices);

            let mut buckets = [FaceBucket::default(); BUCKET_COUNT];
            for (i, bucket_idx) in mesh.bucket_indices.iter().enumerate() {
                buckets[i] = FaceBucket {
                    first_index: all_indices.len() as u32,
                    index_count: bucket_idx.len() as u32,
                };
                all_indices.extend_from_slice(bucket_idx);
            }

            self.draw_params.insert(
                *pos,
                ChunkDrawParams {
                    buckets,
                    vertex_offset,
                    transform_index,
                },
            );
        }

        let host_visible = super::host_visible_coherent();

        let vertex_size = std::mem::size_of_val(all_vertices.as_slice()) as u64;
        let (vb, vm) = allocate_and_fill_buffer(
            &all_vertices,
            vertex_size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            device,
            instance,
            data,
            host_visible,
        )?;
        self.vertex_buffer = vb;
        self.vertex_memory = vm;

        let index_size = std::mem::size_of_val(all_indices.as_slice()) as u64;
        let (ib, im) = allocate_and_fill_buffer(
            &all_indices,
            index_size,
            vk::BufferUsageFlags::INDEX_BUFFER,
            device,
            instance,
            data,
            host_visible,
        )?;
        self.index_buffer = ib;
        self.index_memory = im;

        self.has_gpu_buffers = true;
        Ok(())
    }

    unsafe fn destroy_gpu_buffers(&mut self, device: &Device) {
        if !self.has_gpu_buffers {
            return;
        }
        device.destroy_buffer(self.vertex_buffer, None);
        device.free_memory(self.vertex_memory, None);
        device.destroy_buffer(self.index_buffer, None);
        device.free_memory(self.index_memory, None);
        self.has_gpu_buffers = false;
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        self.destroy_gpu_buffers(device);
        self.chunk_data.clear();
        self.draw_params.clear();
    }
}
