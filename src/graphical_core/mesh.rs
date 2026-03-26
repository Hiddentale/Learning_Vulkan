use crate::graphical_core::buffers::allocate_and_fill_buffer;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use vulkanalia::vk::{self, DeviceV1_0};
use vulkanalia::{Device, Instance};

/// `#[repr(C)]` ensures a predictable memory layout matching Vulkan's expectations.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv_coordinate: [f32; 2],
    pub color: [f32; 3],
}

/// A GPU-uploaded mesh: vertex and index buffers with their backing memory.
#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertex_buffer: vk::Buffer,
    pub vertex_buffer_memory: vk::DeviceMemory,
    pub index_buffer: vk::Buffer,
    pub index_buffer_memory: vk::DeviceMemory,
    pub index_count: u32,
}

/// Uploads vertex and index data to GPU buffers and returns a `Mesh` handle.
pub unsafe fn create_mesh(
    vertices: &[Vertex],
    indices: &[u32],
    device: &Device,
    instance: &Instance,
    data: &mut VulkanApplicationData,
) -> anyhow::Result<Mesh> {
    let host_visible = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

    let vertex_size = std::mem::size_of_val(vertices) as u64;
    let (vertex_buffer, vertex_buffer_memory) = allocate_and_fill_buffer(
        vertices,
        vertex_size,
        vk::BufferUsageFlags::VERTEX_BUFFER,
        device,
        instance,
        data,
        host_visible,
    )?;

    let index_size = std::mem::size_of_val(indices) as u64;
    let (index_buffer, index_buffer_memory) = allocate_and_fill_buffer(
        indices,
        index_size,
        vk::BufferUsageFlags::INDEX_BUFFER,
        device,
        instance,
        data,
        host_visible,
    )?;

    Ok(Mesh {
        vertex_buffer,
        vertex_buffer_memory,
        index_buffer,
        index_buffer_memory,
        index_count: indices.len() as u32,
    })
}

/// Destroys a mesh's GPU buffers and frees their memory.
pub unsafe fn destroy_mesh(device: &Device, mesh: &Mesh) {
    device.destroy_buffer(mesh.vertex_buffer, None);
    device.free_memory(mesh.vertex_buffer_memory, None);
    device.destroy_buffer(mesh.index_buffer, None);
    device.free_memory(mesh.index_buffer_memory, None);
}
