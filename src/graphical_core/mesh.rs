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

    let vertex_size = (vertices.len() * std::mem::size_of::<Vertex>()) as u64;
    let (vertex_buffer, vertex_buffer_memory) =
        allocate_and_fill_buffer(vertices, vertex_size, vk::BufferUsageFlags::VERTEX_BUFFER, device, instance, data, host_visible)?;

    let index_size = (indices.len() * std::mem::size_of::<u32>()) as u64;
    let (index_buffer, index_buffer_memory) =
        allocate_and_fill_buffer(indices, index_size, vk::BufferUsageFlags::INDEX_BUFFER, device, instance, data, host_visible)?;

    Ok(Mesh {
        vertex_buffer,
        vertex_buffer_memory,
        index_buffer,
        index_buffer_memory,
        index_count: indices.len() as u32,
    })
}

/// Creates a unit cube mesh on the GPU.
pub unsafe fn create_cube(device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<Mesh> {
    create_mesh(&CUBE_VERTICES, &CUBE_INDICES, device, instance, data)
}

/// Destroys a mesh's GPU buffers and frees their memory.
pub unsafe fn destroy_mesh(device: &Device, mesh: &Mesh) {
    device.destroy_buffer(mesh.vertex_buffer, None);
    device.free_memory(mesh.vertex_buffer_memory, None);
    device.destroy_buffer(mesh.index_buffer, None);
    device.free_memory(mesh.index_buffer_memory, None);
}

pub const CUBE_VERTICES: [Vertex; 24] = [
    // Front face (z = -0.5)
    Vertex {
        position: [-0.5, -0.5, -0.5],
        uv_coordinate: [0.0, 1.0],
    },
    Vertex {
        position: [0.5, -0.5, -0.5],
        uv_coordinate: [1.0, 1.0],
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
        uv_coordinate: [1.0, 0.0],
    },
    Vertex {
        position: [-0.5, 0.5, -0.5],
        uv_coordinate: [0.0, 0.0],
    },
    // Back face (z = 0.5)
    Vertex {
        position: [0.5, -0.5, 0.5],
        uv_coordinate: [0.0, 1.0],
    },
    Vertex {
        position: [-0.5, -0.5, 0.5],
        uv_coordinate: [1.0, 1.0],
    },
    Vertex {
        position: [-0.5, 0.5, 0.5],
        uv_coordinate: [1.0, 0.0],
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        uv_coordinate: [0.0, 0.0],
    },
    // Right face (x = 0.5)
    Vertex {
        position: [0.5, -0.5, -0.5],
        uv_coordinate: [0.0, 1.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.5],
        uv_coordinate: [1.0, 1.0],
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        uv_coordinate: [1.0, 0.0],
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
        uv_coordinate: [0.0, 0.0],
    },
    // Left face (x = -0.5)
    Vertex {
        position: [-0.5, -0.5, 0.5],
        uv_coordinate: [0.0, 1.0],
    },
    Vertex {
        position: [-0.5, -0.5, -0.5],
        uv_coordinate: [1.0, 1.0],
    },
    Vertex {
        position: [-0.5, 0.5, -0.5],
        uv_coordinate: [1.0, 0.0],
    },
    Vertex {
        position: [-0.5, 0.5, 0.5],
        uv_coordinate: [0.0, 0.0],
    },
    // Top face (y = 0.5)
    Vertex {
        position: [-0.5, 0.5, -0.5],
        uv_coordinate: [0.0, 1.0],
    },
    Vertex {
        position: [0.5, 0.5, -0.5],
        uv_coordinate: [1.0, 1.0],
    },
    Vertex {
        position: [0.5, 0.5, 0.5],
        uv_coordinate: [1.0, 0.0],
    },
    Vertex {
        position: [-0.5, 0.5, 0.5],
        uv_coordinate: [0.0, 0.0],
    },
    // Bottom face (y = -0.5)
    Vertex {
        position: [-0.5, -0.5, 0.5],
        uv_coordinate: [0.0, 1.0],
    },
    Vertex {
        position: [0.5, -0.5, 0.5],
        uv_coordinate: [1.0, 1.0],
    },
    Vertex {
        position: [0.5, -0.5, -0.5],
        uv_coordinate: [1.0, 0.0],
    },
    Vertex {
        position: [-0.5, -0.5, -0.5],
        uv_coordinate: [0.0, 0.0],
    },
];

pub const CUBE_INDICES: [u32; 36] = [
    0, 1, 2, 0, 2, 3, // Front
    4, 5, 6, 4, 6, 7, // Back
    8, 9, 10, 8, 10, 11, // Right
    12, 13, 14, 12, 14, 15, // Left
    16, 17, 18, 16, 18, 19, // Top
    20, 21, 22, 20, 22, 23, // Bottom
];
