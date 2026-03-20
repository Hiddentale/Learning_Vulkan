/// Represents a single vertex with position and texture coordinate data.
///
/// `#[repr(C)]` ensures the struct has a predictable memory layout matching C/Vulkan expectations.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv_coordinate: [f32; 2],
}

pub const CUBE_VERTICES: [Vertex; 24] = [
    // Front face (z = -0.5)
    Vertex { position: [-0.5, -0.5, -0.5], uv_coordinate: [0.0, 1.0] },
    Vertex { position: [ 0.5, -0.5, -0.5], uv_coordinate: [1.0, 1.0] },
    Vertex { position: [ 0.5,  0.5, -0.5], uv_coordinate: [1.0, 0.0] },
    Vertex { position: [-0.5,  0.5, -0.5], uv_coordinate: [0.0, 0.0] },
    // Back face (z = 0.5)
    Vertex { position: [ 0.5, -0.5,  0.5], uv_coordinate: [0.0, 1.0] },
    Vertex { position: [-0.5, -0.5,  0.5], uv_coordinate: [1.0, 1.0] },
    Vertex { position: [-0.5,  0.5,  0.5], uv_coordinate: [1.0, 0.0] },
    Vertex { position: [ 0.5,  0.5,  0.5], uv_coordinate: [0.0, 0.0] },
    // Right face (x = 0.5)
    Vertex { position: [ 0.5, -0.5, -0.5], uv_coordinate: [0.0, 1.0] },
    Vertex { position: [ 0.5, -0.5,  0.5], uv_coordinate: [1.0, 1.0] },
    Vertex { position: [ 0.5,  0.5,  0.5], uv_coordinate: [1.0, 0.0] },
    Vertex { position: [ 0.5,  0.5, -0.5], uv_coordinate: [0.0, 0.0] },
    // Left face (x = -0.5)
    Vertex { position: [-0.5, -0.5,  0.5], uv_coordinate: [0.0, 1.0] },
    Vertex { position: [-0.5, -0.5, -0.5], uv_coordinate: [1.0, 1.0] },
    Vertex { position: [-0.5,  0.5, -0.5], uv_coordinate: [1.0, 0.0] },
    Vertex { position: [-0.5,  0.5,  0.5], uv_coordinate: [0.0, 0.0] },
    // Top face (y = 0.5)
    Vertex { position: [-0.5,  0.5, -0.5], uv_coordinate: [0.0, 1.0] },
    Vertex { position: [ 0.5,  0.5, -0.5], uv_coordinate: [1.0, 1.0] },
    Vertex { position: [ 0.5,  0.5,  0.5], uv_coordinate: [1.0, 0.0] },
    Vertex { position: [-0.5,  0.5,  0.5], uv_coordinate: [0.0, 0.0] },
    // Bottom face (y = -0.5)
    Vertex { position: [-0.5, -0.5,  0.5], uv_coordinate: [0.0, 1.0] },
    Vertex { position: [ 0.5, -0.5,  0.5], uv_coordinate: [1.0, 1.0] },
    Vertex { position: [ 0.5, -0.5, -0.5], uv_coordinate: [1.0, 0.0] },
    Vertex { position: [-0.5, -0.5, -0.5], uv_coordinate: [0.0, 0.0] },
];

pub const CUBE_INDICES: [u16; 36] = [
     0,  1,  2,  0,  2,  3, // Front
     4,  5,  6,  4,  6,  7, // Back
     8,  9, 10,  8, 10, 11, // Right
    12, 13, 14, 12, 14, 15, // Left
    16, 17, 18, 16, 18, 19, // Top
    20, 21, 22, 20, 22, 23, // Bottom
];
