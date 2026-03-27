/// `#[repr(C)]` ensures a predictable memory layout matching Vulkan's expectations.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Vertex {
    pub position: [f32; 3],
    pub uv_coordinate: [f32; 2],
    pub normal: [f32; 3],
    pub material_id: u32,
}
