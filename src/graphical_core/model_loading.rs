use crate::graphical_core::mesh::{create_mesh, Mesh, Vertex};
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use anyhow::Context;
use vulkanalia::{Device, Instance};

/// Loads an OBJ file from disk and uploads it as a GPU mesh.
///
/// All sub-meshes in the file are merged into a single `Mesh`.
/// UV coordinates are flipped vertically (OBJ uses bottom-left origin,
/// Vulkan uses top-left).
pub unsafe fn load_obj(path: &str, device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<Mesh> {
    let (models, _materials) = tobj::load_obj(path, &tobj::GPU_LOAD_OPTIONS).with_context(|| format!("Failed to load OBJ: {path}"))?;

    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for model in &models {
        let obj_mesh = &model.mesh;
        let vertex_offset = vertices.len() as u32;

        let vertex_count = obj_mesh.positions.len() / 3;
        for i in 0..vertex_count {
            let has_uvs = obj_mesh.texcoords.len() > i * 2 + 1;
            let uv = if has_uvs {
                [obj_mesh.texcoords[i * 2], 1.0 - obj_mesh.texcoords[i * 2 + 1]]
            } else {
                [0.0, 0.0]
            };

            vertices.push(Vertex {
                position: [obj_mesh.positions[i * 3], obj_mesh.positions[i * 3 + 1], obj_mesh.positions[i * 3 + 2]],
                uv_coordinate: uv,
                color: [1.0, 1.0, 1.0],
            });
        }

        for &index in &obj_mesh.indices {
            indices.push(vertex_offset + index);
        }
    }

    create_mesh(&vertices, &indices, device, instance, data)
}
