use super::chunk::{Chunk, CHUNK_SIZE};
use crate::graphical_core::mesh::Vertex;

/// Face directions with their normal vectors and axis info.
const FACES: [Face; 6] = [
    Face {
        normal: [1, 0, 0],
        tangent: [0, 0, 1],
        bitangent: [0, 1, 0],
    }, // +X
    Face {
        normal: [-1, 0, 0],
        tangent: [0, 0, -1],
        bitangent: [0, 1, 0],
    }, // -X
    Face {
        normal: [0, 1, 0],
        tangent: [1, 0, 0],
        bitangent: [0, 0, 1],
    }, // +Y
    Face {
        normal: [0, -1, 0],
        tangent: [1, 0, 0],
        bitangent: [0, 0, -1],
    }, // -Y
    Face {
        normal: [0, 0, 1],
        tangent: [-1, 0, 0],
        bitangent: [0, 1, 0],
    }, // +Z
    Face {
        normal: [0, 0, -1],
        tangent: [1, 0, 0],
        bitangent: [0, 1, 0],
    }, // -Z
];

struct Face {
    normal: [i32; 3],
    tangent: [i32; 3],
    bitangent: [i32; 3],
}

/// Optional references to the 4 horizontal neighbor chunks for boundary culling.
pub struct ChunkNeighbors<'a> {
    pub pos_x: Option<&'a Chunk>,
    pub neg_x: Option<&'a Chunk>,
    pub pos_z: Option<&'a Chunk>,
    pub neg_z: Option<&'a Chunk>,
}

/// Generates vertices and indices for all visible faces in a chunk.
/// Neighbor chunks are used for hidden-face culling at chunk boundaries.
pub fn mesh_chunk(chunk: &Chunk, neighbors: &ChunkNeighbors) -> (Vec<Vertex>, Vec<u32>) {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for y in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let block = chunk.get(x, y, z);
                if !block.is_opaque() {
                    continue;
                }
                let color = block.color();
                for face in &FACES {
                    if !is_face_visible(chunk, neighbors, x, y, z, face.normal) {
                        continue;
                    }
                    emit_face(&mut vertices, &mut indices, x, y, z, face, color);
                }
            }
        }
    }

    (vertices, indices)
}

/// A face is visible when its neighbor is air or outside the world.
fn is_face_visible(chunk: &Chunk, neighbors: &ChunkNeighbors, x: usize, y: usize, z: usize, normal: [i32; 3]) -> bool {
    let nx = x as i32 + normal[0];
    let ny = y as i32 + normal[1];
    let nz = z as i32 + normal[2];

    // Vertical out-of-bounds: always visible (no vertical neighbors)
    if ny < 0 || ny >= CHUNK_SIZE as i32 {
        return true;
    }

    // Horizontal out-of-bounds: check neighbor chunk
    if nx < 0 || nx >= CHUNK_SIZE as i32 || nz < 0 || nz >= CHUNK_SIZE as i32 {
        return check_neighbor(neighbors, nx, ny as usize, nz);
    }

    !chunk.get(nx as usize, ny as usize, nz as usize).is_opaque()
}

/// Checks a neighboring chunk for the block at the wrapped coordinate.
/// Returns true (face visible) if no neighbor exists.
fn check_neighbor(neighbors: &ChunkNeighbors, nx: i32, ny: usize, nz: i32) -> bool {
    let last = CHUNK_SIZE - 1;

    let (neighbor, lx, lz) = if nx < 0 {
        (neighbors.neg_x, last, nz as usize)
    } else if nx >= CHUNK_SIZE as i32 {
        (neighbors.pos_x, 0, nz as usize)
    } else if nz < 0 {
        (neighbors.neg_z, nx as usize, last)
    } else {
        (neighbors.pos_z, nx as usize, 0)
    };

    match neighbor {
        Some(chunk) => !chunk.get(lx, ny, lz).is_opaque(),
        None => true,
    }
}

/// Emits 4 vertices and 6 indices (2 triangles) for one block face.
fn emit_face(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, x: usize, y: usize, z: usize, face: &Face, color: [f32; 3]) {
    let base_index = vertices.len() as u32;

    let center = [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5];
    let n = [face.normal[0] as f32, face.normal[1] as f32, face.normal[2] as f32];
    let t = [face.tangent[0] as f32, face.tangent[1] as f32, face.tangent[2] as f32];
    let b = [face.bitangent[0] as f32, face.bitangent[1] as f32, face.bitangent[2] as f32];

    // Four corners of the face, offset 0.5 from center along normal
    let face_center = [center[0] + n[0] * 0.5, center[1] + n[1] * 0.5, center[2] + n[2] * 0.5];

    let corners = [
        [-0.5, -0.5], // bottom-left
        [0.5, -0.5],  // bottom-right
        [0.5, 0.5],   // top-right
        [-0.5, 0.5],  // top-left
    ];

    for (i, [u, v]) in corners.iter().enumerate() {
        let position = [
            face_center[0] + t[0] * u + b[0] * v,
            face_center[1] + t[1] * u + b[1] * v,
            face_center[2] + t[2] * u + b[2] * v,
        ];
        let uv = match i {
            0 => [0.0, 1.0],
            1 => [1.0, 1.0],
            2 => [1.0, 0.0],
            3 => [0.0, 0.0],
            _ => unreachable!(),
        };
        vertices.push(Vertex {
            position,
            uv_coordinate: uv,
            color,
        });
    }

    // Two triangles: 0-1-2, 0-2-3
    indices.extend_from_slice(&[base_index, base_index + 1, base_index + 2, base_index, base_index + 2, base_index + 3]);
}
