use super::block::BlockType;
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

/// Optional references to the 6 neighbor chunks for boundary culling.
pub struct ChunkNeighbors<'a> {
    pub pos_x: Option<&'a Chunk>,
    pub neg_x: Option<&'a Chunk>,
    pub pos_y: Option<&'a Chunk>,
    pub neg_y: Option<&'a Chunk>,
    pub pos_z: Option<&'a Chunk>,
    pub neg_z: Option<&'a Chunk>,
}

/// Number of face direction buckets: +X, -X, +Y, -Y, +Z, -Z.
pub const BUCKET_COUNT: usize = 6;

/// Generates vertices and 6 index buckets (one per face direction) for a chunk.
/// Neighbor chunks are used for hidden-face culling at chunk boundaries.
pub fn mesh_chunk(chunk: &Chunk, neighbors: &ChunkNeighbors) -> (Vec<Vertex>, [Vec<u32>; BUCKET_COUNT]) {
    let mut vertices = Vec::new();
    let mut bucket_indices: [Vec<u32>; BUCKET_COUNT] = Default::default();

    for y in 0..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let block = chunk.get(x, y, z);
                if !block.is_opaque() {
                    continue;
                }
                let material_id = block.material_id() as u32;
                for (bucket, face) in FACES.iter().enumerate() {
                    if !is_face_visible(chunk, neighbors, x, y, z, face.normal) {
                        continue;
                    }
                    let normal = [face.normal[0] as f32, face.normal[1] as f32, face.normal[2] as f32];
                    emit_face(&mut vertices, &mut bucket_indices[bucket], x, y, z, face, normal, material_id);
                }
            }
        }
    }

    (vertices, bucket_indices)
}

/// A face is visible when the neighboring block does not hide it.
/// Opaque faces show when neighbor is non-opaque.
/// Water faces show only when neighbor is Air.
fn is_face_visible(chunk: &Chunk, neighbors: &ChunkNeighbors, x: usize, y: usize, z: usize, normal: [i32; 3]) -> bool {
    let block = chunk.get(x, y, z);
    let neighbor_block = get_neighbor_block(chunk, neighbors, x, y, z, normal);
    should_emit_face(block, neighbor_block)
}

/// Returns whether a face on `block` should be drawn given the adjacent `neighbor`.
fn should_emit_face(block: BlockType, neighbor: BlockType) -> bool {
    if block.is_opaque() {
        !neighbor.is_opaque()
    } else if block.is_transparent() {
        // Water: only show face against air
        neighbor == BlockType::Air
    } else {
        false
    }
}

/// Looks up the block adjacent to (x,y,z) in the given normal direction.
/// Returns Air if the neighbor chunk is not loaded.
fn get_neighbor_block(chunk: &Chunk, neighbors: &ChunkNeighbors, x: usize, y: usize, z: usize, normal: [i32; 3]) -> BlockType {
    let nx = x as i32 + normal[0];
    let ny = y as i32 + normal[1];
    let nz = z as i32 + normal[2];

    // Inside this chunk
    if nx >= 0 && nx < CHUNK_SIZE as i32 && ny >= 0 && ny < CHUNK_SIZE as i32 && nz >= 0 && nz < CHUNK_SIZE as i32 {
        return chunk.get(nx as usize, ny as usize, nz as usize);
    }

    let last = CHUNK_SIZE - 1;

    // Vertical boundary
    if ny < 0 {
        return match neighbors.neg_y {
            Some(c) => c.get(x, last, z),
            None => BlockType::Air,
        };
    }
    if ny >= CHUNK_SIZE as i32 {
        return match neighbors.pos_y {
            Some(c) => c.get(x, 0, z),
            None => BlockType::Air,
        };
    }

    // Horizontal boundary
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
        Some(c) => c.get(lx, ny as usize, lz),
        None => BlockType::Air,
    }
}

/// Emits 4 vertices and 6 indices (2 triangles) for one block face.
fn emit_face(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, x: usize, y: usize, z: usize, face: &Face, normal: [f32; 3], material_id: u32) {
    let base_index = vertices.len() as u32;

    let center = [x as f32 + 0.5, y as f32 + 0.5, z as f32 + 0.5];
    let t = [face.tangent[0] as f32, face.tangent[1] as f32, face.tangent[2] as f32];
    let b = [face.bitangent[0] as f32, face.bitangent[1] as f32, face.bitangent[2] as f32];

    // Four corners of the face, offset 0.5 from center along normal
    let face_center = [center[0] + normal[0] * 0.5, center[1] + normal[1] * 0.5, center[2] + normal[2] * 0.5];

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
            normal,
            material_id,
        });
    }

    // Two triangles: 0-1-2, 0-2-3
    indices.extend_from_slice(&[base_index, base_index + 1, base_index + 2, base_index, base_index + 2, base_index + 3]);
}
