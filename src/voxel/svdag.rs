#![allow(dead_code)] // SVDAG is wired up in Phase 2

use super::block::BlockType;
use super::chunk::{Chunk, CHUNK_SIZE};
use std::collections::HashMap;

// 3-level SVDAG for a 16³ chunk:
//   Level 0 (root):  16³ → 8 children (8³ octants)
//   Level 1 (mid):   8³  → 8 children (4³ octants)
//   Level 2 (leaf):  4³  → 64-bit occupancy bitmap

const LEAF_SIZE: usize = 4;
const LEAF_BYTES: usize = LEAF_SIZE * LEAF_SIZE * LEAF_SIZE; // 64
const HEADER_SIZE: usize = 4;

/// Compresses a dense chunk into an SVDAG byte buffer with subtree deduplication.
/// First 4 bytes are the root offset (little-endian u32).
pub fn svdag_from_chunk(chunk: &Chunk) -> Vec<u8> {
    let mut buf = vec![0u8; HEADER_SIZE]; // reserve header
    let mut dedup: HashMap<Vec<u8>, u32> = HashMap::new();
    let root = build_node(&mut buf, &mut dedup, chunk, 0, 0, 0, CHUNK_SIZE);
    buf[..4].copy_from_slice(&root.to_le_bytes());
    buf
}

/// Random-access lookup into a compressed SVDAG. Returns the block type at (x, y, z).
pub fn svdag_lookup(dag: &[u8], x: u8, y: u8, z: u8) -> BlockType {
    let root = u32::from_le_bytes(dag[..4].try_into().unwrap()) as usize;
    lookup_node(dag, root, x as usize, y as usize, z as usize, CHUNK_SIZE)
}

/// Merges 8 LOD-0 SVDAGs (one per octant child) into a single LOD-1 SVDAG at half resolution.
/// Parent voxel is solid if majority (>=4 of 8) child voxels are solid.
/// Material chosen by majority vote.
pub fn svdag_lod_merge(children: [&Chunk; 8]) -> Vec<u8> {
    let merged = downsample_chunks(children);
    svdag_from_chunk(&merged)
}

/// DFS-ordered flat material array for an SVDAG.
/// Each solid voxel's material (BlockType as u8) appears in DFS traversal order.
pub fn svdag_materials(chunk: &Chunk) -> Vec<u8> {
    let mut materials = Vec::new();
    collect_materials_dfs(chunk, 0, 0, 0, CHUNK_SIZE, &mut materials);
    materials
}

/// Run-length encode a material array. Format: [count, value, count, value, ...]
pub fn rle_encode(data: &[u8]) -> Vec<u8> {
    if data.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::new();
    let mut current = data[0];
    let mut count: u8 = 1;
    for &val in &data[1..] {
        if val == current && count < 255 {
            count += 1;
        } else {
            result.push(count);
            result.push(current);
            current = val;
            count = 1;
        }
    }
    result.push(count);
    result.push(current);
    result
}

/// Decode an RLE-encoded material array.
pub fn rle_decode(rle: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    for pair in rle.chunks_exact(2) {
        let count = pair[0] as usize;
        let value = pair[1];
        result.extend(std::iter::repeat_n(value, count));
    }
    result
}

// --- Internal ---

fn build_node(buf: &mut Vec<u8>, dedup: &mut HashMap<Vec<u8>, u32>, chunk: &Chunk, ox: usize, oy: usize, oz: usize, size: usize) -> u32 {
    if size == LEAF_SIZE {
        return build_leaf(buf, dedup, chunk, ox, oy, oz);
    }

    let half = size / 2;
    let mut child_mask = 0u8;
    let mut child_offsets = [0u32; 8];

    for i in 0..8u8 {
        let (dx, dy, dz) = octant_offset(i);
        let cx = ox + dx * half;
        let cy = oy + dy * half;
        let cz = oz + dz * half;

        if !is_uniform_air(chunk, cx, cy, cz, half) {
            child_mask |= 1 << i;
            child_offsets[i as usize] = build_node(buf, dedup, chunk, cx, cy, cz, half);
        }
    }

    if child_mask == 0 {
        // Entirely air — encode as a leaf of all zeros
        return build_leaf_raw(buf, dedup, 0);
    }

    let child_count = child_mask.count_ones() as usize;
    let mut node_bytes = Vec::with_capacity(1 + child_count * 4);
    node_bytes.push(child_mask);
    for i in 0..8u8 {
        if child_mask & (1 << i) != 0 {
            node_bytes.extend_from_slice(&child_offsets[i as usize].to_le_bytes());
        }
    }

    if let Some(&existing) = dedup.get(&node_bytes) {
        return existing;
    }
    let offset = buf.len() as u32;
    buf.extend_from_slice(&node_bytes);
    dedup.insert(node_bytes, offset);
    offset
}

/// Leaf: 64 bytes — one u8 material ID per voxel in 4³ block. 0 = air.
/// Layout: x + z * 4 + y * 16 (same as occupancy bitmap bit order).
fn build_leaf(buf: &mut Vec<u8>, dedup: &mut HashMap<Vec<u8>, u32>, chunk: &Chunk, ox: usize, oy: usize, oz: usize) -> u32 {
    let mut leaf = [0u8; LEAF_BYTES];
    for y in 0..LEAF_SIZE {
        for z in 0..LEAF_SIZE {
            for x in 0..LEAF_SIZE {
                let idx = x + z * LEAF_SIZE + y * LEAF_SIZE * LEAF_SIZE;
                leaf[idx] = chunk.get(ox + x, oy + y, oz + z) as u8;
            }
        }
    }
    let leaf_vec = leaf.to_vec();
    if let Some(&existing) = dedup.get(&leaf_vec) {
        return existing;
    }
    let offset = buf.len() as u32;
    buf.extend_from_slice(&leaf);
    dedup.insert(leaf_vec, offset);
    offset
}

fn build_leaf_raw(buf: &mut Vec<u8>, dedup: &mut HashMap<Vec<u8>, u32>, fill: u8) -> u32 {
    let leaf = [fill; LEAF_BYTES];
    let leaf_vec = leaf.to_vec();
    if let Some(&existing) = dedup.get(&leaf_vec) {
        return existing;
    }
    let offset = buf.len() as u32;
    buf.extend_from_slice(&leaf);
    dedup.insert(leaf_vec, offset);
    offset
}

fn lookup_node(dag: &[u8], offset: usize, x: usize, y: usize, z: usize, size: usize) -> BlockType {
    if size == LEAF_SIZE {
        let idx = x + z * LEAF_SIZE + y * LEAF_SIZE * LEAF_SIZE;
        let mat = dag[offset + idx];
        // Safety: BlockType is repr(u8) with values 0..=7
        unsafe { std::mem::transmute::<u8, BlockType>(mat) }
    } else {
        let child_mask = dag[offset];
        let half = size / 2;
        let octant = octant_index(x, y, z, half);
        if child_mask & (1 << octant) == 0 {
            return BlockType::Air;
        }
        let child_rank = (child_mask & ((1 << octant) - 1)).count_ones() as usize;
        let ptr_offset = offset + 1 + child_rank * 4;
        let child_offset = u32::from_le_bytes(dag[ptr_offset..ptr_offset + 4].try_into().unwrap()) as usize;
        lookup_node(dag, child_offset, x % half, y % half, z % half, half)
    }
}

fn octant_offset(index: u8) -> (usize, usize, usize) {
    ((index & 1) as usize, ((index >> 1) & 1) as usize, ((index >> 2) & 1) as usize)
}

fn octant_index(x: usize, y: usize, z: usize, half: usize) -> u8 {
    let xi = if x >= half { 1 } else { 0 };
    let yi = if y >= half { 1 } else { 0 };
    let zi = if z >= half { 1 } else { 0 };
    xi | (yi << 1) | (zi << 2)
}

fn is_uniform_air(chunk: &Chunk, ox: usize, oy: usize, oz: usize, size: usize) -> bool {
    for y in oy..oy + size {
        for z in oz..oz + size {
            for x in ox..ox + size {
                if chunk.get(x, y, z).is_opaque() {
                    return false;
                }
            }
        }
    }
    true
}

fn downsample_chunks(children: [&Chunk; 8]) -> Chunk {
    let mut result = Chunk::new(BlockType::Air);
    let half = CHUNK_SIZE / 2;
    for octant in 0..8u8 {
        let (ox, oy, oz) = octant_offset(octant);
        let child = children[octant as usize];
        for y in 0..half {
            for z in 0..half {
                for x in 0..half {
                    let block = majority_block(child, x * 2, y * 2, z * 2);
                    result.set(ox * half + x, oy * half + y, oz * half + z, block);
                }
            }
        }
    }
    result
}

fn majority_block(chunk: &Chunk, bx: usize, by: usize, bz: usize) -> BlockType {
    let mut counts = [0u8; 8];
    for dy in 0..2 {
        for dz in 0..2 {
            for dx in 0..2 {
                let block = chunk.get(bx + dx, by + dy, bz + dz);
                counts[block as usize] += 1;
            }
        }
    }
    // Majority: >=4 of 8 must be solid. If no majority, use most common non-air.
    let mut best = BlockType::Air;
    let mut best_count = 0u8;
    for (i, &c) in counts.iter().enumerate().skip(1) {
        // skip Air (index 0)
        if c > best_count {
            best_count = c;
            best = unsafe { std::mem::transmute::<u8, BlockType>(i as u8) };
        }
    }
    if best_count >= 4 {
        best
    } else {
        BlockType::Air
    }
}

fn collect_materials_dfs(chunk: &Chunk, ox: usize, oy: usize, oz: usize, size: usize, materials: &mut Vec<u8>) {
    if size == LEAF_SIZE {
        for y in 0..LEAF_SIZE {
            for z in 0..LEAF_SIZE {
                for x in 0..LEAF_SIZE {
                    let block = chunk.get(ox + x, oy + y, oz + z);
                    if block.is_opaque() {
                        materials.push(block as u8);
                    }
                }
            }
        }
        return;
    }

    let half = size / 2;
    for i in 0..8u8 {
        let (dx, dy, dz) = octant_offset(i);
        let cx = ox + dx * half;
        let cy = oy + dy * half;
        let cz = oz + dz * half;
        if !is_uniform_air(chunk, cx, cy, cz, half) {
            collect_materials_dfs(chunk, cx, cy, cz, half, materials);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_all_air_chunk() {
        let chunk = Chunk::new(BlockType::Air);
        let dag = svdag_from_chunk(&chunk);
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    assert_eq!(svdag_lookup(&dag, x as u8, y as u8, z as u8), BlockType::Air);
                }
            }
        }
    }

    #[test]
    fn roundtrip_all_solid_chunk() {
        let chunk = Chunk::new(BlockType::Stone);
        let dag = svdag_from_chunk(&chunk);
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    assert!(svdag_lookup(&dag, x as u8, y as u8, z as u8).is_opaque());
                }
            }
        }
    }

    #[test]
    fn roundtrip_mixed_chunk() {
        let mut chunk = Chunk::new(BlockType::Air);
        // Place blocks in a pattern
        for y in 0..8 {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    chunk.set(x, y, z, BlockType::Dirt);
                }
            }
        }
        let dag = svdag_from_chunk(&chunk);
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    let expected_solid = y < 8;
                    let actual = svdag_lookup(&dag, x as u8, y as u8, z as u8);
                    assert_eq!(actual.is_opaque(), expected_solid, "mismatch at ({x}, {y}, {z})");
                }
            }
        }
    }

    #[test]
    fn compression_ratio_all_air() {
        let chunk = Chunk::new(BlockType::Air);
        let dag = svdag_from_chunk(&chunk);
        // All-air should compress very well
        assert!(dag.len() < 100, "all-air dag should be tiny, got {} bytes", dag.len());
    }

    #[test]
    fn compression_ratio_all_solid() {
        let chunk = Chunk::new(BlockType::Stone);
        let dag = svdag_from_chunk(&chunk);
        let dense_size = CHUNK_SIZE * CHUNK_SIZE * CHUNK_SIZE;
        let ratio = dense_size as f64 / dag.len() as f64;
        assert!(ratio > 5.0, "solid chunk should compress >5x, got {ratio:.1}x");
    }

    #[test]
    fn lod_merge_all_solid() {
        let solid = Chunk::new(BlockType::Stone);
        let children: [&Chunk; 8] = [&solid; 8];
        let merged_dag = svdag_lod_merge(children);
        // Merged chunk at half res should be all solid
        for y in 0..CHUNK_SIZE {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    assert!(svdag_lookup(&merged_dag, x as u8, y as u8, z as u8).is_opaque());
                }
            }
        }
    }

    #[test]
    fn lod_merge_half_solid() {
        // Bottom half solid, top half air in each child
        let mut child = Chunk::new(BlockType::Air);
        for y in 0..8 {
            for z in 0..CHUNK_SIZE {
                for x in 0..CHUNK_SIZE {
                    child.set(x, y, z, BlockType::Stone);
                }
            }
        }
        let children: [&Chunk; 8] = [&child; 8];
        let merged_dag = svdag_lod_merge(children);
        // Each child's 16³ → 8³ at half res. Bottom 8 source rows → bottom 4 result rows per octant.
        // Octant 0 (y=0..7): y=0..3 solid, y=4..7 air
        // Octant 2 (y=8..15): y=8..11 solid, y=12..15 air
        for y in 0..4 {
            assert!(svdag_lookup(&merged_dag, 0, y as u8, 0).is_opaque(), "expected solid at y={y}");
        }
        assert!(!svdag_lookup(&merged_dag, 0, 4, 0).is_opaque(), "expected air at y=4");
    }

    #[test]
    fn materials_dfs_order_matches_traversal() {
        let mut chunk = Chunk::new(BlockType::Air);
        chunk.set(0, 0, 0, BlockType::Stone);
        chunk.set(1, 0, 0, BlockType::Dirt);
        let materials = svdag_materials(&chunk);
        assert_eq!(materials.len(), 2);
        assert_eq!(materials[0], BlockType::Stone as u8);
        assert_eq!(materials[1], BlockType::Dirt as u8);
    }

    #[test]
    fn rle_roundtrip() {
        let data = vec![1, 1, 1, 2, 2, 3, 3, 3, 3];
        let encoded = rle_encode(&data);
        let decoded = rle_decode(&encoded);
        assert_eq!(data, decoded);
    }

    #[test]
    fn rle_empty() {
        assert!(rle_encode(&[]).is_empty());
        assert!(rle_decode(&[]).is_empty());
    }

    #[test]
    fn rle_compression_on_uniform() {
        let data = vec![5u8; 255];
        let encoded = rle_encode(&data);
        assert_eq!(encoded.len(), 2);
        assert_eq!(encoded, vec![255, 5]);
    }
}
