#![allow(dead_code)] // SVDAG is wired up in Phase 2

use super::block::BlockType;
use super::chunk::{Chunk, CHUNK_SIZE};
use std::collections::HashMap;

// SVDAG tree: recursive octree down to 4³ leaves.
//   16³ chunk: 3 levels (16 → 8 → 4 → leaf)
//   64³ super-chunk: 5 levels (64 → 32 → 16 → 8 → 4 → leaf)

const LEAF_SIZE: usize = 4;
const LEAF_BYTES: usize = LEAF_SIZE * LEAF_SIZE * LEAF_SIZE; // 64
const HEADER_SIZE: usize = 4;

/// Trait for looking up voxels at arbitrary coordinates.
/// Allows SVDAG compression over single chunks or multi-chunk grids.
pub trait VoxelSource {
    fn get(&self, x: usize, y: usize, z: usize) -> BlockType;
}

impl VoxelSource for Chunk {
    fn get(&self, x: usize, y: usize, z: usize) -> BlockType {
        self.get(x, y, z)
    }
}

/// A 4×4×4 grid of chunks forming a 64³ voxel volume.
/// Missing chunks (None) are treated as all air.
pub struct SuperChunkGrid {
    pub chunks: Box<[Option<Chunk>; 64]>,
}

impl VoxelSource for SuperChunkGrid {
    fn get(&self, x: usize, y: usize, z: usize) -> BlockType {
        let cx = x / CHUNK_SIZE;
        let cy = y / CHUNK_SIZE;
        let cz = z / CHUNK_SIZE;
        let idx = cx + cz * 4 + cy * 16;
        match &self.chunks[idx] {
            Some(chunk) => chunk.get(x % CHUNK_SIZE, y % CHUNK_SIZE, z % CHUNK_SIZE),
            None => BlockType::Air,
        }
    }
}

/// Compresses a dense chunk into an SVDAG byte buffer with subtree deduplication.
/// First 4 bytes are the root offset (little-endian u32).
pub fn svdag_from_chunk(chunk: &Chunk) -> Vec<u8> {
    svdag_compress(chunk, CHUNK_SIZE)
}

/// Compresses a 64³ super-chunk grid into a 5-level SVDAG.
/// Underground is stripped from all columns — LOD-0 SVDAG (128m+) never shows underground.
pub fn svdag_from_super_chunk(grid: &SuperChunkGrid) -> Vec<u8> {
    let size = CHUNK_SIZE * 4;
    let stripped = SurfaceStrippedSource::new(grid, size, SURFACE_STRIP_DEPTH);
    svdag_compress(&stripped, size)
}

/// Voxels of underground to keep below the surface per column.
/// Enough for surface texture but prevents cross-section walls at LOD boundaries.
const SURFACE_STRIP_DEPTH: usize = 2;

/// Materializes a VoxelSource and strips underground from every column,
/// keeping only the top `depth` solid voxels. Prevents visible terrain
/// cross-section walls at chunk/LOD boundaries.
struct SurfaceStrippedSource {
    blocks: Vec<BlockType>,
    size: usize,
}

impl SurfaceStrippedSource {
    fn new(source: &dyn VoxelSource, size: usize, depth: usize) -> Self {
        let mut blocks = vec![BlockType::Air; size * size * size];
        for y in 0..size {
            for z in 0..size {
                for x in 0..size {
                    blocks[x + z * size + y * size * size] = source.get(x, y, z);
                }
            }
        }
        for z in 0..size {
            for x in 0..size {
                let col = x + z * size;
                let mut top = 0;
                for y in (0..size).rev() {
                    if blocks[col + y * size * size] != BlockType::Air {
                        top = y;
                        break;
                    }
                }
                if top >= depth {
                    for y in 0..top - depth {
                        blocks[col + y * size * size] = BlockType::Air;
                    }
                }
            }
        }
        Self { blocks, size }
    }
}

impl VoxelSource for SurfaceStrippedSource {
    fn get(&self, x: usize, y: usize, z: usize) -> BlockType {
        self.blocks[x + z * self.size + y * self.size * self.size]
    }
}

/// Generic SVDAG compression over any voxel source at any power-of-two size.
pub fn svdag_compress(source: &dyn VoxelSource, size: usize) -> Vec<u8> {
    let mut buf = vec![0u8; HEADER_SIZE];
    let mut dedup: HashMap<Vec<u8>, u32> = HashMap::new();
    let root = build_node(&mut buf, &mut dedup, source, 0, 0, 0, size);
    buf[..4].copy_from_slice(&root.to_le_bytes());
    buf
}

/// Random-access lookup into a 16³ SVDAG. Returns the block type at (x, y, z).
pub fn svdag_lookup(dag: &[u8], x: u8, y: u8, z: u8) -> BlockType {
    svdag_lookup_sized(dag, x as usize, y as usize, z as usize, CHUNK_SIZE)
}

/// Random-access lookup into an SVDAG of arbitrary root size.
pub fn svdag_lookup_sized(dag: &[u8], x: usize, y: usize, z: usize, root_size: usize) -> BlockType {
    let root = u32::from_le_bytes(dag[..4].try_into().unwrap()) as usize;
    lookup_node(dag, root, x, y, z, root_size)
}

/// Merges 8 LOD-0 SVDAGs (one per octant child) into a single LOD-1 SVDAG at half resolution.
/// Parent voxel is solid if majority (>=4 of 8) child voxels are solid.
/// Material chosen by majority vote.
pub fn svdag_lod_merge(children: [&Chunk; 8]) -> Vec<u8> {
    let merged = downsample_chunks(children);
    svdag_from_chunk(&merged)
}

/// Merge 2×2×2 super-chunk SVDAGs into one LOD super-chunk at half resolution.
/// Each child is a 64³ SVDAG. The result is a 64³ SVDAG covering 2× the space.
pub fn svdag_lod_merge_super(children: [&[u8]; 8]) -> Vec<u8> {
    let grid = SvdagOctantGrid {
        children,
        child_size: CHUNK_SIZE * 4,
    };
    let downsampled = DownsampledSource { source: &grid, scale: 2 };
    svdag_compress(&downsampled, CHUNK_SIZE * 4)
}

/// 2×2×2 grid of SVDAGs forming a virtual volume at 2× the child resolution.
struct SvdagOctantGrid<'a> {
    children: [&'a [u8]; 8],
    child_size: usize,
}

impl VoxelSource for SvdagOctantGrid<'_> {
    fn get(&self, x: usize, y: usize, z: usize) -> BlockType {
        let half = self.child_size;
        let octant = (if x >= half { 1 } else { 0 }) | (if y >= half { 2 } else { 0 }) | (if z >= half { 4 } else { 0 });
        svdag_lookup_sized(self.children[octant], x % half, y % half, z % half, half)
    }
}

/// Wraps a VoxelSource, downsampling by majority-voting scale³ blocks.
struct DownsampledSource<'a> {
    source: &'a dyn VoxelSource,
    scale: usize,
}

impl VoxelSource for DownsampledSource<'_> {
    fn get(&self, x: usize, y: usize, z: usize) -> BlockType {
        majority_block_source(self.source, x * self.scale, y * self.scale, z * self.scale, self.scale)
    }
}

fn majority_block_source(source: &dyn VoxelSource, bx: usize, by: usize, bz: usize, scale: usize) -> BlockType {
    let mut counts = [0u16; 8];
    for dy in 0..scale {
        for dz in 0..scale {
            for dx in 0..scale {
                let block = source.get(bx + dx, by + dy, bz + dz);
                counts[block as usize] += 1;
            }
        }
    }
    let threshold = (scale * scale * scale / 2) as u16;
    let mut best = BlockType::Air;
    let mut best_count = 0u16;
    for (i, &c) in counts.iter().enumerate().skip(1) {
        if c > best_count {
            best_count = c;
            best = unsafe { std::mem::transmute::<u8, BlockType>(i as u8) };
        }
    }
    if best_count >= threshold {
        best
    } else {
        BlockType::Air
    }
}

/// DFS-ordered flat material array for an SVDAG.
/// Each solid voxel's material (BlockType as u8) appears in DFS traversal order.
pub fn svdag_materials(chunk: &Chunk) -> Vec<u8> {
    let mut materials = Vec::new();
    collect_materials_dfs(chunk as &dyn VoxelSource, 0, 0, 0, CHUNK_SIZE, &mut materials);
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

fn build_node(buf: &mut Vec<u8>, dedup: &mut HashMap<Vec<u8>, u32>, source: &dyn VoxelSource, ox: usize, oy: usize, oz: usize, size: usize) -> u32 {
    if size == LEAF_SIZE {
        return build_leaf(buf, dedup, source, ox, oy, oz);
    }

    let half = size / 2;
    let mut child_mask = 0u8;
    let mut child_offsets = [0u32; 8];

    for i in 0..8u8 {
        let (dx, dy, dz) = octant_offset(i);
        let cx = ox + dx * half;
        let cy = oy + dy * half;
        let cz = oz + dz * half;

        if !is_uniform_air(source, cx, cy, cz, half) {
            child_mask |= 1 << i;
            child_offsets[i as usize] = build_node(buf, dedup, source, cx, cy, cz, half);
        }
    }

    if child_mask == 0 {
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
fn build_leaf(buf: &mut Vec<u8>, dedup: &mut HashMap<Vec<u8>, u32>, source: &dyn VoxelSource, ox: usize, oy: usize, oz: usize) -> u32 {
    let mut leaf = [0u8; LEAF_BYTES];
    for y in 0..LEAF_SIZE {
        for z in 0..LEAF_SIZE {
            for x in 0..LEAF_SIZE {
                let idx = x + z * LEAF_SIZE + y * LEAF_SIZE * LEAF_SIZE;
                leaf[idx] = source.get(ox + x, oy + y, oz + z) as u8;
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

fn is_uniform_air(source: &dyn VoxelSource, ox: usize, oy: usize, oz: usize, size: usize) -> bool {
    for y in oy..oy + size {
        for z in oz..oz + size {
            for x in ox..ox + size {
                if source.get(x, y, z) != BlockType::Air {
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

fn collect_materials_dfs(source: &dyn VoxelSource, ox: usize, oy: usize, oz: usize, size: usize, materials: &mut Vec<u8>) {
    if size == LEAF_SIZE {
        for y in 0..LEAF_SIZE {
            for z in 0..LEAF_SIZE {
                for x in 0..LEAF_SIZE {
                    let block = source.get(ox + x, oy + y, oz + z);
                    if block != BlockType::Air {
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
        if !is_uniform_air(source, cx, cy, cz, half) {
            collect_materials_dfs(source, cx, cy, cz, half, materials);
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

    #[test]
    fn super_chunk_surface_stripping_keeps_top() {
        // Build solid ground from y=0..16 (bottom chunk layer fully solid)
        let mut chunks: Vec<Option<Chunk>> = Vec::with_capacity(64);
        for cy in 0..4 {
            for _cz in 0..4 {
                for _cx in 0..4 {
                    let chunk = if cy == 0 {
                        Chunk::new(BlockType::Stone)
                    } else {
                        Chunk::new(BlockType::Air)
                    };
                    chunks.push(Some(chunk));
                }
            }
        }
        let grid = SuperChunkGrid {
            chunks: match chunks.into_boxed_slice().try_into() {
                Ok(b) => b,
                Err(_) => unreachable!(),
            },
        };
        let dag = svdag_from_super_chunk(&grid);
        let size = CHUNK_SIZE * 4;

        // Surface (y=15) and 1 below (y=14) should survive stripping
        assert!(svdag_lookup_sized(&dag, 0, 15, 0, size).is_opaque(), "surface should be solid");
        assert!(svdag_lookup_sized(&dag, 0, 14, 0, size).is_opaque(), "1 below surface should be solid");
        // Deep underground should be stripped
        assert!(!svdag_lookup_sized(&dag, 0, 0, 0, size).is_opaque(), "underground should be stripped");
        assert!(
            !svdag_lookup_sized(&dag, 0, 10, 0, size).is_opaque(),
            "deep underground should be stripped"
        );
    }

    #[test]
    fn super_chunk_cross_boundary_lookup() {
        // Place a block at (17, 0, 0) — that's chunk (1,0,0) local (1,0,0)
        let mut chunks: Vec<Option<Chunk>> = (0..64).map(|_| Some(Chunk::new(BlockType::Air))).collect();
        // chunk index for (cx=1, cy=0, cz=0) = 1 + 0*4 + 0*16 = 1
        chunks[1].as_mut().unwrap().set(1, 0, 0, BlockType::Stone);
        let grid = SuperChunkGrid {
            chunks: match chunks.into_boxed_slice().try_into() {
                Ok(b) => b,
                Err(_) => unreachable!(),
            },
        };
        // Global coordinate (17, 0, 0) should be stone
        assert_eq!(grid.get(17, 0, 0), BlockType::Stone);
        assert_eq!(grid.get(16, 0, 0), BlockType::Air);
        assert_eq!(grid.get(0, 0, 0), BlockType::Air);

        let dag = svdag_from_super_chunk(&grid);
        assert_eq!(svdag_lookup_sized(&dag, 17, 0, 0, 64), BlockType::Stone);
        assert_eq!(svdag_lookup_sized(&dag, 16, 0, 0, 64), BlockType::Air);
    }

    #[test]
    fn super_chunk_all_air() {
        let chunks: Vec<Option<Chunk>> = (0..64).map(|_| None).collect();
        let grid = SuperChunkGrid {
            chunks: match chunks.into_boxed_slice().try_into() {
                Ok(b) => b,
                Err(_) => unreachable!(),
            },
        };
        let dag = svdag_from_super_chunk(&grid);
        assert!(dag.len() < 200, "all-air super-chunk should be small, got {} bytes", dag.len());
        assert_eq!(svdag_lookup_sized(&dag, 0, 0, 0, 64), BlockType::Air);
        assert_eq!(svdag_lookup_sized(&dag, 63, 63, 63, 64), BlockType::Air);
    }

    #[test]
    fn lod_merge_super_all_solid() {
        // Use svdag_compress directly to test merge logic without surface stripping
        let solid_grid = make_solid_super_chunk(BlockType::Stone);
        let size = CHUNK_SIZE * 4;
        let dag = svdag_compress(&solid_grid, size);
        let children: [&[u8]; 8] = [&dag; 8];
        let merged = svdag_lod_merge_super(children);
        for y in 0..size {
            assert!(svdag_lookup_sized(&merged, 0, y, 0, size).is_opaque(), "expected solid at y={y}");
        }
    }

    #[test]
    fn lod_merge_super_half_solid_preserves_boundary() {
        // Use svdag_compress directly to test merge logic without surface stripping
        let solid_grid = make_solid_super_chunk(BlockType::Stone);
        let air_grid = make_air_super_chunk();
        let size = CHUNK_SIZE * 4;
        let solid_dag = svdag_compress(&solid_grid, size);
        let air_dag = svdag_compress(&air_grid, size);
        // Octant bit1 = Y. Bottom octants (Y=0): indices 0,1,4,5. Top (Y=1): 2,3,6,7.
        let children: [&[u8]; 8] = [&solid_dag, &solid_dag, &air_dag, &air_dag, &solid_dag, &solid_dag, &air_dag, &air_dag];
        let merged = svdag_lod_merge_super(children);
        assert!(svdag_lookup_sized(&merged, 0, 0, 0, size).is_opaque());
        assert!(!svdag_lookup_sized(&merged, 0, size - 1, 0, size).is_opaque());
    }

    fn make_solid_super_chunk(block: BlockType) -> SuperChunkGrid {
        let chunks: Vec<Option<Chunk>> = (0..64).map(|_| Some(Chunk::new(block))).collect();
        SuperChunkGrid {
            chunks: match chunks.into_boxed_slice().try_into() {
                Ok(b) => b,
                Err(_) => unreachable!(),
            },
        }
    }

    fn make_air_super_chunk() -> SuperChunkGrid {
        let chunks: Vec<Option<Chunk>> = (0..64).map(|_| Some(Chunk::new(BlockType::Air))).collect();
        SuperChunkGrid {
            chunks: match chunks.into_boxed_slice().try_into() {
                Ok(b) => b,
                Err(_) => unreachable!(),
            },
        }
    }
}
