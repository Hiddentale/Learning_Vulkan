#![allow(dead_code)] // Wired up incrementally
use crate::graphical_core::buffers::allocate_buffer;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::voxel::heightmap_generator::{HeightmapTileMesh, HeightmapVertex};
use glam::Vec3;
use std::collections::HashMap;
use vulkan_rust::{vk, Device, Instance};

/// Maximum number of heightmap tiles in the pool.
const MAX_TILES: u32 = 350;
/// Max vertices per tile (129x129 grid posts for HM-A).
const MAX_VERTICES_PER_TILE: usize = 129 * 129;
/// Max indices per tile (128x128 quads * 6 indices).
const MAX_INDICES_PER_TILE: usize = 128 * 128 * 6;

/// Per-tile CPU metadata for issuing draw calls.
#[derive(Copy, Clone, Debug)]
pub struct TileDrawInfo {
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub index_count: u32,
    pub lod: u8,
    pub pos: [i32; 2],
    pub morph_factor: f32,
}

/// Manages GPU vertex/index buffers for heightmap tile meshes.
pub struct HeightmapPool {
    pub vertex_buffer: vk::Buffer,
    vertex_memory: vk::DeviceMemory,
    vertex_ptr: *mut HeightmapVertex,

    pub index_buffer: vk::Buffer,
    index_memory: vk::DeviceMemory,
    index_ptr: *mut u16,

    tiles: HashMap<[i32; 2], u32>,
    tile_info: Vec<Option<TileDrawInfo>>,
    free_slots: Vec<u32>,
    next_slot: u32,
    pending_free: Vec<(u32, u32)>, // (slot, frames_remaining)
}

impl HeightmapPool {
    pub unsafe fn new(device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<Self> {
        let host_visible = super::host_visible_coherent();

        let vertex_size = MAX_TILES as u64 * MAX_VERTICES_PER_TILE as u64 * std::mem::size_of::<HeightmapVertex>() as u64;
        let (vertex_buffer, vertex_memory, vertex_ptr) =
            allocate_buffer::<HeightmapVertex>(vertex_size, vk::BufferUsageFlags::VERTEX_BUFFER, device, instance, data, host_visible)?;

        let index_size = MAX_TILES as u64 * MAX_INDICES_PER_TILE as u64 * std::mem::size_of::<u16>() as u64;
        let (index_buffer, index_memory, index_ptr) =
            allocate_buffer::<u16>(index_size, vk::BufferUsageFlags::INDEX_BUFFER, device, instance, data, host_visible)?;

        let mut tile_info = Vec::with_capacity(MAX_TILES as usize);
        tile_info.resize(MAX_TILES as usize, None);

        Ok(Self {
            vertex_buffer,
            vertex_memory,
            vertex_ptr,
            index_buffer,
            index_memory,
            index_ptr,
            tiles: HashMap::new(),
            tile_info,
            free_slots: Vec::new(),
            next_slot: 0,
            pending_free: Vec::new(),
        })
    }

    pub unsafe fn upload_tile(&mut self, mesh: &HeightmapTileMesh) {
        let slot = self.alloc_slot();
        let vertex_offset = slot as usize * MAX_VERTICES_PER_TILE;
        let index_offset = slot as usize * MAX_INDICES_PER_TILE;

        assert!(mesh.vertices.len() <= MAX_VERTICES_PER_TILE);
        assert!(mesh.indices.len() <= MAX_INDICES_PER_TILE);

        std::ptr::copy_nonoverlapping(mesh.vertices.as_ptr(), self.vertex_ptr.add(vertex_offset), mesh.vertices.len());
        std::ptr::copy_nonoverlapping(mesh.indices.as_ptr(), self.index_ptr.add(index_offset), mesh.indices.len());

        self.tile_info[slot as usize] = Some(TileDrawInfo {
            aabb_min: mesh.aabb_min,
            aabb_max: mesh.aabb_max,
            vertex_offset: vertex_offset as u32,
            index_offset: index_offset as u32,
            index_count: mesh.indices.len() as u32,
            lod: mesh.lod,
            pos: mesh.pos,
            morph_factor: 0.0,
        });
        self.tiles.insert(mesh.pos, slot);
    }

    pub unsafe fn remove_tile(&mut self, pos: &[i32; 2]) {
        if let Some(slot) = self.tiles.remove(pos) {
            self.tile_info[slot as usize] = None;
            self.pending_free.push((slot, 3));
        }
    }

    /// Call once per frame. Promotes deferred frees after GPU finishes reading.
    pub fn tick(&mut self) {
        self.pending_free.retain_mut(|(slot, frames)| {
            if *frames == 0 {
                self.free_slots.push(*slot);
                false
            } else {
                *frames -= 1;
                true
            }
        });
    }

    pub fn has_tile(&self, pos: &[i32; 2]) -> bool {
        self.tiles.contains_key(pos)
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Returns draw info for all visible tiles that pass frustum culling.
    /// Computes per-tile morph_factor based on distance to band boundary.
    pub fn visible_tiles(
        &self,
        frustum: &crate::graphical_core::frustum::Frustum,
        player_cx: i32,
        player_cz: i32,
        bands: &[(i32, i32)], // (min_dist, max_dist) per band index
    ) -> Vec<TileDrawInfo> {
        let mut visible = Vec::new();
        for (&_pos, &slot) in &self.tiles {
            if let Some(info) = &self.tile_info[slot as usize] {
                let min = Vec3::new(info.aabb_min[0], info.aabb_min[1], info.aabb_min[2]);
                let max = Vec3::new(info.aabb_max[0], info.aabb_max[1], info.aabb_max[2]);
                if !frustum.intersects_aabb(min, max) {
                    continue;
                }
                let mut tile = *info;
                // Compute morph_factor: 0 at inner edge, 1 at outer edge of band
                let dist = (tile.pos[0] - player_cx).abs().max((tile.pos[1] - player_cz).abs());
                if let Some(&(band_min, band_max)) = bands.get(tile.lod as usize) {
                    let range = (band_max - band_min).max(1) as f32;
                    let t = (dist - band_min) as f32 / range;
                    tile.morph_factor = t.clamp(0.0, 1.0);
                }
                visible.push(tile);
            }
        }
        visible
    }

    /// Evict the farthest tiles to make room.
    pub unsafe fn evict_farthest(&mut self, count: usize, player_cx: i32, player_cz: i32) {
        let mut positions: Vec<[i32; 2]> = self.tiles.keys().copied().collect();
        positions.sort_by_key(|pos| std::cmp::Reverse((pos[0] - player_cx).abs().max((pos[1] - player_cz).abs())));
        for pos in positions.into_iter().take(count) {
            self.remove_tile(&pos);
        }
    }

    /// Remove all tiles beyond max_dist from the player (chunk coords).
    pub unsafe fn evict_out_of_range(&mut self, player_cx: i32, player_cz: i32, max_dist: i32) {
        let to_remove: Vec<[i32; 2]> = self
            .tiles
            .keys()
            .filter(|pos| (pos[0] - player_cx).abs().max((pos[1] - player_cz).abs()) > max_dist)
            .copied()
            .collect();
        for pos in to_remove {
            self.remove_tile(&pos);
        }
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        device.unmap_memory(self.vertex_memory);
        device.destroy_buffer(self.vertex_buffer, None);
        device.free_memory(self.vertex_memory, None);

        device.unmap_memory(self.index_memory);
        device.destroy_buffer(self.index_buffer, None);
        device.free_memory(self.index_memory, None);
    }

    fn alloc_slot(&mut self) -> u32 {
        if let Some(slot) = self.free_slots.pop() {
            slot
        } else {
            let slot = self.next_slot;
            assert!(slot < MAX_TILES, "HeightmapPool: exceeded max tile count");
            self.next_slot += 1;
            slot
        }
    }
}
