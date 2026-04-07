#![allow(dead_code)] // Wired up incrementally
use crate::graphical_core::buffers::allocate_buffer;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::voxel::heightmap_generator::{angular_distance, HeightmapTileMesh, HeightmapVertex, TileKey};
use glam::{DVec3, Vec3};
use std::collections::HashMap;
use vulkan_rust::{vk, Device, Instance};

/// Maximum number of heightmap tiles in the pool.
const MAX_TILES: u32 = 350;
/// Max vertices per tile (129x129 grid posts for HM-A).
const MAX_VERTICES_PER_TILE: usize = 129 * 129;
/// Max indices per tile (128x128 quads * 6 indices).
const MAX_INDICES_PER_TILE: usize = 128 * 128 * 6;

/// Per-tile CPU metadata for issuing draw calls. Vertices already live in
/// world cartesian (the CPU mesh is curved), so the GPU AABB is the curved
/// AABB and the draw needs no per-tile face id.
#[derive(Copy, Clone, Debug)]
pub struct TileDrawInfo {
    pub aabb_min: [f32; 3],
    pub aabb_max: [f32; 3],
    pub vertex_offset: u32,
    pub index_offset: u32,
    pub index_count: u32,
    pub lod: u8,
    pub key: TileKey,
    pub center: [f32; 3],
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

    tiles: HashMap<TileKey, u32>,
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

    pub unsafe fn upload_tile(&mut self, mesh: &HeightmapTileMesh, player_world: DVec3) {
        if self.free_slots.is_empty() && self.next_slot >= MAX_TILES {
            self.evict_farthest(16, player_world);
        }
        let slot = self.alloc_slot();
        let vertex_offset = slot as usize * MAX_VERTICES_PER_TILE;
        let index_offset = slot as usize * MAX_INDICES_PER_TILE;

        assert!(mesh.vertices.len() <= MAX_VERTICES_PER_TILE);
        assert!(mesh.indices.len() <= MAX_INDICES_PER_TILE);

        std::ptr::copy_nonoverlapping(mesh.vertices.as_ptr(), self.vertex_ptr.add(vertex_offset), mesh.vertices.len());
        std::ptr::copy_nonoverlapping(mesh.indices.as_ptr(), self.index_ptr.add(index_offset), mesh.indices.len());

        let center = [
            (mesh.aabb_min[0] + mesh.aabb_max[0]) * 0.5,
            (mesh.aabb_min[1] + mesh.aabb_max[1]) * 0.5,
            (mesh.aabb_min[2] + mesh.aabb_max[2]) * 0.5,
        ];

        self.tile_info[slot as usize] = Some(TileDrawInfo {
            aabb_min: mesh.aabb_min,
            aabb_max: mesh.aabb_max,
            vertex_offset: vertex_offset as u32,
            index_offset: index_offset as u32,
            index_count: mesh.indices.len() as u32,
            lod: mesh.lod,
            key: mesh.key,
            center,
            morph_factor: 0.0,
        });
        self.tiles.insert(mesh.key, slot);
    }

    pub unsafe fn remove_tile(&mut self, key: &TileKey) {
        if let Some(slot) = self.tiles.remove(key) {
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

    pub fn has_tile(&self, key: &TileKey) -> bool {
        self.tiles.contains_key(key)
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }

    /// Returns draw info for all visible tiles that pass frustum culling.
    /// Computes per-tile morph_factor based on angular distance to band edge.
    pub fn visible_tiles(
        &self,
        frustum: &crate::graphical_core::frustum::Frustum,
        player_world: DVec3,
        bands: &[(f32, f32)], // (min_angle, max_angle) in radians per band index
    ) -> Vec<TileDrawInfo> {
        let mut visible = Vec::new();
        for (_key, &slot) in &self.tiles {
            if let Some(info) = &self.tile_info[slot as usize] {
                let min = Vec3::new(info.aabb_min[0], info.aabb_min[1], info.aabb_min[2]);
                let max = Vec3::new(info.aabb_max[0], info.aabb_max[1], info.aabb_max[2]);
                if !frustum.intersects_aabb(min, max) {
                    continue;
                }
                let mut tile = *info;
                let center_w = DVec3::new(tile.center[0] as f64, tile.center[1] as f64, tile.center[2] as f64);
                let dist = angular_distance(player_world, center_w);
                if let Some(&(band_min, band_max)) = bands.get(tile.lod as usize) {
                    let range = (band_max - band_min).max(1e-6);
                    let t = (dist - band_min) / range;
                    tile.morph_factor = t.clamp(0.0, 1.0);
                }
                visible.push(tile);
            }
        }
        visible
    }

    /// Evict the farthest tiles (by angular distance to player) to make room.
    pub unsafe fn evict_farthest(&mut self, count: usize, player_world: DVec3) {
        let mut by_dist: Vec<(TileKey, f32)> = self
            .tiles
            .iter()
            .filter_map(|(&key, &slot)| {
                self.tile_info[slot as usize]
                    .as_ref()
                    .map(|info| (key, angular_distance(player_world, DVec3::new(info.center[0] as f64, info.center[1] as f64, info.center[2] as f64))))
            })
            .collect();
        by_dist.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for (key, _) in by_dist.into_iter().take(count) {
            self.remove_tile(&key);
        }
    }

    /// Remove all tiles beyond `max_angle` (radians) from the player.
    pub unsafe fn evict_out_of_range(&mut self, player_world: DVec3, max_angle: f32) {
        let to_remove: Vec<TileKey> = self
            .tiles
            .iter()
            .filter_map(|(&key, &slot)| {
                let info = self.tile_info[slot as usize].as_ref()?;
                let center_w = DVec3::new(info.center[0] as f64, info.center[1] as f64, info.center[2] as f64);
                if angular_distance(player_world, center_w) > max_angle {
                    Some(key)
                } else {
                    None
                }
            })
            .collect();
        for key in to_remove {
            self.remove_tile(&key);
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
