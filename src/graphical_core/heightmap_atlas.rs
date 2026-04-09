//! GPU heightmap atlas: a single R32_SFLOAT 2D image holding one
//! `HEIGHT_PAGE_SIZE × HEIGHT_PAGE_SIZE` page per resident quadtree leaf.
//!
//! Layout: `ATLAS_COLS × ATLAS_ROWS` pages packed into a single image. Page
//! `p` lives at texel offset `(p % ATLAS_COLS, p / ATLAS_COLS) * HEIGHT_PAGE_SIZE`.
//! The CPU side owns a host-visible staging buffer of the same byte size as
//! the image; `insert` writes new heights into the staging slot and queues
//! a copy. `record_uploads` records `cmd_copy_buffer_to_image` for every
//! pending page, surrounded by the layout transitions needed to make the
//! atlas sample-able again.

#![allow(dead_code)] // Wired up incrementally across phases.

use crate::graphical_core::buffers::allocate_buffer;
use crate::graphical_core::memory::find_memory_type;
use crate::graphical_core::vulkan_object::VulkanApplicationData;
use crate::voxel::heightmap_quadtree::{HEIGHT_PAGE_SIZE, MAX_RESIDENT_TILES};
use std::collections::HashMap;
use vulkan_rust::{vk, Device, Instance};

/// One height per post on a `HEIGHT_PAGE_SIZE × HEIGHT_PAGE_SIZE` grid.
pub const FLOATS_PER_PAGE: usize = (HEIGHT_PAGE_SIZE as usize) * (HEIGHT_PAGE_SIZE as usize);
pub const BYTES_PER_PAGE: usize = FLOATS_PER_PAGE * 4;

/// Atlas grid: how many pages tile across one row of the image. The
/// remainder lives in subsequent rows. Chosen as the ceiling square root
/// of `MAX_RESIDENT_TILES` so the image stays roughly square.
pub const ATLAS_COLS: u32 = 32;
pub const ATLAS_ROWS: u32 = (MAX_RESIDENT_TILES as u32 + ATLAS_COLS - 1) / ATLAS_COLS;
/// Number of pages addressable by the atlas. Sized to the quadtree's
/// `MAX_RESIDENT_TILES` working set.
pub const ATLAS_PAGE_COUNT: usize = (ATLAS_COLS * ATLAS_ROWS) as usize;

const _: () = assert!(ATLAS_PAGE_COUNT >= MAX_RESIDENT_TILES);

pub const ATLAS_WIDTH: u32 = ATLAS_COLS * HEIGHT_PAGE_SIZE;
pub const ATLAS_HEIGHT: u32 = ATLAS_ROWS * HEIGHT_PAGE_SIZE;
pub const ATLAS_FORMAT: vk::Format = vk::Format::R32_SFLOAT;
pub const MATERIAL_ATLAS_FORMAT: vk::Format = vk::Format::R8_UINT;

/// One material id (u8) per post.
pub const MATERIAL_BYTES_PER_PAGE: usize = FLOATS_PER_PAGE; // 65*65 = 4225 bytes

/// Sentinel meaning "no page assigned to this morton".
pub const NO_PAGE: u32 = u32::MAX;

/// Pure-CPU LRU page allocator. Owns the morton↔page mapping that the
/// GPU atlas wraps. Lives in its own struct so it can be unit-tested
/// without standing up Vulkan resources.
pub struct PageAllocator {
    free_pages: Vec<u32>,
    morton_to_page: HashMap<u64, u32>,
    page_to_morton: Vec<Option<u64>>,
    lru: std::collections::VecDeque<u64>,
}

impl PageAllocator {
    pub fn new() -> Self {
        let mut free_pages: Vec<u32> = (0..ATLAS_PAGE_COUNT as u32).rev().collect();
        free_pages.shrink_to_fit();
        Self {
            free_pages,
            morton_to_page: HashMap::with_capacity(ATLAS_PAGE_COUNT),
            page_to_morton: vec![None; ATLAS_PAGE_COUNT],
            lru: std::collections::VecDeque::with_capacity(ATLAS_PAGE_COUNT),
        }
    }

    pub fn resident(&self) -> usize { self.morton_to_page.len() }
    pub fn page_of(&self, morton: u64) -> Option<u32> { self.morton_to_page.get(&morton).copied() }

    /// Returns `(page_index, evicted_morton)`. `evicted_morton` is `Some`
    /// when an LRU eviction happened to make room for `morton`.
    pub fn alloc(&mut self, morton: u64) -> (u32, Option<u64>) {
        if let Some(&page) = self.morton_to_page.get(&morton) {
            self.touch(morton);
            return (page, None);
        }
        let mut evicted = None;
        let page = match self.free_pages.pop() {
            Some(p) => p,
            None => {
                let m = self.lru.pop_front().expect("LRU empty but no free pages");
                let p = self.morton_to_page.remove(&m).expect("LRU referenced unknown morton");
                self.page_to_morton[p as usize] = None;
                evicted = Some(m);
                p
            }
        };
        self.morton_to_page.insert(morton, page);
        self.page_to_morton[page as usize] = Some(morton);
        self.lru.push_back(morton);
        (page, evicted)
    }

    pub fn free(&mut self, morton: u64) -> Option<u32> {
        let page = self.morton_to_page.remove(&morton)?;
        self.page_to_morton[page as usize] = None;
        if let Some(idx) = self.lru.iter().position(|&m| m == morton) {
            self.lru.remove(idx);
        }
        self.free_pages.push(page);
        Some(page)
    }

    fn touch(&mut self, morton: u64) {
        if let Some(idx) = self.lru.iter().position(|&m| m == morton) {
            self.lru.remove(idx);
        }
        self.lru.push_back(morton);
    }
}

impl Default for PageAllocator {
    fn default() -> Self { Self::new() }
}

pub struct HeightmapAtlas {
    // Height atlas (R32_SFLOAT).
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub sampler: vk::Sampler,
    image_memory: vk::DeviceMemory,
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    staging_ptr: *mut f32,

    // Material atlas (R8_UINT).
    pub mat_image: vk::Image,
    pub mat_image_view: vk::ImageView,
    pub mat_sampler: vk::Sampler,
    mat_image_memory: vk::DeviceMemory,
    mat_staging_buffer: vk::Buffer,
    mat_staging_memory: vk::DeviceMemory,
    mat_staging_ptr: *mut u8,

    /// Pages with new contents written to staging since the last upload
    /// flush. `record_uploads` drains this.
    pending: Vec<u32>,
    /// True before the first `record_uploads` call — the image starts in
    /// `UNDEFINED` and the first transition must come from `UNDEFINED`
    /// rather than `SHADER_READ_ONLY_OPTIMAL`.
    initial_layout_done: bool,
    allocator: PageAllocator,
}

unsafe impl Send for HeightmapAtlas {}
unsafe impl Sync for HeightmapAtlas {}

impl HeightmapAtlas {
    pub unsafe fn new(device: &Device, instance: &Instance, data: &mut VulkanApplicationData) -> anyhow::Result<Self> {
        let (image, image_memory) = create_atlas_image(device, instance, data, ATLAS_FORMAT)?;
        let image_view = create_atlas_view(device, image, ATLAS_FORMAT)?;
        let sampler = create_atlas_sampler(device, vk::Filter::LINEAR)?;

        let staging_size = ATLAS_PAGE_COUNT as u64 * BYTES_PER_PAGE as u64;
        let (staging_buffer, staging_memory, staging_ptr) = allocate_buffer::<f32>(
            staging_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            device,
            instance,
            data,
            super::host_visible_coherent(),
        )?;

        // Material atlas (R8_UINT, NEAREST sampling).
        let (mat_image, mat_image_memory) = create_atlas_image(device, instance, data, MATERIAL_ATLAS_FORMAT)?;
        let mat_image_view = create_atlas_view(device, mat_image, MATERIAL_ATLAS_FORMAT)?;
        let mat_sampler = create_atlas_sampler(device, vk::Filter::NEAREST)?;

        let mat_staging_size = ATLAS_PAGE_COUNT as u64 * MATERIAL_BYTES_PER_PAGE as u64;
        let (mat_staging_buffer, mat_staging_memory, mat_staging_ptr) = allocate_buffer::<u8>(
            mat_staging_size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            device,
            instance,
            data,
            super::host_visible_coherent(),
        )?;

        Ok(Self {
            image,
            image_view,
            sampler,
            image_memory,
            staging_buffer,
            staging_memory,
            staging_ptr,
            mat_image,
            mat_image_view,
            mat_sampler,
            mat_image_memory,
            mat_staging_buffer,
            mat_staging_memory,
            mat_staging_ptr,
            pending: Vec::with_capacity(ATLAS_PAGE_COUNT),
            initial_layout_done: false,
            allocator: PageAllocator::new(),
        })
    }

    pub unsafe fn destroy(&mut self, device: &Device) {
        device.destroy_sampler(self.sampler, None);
        device.destroy_image_view(self.image_view, None);
        device.destroy_image(self.image, None);
        device.free_memory(self.image_memory, None);
        device.unmap_memory(self.staging_memory);
        device.destroy_buffer(self.staging_buffer, None);
        device.free_memory(self.staging_memory, None);

        device.destroy_sampler(self.mat_sampler, None);
        device.destroy_image_view(self.mat_image_view, None);
        device.destroy_image(self.mat_image, None);
        device.free_memory(self.mat_image_memory, None);
        device.unmap_memory(self.mat_staging_memory);
        device.destroy_buffer(self.mat_staging_buffer, None);
        device.free_memory(self.mat_staging_memory, None);
    }

    pub fn resident(&self) -> usize { self.allocator.resident() }
    pub fn page_of(&self, morton: u64) -> Option<u32> { self.allocator.page_of(morton) }

    /// Insert a heights + material page for `morton`. Allocates a page
    /// (evicting LRU if full), writes into both staging buffers, and
    /// queues a GPU upload for the next `record_uploads` call.
    pub fn insert(&mut self, morton: u64, heights: &[f32], materials: &[u8]) -> u32 {
        assert_eq!(heights.len(), FLOATS_PER_PAGE);
        assert_eq!(materials.len(), MATERIAL_BYTES_PER_PAGE);
        let (page, _evicted) = self.allocator.alloc(morton);
        unsafe {
            self.write_staging(page, heights);
            self.write_mat_staging(page, materials);
        }
        self.pending.push(page);
        page
    }

    pub fn evict(&mut self, morton: u64) -> Option<u32> {
        self.allocator.free(morton)
    }

    /// Record `cmd_copy_buffer_to_image` for every page modified since the
    /// last call. Brackets the copies with the layout transitions needed
    /// to land both atlases back in `SHADER_READ_ONLY_OPTIMAL`.
    pub unsafe fn record_uploads(&mut self, device: &Device, cmd: vk::CommandBuffer) {
        if self.pending.is_empty() && self.initial_layout_done {
            return;
        }
        let old_layout = if self.initial_layout_done {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else {
            vk::ImageLayout::UNDEFINED
        };
        let (src_access, src_stage) = if self.initial_layout_done {
            (vk::AccessFlags::SHADER_READ, vk::PipelineStageFlags::FRAGMENT_SHADER)
        } else {
            (vk::AccessFlags::empty(), vk::PipelineStageFlags::TOP_OF_PIPE)
        };
        // Transition both atlases to TRANSFER_DST.
        transition_image(device, cmd, self.image,
            old_layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            src_access, vk::AccessFlags::TRANSFER_WRITE,
            src_stage, vk::PipelineStageFlags::TRANSFER);
        transition_image(device, cmd, self.mat_image,
            old_layout, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            src_access, vk::AccessFlags::TRANSFER_WRITE,
            src_stage, vk::PipelineStageFlags::TRANSFER);
        for &page in &self.pending {
            let (px, py) = page_origin_texels(page);
            let subresource = *vk::ImageSubresourceLayers::builder()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1);
            let offset = vk::Offset3D { x: px as i32, y: py as i32, z: 0 };
            let extent = vk::Extent3D { width: HEIGHT_PAGE_SIZE, height: HEIGHT_PAGE_SIZE, depth: 1 };
            // Height page.
            let h_region = *vk::BufferImageCopy::builder()
                .buffer_offset(page as u64 * BYTES_PER_PAGE as u64)
                .image_subresource(subresource)
                .image_offset(offset)
                .image_extent(extent);
            device.cmd_copy_buffer_to_image(cmd, self.staging_buffer, self.image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[h_region]);
            // Material page.
            let m_region = *vk::BufferImageCopy::builder()
                .buffer_offset(page as u64 * MATERIAL_BYTES_PER_PAGE as u64)
                .image_subresource(subresource)
                .image_offset(offset)
                .image_extent(extent);
            device.cmd_copy_buffer_to_image(cmd, self.mat_staging_buffer, self.mat_image, vk::ImageLayout::TRANSFER_DST_OPTIMAL, &[m_region]);
        }
        // Transition both atlases back to SHADER_READ_ONLY.
        transition_image(device, cmd, self.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER);
        transition_image(device, cmd, self.mat_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::TRANSFER_WRITE, vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::TRANSFER, vk::PipelineStageFlags::FRAGMENT_SHADER);
        self.pending.clear();
        self.initial_layout_done = true;
    }

    unsafe fn write_staging(&mut self, page: u32, heights: &[f32]) {
        let dst = self.staging_ptr.add(page as usize * FLOATS_PER_PAGE);
        std::ptr::copy_nonoverlapping(heights.as_ptr(), dst, FLOATS_PER_PAGE);
    }

    unsafe fn write_mat_staging(&mut self, page: u32, materials: &[u8]) {
        let dst = self.mat_staging_ptr.add(page as usize * MATERIAL_BYTES_PER_PAGE);
        std::ptr::copy_nonoverlapping(materials.as_ptr(), dst, MATERIAL_BYTES_PER_PAGE);
    }
}

fn page_origin_texels(page: u32) -> (u32, u32) {
    let px = page % ATLAS_COLS;
    let py = page / ATLAS_COLS;
    (px * HEIGHT_PAGE_SIZE, py * HEIGHT_PAGE_SIZE)
}

unsafe fn create_atlas_image(device: &Device, instance: &Instance, data: &VulkanApplicationData, format: vk::Format) -> anyhow::Result<(vk::Image, vk::DeviceMemory)> {
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .format(format)
        .extent(vk::Extent3D { width: ATLAS_WIDTH, height: ATLAS_HEIGHT, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = device.create_image(&info, None)?;
    let req = device.get_image_memory_requirements(image);
    let props = instance.get_physical_device_memory_properties(data.physical_device);
    let mem_type = find_memory_type(&props, req.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    let alloc = vk::MemoryAllocateInfo::builder().allocation_size(req.size).memory_type_index(mem_type);
    let memory = device.allocate_memory(&alloc, None)?;
    device.bind_image_memory(image, memory, 0)?;
    Ok((image, memory))
}

unsafe fn create_atlas_view(device: &Device, image: vk::Image, format: vk::Format) -> anyhow::Result<vk::ImageView> {
    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .format(format)
        .view_type(vk::ImageViewType::_2D)
        .subresource_range(*vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1));
    Ok(device.create_image_view(&info, None)?)
}

unsafe fn create_atlas_sampler(device: &Device, filter: vk::Filter) -> anyhow::Result<vk::Sampler> {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(filter)
        .min_filter(filter)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .anisotropy_enable(false)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK)
        .compare_enable(false)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST);
    Ok(device.create_sampler(&info, None)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_allocator_has_no_residents() {
        let a = PageAllocator::new();
        assert_eq!(a.resident(), 0);
        assert_eq!(a.page_of(42), None);
    }

    #[test]
    fn alloc_and_lookup_round_trip() {
        let mut a = PageAllocator::new();
        let (idx, evicted) = a.alloc(123);
        assert_eq!(evicted, None);
        assert_eq!(a.page_of(123), Some(idx));
        assert_eq!(a.resident(), 1);
    }

    #[test]
    fn re_alloc_existing_morton_returns_same_page() {
        let mut a = PageAllocator::new();
        let (idx1, _) = a.alloc(7);
        let (idx2, evicted) = a.alloc(7);
        assert_eq!(idx1, idx2);
        assert_eq!(evicted, None);
        assert_eq!(a.resident(), 1);
    }

    #[test]
    fn free_returns_page_for_reuse() {
        let mut a = PageAllocator::new();
        let (idx, _) = a.alloc(99);
        let freed = a.free(99).unwrap();
        assert_eq!(freed, idx);
        assert_eq!(a.page_of(99), None);
        let (idx2, _) = a.alloc(100);
        assert_eq!(idx2, freed);
    }

    #[test]
    fn full_allocator_evicts_lru_on_alloc() {
        let mut a = PageAllocator::new();
        for i in 0..ATLAS_PAGE_COUNT as u64 {
            a.alloc(i);
        }
        assert_eq!(a.resident(), ATLAS_PAGE_COUNT);
        // Touch morton 5 to bump it to MRU.
        a.alloc(5);
        // Insert a new morton — the LRU (morton 0) gets evicted.
        let (_, evicted) = a.alloc(9999);
        assert_eq!(evicted, Some(0));
        assert!(a.page_of(0).is_none());
        assert!(a.page_of(5).is_some());
        assert!(a.page_of(9999).is_some());
    }

    #[test]
    fn pages_are_independent_indices() {
        let mut a = PageAllocator::new();
        let (i1, _) = a.alloc(1);
        let (i2, _) = a.alloc(2);
        assert_ne!(i1, i2);
    }
}

unsafe fn transition_image(
    device: &Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
) {
    let barrier = vk::ImageMemoryBarrier::builder()
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(*vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1))
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED);
    device.cmd_pipeline_barrier(
        cmd, src_stage, dst_stage, vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[*barrier],
    );
}
