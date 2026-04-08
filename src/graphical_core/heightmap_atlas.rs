//! CPU page allocator and storage backing the heightmap atlas.
//!
//! Phase 2: a pure-Rust LRU page allocator that owns the height grid for
//! every resident quadtree leaf. Each page is `HEIGHT_PAGE_SIZE²` floats
//! encoding the surface offset above `PLANET_RADIUS_BLOCKS` (positive
//! upward, clamped to `±MAX_TERRAIN_AMPLITUDE`).
//!
//! Phase 3 will wrap this with a `VkImage` and a staging buffer; the page
//! index will become the (x, y) position in a single 2D texture atlas
//! laid out as `ATLAS_COLS × ATLAS_ROWS` pages, so existing CPU code keeps
//! the same `u32` page index API. The `Vec<f32>` per-page storage will
//! disappear once the GPU image is the source of truth.

#![allow(dead_code)] // Wired up incrementally across phases.

use crate::voxel::heightmap_quadtree::{HEIGHT_PAGE_SIZE, MAX_RESIDENT_TILES};
use std::collections::HashMap;

/// One height per post on a `HEIGHT_PAGE_SIZE × HEIGHT_PAGE_SIZE` grid.
pub const FLOATS_PER_PAGE: usize = (HEIGHT_PAGE_SIZE as usize) * (HEIGHT_PAGE_SIZE as usize);

/// Number of pages addressable by the atlas. Sized to the quadtree's
/// `MAX_RESIDENT_TILES` so a fully-populated working set always fits.
pub const ATLAS_PAGE_COUNT: usize = MAX_RESIDENT_TILES;

/// Sentinel index meaning "no page assigned to this morton".
pub const NO_PAGE: u32 = u32::MAX;

/// Pure-CPU heightmap atlas. Owns the height storage and an LRU page
/// allocator. The Phase 3 Vulkan wrapper holds a `VkImage` whose contents
/// mirror this struct's `pages` array.
pub struct HeightmapAtlas {
    /// Flat backing storage. Page `i` lives in `pages[i*FLOATS_PER_PAGE..(i+1)*FLOATS_PER_PAGE]`.
    pages: Vec<f32>,
    /// Free page indices, popped on alloc.
    free_pages: Vec<u32>,
    /// Morton ↔ page index. Both directions are needed: morton→page for
    /// "where do I draw this tile from", page→morton for eviction lookup.
    morton_to_page: HashMap<u64, u32>,
    page_to_morton: Vec<Option<u64>>,
    /// Most-recently-used queue, oldest at the front. Mortons get pushed
    /// onto the back on every touch; eviction picks from the front.
    lru: std::collections::VecDeque<u64>,
}

impl HeightmapAtlas {
    pub fn new() -> Self {
        let mut free_pages: Vec<u32> = (0..ATLAS_PAGE_COUNT as u32).rev().collect();
        free_pages.shrink_to_fit();
        Self {
            pages: vec![0.0; ATLAS_PAGE_COUNT * FLOATS_PER_PAGE],
            free_pages,
            morton_to_page: HashMap::with_capacity(ATLAS_PAGE_COUNT),
            page_to_morton: vec![None; ATLAS_PAGE_COUNT],
            lru: std::collections::VecDeque::with_capacity(ATLAS_PAGE_COUNT),
        }
    }

    /// Number of pages currently holding tile data.
    pub fn resident(&self) -> usize {
        self.morton_to_page.len()
    }

    pub fn page_of(&self, morton: u64) -> Option<u32> {
        self.morton_to_page.get(&morton).copied()
    }

    /// Insert a heights page for `morton`. If the morton is already
    /// resident the existing page is overwritten in place. Otherwise a
    /// page is allocated; if the atlas is full, the least-recently-used
    /// resident page is evicted to make room.
    ///
    /// Panics if `heights.len() != FLOATS_PER_PAGE`.
    pub fn insert(&mut self, morton: u64, heights: &[f32]) -> u32 {
        assert_eq!(
            heights.len(),
            FLOATS_PER_PAGE,
            "heights page must be exactly {} floats",
            FLOATS_PER_PAGE
        );
        if let Some(&page) = self.morton_to_page.get(&morton) {
            self.write_page(page, heights);
            self.touch_lru(morton);
            return page;
        }
        let page = match self.free_pages.pop() {
            Some(p) => p,
            None => {
                let evict = self.lru.pop_front().expect("LRU empty but no free pages");
                let p = self
                    .morton_to_page
                    .remove(&evict)
                    .expect("LRU referenced an unknown morton");
                self.page_to_morton[p as usize] = None;
                p
            }
        };
        self.morton_to_page.insert(morton, page);
        self.page_to_morton[page as usize] = Some(morton);
        self.lru.push_back(morton);
        self.write_page(page, heights);
        page
    }

    /// Free the page belonging to `morton`, if any. Returns the freed
    /// page index for the caller to log/track.
    pub fn evict(&mut self, morton: u64) -> Option<u32> {
        let page = self.morton_to_page.remove(&morton)?;
        self.page_to_morton[page as usize] = None;
        // Remove from LRU. Linear scan is fine — the queue is bounded by
        // ATLAS_PAGE_COUNT (~1024) and eviction is rare.
        if let Some(idx) = self.lru.iter().position(|&m| m == morton) {
            self.lru.remove(idx);
        }
        self.free_pages.push(page);
        Some(page)
    }

    /// Read-only view of one page's heights.
    pub fn page_heights(&self, page: u32) -> &[f32] {
        let start = page as usize * FLOATS_PER_PAGE;
        &self.pages[start..start + FLOATS_PER_PAGE]
    }

    fn write_page(&mut self, page: u32, heights: &[f32]) {
        let start = page as usize * FLOATS_PER_PAGE;
        self.pages[start..start + FLOATS_PER_PAGE].copy_from_slice(heights);
    }

    fn touch_lru(&mut self, morton: u64) {
        if let Some(idx) = self.lru.iter().position(|&m| m == morton) {
            self.lru.remove(idx);
        }
        self.lru.push_back(morton);
    }
}

impl Default for HeightmapAtlas {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_page(seed: f32) -> Vec<f32> {
        (0..FLOATS_PER_PAGE).map(|i| seed + i as f32 * 0.1).collect()
    }

    #[test]
    fn empty_atlas_has_no_residents() {
        let a = HeightmapAtlas::new();
        assert_eq!(a.resident(), 0);
        assert_eq!(a.page_of(42), None);
    }

    #[test]
    fn insert_and_lookup_round_trip() {
        let mut a = HeightmapAtlas::new();
        let p = fake_page(7.0);
        let idx = a.insert(123, &p);
        assert_eq!(a.page_of(123), Some(idx));
        assert_eq!(a.page_heights(idx), p.as_slice());
        assert_eq!(a.resident(), 1);
    }

    #[test]
    fn re_insert_overwrites_in_place() {
        let mut a = HeightmapAtlas::new();
        let idx1 = a.insert(7, &fake_page(1.0));
        let idx2 = a.insert(7, &fake_page(2.0));
        assert_eq!(idx1, idx2);
        assert_eq!(a.page_heights(idx1)[0], 2.0);
        assert_eq!(a.resident(), 1);
    }

    #[test]
    fn evict_returns_page_to_free_list() {
        let mut a = HeightmapAtlas::new();
        let idx = a.insert(99, &fake_page(0.0));
        let freed = a.evict(99).unwrap();
        assert_eq!(freed, idx);
        assert_eq!(a.page_of(99), None);
        // Next insert should reuse the freed page.
        let idx2 = a.insert(100, &fake_page(0.0));
        assert_eq!(idx2, freed);
    }

    #[test]
    fn full_atlas_evicts_lru_on_insert() {
        let mut a = HeightmapAtlas::new();
        // Fill it up.
        for i in 0..ATLAS_PAGE_COUNT as u64 {
            a.insert(i, &fake_page(i as f32));
        }
        assert_eq!(a.resident(), ATLAS_PAGE_COUNT);
        // Touch morton 5 to bump it to MRU.
        a.insert(5, &fake_page(5.0));
        // Insert a new morton — the LRU (morton 0, since 5 was just touched) gets evicted.
        a.insert(9999, &fake_page(0.0));
        assert_eq!(a.resident(), ATLAS_PAGE_COUNT);
        assert!(a.page_of(0).is_none(), "expected morton 0 to be evicted");
        assert!(a.page_of(5).is_some(), "expected morton 5 to survive (was touched)");
        assert!(a.page_of(9999).is_some());
    }

    #[test]
    fn pages_are_independent() {
        let mut a = HeightmapAtlas::new();
        let p1 = fake_page(1.0);
        let p2 = fake_page(2.0);
        let i1 = a.insert(1, &p1);
        let i2 = a.insert(2, &p2);
        assert_ne!(i1, i2);
        assert_eq!(a.page_heights(i1), p1.as_slice());
        assert_eq!(a.page_heights(i2), p2.as_slice());
    }
}
