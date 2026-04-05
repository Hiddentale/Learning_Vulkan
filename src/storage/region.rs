use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

/// Chunks per region axis (8x8x8 = 512 super-chunks per region).
const REGION_SIZE: i32 = 8;
const ENTRIES_PER_REGION: usize = (REGION_SIZE * REGION_SIZE * REGION_SIZE) as usize;
/// On-disk sector size for alignment.
const SECTOR_SIZE: usize = 4096;
/// Header: one Entry per slot.
const HEADER_BYTES: usize = ENTRIES_PER_REGION * std::mem::size_of::<EntryHeader>();

/// On-disk entry in the region header. 8 bytes per slot.
#[repr(C)]
#[derive(Copy, Clone, Default)]
struct EntryHeader {
    /// Byte offset from file start. 0 = empty slot.
    offset: u32,
    /// Actual data length in bytes.
    length: u32,
}

struct MappedRegion {
    mmap: Mmap,
    headers: [EntryHeader; ENTRIES_PER_REGION],
}

/// Manages a directory of region files for SVDAG cache persistence.
/// Each region covers 8x8x8 super-chunk positions.
pub struct RegionStore {
    dir: PathBuf,
    regions: HashMap<[i32; 3], MappedRegion>,
}

impl RegionStore {
    pub fn new(dir: &Path) -> anyhow::Result<Self> {
        fs::create_dir_all(dir)?;
        Ok(Self {
            dir: dir.to_path_buf(),
            regions: HashMap::new(),
        })
    }

    /// Read a cached super-chunk SVDAG blob. Returns None if not cached.
    pub fn read(&mut self, pos: [i32; 3]) -> Option<&[u8]> {
        let (region_coord, local) = split_coord(pos);
        self.ensure_region_loaded(region_coord);
        let region = self.regions.get(&region_coord)?;
        let entry = &region.headers[local];
        if entry.offset == 0 || entry.length == 0 {
            return None;
        }
        let off = entry.offset as usize;
        let len = entry.length as usize;
        if off + len > region.mmap.len() {
            return None;
        }
        Some(&region.mmap[off..off + len])
    }

    /// Write a super-chunk SVDAG blob to the region file.
    /// Data is LZ4-compressed before calling this.
    pub fn write(&mut self, pos: [i32; 3], data: &[u8]) -> anyhow::Result<()> {
        let (region_coord, local) = split_coord(pos);
        let path = self.region_path(region_coord);

        // Append data to file, update header
        let mut file = OpenOptions::new().read(true).write(true).create(true).truncate(false).open(&path)?;

        let file_len = file.metadata()?.len();
        let data_offset = if file_len < HEADER_BYTES as u64 {
            // New file — write zeroed header
            let header_pad = vec![0u8; HEADER_BYTES];
            file.seek(SeekFrom::Start(0))?;
            file.write_all(&header_pad)?;
            align_to_sector(HEADER_BYTES) as u64
        } else {
            align_to_sector(file_len as usize) as u64
        };

        // Write data at sector-aligned offset
        file.seek(SeekFrom::Start(data_offset))?;
        file.write_all(data)?;

        // Pad to sector boundary
        let padded_end = align_to_sector(data_offset as usize + data.len());
        let pad_bytes = padded_end - (data_offset as usize + data.len());
        if pad_bytes > 0 {
            file.write_all(&vec![0u8; pad_bytes])?;
        }

        // Update header entry
        let entry = EntryHeader {
            offset: data_offset as u32,
            length: data.len() as u32,
        };
        let entry_offset = local * std::mem::size_of::<EntryHeader>();
        file.seek(SeekFrom::Start(entry_offset as u64))?;
        let entry_bytes: &[u8] = unsafe { std::slice::from_raw_parts(&entry as *const EntryHeader as *const u8, std::mem::size_of::<EntryHeader>()) };
        file.write_all(entry_bytes)?;

        // Invalidate old mmap so next read picks up the new data
        self.regions.remove(&region_coord);

        Ok(())
    }

    /// Check if a super-chunk is cached without reading the data.
    #[allow(dead_code)]
    pub fn has(&mut self, pos: [i32; 3]) -> bool {
        let (region_coord, local) = split_coord(pos);
        self.ensure_region_loaded(region_coord);
        match self.regions.get(&region_coord) {
            Some(region) => region.headers[local].offset != 0 && region.headers[local].length != 0,
            None => false,
        }
    }

    fn ensure_region_loaded(&mut self, region_coord: [i32; 3]) {
        if self.regions.contains_key(&region_coord) {
            return;
        }
        let path = self.region_path(region_coord);
        if !path.exists() {
            return;
        }
        if let Ok(region) = load_region(&path) {
            self.regions.insert(region_coord, region);
        }
    }

    fn region_path(&self, coord: [i32; 3]) -> PathBuf {
        self.dir.join(format!("r.{}.{}.{}.bin", coord[0], coord[1], coord[2]))
    }
}

fn load_region(path: &Path) -> anyhow::Result<MappedRegion> {
    let file = File::open(path)?;
    let mmap = unsafe { Mmap::map(&file)? };
    if mmap.len() < HEADER_BYTES {
        anyhow::bail!("Region file too small");
    }
    let mut headers = [EntryHeader::default(); ENTRIES_PER_REGION];
    let header_bytes = &mmap[..HEADER_BYTES];
    unsafe {
        std::ptr::copy_nonoverlapping(header_bytes.as_ptr(), headers.as_mut_ptr() as *mut u8, HEADER_BYTES);
    }
    Ok(MappedRegion { mmap, headers })
}

fn split_coord(pos: [i32; 3]) -> ([i32; 3], usize) {
    let region = [
        pos[0].div_euclid(REGION_SIZE),
        pos[1].div_euclid(REGION_SIZE),
        pos[2].div_euclid(REGION_SIZE),
    ];
    let local = [
        pos[0].rem_euclid(REGION_SIZE) as usize,
        pos[1].rem_euclid(REGION_SIZE) as usize,
        pos[2].rem_euclid(REGION_SIZE) as usize,
    ];
    let idx = local[0] + local[1] * REGION_SIZE as usize + local[2] * (REGION_SIZE * REGION_SIZE) as usize;
    (region, idx)
}

fn align_to_sector(n: usize) -> usize {
    (n + SECTOR_SIZE - 1) & !(SECTOR_SIZE - 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use std::sync::atomic::{AtomicU32, Ordering};
    static TEST_ID: AtomicU32 = AtomicU32::new(0);

    fn temp_dir() -> PathBuf {
        let id = TEST_ID.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("manifold_test_{}_{}", std::process::id(), id));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn write_read_roundtrip() {
        let dir = temp_dir();
        let mut store = RegionStore::new(&dir).unwrap();
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        store.write([0, 0, 0], &data).unwrap();
        let read = store.read([0, 0, 0]).unwrap();
        assert_eq!(read, &data);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_missing_returns_none() {
        let dir = temp_dir();
        let mut store = RegionStore::new(&dir).unwrap();
        assert!(store.read([5, 5, 5]).is_none());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn multiple_chunks_in_same_region() {
        let dir = temp_dir();
        let mut store = RegionStore::new(&dir).unwrap();
        store.write([0, 0, 0], &[10, 20]).unwrap();
        store.write([1, 0, 0], &[30, 40, 50]).unwrap();
        store.write([0, 1, 0], &[60]).unwrap();
        assert_eq!(store.read([0, 0, 0]).unwrap(), &[10, 20]);
        assert_eq!(store.read([1, 0, 0]).unwrap(), &[30, 40, 50]);
        assert_eq!(store.read([0, 1, 0]).unwrap(), &[60]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn chunks_in_different_regions() {
        let dir = temp_dir();
        let mut store = RegionStore::new(&dir).unwrap();
        store.write([0, 0, 0], &[1, 2]).unwrap();
        store.write([8, 0, 0], &[3, 4]).unwrap(); // different region
        assert_eq!(store.read([0, 0, 0]).unwrap(), &[1, 2]);
        assert_eq!(store.read([8, 0, 0]).unwrap(), &[3, 4]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn negative_coordinates() {
        let dir = temp_dir();
        let mut store = RegionStore::new(&dir).unwrap();
        store.write([-4, -8, -12], &[99]).unwrap();
        assert_eq!(store.read([-4, -8, -12]).unwrap(), &[99]);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn has_returns_correct_state() {
        let dir = temp_dir();
        let mut store = RegionStore::new(&dir).unwrap();
        assert!(!store.has([0, 0, 0]));
        store.write([0, 0, 0], &[1]).unwrap();
        assert!(store.has([0, 0, 0]));
        assert!(!store.has([1, 0, 0]));
        let _ = fs::remove_dir_all(&dir);
    }
}
