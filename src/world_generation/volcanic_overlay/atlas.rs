use std::path::Path;

const ATLAS_PATH: &str = "data/volcanic_atlas.bin";
const PATCH_SIZE: usize = 256;
const META_SIZE: usize = 16;

/// Per-patch metadata from the atlas header.
pub(super) struct PatchMeta {
    pub min_elev: i16,
    pub max_elev: i16,
    pub mean_elev: i16,
    pub land_pct: u8,
    pub valid: bool,
    pub base_level: i16,
    pub peak_height: i16,
}

/// Loaded volcanic atlas: 4 categories of 256x256 i16 elevation patches.
pub(super) struct VolcanicAtlas {
    pub n_categories: u32,
    pub patches_per_cat: u32,
    pub patch_size: u32,
    pub meta: Vec<PatchMeta>,
    /// Raw i16 elevation data, row-major per patch.
    pub pixels: Vec<i16>,
}

impl VolcanicAtlas {
    pub fn load() -> Result<Self, String> {
        Self::load_from(Path::new(ATLAS_PATH))
    }

    pub fn load_from(path: &Path) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        if data.len() < 20 {
            return Err("file too small".into());
        }
        if &data[0..4] != b"VOLC" {
            return Err(format!("bad magic: {:?}", &data[0..4]));
        }

        let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
        if version != 1 {
            return Err(format!("unsupported version {version}"));
        }

        let n_categories = u32::from_le_bytes(data[8..12].try_into().unwrap());
        let patches_per_cat = u32::from_le_bytes(data[12..16].try_into().unwrap());
        let patch_size = u32::from_le_bytes(data[16..20].try_into().unwrap());

        if patch_size as usize != PATCH_SIZE {
            return Err(format!("expected patch_size={PATCH_SIZE}, got {patch_size}"));
        }

        let total_patches = (n_categories * patches_per_cat) as usize;
        let meta_offset = 20;
        let pixel_offset = meta_offset + total_patches * META_SIZE;
        let expected_pixels = total_patches * PATCH_SIZE * PATCH_SIZE;
        let expected_size = pixel_offset + expected_pixels * 2;

        if data.len() < expected_size {
            return Err(format!(
                "file too small: {} < {expected_size}",
                data.len()
            ));
        }

        // Parse metadata
        let mut meta = Vec::with_capacity(total_patches);
        for i in 0..total_patches {
            let off = meta_offset + i * META_SIZE;
            let min_elev = i16::from_le_bytes(data[off..off + 2].try_into().unwrap());
            let max_elev = i16::from_le_bytes(data[off + 2..off + 4].try_into().unwrap());
            let mean_elev = i16::from_le_bytes(data[off + 4..off + 6].try_into().unwrap());
            let land_pct = data[off + 6];
            let valid = data[off + 7] != 0;
            let base_level = i16::from_le_bytes(data[off + 8..off + 10].try_into().unwrap());
            let peak_height = i16::from_le_bytes(data[off + 10..off + 12].try_into().unwrap());

            meta.push(PatchMeta {
                min_elev,
                max_elev,
                mean_elev,
                land_pct,
                valid,
                base_level,
                peak_height,
            });
        }

        // Parse pixel data
        let mut pixels = vec![0i16; expected_pixels];
        for i in 0..expected_pixels {
            let off = pixel_offset + i * 2;
            pixels[i] = i16::from_le_bytes(data[off..off + 2].try_into().unwrap());
        }

        eprintln!(
            "[volcanic_atlas] loaded: {} categories × {} patches, {}×{} pixels",
            n_categories, patches_per_cat, patch_size, patch_size
        );

        Ok(Self {
            n_categories,
            patches_per_cat,
            patch_size,
            meta,
            pixels,
        })
    }

    /// Get the pixel data for a specific patch. Returns a PATCH_SIZE×PATCH_SIZE slice.
    pub fn patch_pixels(&self, category: u32, patch_idx: u32) -> &[i16] {
        let idx = (category * self.patches_per_cat + patch_idx) as usize;
        let ppx = PATCH_SIZE * PATCH_SIZE;
        &self.pixels[idx * ppx..(idx + 1) * ppx]
    }

    /// Get metadata for a specific patch.
    pub fn patch_meta(&self, category: u32, patch_idx: u32) -> &PatchMeta {
        let idx = (category * self.patches_per_cat + patch_idx) as usize;
        &self.meta[idx]
    }
}
