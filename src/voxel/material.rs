/// GPU-side material palette entry. Must match the GLSL `MaterialEntry` layout (std140).
///
/// Each `vec3` is followed by a `float` to avoid the std140 padding trap where
/// a standalone `vec3` rounds up to 16 bytes.
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct MaterialEntry {
    pub color: [f32; 3],
    pub roughness: f32,
    pub emissive: [f32; 3],
    pub _padding: f32,
}

pub const PALETTE_SIZE: usize = 256;

#[repr(C)]
#[derive(Copy, Clone)]
pub struct MaterialPalette {
    pub entries: [MaterialEntry; PALETTE_SIZE],
}

const EMPTY_ENTRY: MaterialEntry = MaterialEntry {
    color: [0.0, 0.0, 0.0],
    roughness: 1.0,
    emissive: [0.0, 0.0, 0.0],
    _padding: 0.0,
};

pub fn default_palette() -> MaterialPalette {
    let mut entries = [EMPTY_ENTRY; PALETTE_SIZE];

    // Index 0: Air (unused, but zero-initialized for safety)
    // Index 1: Grass
    entries[1] = MaterialEntry {
        color: [0.3, 0.7, 0.2],
        roughness: 0.9,
        ..EMPTY_ENTRY
    };
    // Index 2: Dirt
    entries[2] = MaterialEntry {
        color: [0.55, 0.35, 0.15],
        roughness: 0.95,
        ..EMPTY_ENTRY
    };
    // Index 3: Stone
    entries[3] = MaterialEntry {
        color: [0.5, 0.5, 0.5],
        roughness: 0.8,
        ..EMPTY_ENTRY
    };

    MaterialPalette { entries }
}
