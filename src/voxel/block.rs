#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BlockType {
    Air,
    Grass,
    Dirt,
    Stone,
    Water,
    Sand,
    Snow,
    Gravel,
}

impl BlockType {
    pub fn is_opaque(self) -> bool {
        // Water is rendered as a solid opaque block for now (its own texture,
        // no transparency). Real water shading is a separate followup.
        !matches!(self, BlockType::Air)
    }

    #[allow(dead_code)] // needed for block placement
    pub fn is_transparent(self) -> bool {
        false
    }

    #[allow(dead_code)] // needed for block placement
    pub fn material_id(self) -> u8 {
        self as u8
    }

    /// Bitmask where bit `i` is set if block type `i` is opaque.
    /// Used by mesh shaders for GPU-side face culling.
    #[allow(dead_code)] // Used in Phase 2 (mesh shader pipeline)
    pub fn opaque_mask() -> u32 {
        let mut mask = 0u32;
        for i in 0..=7u8 {
            // Safety: repr(u8) guarantees valid transmute for 0..=7
            let block: BlockType = unsafe { std::mem::transmute(i) };
            if block.is_opaque() {
                mask |= 1 << i;
            }
        }
        mask
    }
}
