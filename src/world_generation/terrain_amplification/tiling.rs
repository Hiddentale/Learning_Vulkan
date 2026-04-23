/// Tile iteration and overlap blending for the diffusion pipeline.
/// Tiles overlap at edges; a linear weight window ensures smooth blending
/// where multiple tiles contribute to the same output pixel.

/// Grid of accumulated tile outputs with weight tracking for overlap blending.
pub(super) struct BlendGrid {
    pub data: Vec<f32>,
    pub weights: Vec<f32>,
    pub channels: u32,
    pub width: u32,
    pub height: u32,
}

impl BlendGrid {
    pub fn new(channels: u32, width: u32, height: u32) -> Self {
        let size = (channels * width * height) as usize;
        Self {
            data: vec![0.0; size],
            weights: vec![0.0; (width * height) as usize],
            channels,
            width,
            height,
        }
    }

    /// Blend a tile into the grid at position (tx, ty) using the weight window.
    pub fn blend_tile(
        &mut self,
        tile: &[f32],
        tile_size: u32,
        tx: u32,
        ty: u32,
        window: &[f32],
    ) {
        for r in 0..tile_size {
            let oy = ty + r;
            if oy >= self.height {
                break;
            }
            for c in 0..tile_size {
                let ox = tx + c;
                if ox >= self.width {
                    break;
                }
                let w = window[(r * tile_size + c) as usize];
                let out_idx = (oy * self.width + ox) as usize;
                self.weights[out_idx] += w;

                for ch in 0..self.channels {
                    let tile_idx = (ch * tile_size * tile_size + r * tile_size + c) as usize;
                    let grid_idx =
                        (ch * self.width * self.height + oy * self.width + ox) as usize;
                    self.data[grid_idx] += tile[tile_idx] * w;
                }
            }
        }
    }

    /// Divide accumulated values by weights to produce the final blended output.
    pub fn finalize(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let idx = (y * self.width + x) as usize;
                let w = self.weights[idx];
                if w > 0.0 {
                    for ch in 0..self.channels {
                        let i = (ch * self.width * self.height + y * self.width + x) as usize;
                        self.data[i] /= w;
                    }
                }
            }
        }
    }

    /// Extract a single channel as a flat Vec.
    pub fn channel(&self, ch: u32) -> Vec<f32> {
        let start = (ch * self.width * self.height) as usize;
        let end = start + (self.width * self.height) as usize;
        self.data[start..end].to_vec()
    }
}

/// Linear weight window: triangular ramp from edges to center.
/// Produces a size×size grid where edge pixels have weight ≈ eps and center ≈ 1.
pub(super) fn linear_weight_window(size: u32) -> Vec<f32> {
    let mid = (size - 1) as f32 / 2.0;
    let eps = 1e-3;
    let mut window = vec![0.0f32; (size * size) as usize];
    for r in 0..size {
        let wy = 1.0 - (1.0 - eps) * ((r as f32 - mid).abs() / mid).min(1.0);
        for c in 0..size {
            let wx = 1.0 - (1.0 - eps) * ((c as f32 - mid).abs() / mid).min(1.0);
            window[(r * size + c) as usize] = wy * wx;
        }
    }
    window
}

/// Iterate tile positions for a given output size, tile size, and stride.
pub(super) fn tile_positions(output_size: u32, tile_size: u32, stride: u32) -> Vec<(u32, u32)> {
    let mut positions = Vec::new();
    let mut y = 0u32;
    loop {
        let mut x = 0u32;
        loop {
            positions.push((x, y));
            if x + tile_size >= output_size {
                break;
            }
            x += stride;
        }
        if y + tile_size >= output_size {
            break;
        }
        y += stride;
    }
    positions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_window_center_is_max() {
        let w = linear_weight_window(64);
        let center = w[(32 * 64 + 32) as usize];
        let corner = w[0];
        assert!(center > 0.95, "center weight {center}");
        assert!(corner < 0.01, "corner weight {corner}");
    }

    #[test]
    fn weight_window_symmetric() {
        let w = linear_weight_window(16);
        for r in 0..16u32 {
            for c in 0..16u32 {
                let mirror_r = 15 - r;
                let mirror_c = 15 - c;
                let a = w[(r * 16 + c) as usize];
                let b = w[(mirror_r * 16 + mirror_c) as usize];
                assert!((a - b).abs() < 1e-6, "asymmetric at ({r},{c})");
            }
        }
    }

    #[test]
    fn tile_positions_covers_output() {
        let positions = tile_positions(256, 64, 48);
        // With stride 48 and tile 64: 0, 48, 96, 144, 192 = 5 per axis
        assert!(positions.len() >= 25);
        // Last tile should reach the end
        let (last_x, last_y) = positions.last().unwrap();
        assert!(last_x + 64 >= 256);
        assert!(last_y + 64 >= 256);
    }

    #[test]
    fn blend_grid_single_tile_identity() {
        let mut grid = BlendGrid::new(1, 4, 4);
        let tile: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let window = vec![1.0f32; 16];
        grid.blend_tile(&tile, 4, 0, 0, &window);
        grid.finalize();
        assert_eq!(grid.channel(0), tile);
    }

    #[test]
    fn blend_grid_overlap_averages() {
        let mut grid = BlendGrid::new(1, 4, 4);
        let window = vec![1.0f32; 4];
        // Two 2×2 tiles overlapping at column 1
        let tile_a = vec![1.0, 2.0, 3.0, 4.0];
        let tile_b = vec![10.0, 20.0, 30.0, 40.0];
        grid.blend_tile(&tile_a, 2, 0, 0, &window);
        grid.blend_tile(&tile_b, 2, 1, 0, &window);
        grid.finalize();
        let data = grid.channel(0);
        // (0,0)=1.0 (only tile_a), (1,0)=(2+10)/2=6.0 (overlap)
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 6.0).abs() < 1e-6);
    }
}
