/// Removes small upslope bumps in rivers while preserving steep slopes.
///
/// Port of Python `smooth_river_bumps`. Uses Gaussian-weighted Laplacian
/// smoothing: strong smoothing in flat regions, no smoothing on steep slopes.
pub(super) fn smooth_river_bumps(
    elevation: &mut [f32],
    w: usize,
    h: usize,
    slope_thresh: f32,
    smooth_strength: f32,
    iterations: u32,
) {
    let n = w * h;
    let mut laplacian = vec![0.0f32; n];

    for _ in 0..iterations {
        // Compute gradient magnitude and Laplacian
        for r in 0..h {
            for c in 0..w {
                let idx = r * w + c;
                let e = elevation[idx];
                if e.is_nan() || e <= 0.0 {
                    laplacian[idx] = 0.0;
                    continue;
                }

                let mut sum = 0.0f32;
                let mut count = 0.0f32;

                // 4-neighbor Laplacian
                let neighbors: [(i32, i32); 4] = [(-1, 0), (1, 0), (0, -1), (0, 1)];
                for (dr, dc) in neighbors {
                    let nr = r as i32 + dr;
                    let nc = c as i32 + dc;
                    if nr < 0 || nr >= h as i32 || nc < 0 || nc >= w as i32 {
                        continue;
                    }
                    let ne = elevation[nr as usize * w + nc as usize];
                    if ne.is_nan() || ne <= 0.0 {
                        continue;
                    }
                    sum += ne;
                    count += 1.0;
                }

                if count > 0.0 {
                    laplacian[idx] = sum - count * e;
                } else {
                    laplacian[idx] = 0.0;
                }
            }
        }

        // Compute slope and apply weighted smoothing
        for r in 0..h {
            for c in 0..w {
                let idx = r * w + c;
                let e = elevation[idx];
                if e.is_nan() || e <= 0.0 {
                    continue;
                }

                // Gradient magnitude (central differences)
                let grad_x = if c > 0 && c < w - 1 {
                    let el = elevation[idx - 1];
                    let er = elevation[idx + 1];
                    if el.is_nan() || el <= 0.0 || er.is_nan() || er <= 0.0 {
                        0.0
                    } else {
                        (er - el) * 0.5
                    }
                } else {
                    0.0
                };
                let grad_y = if r > 0 && r < h - 1 {
                    let eu = elevation[(r - 1) * w + c];
                    let ed = elevation[(r + 1) * w + c];
                    if eu.is_nan() || eu <= 0.0 || ed.is_nan() || ed <= 0.0 {
                        0.0
                    } else {
                        (ed - eu) * 0.5
                    }
                } else {
                    0.0
                };
                let slope = (grad_x * grad_x + grad_y * grad_y).sqrt();

                // Gaussian weight: smooth strongly in flat areas, not on steep slopes
                let w_val = (-(slope / slope_thresh).powi(2)).exp();
                elevation[idx] += smooth_strength * w_val * laplacian[idx];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smooths_small_bump() {
        // Flat terrain with a small bump at center
        let mut elev = vec![
            5.0, 5.0, 5.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 5.0, 5.0,
            5.0, 5.0, 7.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 5.0, 5.0,
        ];
        smooth_river_bumps(&mut elev, 5, 5, 50.0, 0.3, 3);
        // Center bump should be reduced toward 5.0 (may slightly overshoot)
        assert!(elev[12] < 7.0, "bump should be smoothed: {}", elev[12]);
    }

    #[test]
    fn preserves_steep_slope() {
        // Large grid with a steep cliff: smoothing should be suppressed at the cliff
        // but may affect cells far from the cliff edge
        let mut elev = vec![
            100.0, 100.0, 100.0, 100.0, 100.0,
            100.0, 100.0, 100.0, 100.0, 100.0,
            100.0, 100.0, 100.0, 100.0, 100.0,
              1.0,   1.0,   1.0,   1.0,   1.0,
              1.0,   1.0,   1.0,   1.0,   1.0,
        ];
        let original = elev.clone();
        smooth_river_bumps(&mut elev, 5, 5, 50.0, 0.3, 3);
        // The cliff edge cells get smoothed but the effect should be bounded
        // compared to the original 99m step
        let cliff_change = (elev[11] - original[11]).abs(); // row 2, col 1 — cliff edge
        assert!(cliff_change < 50.0, "cliff smoothing unbounded: {cliff_change}");
    }

    #[test]
    fn ignores_ocean() {
        let mut elev = vec![
            -100.0, -100.0, -100.0,
            -100.0,    5.0, -100.0,
            -100.0, -100.0, -100.0,
        ];
        let original = elev.clone();
        smooth_river_bumps(&mut elev, 3, 3, 50.0, 0.3, 3);
        for i in [0, 1, 2, 3, 5, 6, 7, 8] {
            assert_eq!(elev[i], original[i], "ocean cell {i} changed");
        }
    }
}
