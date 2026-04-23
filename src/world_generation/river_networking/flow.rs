const D8_DR: [i32; 8] = [-1, 1, 0, 0, -1, -1, 1, 1];
const D8_DC: [i32; 8] = [0, 0, -1, 1, -1, 1, -1, 1];
const D8_DIST: [f32; 8] = [1.0, 1.0, 1.0, 1.0, 1.4142, 1.4142, 1.4142, 1.4142];

/// Compute D8 flow direction for each pixel.
/// Returns (flow_dir, is_sink) where flow_dir[i] is 0-7 neighbor index.
pub(super) fn d8_flow(elevation: &[f32], w: usize, h: usize) -> (Vec<u8>, Vec<bool>) {
    let n = w * h;
    let mut flow_dir = vec![u8::MAX; n];
    let mut is_sink = vec![false; n];

    for r in 0..h {
        for c in 0..w {
            let idx = r * w + c;
            let e = elevation[idx];

            if e.is_nan() || e <= 0.0 {
                is_sink[idx] = true;
                continue;
            }

            let mut best_k = u8::MAX;
            let mut best_slope = f32::NEG_INFINITY;

            for k in 0..8u8 {
                let nr = r as i32 + D8_DR[k as usize];
                let nc = c as i32 + D8_DC[k as usize];
                if nr < 0 || nr >= h as i32 || nc < 0 || nc >= w as i32 {
                    continue;
                }
                let nidx = nr as usize * w + nc as usize;
                let ne = elevation[nidx];

                // Prefer routing into ocean
                if ne.is_nan() || ne <= 0.0 {
                    best_k = k;
                    best_slope = f32::INFINITY;
                    break;
                }

                let slope = (e - ne) / D8_DIST[k as usize];
                if slope > best_slope {
                    best_slope = slope;
                    best_k = k;
                }
            }

            if best_slope <= 0.0 {
                is_sink[idx] = true;
            } else {
                flow_dir[idx] = best_k;
            }
        }
    }

    (flow_dir, is_sink)
}

/// Compute flow accumulation by processing cells from high to low.
pub(super) fn accumulate(
    elevation: &[f32],
    flow_dir: &[u8],
    is_sink: &[bool],
    w: usize,
    h: usize,
) -> Vec<f32> {
    let n = w * h;
    let mut accum = vec![0.0f32; n];

    // Collect valid (land) cells
    let mut order: Vec<usize> = (0..n)
        .filter(|&i| !elevation[i].is_nan() && elevation[i] > 0.0)
        .collect();

    // Sort descending by elevation
    order.sort_by(|&a, &b| {
        elevation[b]
            .partial_cmp(&elevation[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Initialize valid cells with 1.0
    for &i in &order {
        accum[i] = 1.0;
    }

    // Route flow downhill
    for &i in &order {
        if is_sink[i] || flow_dir[i] == u8::MAX {
            continue;
        }

        let r = i / w;
        let c = i % w;
        let k = flow_dir[i] as usize;
        let nr = (r as i32 + D8_DR[k]) as usize;
        let nc = (c as i32 + D8_DC[k]) as usize;
        let ni = nr * w + nc;

        accum[ni] += accum[i];
    }

    accum
}

/// Strahler stream ordering.
/// Headwaters = 1. When two streams of order N merge, the result is order N+1.
/// When streams of different order merge, the result keeps the higher order.
/// Only assigns orders to pixels with accumulation above `river_threshold`.
pub(super) fn strahler_order(
    flow_dir: &[u8],
    is_sink: &[bool],
    accumulation: &[f32],
    elevation: &[f32],
    w: usize,
    h: usize,
    river_threshold: f32,
) -> Vec<u8> {
    let n = w * h;
    let mut order = vec![0u8; n];

    // Count how many upstream river pixels flow into each pixel
    let mut in_count = vec![0u32; n];
    for i in 0..n {
        if flow_dir[i] == u8::MAX || is_sink[i] {
            continue;
        }
        if accumulation[i] < river_threshold {
            continue;
        }
        let r = i / w;
        let c = i % w;
        let k = flow_dir[i] as usize;
        let nr = (r as i32 + D8_DR[k]) as usize;
        let nc = (c as i32 + D8_DC[k]) as usize;
        let ni = nr * w + nc;
        in_count[ni] += 1;
    }

    // Process from high to low elevation (headwaters first)
    let mut river_cells: Vec<usize> = (0..n)
        .filter(|&i| accumulation[i] >= river_threshold && elevation[i] > 0.0 && !elevation[i].is_nan())
        .collect();
    river_cells.sort_by(|&a, &b| {
        elevation[b]
            .partial_cmp(&elevation[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Assign headwater order
    for &i in &river_cells {
        if in_count[i] == 0 {
            order[i] = 1;
        }
    }

    // Propagate downstream: track max and second-max upstream order per pixel
    let mut max_upstream = vec![0u8; n];
    let mut second_upstream = vec![0u8; n];
    let mut upstream_processed = vec![0u32; n];

    for &i in &river_cells {
        if order[i] == 0 && upstream_processed[i] == in_count[i] && in_count[i] > 0 {
            // All upstream processed — compute order from upstream orders
            if max_upstream[i] == second_upstream[i] && max_upstream[i] > 0 {
                order[i] = max_upstream[i] + 1; // two equal-order streams merge
            } else {
                order[i] = max_upstream[i]; // keep highest
            }
        }
        if order[i] == 0 {
            order[i] = 1; // fallback for unresolved
        }

        // Propagate to downstream
        if flow_dir[i] == u8::MAX || is_sink[i] {
            continue;
        }
        let r = i / w;
        let c = i % w;
        let k = flow_dir[i] as usize;
        let nr = (r as i32 + D8_DR[k]) as usize;
        let nc = (c as i32 + D8_DC[k]) as usize;
        let ni = nr * w + nc;

        if accumulation[ni] >= river_threshold {
            upstream_processed[ni] += 1;
            let o = order[i];
            if o >= max_upstream[ni] {
                second_upstream[ni] = max_upstream[ni];
                max_upstream[ni] = o;
            } else if o > second_upstream[ni] {
                second_upstream[ni] = o;
            }
        }
    }

    order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_slope_flows_downhill() {
        // 3x3 grid sloping from top-left to bottom-right
        let elev = vec![
            9.0, 8.0, 7.0,
            6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        ];
        let (flow_dir, is_sink) = d8_flow(&elev, 3, 3);

        // Bottom-right (1.0) is the lowest land cell
        // It should be a sink or flow to edge
        assert!(flow_dir[0] != u8::MAX, "top-left should flow somewhere");
        assert!(is_sink[8] || flow_dir[8] != u8::MAX); // lowest cell
    }

    #[test]
    fn ocean_cells_are_sinks() {
        let elev = vec![
            -100.0, 5.0,
            -100.0, 3.0,
        ];
        let (_, is_sink) = d8_flow(&elev, 2, 2);
        assert!(is_sink[0], "ocean should be sink");
        assert!(is_sink[2], "ocean should be sink");
    }

    #[test]
    fn accumulation_increases_downstream() {
        // Simple 1D slope
        let elev = vec![
            5.0, 4.0, 3.0, 2.0, 1.0,
        ];
        let (flow_dir, is_sink) = d8_flow(&elev, 5, 1);
        let accum = accumulate(&elev, &flow_dir, &is_sink, 5, 1);
        // Each cell flows right, so accumulation should increase
        assert!(accum[4] >= accum[0], "downstream should have more accumulation");
    }

    #[test]
    fn strahler_headwaters_are_order_1() {
        let elev = vec![
            5.0, 4.0, 3.0, 2.0, 1.0,
        ];
        let (flow_dir, is_sink) = d8_flow(&elev, 5, 1);
        let accum = accumulate(&elev, &flow_dir, &is_sink, 5, 1);
        let order = strahler_order(&flow_dir, &is_sink, &accum, &elev, 5, 1, 1.0);
        // First cell has no upstream — should be order 1
        assert_eq!(order[0], 1, "headwater should be order 1");
        // Downstream cells should be >= 1
        for &o in &order {
            if o > 0 {
                assert!(o >= 1);
            }
        }
    }
}
