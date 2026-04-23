/// Depth of valley carved per stream order level (meters).
const VALLEY_DEPTH_PER_ORDER: f32 = 30.0;

/// Half-width of valley per stream order level (pixels).
const VALLEY_HALFWIDTH_PER_ORDER: f32 = 3.0;

/// Maximum valley carve depth (meters).
const MAX_VALLEY_DEPTH: f32 = 200.0;

/// Carve river valleys into the elevation grid.
///
/// For each river pixel (stream_order > 0), lower the elevation and
/// spread the carving laterally with a parabolic cross-section profile.
/// Higher stream order = wider and deeper valley.
pub(super) fn carve(
    elevation: &mut [f32],
    stream_order: &[u8],
    accumulation: &[f32],
    w: usize,
    h: usize,
    river_threshold: f32,
) {
    let n = w * h;

    // Pre-compute carve depth at each river pixel
    let mut carve_map = vec![0.0f32; n];

    for i in 0..n {
        let order = stream_order[i];
        if order == 0 || accumulation[i] < river_threshold {
            continue;
        }
        if elevation[i] <= 0.0 || elevation[i].is_nan() {
            continue;
        }

        let depth = (VALLEY_DEPTH_PER_ORDER * order as f32).min(MAX_VALLEY_DEPTH);
        let halfwidth = (VALLEY_HALFWIDTH_PER_ORDER * order as f32).max(1.0) as i32;

        let r = (i / w) as i32;
        let c = (i % w) as i32;

        // Spread carving laterally with parabolic profile
        for dr in -halfwidth..=halfwidth {
            for dc in -halfwidth..=halfwidth {
                let nr = r + dr;
                let nc = c + dc;
                if nr < 0 || nr >= h as i32 || nc < 0 || nc >= w as i32 {
                    continue;
                }
                let ni = nr as usize * w + nc as usize;

                if elevation[ni] <= 0.0 || elevation[ni].is_nan() {
                    continue;
                }

                let dist = ((dr * dr + dc * dc) as f32).sqrt();
                let hw = halfwidth as f32;
                if dist > hw {
                    continue;
                }

                // Parabolic profile: deepest at center, zero at edge
                let t = dist / hw;
                let profile = (1.0 - t * t).max(0.0);
                let carve_depth = depth * profile;

                if carve_depth > carve_map[ni] {
                    carve_map[ni] = carve_depth;
                }
            }
        }
    }

    // Apply carving — never push below sea level
    for i in 0..n {
        if carve_map[i] > 0.0 {
            let new_elev = (elevation[i] - carve_map[i]).max(0.5);
            if new_elev < elevation[i] {
                elevation[i] = new_elev;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn carves_river_valley() {
        // 21x9 terrain with a river down the center (col 10)
        let w = 21;
        let h = 9;
        let mut elev = vec![100.0f32; w * h];
        let mut stream_order = vec![0u8; w * h];
        let mut accum = vec![0.0f32; w * h];

        // River down center column with order 2 (halfwidth = 6)
        for r in 0..h {
            let idx = r * w + 10;
            stream_order[idx] = 2;
            accum[idx] = 100.0;
        }

        let original = elev.clone();
        carve(&mut elev, &stream_order, &accum, w, h, 50.0);

        // Center should be carved deepest
        assert!(elev[4 * w + 10] < original[4 * w + 10], "center should be carved");

        // Adjacent should be carved less
        assert!(elev[4 * w + 9] < original[4 * w + 9], "adjacent should be carved");
        assert!(elev[4 * w + 10] < elev[4 * w + 9], "center deeper than adjacent");

        // Far columns (>halfwidth away) should be untouched
        assert_eq!(elev[4 * w + 0], original[4 * w + 0], "far column should be untouched");
    }

    #[test]
    fn does_not_carve_below_sea_level() {
        let w = 5;
        let h = 1;
        let mut elev = vec![10.0, 10.0, 10.0, 10.0, 10.0];
        let stream_order = vec![0, 0, 5, 0, 0]; // high order in center
        let accum = vec![0.0, 0.0, 200.0, 0.0, 0.0];

        carve(&mut elev, &stream_order, &accum, w, h, 50.0);

        // Should not go below 0.5
        for &e in &elev {
            assert!(e >= 0.5, "carved below minimum: {e}");
        }
    }
}
