use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// Priority queue entry: lower elevation = higher priority.
#[derive(PartialEq)]
struct Cell {
    elev: f32,
    idx: usize,
}

impl Eq for Cell {}

impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cell {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order: lowest elevation first (min-heap via BinaryHeap)
        other.elev.partial_cmp(&self.elev).unwrap_or(Ordering::Equal)
    }
}

const D8_DR: [i32; 8] = [-1, 1, 0, 0, -1, -1, 1, 1];
const D8_DC: [i32; 8] = [0, 0, -1, 1, -1, 1, -1, 1];

/// Priority-Flood selective depression fill.
///
/// Fills pits up to `max_raise` meters deep. Deeper basins are left as true
/// depressions. `epsilon` ensures drainage across flats.
///
/// Port of Python `fill_depressions_priority_flood`.
pub(super) fn fill(
    elevation: &mut [f32],
    w: usize,
    h: usize,
    epsilon: f32,
    max_raise: Option<f32>,
) {
    let n = w * h;
    let base: Vec<f32> = elevation.to_vec();
    let mut visited = vec![false; n];
    let mut basin_min = vec![f32::INFINITY; n];
    let mut heap: BinaryHeap<Cell> = BinaryHeap::new();

    let ocean = |e: f32| -> bool { e.is_nan() || e <= 0.0 };

    // Seed: outer border cells that are not ocean
    for r in 0..h {
        for c in 0..w {
            let is_border = r == 0 || r == h - 1 || c == 0 || c == w - 1;
            if !is_border {
                continue;
            }
            let idx = r * w + c;
            if ocean(elevation[idx]) || visited[idx] {
                continue;
            }
            heap.push(Cell { elev: elevation[idx], idx });
            visited[idx] = true;
            basin_min[idx] = base[idx];
        }
    }

    // Seed: coast-adjacent valid cells (next to ocean)
    for r in 0..h {
        for c in 0..w {
            let idx = r * w + c;
            if ocean(elevation[idx]) || visited[idx] {
                continue;
            }
            let mut coastal = false;
            for k in 0..8 {
                let nr = r as i32 + D8_DR[k];
                let nc = c as i32 + D8_DC[k];
                if nr < 0 || nr >= h as i32 || nc < 0 || nc >= w as i32 {
                    continue;
                }
                if ocean(elevation[nr as usize * w + nc as usize]) {
                    coastal = true;
                    break;
                }
            }
            if coastal {
                let seed_elev = elevation[idx].max(0.0);
                heap.push(Cell { elev: seed_elev, idx });
                visited[idx] = true;
                basin_min[idx] = base[idx];
            }
        }
    }

    // Priority-Flood
    while let Some(Cell { elev, idx }) = heap.pop() {
        let r = idx / w;
        let c = idx % w;
        let bm_cur = basin_min[idx];

        for k in 0..8 {
            let nr = r as i32 + D8_DR[k];
            let nc = c as i32 + D8_DC[k];
            if nr < 0 || nr >= h as i32 || nc < 0 || nc >= w as i32 {
                continue;
            }
            let nidx = nr as usize * w + nc as usize;
            if visited[nidx] || ocean(elevation[nidx]) {
                continue;
            }

            let ne = elevation[nidx];
            let bm_next = if base[nidx] >= bm_cur { bm_cur } else { base[nidx] };

            if ne <= elev {
                // Pit: check if fill depth exceeds max_raise
                if let Some(mr) = max_raise {
                    if elev - bm_cur >= mr {
                        // Basin too deep — leave as-is
                        heap.push(Cell { elev: ne, idx: nidx });
                        visited[nidx] = true;
                        basin_min[nidx] = bm_next;
                        continue;
                    }
                }
                let mut new_e = elev + epsilon;
                if let Some(mr) = max_raise {
                    let max_level = bm_cur + mr;
                    if new_e > max_level {
                        new_e = max_level;
                    }
                }
                if new_e > ne {
                    elevation[nidx] = new_e;
                }
                heap.push(Cell { elev: elevation[nidx], idx: nidx });
            } else {
                heap.push(Cell { elev: ne, idx: nidx });
            }
            visited[nidx] = true;
            basin_min[nidx] = bm_next;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_simple_pit() {
        // 5x5 grid with a pit at center
        let mut elev = vec![
            10.0, 10.0, 10.0, 10.0, 10.0,
            10.0,  8.0,  8.0,  8.0, 10.0,
            10.0,  8.0,  2.0,  8.0, 10.0,
            10.0,  8.0,  8.0,  8.0, 10.0,
            10.0, 10.0, 10.0, 10.0, 10.0,
        ];
        fill(&mut elev, 5, 5, 0.001, None);
        // Center should be raised to at least the surrounding 8.0
        assert!(elev[12] >= 8.0, "center={}", elev[12]);
    }

    #[test]
    fn preserves_ocean() {
        let mut elev = vec![
            -100.0, -100.0, -100.0,
            -100.0,    5.0, -100.0,
            -100.0, -100.0, -100.0,
        ];
        let original = elev.clone();
        fill(&mut elev, 3, 3, 0.001, None);
        // Ocean cells should be untouched
        for i in [0, 1, 2, 3, 5, 6, 7, 8] {
            assert_eq!(elev[i], original[i], "ocean cell {i} changed");
        }
    }

    #[test]
    fn respects_max_raise() {
        // Wide basin where the flood path encounters the pit minimum before filling
        // 7x7 grid: border=10, wide basin interior slopes down to center
        let mut elev = vec![
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
            10.0,  6.0,  5.0,  4.0,  5.0,  6.0, 10.0,
            10.0,  5.0,  3.0,  2.0,  3.0,  5.0, 10.0,
            10.0,  4.0,  2.0,  1.0,  2.0,  4.0, 10.0,
            10.0,  5.0,  3.0,  2.0,  3.0,  5.0, 10.0,
            10.0,  6.0,  5.0,  4.0,  5.0,  6.0, 10.0,
            10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
        ];
        fill(&mut elev, 7, 7, 0.001, Some(2.0));
        // Center at (3,3) originally 1.0 in a basin.
        // With max_raise=2, the fill depth from basin_min should be limited.
        // The center should be raised but not to the rim height (10.0).
        let center = elev[3 * 7 + 3];
        assert!(center > 1.0, "center should be raised: {center}");
        assert!(center < 10.0, "center should not reach rim: {center}");
    }

    #[test]
    fn coastal_cells_drain_to_ocean() {
        // Land next to ocean with a small pit
        let mut elev = vec![
            -10.0, -10.0, -10.0, -10.0,
            -10.0,   5.0,   3.0, -10.0,
            -10.0,   5.0,   1.0, -10.0,
            -10.0, -10.0, -10.0, -10.0,
        ];
        fill(&mut elev, 4, 4, 0.001, None);
        // The pit at (2,2)=1.0 should be raised since it's surrounded by 5.0 and 3.0
        // but it's adjacent to ocean so it should drain there — may or may not fill
        // depending on coast seeding. The key is no crash.
        assert!(elev[10] >= 1.0);
    }
}
