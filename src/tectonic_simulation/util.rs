use std::cmp::Ordering;

use glam::DVec3;

use super::plates::Plate;

/// Splitmix64 hash — deterministic, fast, good avalanche.
/// Used throughout the tectonic simulation for seeded pseudo-random decisions.
pub(super) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Min-heap entry for Dijkstra flood-fill.
/// `BinaryHeap` is a max-heap, so `Ord` is reversed.
pub(super) struct MinHeapEntry {
    pub cost: f64,
    pub index: u32,
}

impl PartialEq for MinHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for MinHeapEntry {}

impl Ord for MinHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for MinHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Dot product threshold for choosing a stable cross-product reference axis.
/// Avoids near-parallel vectors when building a tangent frame.
const TANGENT_AXIS_THRESHOLD: f64 = 0.9;

/// Compute the centroid of a plate's points on the unit sphere.
pub(super) fn plate_centroid(plate: &Plate, points: &[DVec3]) -> DVec3 {
    let sum: DVec3 = plate.point_indices.iter().map(|&i| points[i as usize]).sum();
    sum.normalize_or_zero()
}

/// Arbitrary tangent vector perpendicular to a sphere-surface normal.
pub(super) fn arbitrary_tangent(normal: DVec3) -> DVec3 {
    let up = if normal.y.abs() < TANGENT_AXIS_THRESHOLD { DVec3::Y } else { DVec3::X };
    normal.cross(up).normalize()
}
