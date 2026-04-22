use glam::DVec3;

/// Splitmix64 hash — deterministic, fast, good avalanche.
pub(super) fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Dot product threshold for choosing a stable cross-product reference axis.
const TANGENT_AXIS_THRESHOLD: f64 = 0.9;

/// Arbitrary tangent vector perpendicular to a sphere-surface normal.
pub(super) fn arbitrary_tangent(normal: DVec3) -> DVec3 {
    let up = if normal.y.abs() < TANGENT_AXIS_THRESHOLD {
        DVec3::Y
    } else {
        DVec3::X
    };
    normal.cross(up).normalize()
}
