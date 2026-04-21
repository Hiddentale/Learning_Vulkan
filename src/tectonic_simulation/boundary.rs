use super::plates::CrustType;

/// A boundary edge: a point where two plates meet.
///
/// Retained for compatibility with physics sub-modules that haven't been
/// migrated to the sample-grid boundary detection in simulate.rs.
pub(super) struct BoundaryEdge {
    pub(super) point: u32,
    pub(super) neighbor: u32,
    pub(super) plate_a: u32,
    pub(super) plate_b: u32,
    pub(super) crust_a: CrustType,
    pub(super) crust_b: CrustType,
    pub(super) age_a: f64,
    pub(super) age_b: f64,
}
