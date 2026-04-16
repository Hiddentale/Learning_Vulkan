use glam::DVec3;

use super::plates::CrustType;

/// Maximum subduction distance r_s in km. Beyond this, f(d) = 0.
const SUBDUCTION_DISTANCE: f64 = 1800.0;
/// Distance from boundary where f(d) peaks, approximately r_s / 3.
const SUBDUCTION_PEAK_DISTANCE: f64 = 600.0;
/// Base subduction uplift u_0 in mm/yr.
const BASE_UPLIFT: f64 = 0.6;
/// Maximum plate speed v_0 in mm/yr, used to normalize g(v).
const MAX_PLATE_SPEED: f64 = 100.0;
/// Highest continental altitude z_c in km.
const MAX_CONTINENTAL_ALTITUDE: f64 = 10.0;
/// Maximum depth of oceanic trenches z_t in km.
const OCEANIC_TRENCH_DEPTH: f64 = -10.0;
/// Weight for fold direction update from relative plate velocity.
const FOLD_DIRECTION_WEIGHT: f64 = 0.3;
/// Slab pull scaling factor ε. Small so each boundary point has limited individual influence,
/// but long subduction fronts with many samples accumulate a noticeable effect.
const SLAB_PULL_EPSILON: f64 = 1e-4;
/// Minimum cross-product magnitude to consider a slab-pull direction meaningful.
const CROSS_PRODUCT_EPSILON: f64 = 1e-12;

/// Which plate subducts beneath the other at a converging boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubductionResult {
    /// Plate `i` dives beneath plate `j` (normal subduction).
    PlateSubducts(u32),
    /// Continental collision — forced partial subduction followed by terrane accretion.
    ContinentalCollision,
}

/// Determines the subduction outcome for two converging plates at a boundary point.
///
/// - Oceanic vs oceanic: the older plate subducts.
/// - Oceanic vs continental: the oceanic plate always subducts.
/// - Continental vs continental: partial forced subduction, evolving into collision.
pub fn resolve_subduction(
    plate_i: u32,
    plate_j: u32,
    crust_i: CrustType,
    age_i: f64,
    crust_j: CrustType,
    age_j: f64,
) -> SubductionResult {
    match (crust_i, crust_j) {
        (CrustType::Oceanic, CrustType::Oceanic) => {
            if age_i >= age_j {
                SubductionResult::PlateSubducts(plate_i)
            } else {
                SubductionResult::PlateSubducts(plate_j)
            }
        }
        (CrustType::Oceanic, CrustType::Continental) => SubductionResult::PlateSubducts(plate_i),
        (CrustType::Continental, CrustType::Oceanic) => SubductionResult::PlateSubducts(plate_j),
        (CrustType::Continental, CrustType::Continental) => SubductionResult::ContinentalCollision,
    }
}

/// Compute subduction uplift ũ_j(p) on the overriding plate.
///
/// ũ_j(p) = u_0 · f(d(p)) · g(v(p)) · h(z̃_i(p))
///
/// - d: distance from point to the subduction front, in km
/// - relative_speed: ||s_i(p) - s_j(p)|| in mm/yr
/// - subducting_elevation: elevation z of the subducting plate at this point, in km
pub fn subduction_uplift(d: f64, relative_speed: f64, subducting_elevation: f64) -> f64 {
    BASE_UPLIFT
        * distance_transfer(d)
        * speed_transfer(relative_speed)
        * height_transfer(subducting_elevation)
}

/// Apply one subduction timestep to a point on the overriding plate.
///
/// Updates elevation and fold direction:
/// - z_j(p, t+δt) = z_j(p, t) + ũ_j(p) · δt
/// - f_j(p, t+δt) = f_j(p, t) + β · (s_i(p) - s_j(p)) · δt
///
/// For newly emerged land (elevation crosses from ≤0 to >0), orogeny age resets to 0.
///
/// Returns the updated (elevation, fold_direction, orogeny_age).
pub fn apply_subduction_step(
    elevation: f64,
    fold_direction: DVec3,
    orogeny_age: f64,
    d: f64,
    velocity_i: DVec3,
    velocity_j: DVec3,
    dt: f64,
) -> (f64, DVec3, f64) {
    let relative = velocity_i - velocity_j;
    let relative_speed = relative.length();
    let uplift = subduction_uplift(d, relative_speed, elevation);

    let new_elevation = elevation + uplift * dt;
    let new_fold = (fold_direction + FOLD_DIRECTION_WEIGHT * relative * dt).normalize_or_zero();

    // Reset orogeny age for newly emerged land.
    let new_age = if elevation <= 0.0 && new_elevation > 0.0 {
        0.0
    } else {
        orogeny_age
    };

    (new_elevation, new_fold, new_age)
}

/// Distance transfer f(d): piecewise cubic, 0 at d=0, peaks at SUBDUCTION_PEAK_DISTANCE,
/// fades to 0 at SUBDUCTION_DISTANCE.
fn distance_transfer(d: f64) -> f64 {
    if d <= 0.0 || d >= SUBDUCTION_DISTANCE {
        return 0.0;
    }
    let peak = SUBDUCTION_PEAK_DISTANCE;
    let rs = SUBDUCTION_DISTANCE;

    if d <= peak {
        let t = d / peak;
        t * t * (3.0 - 2.0 * t)
    } else {
        let t = (d - peak) / (rs - peak);
        let s = 1.0 - t;
        s * s * (3.0 - 2.0 * s)
    }
}

/// Speed transfer g(v) = v / v_0, clamped to [0, 1].
fn speed_transfer(relative_speed: f64) -> f64 {
    (relative_speed / MAX_PLATE_SPEED).clamp(0.0, 1.0)
}

/// Height transfer h(z̃) = z̃², where z̃ = (z - z_t) / (z_c - z_t) normalized to [0, 1].
/// Elevations above sea level have much more impact than abyssal depths.
fn height_transfer(elevation: f64) -> f64 {
    let z_norm =
        ((elevation - OCEANIC_TRENCH_DEPTH) / (MAX_CONTINENTAL_ALTITUDE - OCEANIC_TRENCH_DEPTH))
            .clamp(0.0, 1.0);
    z_norm * z_norm
}

/// Compute the slab pull adjustment to a subducting plate's rotation axis.
///
/// w_i(t+δt) = w_i(t) + ε · Σ (c_i × q_k) / ||c_i × q_k|| · δt
///
/// The subducting plate's rotation axis is drawn toward its own subduction front,
/// pulling the plate toward where it is being consumed.
///
/// - rotation_axis: current normalized rotation axis w_i of the subducting plate
/// - plate_center: centroid c_i of the subducting plate
/// - boundary_points: positions q_k of points along the subduction front
/// - dt: timestep in Myr
///
/// Returns the updated (and re-normalized) rotation axis.
pub fn apply_slab_pull(
    rotation_axis: DVec3,
    plate_center: DVec3,
    boundary_points: &[DVec3],
    dt: f64,
) -> DVec3 {
    let mut pull = DVec3::ZERO;
    for &q in boundary_points {
        let cross = plate_center.cross(q);
        let len = cross.length();
        if len > CROSS_PRODUCT_EPSILON {
            pull += cross / len;
        }
    }
    (rotation_axis + SLAB_PULL_EPSILON * pull * dt).normalize()
}

/// Data tracked at an active subduction boundary point.
pub struct SubductionSite {
    /// Index of the subducting plate.
    pub subducting_plate: u32,
    /// Index of the overriding plate.
    pub overriding_plate: u32,
    /// Global point index of the boundary point.
    pub boundary_point: u32,
    /// How far the subducting slab has penetrated beneath the overriding plate, in km.
    pub subduction_distance: f64,
}

impl SubductionSite {
    pub fn new(subducting: u32, overriding: u32, boundary_point: u32) -> Self {
        Self {
            subducting_plate: subducting,
            overriding_plate: overriding,
            boundary_point,
            subduction_distance: 0.0,
        }
    }

    /// Advance subduction by the relative convergence speed over a time step.
    pub fn advance(&mut self, relative_speed: f64, dt: f64) {
        self.subduction_distance += relative_speed * dt;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oceanic_oceanic_older_subducts() {
        let result = resolve_subduction(0, 1, CrustType::Oceanic, 100.0, CrustType::Oceanic, 50.0);
        assert_eq!(result, SubductionResult::PlateSubducts(0));
    }

    #[test]
    fn oceanic_oceanic_younger_stays() {
        let result = resolve_subduction(0, 1, CrustType::Oceanic, 20.0, CrustType::Oceanic, 80.0);
        assert_eq!(result, SubductionResult::PlateSubducts(1));
    }

    #[test]
    fn oceanic_always_subducts_under_continental() {
        let result = resolve_subduction(0, 1, CrustType::Oceanic, 10.0, CrustType::Continental, 500.0);
        assert_eq!(result, SubductionResult::PlateSubducts(0));

        let result = resolve_subduction(0, 1, CrustType::Continental, 500.0, CrustType::Oceanic, 10.0);
        assert_eq!(result, SubductionResult::PlateSubducts(1));
    }

    #[test]
    fn continental_continental_is_collision() {
        let result = resolve_subduction(0, 1, CrustType::Continental, 100.0, CrustType::Continental, 200.0);
        assert_eq!(result, SubductionResult::ContinentalCollision);
    }

    #[test]
    fn equal_age_oceanic_prefers_first() {
        let result = resolve_subduction(0, 1, CrustType::Oceanic, 50.0, CrustType::Oceanic, 50.0);
        assert_eq!(result, SubductionResult::PlateSubducts(0));
    }

    #[test]
    fn subduction_distance_advances() {
        let mut site = SubductionSite::new(0, 1, 42);
        assert_eq!(site.subduction_distance, 0.0);
        site.advance(5.0, 10.0);
        assert!((site.subduction_distance - 50.0).abs() < 1e-10);
        site.advance(3.0, 5.0);
        assert!((site.subduction_distance - 65.0).abs() < 1e-10);
    }

    // --- Transfer function tests ---

    #[test]
    fn distance_transfer_zero_at_boundaries() {
        assert_eq!(distance_transfer(0.0), 0.0);
        assert_eq!(distance_transfer(SUBDUCTION_DISTANCE), 0.0);
        assert_eq!(distance_transfer(-10.0), 0.0);
        assert_eq!(distance_transfer(2000.0), 0.0);
    }

    #[test]
    fn distance_transfer_peaks_at_control_distance() {
        let peak_val = distance_transfer(SUBDUCTION_PEAK_DISTANCE);
        assert!((peak_val - 1.0).abs() < 1e-10, "peak should be 1.0, got {peak_val}");

        assert!(distance_transfer(SUBDUCTION_PEAK_DISTANCE - 100.0) < peak_val);
        assert!(distance_transfer(SUBDUCTION_PEAK_DISTANCE + 100.0) < peak_val);
    }

    #[test]
    fn distance_transfer_monotonic_rise_and_fall() {
        let mut prev = 0.0;
        for i in 1..=10 {
            let d = SUBDUCTION_PEAK_DISTANCE * i as f64 / 10.0;
            let val = distance_transfer(d);
            assert!(val >= prev, "not monotonically rising at d={d}");
            prev = val;
        }
        prev = 1.0;
        for i in 1..=10 {
            let d = SUBDUCTION_PEAK_DISTANCE
                + (SUBDUCTION_DISTANCE - SUBDUCTION_PEAK_DISTANCE) * i as f64 / 10.0;
            let val = distance_transfer(d);
            assert!(val <= prev, "not monotonically falling at d={d}");
            prev = val;
        }
    }

    #[test]
    fn speed_transfer_linear() {
        assert!((speed_transfer(0.0)).abs() < 1e-10);
        assert!((speed_transfer(50.0) - 0.5).abs() < 1e-10);
        assert!((speed_transfer(100.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn speed_transfer_clamps() {
        assert_eq!(speed_transfer(-10.0), 0.0);
        assert_eq!(speed_transfer(200.0), 1.0);
    }

    #[test]
    fn height_transfer_squared_with_trench_normalization() {
        // z̃ = (z - z_t) / (z_c - z_t) = (z + 10) / 20
        // h(z_t = -10) => z̃ = 0, h = 0
        assert!((height_transfer(-10.0)).abs() < 1e-10);
        // h(z_c = 10) => z̃ = 1, h = 1
        assert!((height_transfer(10.0) - 1.0).abs() < 1e-10);
        // h(0km) => z̃ = 10/20 = 0.5, h = 0.25
        assert!((height_transfer(0.0) - 0.25).abs() < 1e-10);
        // h(-4km) => z̃ = 6/20 = 0.3, h = 0.09
        assert!((height_transfer(-4.0) - 0.09).abs() < 1e-10);
    }

    #[test]
    fn height_transfer_clamps() {
        assert_eq!(height_transfer(-20.0), 0.0);
        assert_eq!(height_transfer(20.0), 1.0);
    }

    #[test]
    fn uplift_combines_all_transfers() {
        let uplift = subduction_uplift(
            SUBDUCTION_PEAK_DISTANCE,
            MAX_PLATE_SPEED,
            MAX_CONTINENTAL_ALTITUDE,
        );
        assert!((uplift - BASE_UPLIFT).abs() < 1e-10);

        // Any zero factor kills the uplift.
        assert_eq!(subduction_uplift(0.0, 50.0, 5.0), 0.0);
        assert_eq!(subduction_uplift(600.0, 0.0, 5.0), 0.0);
        // At z_t elevation, h(z̃) = 0.
        assert!(subduction_uplift(600.0, 50.0, -10.0).abs() < 1e-10);
    }

    // --- Subduction step tests ---

    #[test]
    fn step_increases_elevation() {
        let (new_z, _, _) = apply_subduction_step(
            -4.0,
            DVec3::X,
            100.0,
            600.0,
            DVec3::new(50.0, 0.0, 0.0),
            DVec3::new(-20.0, 0.0, 0.0),
            2.0,
        );
        assert!(new_z > -4.0, "elevation should increase from uplift");
    }

    #[test]
    fn step_rotates_fold_direction() {
        let initial_fold = DVec3::X;
        let (_, new_fold, _) = apply_subduction_step(
            0.0,
            initial_fold,
            100.0,
            600.0,
            DVec3::new(0.0, 50.0, 0.0),
            DVec3::ZERO,
            2.0,
        );
        // Fold should pick up a Y component from the relative velocity.
        assert!(new_fold.y > 0.0, "fold should rotate toward relative velocity");
        assert!((new_fold.length() - 1.0).abs() < 1e-10, "fold should stay normalized");
    }

    #[test]
    fn step_resets_orogeny_age_on_emergence() {
        let (new_z, _, new_age) = apply_subduction_step(
            -0.001,
            DVec3::X,
            500.0,
            600.0,
            DVec3::new(80.0, 0.0, 0.0),
            DVec3::ZERO,
            100.0,
        );
        assert!(new_z > 0.0, "should have emerged");
        assert_eq!(new_age, 0.0, "orogeny age should reset on emergence");
    }

    // --- Slab pull tests ---

    #[test]
    fn slab_pull_shifts_axis_toward_boundary() {
        let axis = DVec3::Z;
        let center = DVec3::Z;
        // Boundary points along the equator to the "east".
        let boundary = vec![DVec3::X, DVec3::new(0.707, 0.707, 0.0).normalize()];
        let new_axis = apply_slab_pull(axis, center, &boundary, 2.0);

        // Axis should still be normalized.
        assert!((new_axis.length() - 1.0).abs() < 1e-10);
        // Should have shifted away from pure Z (toward the boundary).
        assert!(new_axis.z < 1.0, "axis should shift away from pure Z");
    }

    #[test]
    fn slab_pull_no_boundary_preserves_axis() {
        let axis = DVec3::Y.normalize();
        let new_axis = apply_slab_pull(axis, DVec3::Y, &[], 2.0);
        assert!((new_axis - axis).length() < 1e-10);
    }

    #[test]
    fn slab_pull_symmetric_boundary_cancels() {
        let axis = DVec3::Z;
        let center = DVec3::Z;
        // Opposing boundary points should cancel out.
        let boundary = vec![DVec3::X, DVec3::NEG_X];
        let new_axis = apply_slab_pull(axis, center, &boundary, 2.0);
        assert!((new_axis - axis).length() < 1e-10, "symmetric pull should cancel");
    }

    #[test]
    fn slab_pull_scales_with_boundary_length() {
        let axis = DVec3::Z;
        let center = DVec3::Z;
        let short = vec![DVec3::X];
        let long = vec![DVec3::X; 10];
        let shift_short = apply_slab_pull(axis, center, &short, 2.0);
        let shift_long = apply_slab_pull(axis, center, &long, 2.0);
        let delta_short = (shift_short - axis).length();
        let delta_long = (shift_long - axis).length();
        assert!(delta_long > delta_short, "longer boundary should pull more");
    }

    #[test]
    fn step_preserves_orogeny_age_when_already_above() {
        let (_, _, new_age) = apply_subduction_step(
            5.0,
            DVec3::X,
            250.0,
            600.0,
            DVec3::new(50.0, 0.0, 0.0),
            DVec3::ZERO,
            2.0,
        );
        assert_eq!(new_age, 250.0, "age should not reset when already above sea level");
    }
}
