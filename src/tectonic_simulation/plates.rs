use glam::DVec3;

/// Crust classification determines which per-point parameters apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrustType {
    Oceanic,
    Continental,
}

/// Orogeny origin for continental crust.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrogenyType {
    /// Subduction-driven mountain building.
    Andean,
    /// Collision-driven mountain building.
    Himalayan,
}

/// Per-point crustal and tectonic data.
#[derive(Debug, Clone)]
pub struct CrustData {
    pub crust_type: CrustType,
    /// Crust thickness e(p) in km.
    pub thickness: f64,
    /// Surface elevation z(p) in km. Negative for ocean floor.
    pub elevation: f64,
    /// Oceanic: age since formation at ridge. Continental: orogeny age. In Myr.
    pub age: f64,
    /// Oceanic: local ridge direction r(p). Continental: local fold direction f(p).
    pub local_direction: DVec3,
    /// Continental only — `None` for oceanic crust.
    pub orogeny_type: Option<OrogenyType>,
    /// Accumulated travel past a convergent front for subducting vertices (km).
    /// Advances per step as `relative_speed * dt` while the vertex is on the
    /// subducting side of a convergent boundary. At `SUBDUCTION_DISTANCE` the
    /// vertex is considered fully subducted and removed from the simulation.
    pub subducted_distance: f64,
}

impl CrustData {
    pub fn oceanic(thickness: f64, elevation: f64, age: f64, ridge_direction: DVec3) -> Self {
        Self {
            crust_type: CrustType::Oceanic,
            thickness,
            elevation,
            age,
            local_direction: ridge_direction,
            orogeny_type: None,
            subducted_distance: 0.0,
        }
    }

    pub fn continental(
        thickness: f64,
        elevation: f64,
        age: f64,
        fold_direction: DVec3,
        orogeny: OrogenyType,
    ) -> Self {
        Self {
            crust_type: CrustType::Continental,
            thickness,
            elevation,
            age,
            local_direction: fold_direction,
            orogeny_type: Some(orogeny),
            subducted_distance: 0.0,
        }
    }
}

/// A tectonic plate: a region of crust moving as a rigid body over the sphere.
///
/// Points sample the plate surface and store crustal data. Barycentric interpolation
/// over the plate's triangulation defines continuous fields.
pub struct Plate {
    /// Indices into the global point array for this plate's vertices.
    pub point_indices: Vec<u32>,
    /// Per-vertex crust data, parallel to `point_indices`.
    pub crust: Vec<CrustData>,
    /// Normalized rotation axis through the planet center.
    pub rotation_axis: DVec3,
    /// Angular speed in rad/Myr. Surface speed at point p = angular_speed * (rotation_axis × p).
    pub angular_speed: f64,
}

impl Plate {
    pub fn surface_velocity(&self, point: DVec3) -> DVec3 {
        self.angular_speed * self.rotation_axis.cross(point)
    }

    pub fn point_count(&self) -> usize {
        self.point_indices.len()
    }
}
