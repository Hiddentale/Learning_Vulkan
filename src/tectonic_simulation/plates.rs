use glam::{DQuat, DVec3};

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

    /// Barycentric blend of three crust data values.
    /// Discrete fields (crust_type, orogeny_type) come from the dominant vertex.
    pub fn barycentric_blend(a: &CrustData, b: &CrustData, c: &CrustData, w: [f64; 3]) -> Self {
        let dom = if w[0] >= w[1] && w[0] >= w[2] {
            0
        } else if w[1] >= w[2] {
            1
        } else {
            2
        };
        let dominant = [a, b, c][dom];

        Self {
            crust_type: dominant.crust_type,
            thickness: a.thickness * w[0] + b.thickness * w[1] + c.thickness * w[2],
            elevation: a.elevation * w[0] + b.elevation * w[1] + c.elevation * w[2],
            age: a.age * w[0] + b.age * w[1] + c.age * w[2],
            local_direction: (a.local_direction * w[0]
                + b.local_direction * w[1]
                + c.local_direction * w[2])
                .normalize_or_zero(),
            orogeny_type: dominant.orogeny_type,
            subducted_distance: a.subducted_distance * w[0]
                + b.subducted_distance * w[1]
                + c.subducted_distance * w[2],
        }
    }
}

/// Per-triangle adjacency: for each of the three edges, the index of the
/// neighboring triangle across that edge, or `None` at a plate boundary.
///
/// Edge ordering matches the triangle vertex ordering:
/// - edge 0: v0 → v1  (opposite v2)
/// - edge 1: v1 → v2  (opposite v0)
/// - edge 2: v2 → v0  (opposite v1)
pub type TriangleNeighbors = [Option<u32>; 3];

/// Bounding spherical cap for fast candidate-plate rejection (step 4).
/// A plate's reference-frame cap is (center, cos_radius). In world space
/// the center rotates to `R_k * center`; the angular radius is unchanged.
pub struct BoundingCap {
    /// Cap center in the plate's reference frame (unit vector).
    pub center: DVec3,
    /// Cosine of the angular radius. A point p is inside when
    /// `p.dot(rotated_center) >= cos_radius`.
    pub cos_radius: f64,
}

/// A tectonic plate: a region of crust moving as a rigid body over the sphere.
///
/// The plate owns a reference sub-mesh T_k — vertex positions in the plate's
/// birth frame that never move due to rotation. World-space positions are
/// `rotation * reference_points[i]`. Between resamples only `rotation` changes;
/// the mesh topology and reference positions are stable. Boundary operations
/// (rifting, subduction, oceanic generation) perform local mesh surgery.
pub struct Plate {
    /// Vertex positions in the plate's reference frame (unit sphere).
    /// These are the "birth positions" and do not change due to plate motion.
    pub reference_points: Vec<DVec3>,
    /// Per-vertex crust data, parallel to `reference_points`.
    pub crust: Vec<CrustData>,
    /// Triangle connectivity — indices into `reference_points`.
    pub triangles: Vec<[u32; 3]>,
    /// Per-triangle adjacency. `adjacency[t][e]` is the triangle across edge e
    /// of triangle t, or `None` at a plate boundary. Parallel to `triangles`.
    pub adjacency: Vec<TriangleNeighbors>,
    /// Bounding spherical cap in reference frame, for fast candidate rejection.
    pub bounding_cap: BoundingCap,
    /// Accumulated rotation since t=0 (R_k). Apply to reference_points
    /// to get current world-space positions: `world = rotation * ref`.
    /// Inverse maps world to reference: `ref = rotation.inverse() * world`.
    pub rotation: DQuat,
    /// Normalized Euler pole through the planet center.
    pub rotation_axis: DVec3,
    /// Angular speed in rad/Myr. Surface speed at point p = angular_speed * (rotation_axis × p).
    pub angular_speed: f64,
}

impl Plate {
    /// Surface velocity in km/Myr (= mm/yr) at a world-space point.
    /// Converts from angular (rad/Myr) to linear by scaling with planet radius.
    pub fn surface_velocity(&self, world_point: DVec3) -> DVec3 {
        const PLANET_RADIUS: f64 = 6370.0;
        self.angular_speed * PLANET_RADIUS * self.rotation_axis.cross(world_point)
    }

    pub fn point_count(&self) -> usize {
        self.reference_points.len()
    }

    pub fn triangle_count(&self) -> usize {
        self.triangles.len()
    }

    /// Map a world-space point into this plate's reference frame.
    pub fn to_reference(&self, world_point: DVec3) -> DVec3 {
        self.rotation.inverse().mul_vec3(world_point)
    }

    /// Map a reference-frame point into world space.
    pub fn to_world(&self, ref_point: DVec3) -> DVec3 {
        self.rotation.mul_vec3(ref_point)
    }

    /// Get the world-space position of vertex `i`.
    pub fn world_point(&self, i: usize) -> DVec3 {
        self.to_world(self.reference_points[i])
    }

    /// Test whether reference-frame point `q` is inside triangle `tri`.
    /// Returns the barycentric weights if inside (all >= 0), or `None`.
    ///
    /// Uses the spherical orientation test: for edge (a, b), q is on the
    /// inside if `q.dot(a.cross(b)) >= 0` (same hemisphere as the triangle
    /// interior with respect to the great circle through a and b).
    pub fn point_in_triangle(&self, tri: usize, q: DVec3) -> Option<[f64; 3]> {
        let [vi, vj, vk] = self.triangles[tri];
        let a = self.reference_points[vi as usize];
        let b = self.reference_points[vj as usize];
        let c = self.reference_points[vk as usize];

        // Flip signs for CW-wound triangles so the inside test works uniformly.
        let det = a.dot(b.cross(c));
        let flip = det.signum();
        let w0 = flip * q.dot(b.cross(c));
        let w1 = flip * q.dot(c.cross(a));
        let w2 = flip * q.dot(a.cross(b));

        // Tolerance for vertices/edges where weights are numerically ~0.
        const EPS: f64 = -1e-12;
        if w0 < EPS || w1 < EPS || w2 < EPS {
            return None;
        }

        let sum = w0 + w1 + w2;
        if sum < 1e-30 {
            return Some([1.0 / 3.0; 3]);
        }
        Some([w0 / sum, w1 / sum, w2 / sum])
    }

    /// Walk the reference mesh from `start_tri` toward `q`, returning the
    /// enclosing triangle and its barycentric weights, or `None` if the walk
    /// exits the plate boundary (step 3).
    ///
    /// `max_steps` caps the walk to prevent infinite loops on degenerate meshes.
    pub fn walk_to_point(
        &self,
        q: DVec3,
        start_tri: u32,
        max_steps: u32,
    ) -> WalkResult {
        if self.triangles.is_empty() {
            return WalkResult::MaxStepsReached { last_triangle: 0 };
        }
        let mut tri = (start_tri as usize).min(self.triangles.len() - 1);

        for _ in 0..max_steps {
            let [vi, vj, vk] = self.triangles[tri];
            let a = self.reference_points[vi as usize];
            let b = self.reference_points[vj as usize];
            let c = self.reference_points[vk as usize];

            // Flip signs for CW-wound triangles so the walk works uniformly.
            let det = a.dot(b.cross(c));
            let flip = det.signum();
            let bary = [
                flip * q.dot(b.cross(c)), // weight for v0; negative → outside edge 1
                flip * q.dot(c.cross(a)), // weight for v1; negative → outside edge 2
                flip * q.dot(a.cross(b)), // weight for v2; negative → outside edge 0
            ];

            const EPS: f64 = -1e-12;
            if bary[0] >= EPS && bary[1] >= EPS && bary[2] >= EPS {
                let sum = bary[0] + bary[1] + bary[2];
                let w = if sum < 1e-30 {
                    [1.0 / 3.0; 3]
                } else {
                    [bary[0] / sum, bary[1] / sum, bary[2] / sum]
                };
                return WalkResult::Found {
                    triangle: tri as u32,
                    bary: w,
                };
            }

            // Cross the edge opposite the vertex with the most-negative weight.
            // If that edge is a plate boundary (None), try the next-most-negative.
            // This handles concave plate shapes where the direct path exits but
            // an alternative path stays inside the mesh.
            let mut candidates = [
                (bary[0], 1usize), // bary[0] < 0 → cross edge 1 (opposite v0)
                (bary[1], 2),      // bary[1] < 0 → cross edge 2 (opposite v1)
                (bary[2], 0),      // bary[2] < 0 → cross edge 0 (opposite v2)
            ];
            candidates.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            let mut crossed = false;
            let mut last_exit_edge = candidates[0].1 as u8;
            for &(weight, edge) in &candidates {
                if weight >= EPS {
                    break;
                }
                last_exit_edge = edge as u8;
                if let Some(neighbor) = self.adjacency[tri][edge] {
                    tri = neighbor as usize;
                    crossed = true;
                    break;
                }
            }
            if !crossed {
                return WalkResult::ExitedPlate {
                    boundary_triangle: tri as u32,
                    exit_edge: last_exit_edge,
                };
            }
        }

        WalkResult::MaxStepsReached { last_triangle: tri as u32 }
    }

    /// Test whether world-space point `p` is inside this plate's bounding cap.
    pub fn world_point_in_cap(&self, p: DVec3) -> bool {
        let world_center = self.to_world(self.bounding_cap.center);
        p.dot(world_center) >= self.bounding_cap.cos_radius
    }

    /// Insert a new vertex at a plate boundary edge, creating one triangle
    /// that connects the boundary edge to the new vertex.
    pub fn insert_boundary_vertex(
        &mut self,
        ref_pos: DVec3,
        crust: CrustData,
        boundary_tri: usize,
        boundary_edge: usize,
    ) {
        let new_vi = self.reference_points.len() as u32;
        self.reference_points.push(ref_pos);
        self.crust.push(crust);

        let tri = self.triangles[boundary_tri];
        let e0 = tri[boundary_edge];
        let e1 = tri[(boundary_edge + 1) % 3];

        let new_tri_idx = self.triangles.len() as u32;
        self.triangles.push([e0, e1, new_vi]);

        // New triangle adjacency:
        //   edge 0 (e0→e1): shared with boundary_tri
        //   edge 1 (e1→new_vi): plate boundary
        //   edge 2 (new_vi→e0): plate boundary
        self.adjacency.push([Some(boundary_tri as u32), None, None]);
        self.adjacency[boundary_tri][boundary_edge] = Some(new_tri_idx);
    }

    /// Find the boundary edge nearest to reference-frame point `q`.
    /// Returns (triangle_index, edge_index) or None if no boundary edges exist.
    pub fn nearest_boundary_edge(&self, q: DVec3) -> Option<(usize, usize)> {
        let mut best_dot = f64::NEG_INFINITY;
        let mut best = None;

        for (t, adj) in self.adjacency.iter().enumerate() {
            for e in 0..3 {
                if adj[e].is_some() {
                    continue;
                }
                let v0 = self.triangles[t][e] as usize;
                let v1 = self.triangles[t][(e + 1) % 3] as usize;
                let midpoint =
                    (self.reference_points[v0] + self.reference_points[v1]).normalize();
                let dot = midpoint.dot(q);
                if dot > best_dot {
                    best_dot = dot;
                    best = Some((t, e));
                }
            }
        }

        best
    }

    /// Find the triangle nearest to reference-frame point `q` by scanning
    /// vertices. Used as a cold-start for the walk when no warm-start is
    /// available — O(vertices) but only fires for boundary samples.
    pub fn nearest_start_triangle(&self, q: DVec3) -> u32 {
        let mut best_dot = f64::NEG_INFINITY;
        let mut best_vert = 0usize;
        for (i, &v) in self.reference_points.iter().enumerate() {
            let dot = v.dot(q);
            if dot > best_dot {
                best_dot = dot;
                best_vert = i;
            }
        }
        self.triangles
            .iter()
            .position(|tri| tri.contains(&(best_vert as u32)))
            .unwrap_or(0) as u32
    }

    /// Recompute the bounding cap from the current reference points.
    pub fn recompute_bounding_cap(&mut self) {
        self.bounding_cap = compute_bounding_cap(&self.reference_points);
    }
}

/// Result of `walk_to_point`.
pub enum WalkResult {
    /// Point is inside this triangle.
    Found { triangle: u32, bary: [f64; 3] },
    /// Walk exited the plate through a boundary edge (step 3).
    ExitedPlate { boundary_triangle: u32, exit_edge: u8 },
    /// Safety limit reached without converging.
    MaxStepsReached { last_triangle: u32 },
}

/// Per-sample warm-start cache (step 8).
/// Stored on the global sample grid, not on the plate.
#[derive(Clone, Copy)]
pub struct SampleCache {
    /// Previous plate assignment (k*). `u32::MAX` = unassigned (gap).
    pub plate: u32,
    /// Previous triangle within that plate's reference mesh (τ*).
    pub triangle: u32,
    /// Barycentric coordinates within the triangle.
    pub bary: [f64; 3],
}

/// Compute a tight bounding spherical cap for a set of reference-frame points.
pub fn compute_bounding_cap(points: &[DVec3]) -> BoundingCap {
    if points.is_empty() {
        return BoundingCap {
            center: DVec3::Y,
            cos_radius: 1.0,
        };
    }

    // Centroid as cap center.
    let sum: DVec3 = points.iter().copied().sum();
    let center = sum.normalize_or_zero();

    // Angular radius: smallest dot product (= largest angular distance).
    let min_dot = points
        .iter()
        .map(|p| p.dot(center))
        .fold(f64::MAX, f64::min);

    // Expand slightly to handle numerical imprecision.
    let cos_radius = (min_dot - 1e-6).max(-1.0);

    BoundingCap { center, cos_radius }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn single_tri_plate() -> Plate {
        let a = DVec3::new(0.0, 0.0, 1.0);
        let b = DVec3::new(1.0, 0.0, 0.0);
        let c = DVec3::new(0.0, 1.0, 0.0);
        Plate {
            reference_points: vec![a, b, c],
            crust: vec![
                CrustData::continental(35.0, 2.0, 10.0, DVec3::X, OrogenyType::Andean),
                CrustData::oceanic(7.0, -4.0, 50.0, DVec3::Y),
                CrustData::oceanic(7.0, -3.0, 30.0, DVec3::Z),
            ],
            triangles: vec![[0, 1, 2]],
            adjacency: vec![[None, None, None]],
            bounding_cap: compute_bounding_cap(&[a, b, c]),
            rotation: DQuat::IDENTITY,
            rotation_axis: DVec3::Y,
            angular_speed: 0.01,
        }
    }

    /// Two-triangle plate sharing edge 1→2:
    ///   0
    ///  / \
    /// 1───2
    ///  \ /
    ///   3
    fn diamond_plate() -> Plate {
        let v0 = DVec3::new(0.0, 0.0, 1.0);
        let v1 = DVec3::new(-0.7, -0.7, 0.1).normalize();
        let v2 = DVec3::new(0.7, -0.7, 0.1).normalize();
        let v3 = DVec3::new(0.0, -1.0, 0.0).normalize();
        Plate {
            reference_points: vec![v0, v1, v2, v3],
            crust: vec![
                CrustData::oceanic(7.0, -4.0, 0.0, DVec3::X),
                CrustData::oceanic(7.0, -4.0, 0.0, DVec3::X),
                CrustData::oceanic(7.0, -4.0, 0.0, DVec3::X),
                CrustData::oceanic(7.0, -4.0, 0.0, DVec3::X),
            ],
            // tri 0: [0,1,2], tri 1: [1,3,2]
            triangles: vec![[0, 1, 2], [1, 3, 2]],
            // tri 0 edge1 (1→2) neighbors tri 1; tri 1 edge2 (2→1) neighbors tri 0
            adjacency: vec![
                [None, Some(1), None],
                [None, None, Some(0)],
            ],
            bounding_cap: compute_bounding_cap(&[v0, v1, v2, v3]),
            rotation: DQuat::IDENTITY,
            rotation_axis: DVec3::Y,
            angular_speed: 0.01,
        }
    }

    // --- CrustData ---

    #[test]
    fn barycentric_blend_at_vertex_returns_that_vertex() {
        let a = CrustData::continental(35.0, 2.0, 10.0, DVec3::X, OrogenyType::Andean);
        let b = CrustData::oceanic(7.0, -4.0, 50.0, DVec3::Y);
        let c = CrustData::oceanic(7.0, -3.0, 30.0, DVec3::Z);

        let result = CrustData::barycentric_blend(&a, &b, &c, [1.0, 0.0, 0.0]);
        assert_eq!(result.crust_type, CrustType::Continental);
        assert!((result.elevation - 2.0).abs() < 1e-10);
        assert!((result.thickness - 35.0).abs() < 1e-10);
        assert_eq!(result.orogeny_type, Some(OrogenyType::Andean));
    }

    #[test]
    fn barycentric_blend_midpoint_averages_continuous_fields() {
        let a = CrustData::oceanic(10.0, -2.0, 0.0, DVec3::X);
        let b = CrustData::oceanic(20.0, -6.0, 100.0, DVec3::X);
        let c = CrustData::oceanic(30.0, -4.0, 50.0, DVec3::X);

        let result = CrustData::barycentric_blend(&a, &b, &c, [1.0 / 3.0; 3]);
        assert!((result.thickness - 20.0).abs() < 1e-10);
        assert!((result.elevation - (-4.0)).abs() < 1e-10);
        assert!((result.age - 50.0).abs() < 1e-10);
    }

    #[test]
    fn barycentric_blend_dominant_picks_largest_weight() {
        let a = CrustData::continental(35.0, 2.0, 0.0, DVec3::X, OrogenyType::Himalayan);
        let b = CrustData::oceanic(7.0, -4.0, 0.0, DVec3::Y);
        let c = CrustData::oceanic(7.0, -3.0, 0.0, DVec3::Z);

        let result = CrustData::barycentric_blend(&a, &b, &c, [0.1, 0.6, 0.3]);
        assert_eq!(result.crust_type, CrustType::Oceanic);
        assert_eq!(result.orogeny_type, None);
    }

    #[test]
    fn barycentric_blend_direction_normalized() {
        let a = CrustData::oceanic(7.0, -4.0, 0.0, DVec3::X);
        let b = CrustData::oceanic(7.0, -4.0, 0.0, DVec3::Y);
        let c = CrustData::oceanic(7.0, -4.0, 0.0, DVec3::Z);

        let result = CrustData::barycentric_blend(&a, &b, &c, [1.0 / 3.0; 3]);
        let len = result.local_direction.length();
        assert!((len - 1.0).abs() < 1e-6);
    }

    // --- point_in_triangle ---

    #[test]
    fn point_in_triangle_center_inside() {
        let plate = single_tri_plate();
        let center = (plate.reference_points[0]
            + plate.reference_points[1]
            + plate.reference_points[2])
            .normalize();
        let result = plate.point_in_triangle(0, center);
        assert!(result.is_some());
        let w = result.unwrap();
        assert!(w[0] > 0.0 && w[1] > 0.0 && w[2] > 0.0);
        assert!((w[0] + w[1] + w[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn point_in_triangle_opposite_side_outside() {
        let plate = single_tri_plate();
        let opposite = DVec3::new(-1.0, -1.0, -1.0).normalize();
        assert!(plate.point_in_triangle(0, opposite).is_none());
    }

    #[test]
    fn point_in_triangle_at_vertex_inside() {
        let plate = single_tri_plate();
        let result = plate.point_in_triangle(0, plate.reference_points[0]);
        assert!(result.is_some());
    }

    // --- walk_to_point ---

    #[test]
    fn walk_finds_point_in_starting_triangle() {
        let plate = single_tri_plate();
        let center = (plate.reference_points[0]
            + plate.reference_points[1]
            + plate.reference_points[2])
            .normalize();
        match plate.walk_to_point(center, 0, 10) {
            WalkResult::Found { triangle, bary } => {
                assert_eq!(triangle, 0);
                assert!(bary[0] > 0.0 && bary[1] > 0.0 && bary[2] > 0.0);
            }
            _ => panic!("expected Found"),
        }
    }

    #[test]
    fn walk_crosses_edge_to_neighbor() {
        let plate = diamond_plate();
        // Point in tri 1 (near v3), start walk from tri 0.
        let target = plate.reference_points[3];
        match plate.walk_to_point(target, 0, 10) {
            WalkResult::Found { triangle, .. } => assert_eq!(triangle, 1),
            _ => panic!("expected Found after crossing edge"),
        }
    }

    #[test]
    fn walk_exits_plate_at_boundary() {
        let plate = single_tri_plate();
        let outside = DVec3::new(-1.0, 0.0, 0.0).normalize();
        match plate.walk_to_point(outside, 0, 10) {
            WalkResult::ExitedPlate { .. } => {}
            other => panic!("expected ExitedPlate, got {:?}", match other {
                WalkResult::Found { .. } => "Found",
                WalkResult::MaxStepsReached { .. } => "MaxStepsReached",
                WalkResult::ExitedPlate { .. } => "ExitedPlate",
            }),
        }
    }

    #[test]
    fn walk_max_steps_reached() {
        let plate = diamond_plate();
        let target = plate.reference_points[3];
        match plate.walk_to_point(target, 0, 0) {
            WalkResult::MaxStepsReached { .. } => {}
            _ => panic!("expected MaxStepsReached with 0 max_steps"),
        }
    }

    // --- to_reference / to_world ---

    #[test]
    fn to_world_identity_rotation_is_noop() {
        let plate = single_tri_plate();
        let p = DVec3::new(0.5, 0.5, 0.5).normalize();
        let world = plate.to_world(p);
        assert!((world - p).length() < 1e-10);
    }

    #[test]
    fn to_reference_to_world_roundtrip() {
        let mut plate = single_tri_plate();
        plate.rotation = DQuat::from_axis_angle(DVec3::Y, 0.5);

        let p = DVec3::new(0.3, 0.7, 0.5).normalize();
        let ref_p = plate.to_reference(p);
        let back = plate.to_world(ref_p);
        assert!((back - p).length() < 1e-10);
    }

    #[test]
    fn to_world_rotates_points() {
        let mut plate = single_tri_plate();
        plate.rotation = DQuat::from_axis_angle(DVec3::Y, std::f64::consts::FRAC_PI_2);

        let ref_p = DVec3::X;
        let world = plate.to_world(ref_p);
        // 90° around Y: X → -Z
        assert!((world - DVec3::NEG_Z).length() < 1e-10);
    }

    // --- world_point_in_cap ---

    #[test]
    fn cap_contains_plate_vertices() {
        let plate = single_tri_plate();
        for &p in &plate.reference_points {
            assert!(plate.world_point_in_cap(p));
        }
    }

    #[test]
    fn cap_rejects_antipodal_point() {
        let plate = single_tri_plate();
        let centroid = (plate.reference_points[0]
            + plate.reference_points[1]
            + plate.reference_points[2])
            .normalize();
        assert!(!plate.world_point_in_cap(-centroid));
    }

    #[test]
    fn cap_rotates_with_plate() {
        let mut plate = single_tri_plate();
        let p = plate.reference_points[0];
        assert!(plate.world_point_in_cap(p));

        plate.rotation = DQuat::from_axis_angle(DVec3::Y, std::f64::consts::PI);
        plate.recompute_bounding_cap();
        // After 180° rotation, the original world-space point may be outside.
        let world_p = plate.to_world(plate.reference_points[0]);
        assert!(plate.world_point_in_cap(world_p));
    }

    // --- compute_bounding_cap ---

    #[test]
    fn bounding_cap_empty_points() {
        let cap = compute_bounding_cap(&[]);
        assert!((cap.cos_radius - 1.0).abs() < 1e-10);
    }

    #[test]
    fn bounding_cap_single_point() {
        let cap = compute_bounding_cap(&[DVec3::X]);
        assert!((cap.center - DVec3::X).length() < 1e-10);
    }

    #[test]
    fn bounding_cap_covers_all_points() {
        let points = vec![
            DVec3::X,
            DVec3::Y,
            DVec3::Z,
            DVec3::new(1.0, 1.0, 1.0).normalize(),
        ];
        let cap = compute_bounding_cap(&points);
        for &p in &points {
            assert!(
                p.dot(cap.center) >= cap.cos_radius,
                "point {p:?} outside cap"
            );
        }
    }

    // --- surface_velocity ---

    #[test]
    fn surface_velocity_perpendicular_to_point() {
        let plate = single_tri_plate();
        let p = DVec3::new(0.5, 0.5, 0.5).normalize();
        let v = plate.surface_velocity(p);
        assert!(v.dot(p).abs() < 1e-10);
    }

    #[test]
    fn surface_velocity_zero_at_pole() {
        let plate = single_tri_plate();
        // rotation_axis = Y, so velocity at Y pole is zero
        let v = plate.surface_velocity(DVec3::Y);
        assert!(v.length() < 1e-10);
    }

    // --- walk on initialized plates ---

    #[test]
    fn walk_finds_vertex_from_incident_triangle() {
        use super::super::fibonnaci_spiral::SphericalFibonacci;
        use super::super::plate_initializer::{initialize_plates, InitParams};
        use super::super::plate_seed_placement::assign_plates;
        use super::super::spherical_delaunay_triangulation::SphericalDelaunay;

        let fib = SphericalFibonacci::new(200);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, 4, 42);
        let (plates, _) = initialize_plates(&points, &del, &assignment, &InitParams::default());

        for plate in &plates {
            let mut found = 0;
            for (i, &ref_p) in plate.reference_points.iter().enumerate() {
                let start = plate
                    .triangles
                    .iter()
                    .position(|tri| tri.contains(&(i as u32)))
                    .unwrap_or(0);
                if let WalkResult::Found { .. } =
                    plate.walk_to_point(ref_p, start as u32, 256)
                {
                    found += 1;
                }
            }
            assert_eq!(
                found,
                plate.point_count(),
                "walk missed {} vertices from incident triangle",
                plate.point_count() - found
            );
        }
    }
}
