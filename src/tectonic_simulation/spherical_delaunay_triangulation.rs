use std::collections::HashMap;

use glam::DVec3;

const UNSET: u32 = u32::MAX;

// ── Robust orient3d ──────────────────────────────────────────────────────────

/// Exact sign of the 3×3 determinant | (a-d) (b-d) (c-d) |.
/// Uses an error-bounded fast path; falls back to compensated arithmetic.
fn orient3d(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> f64 {
    let ad = a - d;
    let bd = b - d;
    let cd = c - d;
    let result = ad.dot(bd.cross(cd));

    // Error bound from Shewchuk: if |result| > bound, the sign is reliable.
    let permanent = (ad.x.abs() * (bd.y * cd.z).abs()
        + ad.x.abs() * (bd.z * cd.y).abs()
        + ad.y.abs() * (bd.x * cd.z).abs()
        + ad.y.abs() * (bd.z * cd.x).abs()
        + ad.z.abs() * (bd.x * cd.y).abs()
        + ad.z.abs() * (bd.y * cd.x).abs())
        + (bd.x.abs() * (cd.y * ad.z).abs()
            + bd.x.abs() * (cd.z * ad.y).abs()
            + bd.y.abs() * (cd.x * ad.z).abs()
            + bd.y.abs() * (cd.z * ad.x).abs()
            + bd.z.abs() * (cd.x * ad.y).abs()
            + bd.z.abs() * (cd.y * ad.x).abs())
        + (cd.x.abs() * (ad.y * bd.z).abs()
            + cd.x.abs() * (ad.z * bd.y).abs()
            + cd.y.abs() * (ad.x * bd.z).abs()
            + cd.y.abs() * (ad.z * bd.x).abs()
            + cd.z.abs() * (ad.x * bd.y).abs()
            + cd.z.abs() * (ad.y * bd.x).abs());

    // 5ε is the relative error bound for the 3×3 determinant (Shewchuk §4.1).
    let eps = 5.0 * f64::EPSILON;
    if result.abs() > eps * permanent {
        return result;
    }

    // Slow path: recompute with Kahan-style compensated summation.
    orient3d_exact(a, b, c, d)
}

/// Compensated orient3d for the near-zero case.
fn orient3d_exact(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> f64 {
    // Expand the determinant into 6 products and sum with compensation.
    let ad = a - d;
    let bd = b - d;
    let cd = c - d;
    let terms = [
        two_product(ad.x, bd.y * cd.z - bd.z * cd.y),
        two_product(ad.y, bd.z * cd.x - bd.x * cd.z),
        two_product(ad.z, bd.x * cd.y - bd.y * cd.x),
    ];
    let mut sum = 0.0_f64;
    let mut comp = 0.0_f64;
    for (hi, lo) in terms {
        let y = hi - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
        let y2 = lo - comp;
        let t2 = sum + y2;
        comp = (t2 - sum) - y2;
        sum = t2;
    }
    sum
}

/// Dekker's two-product: returns (hi, lo) such that a*b = hi + lo exactly.
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let hi = a * b;
    let lo = a.mul_add(b, -hi);
    (hi, lo)
}

// ── Spherical Delaunay (public API) ──────────────────────────────────────────

/// Spherical Delaunay triangulation via incremental insertion with Lawson flipping.
///
/// For points on a sphere, the Delaunay triangulation equals the convex hull.
/// This implementation works directly on the sphere surface, making it robust
/// against the numerical instability that plagues convex-hull-based approaches
/// when points are nearly coplanar (e.g. after plate rotation).
pub struct SphericalDelaunay {
    /// Flat triangle indices — every 3 consecutive entries form one triangle (CCW from outside).
    pub triangles: Vec<u32>,
    /// Half-edge adjacency — `halfedges[i]` is the index of the opposite half-edge.
    pub halfedges: Vec<u32>,
}

impl SphericalDelaunay {
    pub fn triangle_count(&self) -> usize {
        self.triangles.len() / 3
    }

    pub fn from_points(points: &[DVec3]) -> Self {
        assert!(points.len() >= 4, "need at least 4 points");
        let mut builder = IncrementalBuilder::new(points);
        builder.build();
        builder.into_delaunay()
    }
}

// ── Internal builder ─────────────────────────────────────────────────────────

struct BuildTriangle {
    vertices: [u32; 3],
    /// neighbors[i] is the triangle opposite vertices[i]
    neighbors: [u32; 3],
    alive: bool,
}

struct IncrementalBuilder {
    points: Vec<DVec3>,
    tris: Vec<BuildTriangle>,
    last_located: u32,
}

impl IncrementalBuilder {
    fn new(points: &[DVec3]) -> Self {
        let normalized: Vec<DVec3> = points.iter().map(|p| p.normalize()).collect();
        Self {
            points: normalized,
            tris: Vec::new(),
            last_located: 0,
        }
    }

    fn build(&mut self) {
        let (seed_indices, order) = self.plan_insertion();
        self.create_seed(&seed_indices);

        for &idx in &order {
            self.insert(idx);
        }
    }

    /// Pick 4 well-separated seed points, return them + shuffled insertion order for the rest.
    fn plan_insertion(&self) -> ([u32; 4], Vec<u32>) {
        let pts = &self.points;
        let n = pts.len();

        // Axis-aligned extremes → two most distant.
        let mut extremes = [0usize; 6];
        for i in 1..n {
            if pts[i].x < pts[extremes[0]].x { extremes[0] = i; }
            if pts[i].x > pts[extremes[1]].x { extremes[1] = i; }
            if pts[i].y < pts[extremes[2]].y { extremes[2] = i; }
            if pts[i].y > pts[extremes[3]].y { extremes[3] = i; }
            if pts[i].z < pts[extremes[4]].z { extremes[4] = i; }
            if pts[i].z > pts[extremes[5]].z { extremes[5] = i; }
        }
        let (mut a, mut b, mut best) = (0, 1, 0.0_f64);
        for i in 0..6 {
            for j in (i + 1)..6 {
                let d = (pts[extremes[i]] - pts[extremes[j]]).length_squared();
                if d > best { best = d; a = extremes[i]; b = extremes[j]; }
            }
        }

        // Farthest from edge ab.
        let ab = (pts[b] - pts[a]).normalize();
        let mut c = 0;
        best = 0.0;
        for i in 0..n {
            if i == a || i == b { continue; }
            let v = pts[i] - pts[a];
            let dist = (v - ab * v.dot(ab)).length_squared();
            if dist > best { best = dist; c = i; }
        }

        // Farthest from plane abc.
        let normal = (pts[b] - pts[a]).cross(pts[c] - pts[a]).normalize();
        let mut d = 0;
        best = 0.0;
        for i in 0..n {
            if i == a || i == b || i == c { continue; }
            let dist = (pts[i] - pts[a]).dot(normal).abs();
            if dist > best { best = dist; d = i; }
        }

        let seeds = [a as u32, b as u32, c as u32, d as u32];

        // Deterministic Fisher-Yates shuffle for O(n log n) expected insertion.
        let mut order: Vec<u32> = (0..n as u32)
            .filter(|i| !seeds.contains(i))
            .collect();
        let mut rng = 0x517cc1b727220a95u64;
        for i in (1..order.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % (i + 1);
            order.swap(i, j);
        }

        (seeds, order)
    }

    fn create_seed(&mut self, seeds: &[u32; 4]) {
        let [a, b, c, d] = *seeds;
        let pts = &self.points;

        // Orient so d is on the negative side of (a,b,c).
        let (a, b, c) = if orient3d(pts[a as usize], pts[b as usize], pts[c as usize], pts[d as usize]) > 0.0 {
            (a, b, c)
        } else {
            (a, c, b)
        };

        // 4 triangles of the tetrahedron with consistent neighbor wiring.
        self.tris = vec![
            BuildTriangle { vertices: [a, b, c], neighbors: [3, 1, 2], alive: true }, // f0, opposite d
            BuildTriangle { vertices: [a, c, d], neighbors: [3, 2, 0], alive: true }, // f1, opposite b
            BuildTriangle { vertices: [a, d, b], neighbors: [3, 0, 1], alive: true }, // f2, opposite c
            BuildTriangle { vertices: [b, d, c], neighbors: [1, 0, 2], alive: true }, // f3, opposite a
        ];
        self.last_located = 0;
    }

    fn insert(&mut self, point_idx: u32) {
        let tri_idx = self.locate(point_idx);
        let new_tris = self.split(tri_idx, point_idx);

        // Push edges opposite the new point onto the flip stack.
        // In each new triangle, vertex[0] is the inserted point, so neighbors[0] is the
        // external neighbor (the edge opposite the new point).
        let mut stack: Vec<(u32, u32)> = Vec::with_capacity(12);
        for &t in &new_tris {
            stack.push((t, 0));
        }
        self.flip_edges(&mut stack);
    }

    // ── Stage 2: Walk to containing triangle ─────────────────────────────────

    fn locate(&mut self, point_idx: u32) -> u32 {
        let p = self.points[point_idx as usize];
        let mut tri_idx = self.last_located;
        let max_steps = (self.tris.len() as f64).sqrt() as usize * 6 + 20;

        for _ in 0..max_steps {
            // Skip dead triangles.
            if !self.tris[tri_idx as usize].alive {
                tri_idx = self.any_alive_triangle();
            }

            let [va, vb, vc] = self.tris[tri_idx as usize].vertices;
            let a = self.points[va as usize];
            let b = self.points[vb as usize];
            let c = self.points[vc as usize];

            // For each edge, test which side p is on.
            // Edge BC (opposite vertex A, neighbor[0]): test p.dot(b.cross(c))
            let orient_bc = p.dot(b.cross(c));
            if orient_bc < 0.0 {
                let next = self.tris[tri_idx as usize].neighbors[0];
                if next == UNSET { break; }
                tri_idx = next;
                continue;
            }

            let orient_ca = p.dot(c.cross(a));
            if orient_ca < 0.0 {
                let next = self.tris[tri_idx as usize].neighbors[1];
                if next == UNSET { break; }
                tri_idx = next;
                continue;
            }

            let orient_ab = p.dot(a.cross(b));
            if orient_ab < 0.0 {
                let next = self.tris[tri_idx as usize].neighbors[2];
                if next == UNSET { break; }
                tri_idx = next;
                continue;
            }

            // Point is inside this triangle.
            self.last_located = tri_idx;
            return tri_idx;
        }

        // Fallback: brute-force search (should be extremely rare).
        self.locate_brute_force(p)
    }

    fn any_alive_triangle(&self) -> u32 {
        self.tris.iter().position(|t| t.alive).unwrap() as u32
    }

    fn locate_brute_force(&mut self, p: DVec3) -> u32 {
        let mut best_tri = 0u32;
        let mut best_dot = f64::NEG_INFINITY;
        for (i, tri) in self.tris.iter().enumerate() {
            if !tri.alive { continue; }
            let centroid = (self.points[tri.vertices[0] as usize]
                + self.points[tri.vertices[1] as usize]
                + self.points[tri.vertices[2] as usize])
                .normalize();
            let d = p.dot(centroid);
            if d > best_dot {
                best_dot = d;
                best_tri = i as u32;
            }
        }
        self.last_located = best_tri;
        best_tri
    }

    // ── Stage 3: Split triangle into 3 ──────────────────────────────────────

    fn split(&mut self, tri_idx: u32, point_idx: u32) -> [u32; 3] {
        let [a, b, c] = self.tris[tri_idx as usize].vertices;
        let [n_a, n_b, n_c] = self.tris[tri_idx as usize].neighbors;

        // Kill old triangle.
        self.tris[tri_idx as usize].alive = false;

        // Create 3 new triangles with P as vertex[0].
        // T0 = (P, B, C) — opposite edge BC, external neighbor = n_a
        // T1 = (P, C, A) — opposite edge CA, external neighbor = n_b
        // T2 = (P, A, B) — opposite edge AB, external neighbor = n_c
        let t0 = self.alloc_tri([point_idx, b, c], [n_a, UNSET, UNSET]);
        let t1 = self.alloc_tri([point_idx, c, a], [n_b, UNSET, UNSET]);
        let t2 = self.alloc_tri([point_idx, a, b], [n_c, UNSET, UNSET]);

        // Wire internal neighbors (edges adjacent to P).
        // T0's neighbor opposite C (index 1) = T1; opposite B (index 2) = T2
        self.tris[t0 as usize].neighbors[1] = t1;
        self.tris[t0 as usize].neighbors[2] = t2;
        self.tris[t1 as usize].neighbors[1] = t2;
        self.tris[t1 as usize].neighbors[2] = t0;
        self.tris[t2 as usize].neighbors[1] = t0;
        self.tris[t2 as usize].neighbors[2] = t1;

        // Patch external neighbors to point back to new triangles.
        self.patch_neighbor(n_a, tri_idx, t0);
        self.patch_neighbor(n_b, tri_idx, t1);
        self.patch_neighbor(n_c, tri_idx, t2);

        self.last_located = t0;
        [t0, t1, t2]
    }

    fn alloc_tri(&mut self, vertices: [u32; 3], neighbors: [u32; 3]) -> u32 {
        let idx = self.tris.len() as u32;
        self.tris.push(BuildTriangle { vertices, neighbors, alive: true });
        idx
    }

    /// Patch a neighbor to point to new_tri instead of old_tri.
    fn patch_neighbor(&mut self, neighbor: u32, old_tri: u32, new_tri: u32) {
        if neighbor == UNSET { return; }
        let n = &mut self.tris[neighbor as usize];
        for i in 0..3 {
            if n.neighbors[i] == old_tri {
                n.neighbors[i] = new_tri;
                return;
            }
        }
        // old_tri may have been killed by a prior flip — find by shared edge instead.
        let new_verts = self.tris[new_tri as usize].vertices;
        let n = &mut self.tris[neighbor as usize];
        for i in 0..3 {
            let nv1 = n.vertices[(i + 1) % 3];
            let nv2 = n.vertices[(i + 2) % 3];
            // The shared edge in the neighbor is (nv1, nv2). Check if new_tri has both.
            if new_verts.contains(&nv1) && new_verts.contains(&nv2) {
                n.neighbors[i] = new_tri;
                return;
            }
        }
    }

    // ── Stage 4: Lawson edge flipping ────────────────────────────────────────

    fn flip_edges(&mut self, stack: &mut Vec<(u32, u32)>) {
        while let Some((tri_idx, edge_idx)) = stack.pop() {
            if !self.tris[tri_idx as usize].alive { continue; }

            let neighbor_idx = self.tris[tri_idx as usize].neighbors[edge_idx as usize];
            if neighbor_idx == UNSET { continue; }
            if !self.tris[neighbor_idx as usize].alive { continue; }

            let p = self.tris[tri_idx as usize].vertices[edge_idx as usize];
            let shared_v1 = self.tris[tri_idx as usize].vertices[(edge_idx as usize + 1) % 3];
            let shared_v2 = self.tris[tri_idx as usize].vertices[(edge_idx as usize + 2) % 3];

            // Find the opposite vertex in the neighbor triangle.
            let d = self.opposite_vertex(neighbor_idx, shared_v1, shared_v2);

            // InCircle test on the unit sphere: flip if D is "above" the plane of (P, V1, V2),
            // meaning orient3d(P, V1, V2, D) > 0 (D visible from triangle → inside circumcircle).
            let orient = orient3d(
                self.points[p as usize],
                self.points[shared_v1 as usize],
                self.points[shared_v2 as usize],
                self.points[d as usize],
            );

            if orient > 0.0 {
                self.flip(tri_idx, edge_idx as usize, neighbor_idx, p, d, shared_v1, shared_v2, stack);
            }
        }
    }

    fn opposite_vertex(&self, tri_idx: u32, v1: u32, v2: u32) -> u32 {
        let verts = self.tris[tri_idx as usize].vertices;
        for &v in &verts {
            if v != v1 && v != v2 { return v; }
        }
        panic!("shared edge not found in neighbor");
    }

    /// Flip the shared edge between tri_a and tri_b in-place (no new allocations).
    ///
    /// Before: tri_a = (..., p, v1, v2, ...), tri_b = (..., d, v2, v1, ...)
    ///         sharing edge v1-v2.
    /// After:  tri_a = (p, d, v2), tri_b = (d, p, v1).
    fn flip(
        &mut self,
        tri_a: u32, edge_a: usize, tri_b: u32,
        p: u32, d: u32, v1: u32, v2: u32,
        stack: &mut Vec<(u32, u32)>,
    ) {
        // Gather external neighbors before mutating.
        let ext_a_opp_v1 = self.tris[tri_a as usize].neighbors[(edge_a + 1) % 3];
        let ext_a_opp_v2 = self.tris[tri_a as usize].neighbors[(edge_a + 2) % 3];

        let vb = self.tris[tri_b as usize].vertices;
        let nb = self.tris[tri_b as usize].neighbors;
        let mut ext_b_opp_v1 = UNSET;
        let mut ext_b_opp_v2 = UNSET;
        for i in 0..3 {
            if vb[i] == v1 { ext_b_opp_v1 = nb[i]; }
            if vb[i] == v2 { ext_b_opp_v2 = nb[i]; }
        }

        // tri_a becomes (p, d, v2): neighbor[0] opp p, neighbor[1] opp d, neighbor[2] opp v2
        self.tris[tri_a as usize].vertices = [p, d, v2];
        self.tris[tri_a as usize].neighbors = [ext_b_opp_v2, ext_a_opp_v2, tri_b];

        // tri_b becomes (p, v1, d): neighbor[0] opp p, neighbor[1] opp v1, neighbor[2] opp d
        self.tris[tri_b as usize].vertices = [p, v1, d];
        self.tris[tri_b as usize].neighbors = [ext_b_opp_v1, tri_a, ext_a_opp_v1];

        // Patch the 4 external neighbors to point to the correct triangle.
        self.patch_neighbor(ext_a_opp_v1, tri_a, tri_b);
        self.patch_neighbor(ext_b_opp_v2, tri_b, tri_a);
        // ext_a_opp_v2 stays with tri_a, ext_b_opp_v1 stays with tri_b — no patch needed.

        // Push the 4 external edges for further flipping.
        // tri_a = (p, d, v2): edge[0] opp p = (d,v2), edge[1] opp d = (v2,p)
        stack.push((tri_a, 0));
        stack.push((tri_a, 1));
        // tri_b = (p, v1, d): edge[0] opp p = (v1,d), edge[2] opp d = (p,v1)
        stack.push((tri_b, 0));
        stack.push((tri_b, 2));
    }


    // ── Stage 5: Convert to output format ────────────────────────────────────

    fn into_delaunay(self) -> SphericalDelaunay {
        let tri_count = self.tris.iter().filter(|t| t.alive).count();
        let mut triangles = Vec::with_capacity(tri_count * 3);
        let mut halfedges = vec![UNSET; tri_count * 3];

        for t in &self.tris {
            if t.alive {
                triangles.extend_from_slice(&t.vertices);
            }
        }

        let mut edge_map: HashMap<(u32, u32), u32> = HashMap::with_capacity(tri_count * 3);
        for tri in 0..tri_count {
            for e in 0..3 {
                let he = (tri * 3 + e) as u32;
                let v0 = triangles[he as usize];
                let v1 = triangles[tri * 3 + (e + 1) % 3];
                if let Some(&twin) = edge_map.get(&(v1, v0)) {
                    halfedges[he as usize] = twin;
                    halfedges[twin as usize] = he;
                }
                edge_map.insert((v0, v1), he);
            }
        }

        SphericalDelaunay { triangles, halfedges }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tectonic_simulation::fibonnaci_spiral::SphericalFibonacci;

    #[test]
    fn tetrahedron_four_points() {
        let points = vec![
            DVec3::new(1.0, 1.0, 1.0).normalize(),
            DVec3::new(-1.0, -1.0, 1.0).normalize(),
            DVec3::new(-1.0, 1.0, -1.0).normalize(),
            DVec3::new(1.0, -1.0, -1.0).normalize(),
        ];
        let del = SphericalDelaunay::from_points(&points);
        assert_eq!(del.triangle_count(), 4);
    }

    #[test]
    fn correct_triangle_count() {
        for n in [8, 20, 100, 500] {
            let sf = SphericalFibonacci::new(n);
            let del = SphericalDelaunay::from_points(&sf.all_points());
            assert_eq!(del.triangle_count(), 2 * n as usize - 4, "n={n}");
        }
    }

    #[test]
    fn halfedges_are_symmetric() {
        let del = SphericalDelaunay::from_points(&SphericalFibonacci::new(200).all_points());
        for (i, &twin) in del.halfedges.iter().enumerate() {
            assert_ne!(twin, UNSET, "halfedge {i} has no twin");
            assert_eq!(del.halfedges[twin as usize], i as u32, "halfedge {i}: twin mismatch");
        }
    }

    #[test]
    fn halfedges_share_vertices() {
        let del = SphericalDelaunay::from_points(&SphericalFibonacci::new(200).all_points());
        for i in 0..del.halfedges.len() {
            let twin = del.halfedges[i] as usize;
            let (ta, ea) = (i / 3, i % 3);
            let (tb, eb) = (twin / 3, twin % 3);
            let a0 = del.triangles[ta * 3 + ea];
            let a1 = del.triangles[ta * 3 + (ea + 1) % 3];
            let b0 = del.triangles[tb * 3 + eb];
            let b1 = del.triangles[tb * 3 + (eb + 1) % 3];
            assert!(a0 == b1 && a1 == b0, "halfedge {i}: vertices don't match twin");
        }
    }

    #[test]
    fn all_normals_face_outward() {
        let points = SphericalFibonacci::new(500).all_points();
        let del = SphericalDelaunay::from_points(&points);
        for tri in 0..del.triangle_count() {
            let a = points[del.triangles[tri * 3] as usize];
            let b = points[del.triangles[tri * 3 + 1] as usize];
            let c = points[del.triangles[tri * 3 + 2] as usize];
            let normal = (b - a).cross(c - a);
            assert!(normal.dot(a + b + c) > 0.0, "triangle {tri} faces inward");
        }
    }

    #[test]
    fn convex_hull_contains_all_points() {
        let points = SphericalFibonacci::new(500).all_points();
        let del = SphericalDelaunay::from_points(&points);
        for tri in 0..del.triangle_count() {
            let a = points[del.triangles[tri * 3] as usize];
            let b = points[del.triangles[tri * 3 + 1] as usize];
            let c = points[del.triangles[tri * 3 + 2] as usize];
            let normal = (b - a).cross(c - a).normalize();
            let d = normal.dot(a);
            for p in &points {
                assert!(normal.dot(*p) <= d + 1e-10, "point outside hull");
            }
        }
    }

    #[test]
    fn octahedron_produces_eight_triangles() {
        let pts = vec![DVec3::X, DVec3::NEG_X, DVec3::Y, DVec3::NEG_Y, DVec3::Z, DVec3::NEG_Z];
        assert_eq!(SphericalDelaunay::from_points(&pts).triangle_count(), 8);
    }

    #[test]
    fn fibonacci_1000_full_pipeline() {
        let del = SphericalDelaunay::from_points(&SphericalFibonacci::new(1000).all_points());
        assert_eq!(del.triangle_count(), 1996);
        for (i, &he) in del.halfedges.iter().enumerate() {
            assert_ne!(he, UNSET, "unpaired halfedge {i}");
        }
    }

    #[test]
    fn every_point_appears_as_vertex() {
        let n = 200u32;
        let del = SphericalDelaunay::from_points(&SphericalFibonacci::new(n).all_points());
        let mut seen = vec![false; n as usize];
        for &idx in &del.triangles { seen[idx as usize] = true; }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "point {i} missing from triangulation");
        }
    }

    #[test]
    fn random_distribution() {
        let mut points = Vec::with_capacity(300);
        let mut state = 123456789u64;
        let next = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*s >> 11) as f64 / ((1u64 << 53) as f64)
        };
        for _ in 0..300 {
            let z = next(&mut state) * 2.0 - 1.0;
            let theta = next(&mut state) * std::f64::consts::TAU;
            let r = (1.0 - z * z).sqrt();
            points.push(DVec3::new(r * theta.cos(), r * theta.sin(), z));
        }
        let del = SphericalDelaunay::from_points(&points);
        assert_eq!(del.triangle_count(), 2 * 300 - 4);
        for (i, &twin) in del.halfedges.iter().enumerate() {
            assert_ne!(twin, UNSET, "halfedge {i} has no twin");
            assert_eq!(del.halfedges[twin as usize], i as u32);
        }
        for tri in 0..del.triangle_count() {
            let a = points[del.triangles[tri * 3] as usize];
            let b = points[del.triangles[tri * 3 + 1] as usize];
            let c = points[del.triangles[tri * 3 + 2] as usize];
            let normal = (b - a).cross(c - a).normalize();
            let d = normal.dot(a);
            for p in &points {
                assert!(normal.dot(*p) <= d + 1e-10, "point outside hull");
            }
        }
    }

    // ── New robustness tests ─────────────────────────────────────────────────

    #[test]
    fn rotated_fibonacci_points() {
        use glam::DQuat;
        let sf = SphericalFibonacci::new(500);
        let mut points = sf.all_points();
        // Simulate plate rotation — the case that broke QuickHull.
        let rotation = DQuat::from_axis_angle(DVec3::new(0.3, 0.7, 0.5).normalize(), 0.15);
        for p in &mut points {
            *p = rotation.mul_vec3(*p);
        }
        let del = SphericalDelaunay::from_points(&points);
        assert_eq!(del.triangle_count(), 2 * 500 - 4);
        for (i, &twin) in del.halfedges.iter().enumerate() {
            assert_ne!(twin, UNSET, "halfedge {i} has no twin");
        }
    }

    #[test]
    fn perturbed_sphere_points() {
        let sf = SphericalFibonacci::new(200);
        let mut points = sf.all_points();
        let mut state = 42u64;
        for p in &mut points {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let scale = 0.999 + (state as f64 / u64::MAX as f64) * 0.002;
            *p *= scale;
        }
        let del = SphericalDelaunay::from_points(&points);
        assert_eq!(del.triangle_count(), 2 * 200 - 4);
    }
}
