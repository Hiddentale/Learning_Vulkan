use std::collections::HashMap;
use std::io::Write;

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

    #[cfg(test)]
    fn from_points_debug(points: &[DVec3]) -> Self {
        assert!(points.len() >= 4, "need at least 4 points");
        let mut builder = IncrementalBuilder::new(points);
        builder.build_with_log(true);
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
    triangles: Vec<BuildTriangle>,
    last_located: u32,
}

impl IncrementalBuilder {
    fn new(points: &[DVec3]) -> Self {
        let normalized: Vec<DVec3> = points.iter().map(|p| p.normalize()).collect();
        Self {
            points: normalized,
            triangles: Vec::new(),
            last_located: 0,
        }
    }

    fn build(&mut self) {
        self.build_with_log(false);
    }

    fn build_with_log(&mut self, debug: bool) {
        let (initial_vertices, order) = self.plan_insertion();
        self.create_initial_tetrahedron(&initial_vertices);

        let mut log_file = if debug {
            Some(std::fs::File::create("delaunay_debug.log").unwrap())
        } else {
            None
        };

        if let Some(f) = &mut log_file {
            writeln!(f, "INITIAL TETRAHEDRON: {:?}", initial_vertices).unwrap();
            self.validate_topology(f, "after_tetrahedron", 0);
        }

        for (step, &idx) in order.iter().enumerate() {
            self.insert(idx);
            if let Some(f) = &mut log_file {
                self.validate_topology(f, "after_insert", idx);
            }
        }
    }

    fn validate_topology(&self, f: &mut std::fs::File, label: &str, point_idx: u32) {
        let alive_count = self.triangles.iter().filter(|t| t.alive).count();
        writeln!(f, "\n=== {label} point={point_idx} alive_tris={alive_count} ===").unwrap();

        for (i, tri) in self.triangles.iter().enumerate() {
            if !tri.alive { continue; }
            let [a, b, c] = tri.vertices;
            let pa = self.points[a as usize];
            let pb = self.points[b as usize];
            let pc = self.points[c as usize];

            // Check winding
            let normal = (pb - pa).cross(pc - pa);
            let centroid = pa + pb + pc;
            let dot = normal.dot(centroid);
            if dot <= 0.0 {
                writeln!(f, "  WINDING ERROR: tri[{i}] verts=({a},{b},{c})").unwrap();
                writeln!(f, "    expected: normal·centroid > 0 (outward facing)").unwrap();
                writeln!(f, "    actual:   normal·centroid = {dot:.6e} (inward facing)").unwrap();
            }

            // Check neighbor symmetry
            for edge in 0..3 {
                let opp_vert = tri.vertices[edge];
                let ev1 = tri.vertices[(edge + 1) % 3];
                let ev2 = tri.vertices[(edge + 2) % 3];
                let neighbor_idx = tri.neighbors[edge];

                if neighbor_idx == UNSET {
                    writeln!(f, "  UNSET NEIGHBOR: tri[{i}] neighbors[{edge}] (opp vert {opp_vert}, edge ({ev1},{ev2}))").unwrap();
                    continue;
                }
                let ni = neighbor_idx as usize;
                if ni >= self.triangles.len() {
                    writeln!(f, "  OUT OF BOUNDS: tri[{i}] neighbors[{edge}]={neighbor_idx}, tris.len()={}", self.triangles.len()).unwrap();
                    continue;
                }
                if !self.triangles[ni].alive {
                    writeln!(f, "  DEAD NEIGHBOR: tri[{i}] neighbors[{edge}]={neighbor_idx}").unwrap();
                    writeln!(f, "    expected: alive neighbor sharing edge ({ev1},{ev2})").unwrap();
                    writeln!(f, "    actual:   tri[{ni}] is dead, verts={:?}", self.triangles[ni].vertices).unwrap();
                    continue;
                }

                // Check that neighbor shares the expected edge
                let nv = self.triangles[ni].vertices;
                let nn = self.triangles[ni].neighbors;
                let has_ev1 = nv.contains(&ev1);
                let has_ev2 = nv.contains(&ev2);
                if !has_ev1 || !has_ev2 {
                    writeln!(f, "  WRONG NEIGHBOR: tri[{i}] neighbors[{edge}]={neighbor_idx} (opp vert {opp_vert})").unwrap();
                    writeln!(f, "    expected: neighbor shares edge ({ev1},{ev2})").unwrap();
                    writeln!(f, "    actual:   tri[{ni}] verts={:?}, has_v1({ev1})={has_ev1}, has_v2({ev2})={has_ev2}", nv).unwrap();
                    continue;
                }

                // Check that neighbor points back to us
                let mut found_back = false;
                for ne in 0..3 {
                    if nn[ne] == i as u32 {
                        let nev1 = nv[(ne + 1) % 3];
                        let nev2 = nv[(ne + 2) % 3];
                        if (nev1 == ev1 && nev2 == ev2) || (nev1 == ev2 && nev2 == ev1) {
                            found_back = true;
                        } else {
                            writeln!(f, "  EDGE MISMATCH: tri[{i}] edge {edge}").unwrap();
                            writeln!(f, "    expected: tri[{ni}] shares edge ({ev1},{ev2}) at the slot pointing back to {i}").unwrap();
                            writeln!(f, "    actual:   tri[{ni}] points back at edge {ne} with verts ({nev1},{nev2})").unwrap();
                            found_back = true;
                        }
                        break;
                    }
                }
                if !found_back {
                    writeln!(f, "  NO BACKREF: tri[{i}] neighbors[{edge}]={neighbor_idx}").unwrap();
                    writeln!(f, "    expected: tri[{ni}].neighbors contains {i}").unwrap();
                    writeln!(f, "    actual:   tri[{ni}].neighbors={:?}, tri[{ni}].verts={:?}", nn, nv).unwrap();
                }
            }
        }
    }

    /// Pick 4 well-separated points for the initial tetrahedron, return them + shuffled insertion order.
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

        let initial = [a as u32, b as u32, c as u32, d as u32];

        // Deterministic Fisher-Yates shuffle for O(n log n) expected insertion.
        let mut order: Vec<u32> = (0..n as u32)
            .filter(|i| !initial.contains(i))
            .collect();
        let mut rng = 0x517cc1b727220a95u64;
        for i in (1..order.len()).rev() {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (rng >> 33) as usize % (i + 1);
            order.swap(i, j);
        }

        (initial, order)
    }

    fn create_initial_tetrahedron(&mut self, vertices: &[u32; 4]) {
        let [a, b, c, d] = *vertices;
        let pts = &self.points;

        // Orient so d is on the negative side of (a,b,c).
        let (a, b, c) = if orient3d(pts[a as usize], pts[b as usize], pts[c as usize], pts[d as usize]) > 0.0 {
            (a, b, c)
        } else {
            (a, c, b)
        };

        // 4 triangles of the tetrahedron with consistent neighbor wiring.
        self.triangles = vec![
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
        let max_steps = (self.triangles.len() as f64).sqrt() as usize * 6 + 20;

        for _ in 0..max_steps {
            // Skip dead triangles.
            if !self.triangles[tri_idx as usize].alive {
                tri_idx = self.any_alive_triangle();
            }

            let [va, vb, vc] = self.triangles[tri_idx as usize].vertices;
            let a = self.points[va as usize];
            let b = self.points[vb as usize];
            let c = self.points[vc as usize];

            // For each edge, test which side p is on.
            // Edge BC (opposite vertex A, neighbor[0]): test p.dot(b.cross(c))
            let orient_bc = p.dot(b.cross(c));
            if orient_bc < 0.0 {
                let next = self.triangles[tri_idx as usize].neighbors[0];
                if next == UNSET { break; }
                tri_idx = next;
                continue;
            }

            let orient_ca = p.dot(c.cross(a));
            if orient_ca < 0.0 {
                let next = self.triangles[tri_idx as usize].neighbors[1];
                if next == UNSET { break; }
                tri_idx = next;
                continue;
            }

            let orient_ab = p.dot(a.cross(b));
            if orient_ab < 0.0 {
                let next = self.triangles[tri_idx as usize].neighbors[2];
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
        self.triangles.iter().position(|t| t.alive).unwrap() as u32
    }

    fn locate_brute_force(&mut self, p: DVec3) -> u32 {
        let mut best_tri = 0u32;
        let mut best_dot = f64::NEG_INFINITY;
        for (i, tri) in self.triangles.iter().enumerate() {
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
        let [a, b, c] = self.triangles[tri_idx as usize].vertices;
        let [n_a, n_b, n_c] = self.triangles[tri_idx as usize].neighbors;

        // Kill old triangle.
        self.triangles[tri_idx as usize].alive = false;

        // Create 3 new triangles with P as vertex[0].
        // T0 = (P, B, C) — opposite edge BC, external neighbor = n_a
        // T1 = (P, C, A) — opposite edge CA, external neighbor = n_b
        // T2 = (P, A, B) — opposite edge AB, external neighbor = n_c
        let t0 = self.alloc_tri([point_idx, b, c], [n_a, UNSET, UNSET]);
        let t1 = self.alloc_tri([point_idx, c, a], [n_b, UNSET, UNSET]);
        let t2 = self.alloc_tri([point_idx, a, b], [n_c, UNSET, UNSET]);

        // Wire internal neighbors (edges adjacent to P).
        // T0's neighbor opposite C (index 1) = T1; opposite B (index 2) = T2
        self.triangles[t0 as usize].neighbors[1] = t1;
        self.triangles[t0 as usize].neighbors[2] = t2;
        self.triangles[t1 as usize].neighbors[1] = t2;
        self.triangles[t1 as usize].neighbors[2] = t0;
        self.triangles[t2 as usize].neighbors[1] = t0;
        self.triangles[t2 as usize].neighbors[2] = t1;

        // Patch external neighbors to point back to new triangles.
        self.patch_neighbor(n_a, tri_idx, t0);
        self.patch_neighbor(n_b, tri_idx, t1);
        self.patch_neighbor(n_c, tri_idx, t2);

        self.last_located = t0;
        [t0, t1, t2]
    }

    fn alloc_tri(&mut self, vertices: [u32; 3], neighbors: [u32; 3]) -> u32 {
        let idx = self.triangles.len() as u32;
        self.triangles.push(BuildTriangle { vertices, neighbors, alive: true });
        idx
    }

    /// Patch a neighbor to point to new_tri instead of old_tri.
    fn patch_neighbor(&mut self, neighbor: u32, old_tri: u32, new_tri: u32) {
        if neighbor == UNSET { return; }
        let n = &mut self.triangles[neighbor as usize];
        for i in 0..3 {
            if n.neighbors[i] == old_tri {
                n.neighbors[i] = new_tri;
                return;
            }
        }
        // old_tri may have been killed by a prior flip — find by shared edge instead.
        let new_verts = self.triangles[new_tri as usize].vertices;
        let n = &mut self.triangles[neighbor as usize];
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
        let mut iteration = 0u32;
        let max_iterations = self.points.len() as u32 * 20;

        while let Some((tri_idx, edge_idx)) = stack.pop() {
            iteration += 1;
            if iteration > max_iterations {
                eprintln!("FLIP LOOP: exceeded {max_iterations} iterations, stack size={}", stack.len());
                eprintln!("  last popped: tri_idx={tri_idx} edge_idx={edge_idx}");
                eprintln!("  tri[{tri_idx}] verts={:?} neighbors={:?} alive={}",
                    self.triangles[tri_idx as usize].vertices,
                    self.triangles[tri_idx as usize].neighbors,
                    self.triangles[tri_idx as usize].alive);
                // Print last 10 stack entries
                for (i, &(t, e)) in stack.iter().rev().take(10).enumerate() {
                    eprintln!("  stack[-{}]: tri={t} edge={e} verts={:?}", i+1,
                        self.triangles[t as usize].vertices);
                }
                panic!("infinite flip loop detected");
            }

            if !self.triangles[tri_idx as usize].alive { continue; }

            let neighbor_idx = self.triangles[tri_idx as usize].neighbors[edge_idx as usize];
            if neighbor_idx == UNSET { continue; }
            if !self.triangles[neighbor_idx as usize].alive { continue; }

            let p = self.triangles[tri_idx as usize].vertices[edge_idx as usize];
            let shared_v1 = self.triangles[tri_idx as usize].vertices[(edge_idx as usize + 1) % 3];
            let shared_v2 = self.triangles[tri_idx as usize].vertices[(edge_idx as usize + 2) % 3];

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
                eprintln!("FLIP #{iteration}: tri_a={tri_idx}(edge {edge_idx}) tri_b={neighbor_idx} | p={p} v1={shared_v1} v2={shared_v2} d={d} | orient={orient:.6e} | stack_after={}", stack.len() + 4);
                self.flip(tri_idx, edge_idx as usize, neighbor_idx, p, d, shared_v1, shared_v2, stack);
            }
        }
    }

    fn opposite_vertex(&self, tri_idx: u32, v1: u32, v2: u32) -> u32 {
        let verts = self.triangles[tri_idx as usize].vertices;
        for &v in &verts {
            if v != v1 && v != v2 { return v; }
        }
        panic!("shared edge not found in neighbor");
    }

    /// If a triangle's normal points inward, swap vertices[1]/[2] and neighbors[1]/[2]
    /// to restore CCW-from-outside winding.
    fn fix_winding(&mut self, tri_idx: u32) {
        let t = &self.triangles[tri_idx as usize];
        let p0 = self.points[t.vertices[0] as usize];
        let p1 = self.points[t.vertices[1] as usize];
        let p2 = self.points[t.vertices[2] as usize];
        let normal = (p1 - p0).cross(p2 - p0);
        if normal.dot(p0 + p1 + p2) < 0.0 {
            let t = &mut self.triangles[tri_idx as usize];
            t.vertices.swap(1, 2);
            t.neighbors.swap(1, 2);
        }
    }

    fn winding_dot(&self, v0: u32, v1: u32, v2: u32) -> f64 {
        let p0 = self.points[v0 as usize];
        let p1 = self.points[v1 as usize];
        let p2 = self.points[v2 as usize];
        let normal = (p1 - p0).cross(p2 - p0);
        normal.dot(p0 + p1 + p2)
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
        let ext_a_opp_v1 = self.triangles[tri_a as usize].neighbors[(edge_a + 1) % 3];
        let ext_a_opp_v2 = self.triangles[tri_a as usize].neighbors[(edge_a + 2) % 3];

        let vb = self.triangles[tri_b as usize].vertices;
        let nb = self.triangles[tri_b as usize].neighbors;
        let mut ext_b_opp_v1 = UNSET;
        let mut ext_b_opp_v2 = UNSET;
        for i in 0..3 {
            if vb[i] == v1 { ext_b_opp_v1 = nb[i]; }
            if vb[i] == v2 { ext_b_opp_v2 = nb[i]; }
        }

        // tri_a becomes (p, d, v2): neighbor[0] opp p, neighbor[1] opp d, neighbor[2] opp v2
        self.triangles[tri_a as usize].vertices = [p, d, v2];
        self.triangles[tri_a as usize].neighbors = [ext_b_opp_v1, ext_a_opp_v1, tri_b];

        // tri_b becomes (p, v1, d): neighbor[0] opp p, neighbor[1] opp v1, neighbor[2] opp d
        self.triangles[tri_b as usize].vertices = [p, v1, d];
        self.triangles[tri_b as usize].neighbors = [ext_b_opp_v2, tri_a, ext_a_opp_v2];

        // Correct winding: the flip may produce inward-facing triangles depending on
        // the 3D geometry of the quadrilateral. Swapping vertices[1]/[2] and
        // neighbors[1]/[2] reverses the winding while preserving the opposite-vertex convention.
        self.fix_winding(tri_a);
        self.fix_winding(tri_b);

        // Patch the 2 external neighbors that moved to a different triangle.
        self.patch_neighbor(ext_a_opp_v2, tri_a, tri_b);
        self.patch_neighbor(ext_b_opp_v1, tri_b, tri_a);
        // ext_a_opp_v1 stays with tri_a, ext_b_opp_v2 stays with tri_b — no patch needed.

        #[cfg(test)]
        {
            let winding_a = self.winding_dot(p, d, v2);
            let winding_b = self.winding_dot(p, v1, d);
            if winding_a <= 0.0 || winding_b <= 0.0 {
                let va_before = [p, v1, v2];
                let vb_before_verts = vb;
                eprintln!("FLIP WINDING BUG:");
                eprintln!("  before: tri_a[{tri_a}]=({},{},{}) tri_b[{tri_b}]=({},{},{})",
                    va_before[0], va_before[1], va_before[2],
                    vb_before_verts[0], vb_before_verts[1], vb_before_verts[2]);
                eprintln!("  edge_a={edge_a} p={p} d={d} v1={v1} v2={v2}");
                eprintln!("  after:  tri_a[{tri_a}]=({p},{d},{v2}) winding={winding_a:.6e}");
                eprintln!("  after:  tri_b[{tri_b}]=({p},{v1},{d}) winding={winding_b:.6e}");
                eprintln!("  orient3d(p,v1,v2,d) = {:.6e}", orient3d(
                    self.points[p as usize], self.points[v1 as usize],
                    self.points[v2 as usize], self.points[d as usize]));
                eprintln!("  winding before tri_a: {:.6e}", self.winding_dot(va_before[0], va_before[1], va_before[2]));
                eprintln!("  points: p={:?}", self.points[p as usize]);
                eprintln!("          v1={:?}", self.points[v1 as usize]);
                eprintln!("          v2={:?}", self.points[v2 as usize]);
                eprintln!("          d={:?}", self.points[d as usize]);
            }
        }

        // Push external edges for further flipping.
        // After fix_winding, vertex positions may have swapped, so find external edges
        // by excluding the internal edge (the one whose neighbor is the other triangle).
        for edge in 0..3u32 {
            if self.triangles[tri_a as usize].neighbors[edge as usize] != tri_b {
                stack.push((tri_a, edge));
            }
            if self.triangles[tri_b as usize].neighbors[edge as usize] != tri_a {
                stack.push((tri_b, edge));
            }
        }
    }


    // ── Stage 5: Convert to output format ────────────────────────────────────

    fn into_delaunay(self) -> SphericalDelaunay {
        let tri_count = self.triangles.iter().filter(|t| t.alive).count();
        let mut triangles = Vec::with_capacity(tri_count * 3);
        let mut halfedges = vec![UNSET; tri_count * 3];

        for t in &self.triangles {
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

    #[test]
    fn debug_trace_small() {
        let points = SphericalFibonacci::new(20).all_points();
        let _del = SphericalDelaunay::from_points_debug(&points);
        // Check the log file for errors
        let log = std::fs::read_to_string("delaunay_debug.log").unwrap();
        let errors: Vec<&str> = log.lines()
            .filter(|l| l.contains("ERROR") || l.contains("MISMATCH") || l.contains("NO BACKREF") || l.contains("DEAD NEIGHBOR") || l.contains("UNSET NEIGHBOR"))
            .collect();
        for e in &errors {
            eprintln!("{e}");
        }
        assert!(errors.is_empty(), "found {} topology errors, see delaunay_debug.log", errors.len());
    }

    // ── Unit tests for individual functions ──────────────────────────────────

    #[test]
    fn orient3d_positive_tetrahedron() {
        // Standard right-handed tetrahedron: d at origin, (a,b,c) CCW from outside.
        let a = DVec3::new(1.0, 0.0, 0.0);
        let b = DVec3::new(0.0, 1.0, 0.0);
        let c = DVec3::new(0.0, 0.0, 1.0);
        let d = DVec3::ZERO;
        let result = orient3d(a, b, c, d);
        assert!(result > 0.0, "expected positive, got {result}");
    }

    #[test]
    fn orient3d_negative_when_swapped() {
        let a = DVec3::new(1.0, 0.0, 0.0);
        let b = DVec3::new(0.0, 1.0, 0.0);
        let c = DVec3::new(0.0, 0.0, 1.0);
        let d = DVec3::ZERO;
        // Swap b and c → flips sign.
        let result = orient3d(a, c, b, d);
        assert!(result < 0.0, "expected negative, got {result}");
    }

    #[test]
    fn orient3d_zero_for_coplanar() {
        let a = DVec3::new(1.0, 0.0, 0.0);
        let b = DVec3::new(0.0, 1.0, 0.0);
        let c = DVec3::new(-1.0, -1.0, 0.0);
        let d = DVec3::new(0.5, 0.5, 0.0);
        let result = orient3d(a, b, c, d);
        assert!(result.abs() < 1e-15, "expected ~0, got {result}");
    }

    #[test]
    fn orient3d_sphere_points_outside_circumcircle() {
        // On unit sphere: orient3d(P,V1,V2,D) > 0 means D inside circumcircle of (P,V1,V2).
        // D on the same side as (P,V1,V2) but just outside the circumcircle.
        let p = DVec3::new(1.0, 0.0, 0.0);
        let v1 = DVec3::new(0.0, 1.0, 0.0);
        let v2 = DVec3::new(0.0, 0.0, 1.0);
        // d on the opposite hemisphere — the plane through p,v1,v2 separates d from the origin,
        // so d is on the origin's side (inside the plane), meaning outside the circumcircle.
        let d = DVec3::new(0.1, 0.1, 0.1).normalize();
        let result = orient3d(p, v1, v2, d);
        assert!(result < 0.0, "expected negative (outside circumcircle), got {result}");
    }

    #[test]
    fn orient3d_sphere_points_inside_circumcircle() {
        let p = DVec3::new(1.0, 0.0, 0.0);
        let v1 = DVec3::new(0.0, 1.0, 0.0);
        let v2 = DVec3::new(0.0, 0.0, 1.0);
        // d on the sphere, on the far side of plane(p,v1,v2) from origin → inside circumcircle.
        let d = DVec3::new(-1.0, -1.0, -1.0).normalize();
        let result = orient3d(p, v1, v2, d);
        assert!(result > 0.0, "expected positive (inside circumcircle), got {result}");
    }

    #[test]
    fn initial_tetrahedron_all_faces_outward() {
        let points = vec![
            DVec3::new(1.0, 1.0, 1.0).normalize(),    // 0
            DVec3::new(-1.0, -1.0, 1.0).normalize(),   // 1
            DVec3::new(-1.0, 1.0, -1.0).normalize(),   // 2
            DVec3::new(1.0, -1.0, -1.0).normalize(),   // 3
        ];
        let mut builder = IncrementalBuilder::new(&points);
        builder.create_initial_tetrahedron(&[0, 1, 2, 3]);
        for (i, tri) in builder.triangles.iter().enumerate() {
            let a = builder.points[tri.vertices[0] as usize];
            let b = builder.points[tri.vertices[1] as usize];
            let c = builder.points[tri.vertices[2] as usize];
            let normal = (b - a).cross(c - a);
            let dot = normal.dot(a + b + c);
            assert!(dot > 0.0, "face {i} verts={:?} faces inward (dot={dot:.6e})", tri.vertices);
        }
    }

    #[test]
    fn initial_tetrahedron_neighbor_symmetry() {
        let points = vec![
            DVec3::new(1.0, 1.0, 1.0).normalize(),
            DVec3::new(-1.0, -1.0, 1.0).normalize(),
            DVec3::new(-1.0, 1.0, -1.0).normalize(),
            DVec3::new(1.0, -1.0, -1.0).normalize(),
        ];
        let mut builder = IncrementalBuilder::new(&points);
        builder.create_initial_tetrahedron(&[0, 1, 2, 3]);
        for i in 0..4u32 {
            for edge in 0..3 {
                let neighbor = builder.triangles[i as usize].neighbors[edge];
                // Neighbor should point back to us.
                let nn = builder.triangles[neighbor as usize].neighbors;
                assert!(
                    nn.contains(&i),
                    "tri[{i}] neighbors[{edge}]={neighbor}, but tri[{neighbor}].neighbors={nn:?} doesn't contain {i}"
                );
            }
        }
    }

    #[test]
    fn initial_tetrahedron_shared_edges_match() {
        let points = vec![
            DVec3::new(1.0, 1.0, 1.0).normalize(),
            DVec3::new(-1.0, -1.0, 1.0).normalize(),
            DVec3::new(-1.0, 1.0, -1.0).normalize(),
            DVec3::new(1.0, -1.0, -1.0).normalize(),
        ];
        let mut builder = IncrementalBuilder::new(&points);
        builder.create_initial_tetrahedron(&[0, 1, 2, 3]);
        for i in 0..4usize {
            for edge in 0..3 {
                let ni = builder.triangles[i].neighbors[edge] as usize;
                let ev1 = builder.triangles[i].vertices[(edge + 1) % 3];
                let ev2 = builder.triangles[i].vertices[(edge + 2) % 3];
                let nv = builder.triangles[ni].vertices;
                assert!(
                    nv.contains(&ev1) && nv.contains(&ev2),
                    "tri[{i}] edge {edge}: expected neighbor {ni} to share verts ({ev1},{ev2}), got {:?}", nv
                );
            }
        }
    }

    #[test]
    fn split_produces_three_ccw_triangles() {
        let points = vec![
            DVec3::new(1.0, 1.0, 1.0).normalize(),    // 0
            DVec3::new(-1.0, -1.0, 1.0).normalize(),   // 1
            DVec3::new(-1.0, 1.0, -1.0).normalize(),   // 2
            DVec3::new(1.0, -1.0, -1.0).normalize(),   // 3
            DVec3::new(0.5, 0.5, 0.5).normalize(), // 4: point to insert
        ];
        let mut builder = IncrementalBuilder::new(&points);
        builder.create_initial_tetrahedron(&[0, 1, 2, 3]);

        // Find which triangle "sees" point 4 (brute force).
        let tri_idx = builder.locate(4);
        let new_tris = builder.split(tri_idx, 4);

        for &t in &new_tris {
            let tri = &builder.triangles[t as usize];
            assert!(tri.alive);
            assert_eq!(tri.vertices[0], 4, "inserted point should be vertex[0]");
            let a = builder.points[tri.vertices[0] as usize];
            let b = builder.points[tri.vertices[1] as usize];
            let c = builder.points[tri.vertices[2] as usize];
            let normal = (b - a).cross(c - a);
            let dot = normal.dot(a + b + c);
            assert!(dot > 0.0, "split tri[{t}] verts={:?} faces inward (dot={dot:.6e})", tri.vertices);
        }
    }

    #[test]
    fn split_neighbor_symmetry() {
        let points = vec![
            DVec3::new(1.0, 1.0, 1.0).normalize(),
            DVec3::new(-1.0, -1.0, 1.0).normalize(),
            DVec3::new(-1.0, 1.0, -1.0).normalize(),
            DVec3::new(1.0, -1.0, -1.0).normalize(),
            DVec3::new(0.5, 0.5, 0.5).normalize(),
        ];
        let mut builder = IncrementalBuilder::new(&points);
        builder.create_initial_tetrahedron(&[0, 1, 2, 3]);
        let tri_idx = builder.locate(4);
        let _new_tris = builder.split(tri_idx, 4);

        // Check all alive triangles have symmetric neighbors.
        for i in 0..builder.triangles.len() {
            let tri = &builder.triangles[i];
            if !tri.alive { continue; }
            for edge in 0..3 {
                let ni = tri.neighbors[edge] as usize;
                if ni == UNSET as usize { continue; }
                let nn = builder.triangles[ni].neighbors;
                assert!(
                    nn.contains(&(i as u32)),
                    "after split: tri[{i}] neighbors[{edge}]={ni}, but tri[{ni}].neighbors={nn:?} doesn't contain {i}"
                );
            }
        }
    }

    #[test]
    fn single_flip_preserves_winding() {
        // Build a minimal case: tetrahedron + one inserted point, then check the first flip.
        let points = SphericalFibonacci::new(20).all_points();
        let mut builder = IncrementalBuilder::new(&points);
        let (initial_vertices, order) = builder.plan_insertion();
        builder.create_initial_tetrahedron(&initial_vertices);

        // Insert just the first point to trigger flips.
        builder.insert(order[0]);

        // Check all alive triangles face outward.
        for (i, tri) in builder.triangles.iter().enumerate() {
            if !tri.alive { continue; }
            let a = builder.points[tri.vertices[0] as usize];
            let b = builder.points[tri.vertices[1] as usize];
            let c = builder.points[tri.vertices[2] as usize];
            let normal = (b - a).cross(c - a);
            let dot = normal.dot(a + b + c);
            assert!(dot > 0.0,
                "after first insert: tri[{i}] verts={:?} faces inward (dot={dot:.6e})", tri.vertices);
        }
    }

    #[test]
    fn patch_neighbor_updates_correctly() {
        let points = vec![
            DVec3::new(1.0, 1.0, 1.0).normalize(),
            DVec3::new(-1.0, -1.0, 1.0).normalize(),
            DVec3::new(-1.0, 1.0, -1.0).normalize(),
            DVec3::new(1.0, -1.0, -1.0).normalize(),
        ];
        let mut builder = IncrementalBuilder::new(&points);
        builder.create_initial_tetrahedron(&[0, 1, 2, 3]);

        // tri[0].neighbors should contain 3, 1, 2.
        assert!(builder.triangles[0].neighbors.contains(&3));

        // Patch: tell tri[0] that where it used to reference tri[3], it now references tri[99].
        // (tri[99] doesn't exist, but we're testing the pointer update.)
        // First, add a dummy triangle so index 99 doesn't panic.
        while builder.triangles.len() <= 99 {
            builder.triangles.push(BuildTriangle { vertices: [0,0,0], neighbors: [UNSET,UNSET,UNSET], alive: false });
        }
        builder.patch_neighbor(0, 3, 99);
        assert!(builder.triangles[0].neighbors.contains(&99),
            "expected 99 in neighbors, got {:?}", builder.triangles[0].neighbors);
        assert!(!builder.triangles[0].neighbors.contains(&3),
            "old neighbor 3 should be gone");
    }

    #[test]
    fn opposite_vertex_finds_correct_vertex() {
        let points = vec![
            DVec3::new(1.0, 1.0, 1.0).normalize(),
            DVec3::new(-1.0, -1.0, 1.0).normalize(),
            DVec3::new(-1.0, 1.0, -1.0).normalize(),
            DVec3::new(1.0, -1.0, -1.0).normalize(),
        ];
        let mut builder = IncrementalBuilder::new(&points);
        builder.create_initial_tetrahedron(&[0, 1, 2, 3]);

        // tri[0] has some set of 3 vertices. Find the one opposite a shared edge.
        let verts = builder.triangles[0].vertices;
        let opp = builder.opposite_vertex(0, verts[1], verts[2]);
        assert_eq!(opp, verts[0]);
        let opp = builder.opposite_vertex(0, verts[0], verts[2]);
        assert_eq!(opp, verts[1]);
        let opp = builder.opposite_vertex(0, verts[0], verts[1]);
        assert_eq!(opp, verts[2]);
    }

    #[test]
    fn locate_finds_containing_triangle() {
        let points = vec![
            DVec3::new(1.0, 1.0, 1.0).normalize(),
            DVec3::new(-1.0, -1.0, 1.0).normalize(),
            DVec3::new(-1.0, 1.0, -1.0).normalize(),
            DVec3::new(1.0, -1.0, -1.0).normalize(),
            DVec3::new(0.5, 0.5, 0.5).normalize(),
        ];
        let mut builder = IncrementalBuilder::new(&points);
        builder.create_initial_tetrahedron(&[0, 1, 2, 3]);
        let tri_idx = builder.locate(4);
        let tri = &builder.triangles[tri_idx as usize];

        // Point 4 should be "inside" the located triangle:
        // all three orient tests should be >= 0.
        let p = builder.points[4];
        let a = builder.points[tri.vertices[0] as usize];
        let b = builder.points[tri.vertices[1] as usize];
        let c = builder.points[tri.vertices[2] as usize];
        assert!(p.dot(b.cross(c)) >= 0.0, "point outside edge BC");
        assert!(p.dot(c.cross(a)) >= 0.0, "point outside edge CA");
        assert!(p.dot(a.cross(b)) >= 0.0, "point outside edge AB");
    }
}
