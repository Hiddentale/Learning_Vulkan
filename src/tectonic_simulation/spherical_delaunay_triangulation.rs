use std::collections::HashMap;

use glam::DVec3;

const UNSET: u32 = u32::MAX;

/// Tolerance for spherical triangle containment tests.
/// Prevents walk cycling at triangle edges due to floating-point error.
const CONTAINMENT_EPS: f64 = -1e-12;

// ── Robust orient3d ──────────────────────────────────────────────────────────

/// Exact sign of the 3×3 determinant | (a-d) (b-d) (c-d) |.
/// Uses an error-bounded fast path; falls back to compensated arithmetic.
fn orient3d(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> f64 {
    let ad = a - d;
    let bd = b - d;
    let cd = c - d;
    let result = ad.dot(bd.cross(cd));

    let permanent = ad.x.abs() * (bd.y * cd.z).abs()
        + ad.x.abs() * (bd.z * cd.y).abs()
        + ad.y.abs() * (bd.x * cd.z).abs()
        + ad.y.abs() * (bd.z * cd.x).abs()
        + ad.z.abs() * (bd.x * cd.y).abs()
        + ad.z.abs() * (bd.y * cd.x).abs();

    let eps = 5.0 * f64::EPSILON;
    if result.abs() > eps * permanent {
        return result;
    }

    orient3d_exact(a, b, c, d)
}

/// Compensated orient3d for the near-zero case.
fn orient3d_exact(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> f64 {
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
/// Uses STRIPACK-style per-node neighbor lists internally, which makes the flip
/// operation structurally correct (no winding invariant to maintain manually).
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
        let mut adj = AdjacencyLists::new(points);
        adj.build();
        adj.into_delaunay()
    }

    /// Locate the triangle containing `p` via halfedge walk.
    /// Returns `(tri_index, b1, b2, b3)` — unnormalized barycentric coordinates.
    /// `start_tri` is a hint for spatial coherence.
    pub fn locate(&self, p: DVec3, points: &[DVec3], start_tri: usize) -> (usize, f64, f64, f64) {
        let mut tri = start_tri.min(self.triangle_count() - 1);

        for _ in 0..self.triangle_count() {
            let base = tri * 3;
            let p1 = points[self.triangles[base] as usize];
            let p2 = points[self.triangles[base + 1] as usize];
            let p3 = points[self.triangles[base + 2] as usize];

            let b1 = p.dot(p2.cross(p3));
            let b2 = p.dot(p3.cross(p1));
            let b3 = p.dot(p1.cross(p2));

            if b1 >= CONTAINMENT_EPS && b2 >= CONTAINMENT_EPS && b3 >= CONTAINMENT_EPS {
                return (tri, b1, b2, b3);
            }

            // Hop across the edge with the most negative barycentric coordinate.
            // Halfedge layout: he[base+0] = v0→v1, he[base+1] = v1→v2, he[base+2] = v2→v0.
            // Edge opposite v0 (b1) = v1→v2 = he[base+1].
            // Edge opposite v1 (b2) = v2→v0 = he[base+2].
            // Edge opposite v2 (b3) = v0→v1 = he[base+0].
            let he = if b1 <= b2 && b1 <= b3 {
                base + 1
            } else if b2 <= b3 {
                base + 2
            } else {
                base
            };

            let twin = self.halfedges[he];
            if twin == UNSET { break; }
            tri = twin as usize / 3;
        }

        // Fallback: brute force (should rarely fire).
        self.locate_brute(p, points)
    }

    fn locate_brute(&self, p: DVec3, points: &[DVec3]) -> (usize, f64, f64, f64) {
        let mut best = (0usize, 0.0, 0.0, 0.0);
        let mut best_min = f64::NEG_INFINITY;

        for tri in 0..self.triangle_count() {
            let base = tri * 3;
            let p1 = points[self.triangles[base] as usize];
            let p2 = points[self.triangles[base + 1] as usize];
            let p3 = points[self.triangles[base + 2] as usize];
            let b1 = p.dot(p2.cross(p3));
            let b2 = p.dot(p3.cross(p1));
            let b3 = p.dot(p1.cross(p2));
            if b1 >= CONTAINMENT_EPS && b2 >= CONTAINMENT_EPS && b3 >= CONTAINMENT_EPS {
                return (tri, b1, b2, b3);
            }
            let m = b1.min(b2).min(b3);
            if m > best_min { best_min = m; best = (tri, b1, b2, b3); }
        }
        best
    }
}

// ── Per-node neighbor lists (STRIPACK-style) ─────────────────────────────────

/// Adjacency structure where each node stores its neighbors in CCW order
/// as a circular linked list. Triangles are implicit — no explicit triangle
/// storage means no winding invariant to maintain through flips.
struct AdjacencyLists {
    /// Neighbor node indices. Each node's neighbors form a circular linked list.
    list: Vec<u32>,
    /// Next-pointers: lptr[i] = index in `list` of the neighbor after list[i].
    lptr: Vec<u32>,
    /// lend[node] = index in `list` of the last neighbor entry for `node`.
    lend: Vec<u32>,
    /// Next free slot in list/lptr.
    lnew: usize,
    points: Vec<DVec3>,
}

impl AdjacencyLists {
    /// Dump a node's circular neighbor list for debugging.
    #[cfg(test)]
    fn dump_node(&self, node: u32) -> Vec<u32> {
        let mut neighbors = Vec::new();
        if self.lend[node as usize] == UNSET { return neighbors; }
        let first_lp = self.lptr[self.lend[node as usize] as usize];
        let mut lp = first_lp;
        loop {
            neighbors.push(self.list[lp as usize]);
            lp = self.lptr[lp as usize];
            if lp == first_lp { break; }
            if neighbors.len() > 20 { neighbors.push(UNSET); break; } // safety
        }
        neighbors
    }

    #[cfg(test)]
    fn dump_all(&self, label: &str) {
        eprintln!("=== {label} ===");
        for i in 0..self.points.len() {
            if self.lend[i] == UNSET { continue; }
            eprintln!("  node {i}: neighbors={:?} (lend={})", self.dump_node(i as u32), self.lend[i]);
        }
    }

    fn new(points: &[DVec3]) -> Self {
        let n = points.len();
        let capacity = 10 * n;
        let normalized: Vec<DVec3> = points.iter().map(|p| p.normalize()).collect();
        Self {
            list: vec![0; capacity],
            lptr: vec![0; capacity],
            lend: vec![UNSET; n],
            lnew: 0,
            points: normalized,
        }
    }

    fn build(&mut self) {
        let (initial, order) = self.plan_insertion();
        self.create_initial_tetrahedron(initial);

        let mut last_start = initial[0];
        for &k in &order {
            let (i1, i2, i3) = self.find(self.points[k as usize], last_start);
            self.insert_interior(k, i1, i2, i3);
            self.enforce_delaunay(k);
            last_start = k;
        }
    }

    #[cfg(test)]
    fn validate_triangles(&self, label: &str) {
        // For every consecutive pair (j, k) in node i's list, j and k must be neighbors.
        for i in 0..self.points.len() {
            if self.lend[i] == UNSET { continue; }
            let first_lp = self.lptr[self.lend[i] as usize];
            let mut lp = first_lp;
            loop {
                let j = self.list[lp as usize];
                let k = self.list[self.lptr[lp as usize] as usize];
                // Check j has k as neighbor.
                let j_neighbors = self.dump_node(j);
                if !j_neighbors.contains(&k) {
                    eprintln!("TRIANGLE EDGE MISSING ({label}): tri ({i},{j},{k}) — node {j} doesn't have neighbor {k}");
                    eprintln!("  node {i}: {:?}", self.dump_node(i as u32));
                    eprintln!("  node {j}: {:?}", j_neighbors);
                    eprintln!("  node {k}: {:?}", self.dump_node(k));
                }
                lp = self.lptr[lp as usize];
                if lp == first_lp { break; }
            }
        }
    }

    #[cfg(test)]
    fn validate_neighbor_symmetry(&self, label: &str) {
        for i in 0..self.points.len() {
            if self.lend[i] == UNSET { continue; }
            let first_lp = self.lptr[self.lend[i] as usize];
            let mut lp = first_lp;
            loop {
                let j = self.list[lp as usize] as usize;
                // j should have i in its neighbor list.
                let j_first = self.lptr[self.lend[j] as usize];
                let mut jlp = j_first;
                let mut found = false;
                loop {
                    if self.list[jlp as usize] == i as u32 { found = true; break; }
                    jlp = self.lptr[jlp as usize];
                    if jlp == j_first { break; }
                }
                if !found {
                    eprintln!("NEIGHBOR ASYMMETRY ({label}): node {i} has neighbor {j}, but node {j} does NOT have neighbor {i}");
                    eprintln!("  node {i}: {:?}", self.dump_node(i as u32));
                    eprintln!("  node {j}: {:?}", self.dump_node(j as u32));
                }
                lp = self.lptr[lp as usize];
                if lp == first_lp { break; }
            }
        }
    }

    // ── Seed selection and insertion order ───────────────────────────────────

    fn plan_insertion(&self) -> ([u32; 4], Vec<u32>) {
        let pts = &self.points;
        let n = pts.len();

        // Find two most distant points via axis-aligned extremes.
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

        // Third point: farthest from line ab.
        let ab = (pts[b] - pts[a]).normalize();
        let mut c = 0;
        best = 0.0;
        for i in 0..n {
            if i == a || i == b { continue; }
            let v = pts[i] - pts[a];
            let dist = (v - ab * v.dot(ab)).length_squared();
            if dist > best { best = dist; c = i; }
        }

        // Fourth point: farthest from plane abc.
        let normal = (pts[b] - pts[a]).cross(pts[c] - pts[a]).normalize();
        let mut d = 0;
        best = 0.0;
        for i in 0..n {
            if i == a || i == b || i == c { continue; }
            let dist = (pts[i] - pts[a]).dot(normal).abs();
            if dist > best { best = dist; d = i; }
        }

        let initial = [a as u32, b as u32, c as u32, d as u32];

        // Deterministic Fisher-Yates shuffle.
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

    fn create_initial_tetrahedron(&mut self, vertices: [u32; 4]) {
        let [a, b, c, d] = vertices;

        // Orient so (a, b, c) is CCW from outside (d on the inside).
        let (a, b, c) = if orient3d(
            self.points[a as usize], self.points[b as usize],
            self.points[c as usize], self.points[d as usize],
        ) > 0.0 {
            (a, b, c)
        } else {
            (a, c, b)
        };

        // 4 faces of tetrahedron, all CCW from outside:
        //   f0 = (a, b, c) — opposite d
        //   f1 = (a, c, d) — opposite b
        //   f2 = (a, d, b) — opposite c
        //   f3 = (b, d, c) — opposite a
        //
        // Each node has 3 neighbors in CCW order:
        //   a: [b, c, d]  (from faces f0, f1, f2)
        //   b: [d, c, a]  (from faces f3, f0, f2)
        //   c: [a, b, d]  (from faces f0, f3, f1) — wait, need to trace carefully
        //
        // CCW from outside at node a: looking from a outward, neighbors go CCW.
        //   a is in faces f0=(a,b,c), f1=(a,c,d), f2=(a,d,b).
        //   Going CCW around a: b → c → d → b. So a's neighbors = [b, c, d].
        //
        // CCW at node b: faces f0=(a,b,c), f2=(a,d,b), f3=(b,d,c).
        //   In f0: a,c are neighbors. In f2: a,d. In f3: d,c.
        //   Going CCW around b: c → a → d → c. So b's neighbors = [c, a, d].
        //
        // CCW at node c: faces f0=(a,b,c), f3=(b,d,c), f1=(a,c,d).
        //   In f0: a,b. In f3: b,d. In f1: a,d.
        //   Going CCW around c: b → d → a → b? No...
        //   f0 has edges a→b→c, so at c the incoming edge is b→c and outgoing is c→a.
        //   f3 has edges b→d→c, so at c the incoming edge is d→c and outgoing is c→b.
        //   f1 has edges a→c→d, so at c the incoming edge is a→c and outgoing is c→d.
        //   CCW at c: after a comes d (f1), after d comes b (f3), after b comes a (f0).
        //   So c's neighbors = [a, d, b].
        //
        // CCW at node d: faces f1=(a,c,d), f2=(a,d,b), f3=(b,d,c).
        //   f1 at d: incoming c→d, outgoing d→a. Next after c is a.
        //   f2 at d: incoming a→d, outgoing d→b. Next after a is b.
        //   f3 at d: incoming b→d, outgoing d→c. Next after b is c.
        //   CCW at d: c → a → b → c. Wait, let me use: [c, a, b]?
        //   Or: f3 has b,c. f1 has a,c. f2 has a,b.
        //   Going CCW: b → c → a → b. So d's neighbors = [b, c, a].

        // CCW fan order: for triangle (A,B,C), at node A pair is (B,C),
        // at node B pair is (C,A), at node C pair is (A,B).
        //
        // Faces: f0=(a,b,c), f1=(a,c,d), f2=(a,d,b), f3=(b,d,c).
        //
        // Node a: f0 gives (b,c), f1 gives (c,d), f2 gives (d,b). Chain: b→c→d→b.
        // Node b: f0@B gives (c,a), f2@C gives (a,d), f3@A gives (d,c). Chain: c→a→d→c.
        //   Wait: f2=(a,d,b), at B=b: pair (A,B)=(a,d)? No — b is C in f2.
        //   f2=(a,d,b): A=a,B=d,C=b. At C=b: pair (A,B)=(a,d).
        //   f3=(b,d,c): A=b,B=d,C=c. At A=b: pair (B,C)=(d,c).
        //   f0=(a,b,c): A=a,B=b,C=c. At B=b: pair (C,A)=(c,a).
        //   Chain at b: a→d (from f2@C), d→c (from f3@A), c→a (from f0@B). = a→d→c→a.
        //   CCW at b: [a, d, c].
        //
        // Node c: f0=(a,b,c) at C=c: pair (A,B)=(a,b).
        //   f3=(b,d,c) at C=c: pair (A,B)=(b,d).
        //   f1=(a,c,d) at B=c: pair (C,A)=(d,a).
        //   Chain at c: a→b (f0), b→d (f3), d→a (f1). = a→b→d→a.
        //   CCW at c: [a, b, d].
        //
        // Node d: f1=(a,c,d) at C=d: pair (A,B)=(a,c).
        //   f2=(a,d,b) at B=d: pair (C,A)=(b,a).
        //   f3=(b,d,c) at B=d: pair (C,A)=(c,b).
        //   Chain at d: a→c (f1), c→b (f3), b→a (f2). = a→c→b→a.
        //   CCW at d: [a, c, b]. Hmm, let me double-check with chain:
        //   b→a (from f2), a→c (from f1), c→b (from f3). = b→a→c→b.
        //   So [b, a, c] or equivalently starting from a: [a, c, b].

        // Node a: [b, c, d]
        self.list[0] = b; self.list[1] = c; self.list[2] = d;
        self.lptr[0] = 1; self.lptr[1] = 2; self.lptr[2] = 0;
        self.lend[a as usize] = 2;

        // Node b: [a, d, c]
        self.list[3] = a; self.list[4] = d; self.list[5] = c;
        self.lptr[3] = 4; self.lptr[4] = 5; self.lptr[5] = 3;
        self.lend[b as usize] = 5;

        // Node c: [a, b, d]
        self.list[6] = a; self.list[7] = b; self.list[8] = d;
        self.lptr[6] = 7; self.lptr[7] = 8; self.lptr[8] = 6;
        self.lend[c as usize] = 8;

        // Node d: [a, c, b]
        self.list[9] = a; self.list[10] = c; self.list[11] = b;
        self.lptr[9] = 10; self.lptr[10] = 11; self.lptr[11] = 9;
        self.lend[d as usize] = 11;

        self.lnew = 12;
    }

    // ── Point location ──────────────────────────────────────────────────────

    /// Visibility walk: start at a triangle near `start`, hop toward `p`.
    /// Expected O(√n) steps. Falls back to brute force on cycle detection.
    fn find(&self, p: DVec3, start: u32) -> (u32, u32, u32) {
        let s = if self.lend[start as usize] != UNSET { start } else { 0 };
        let lp_end = self.lend[s as usize];
        let lp_first = self.lptr[lp_end as usize];
        let mut n1 = s;
        let mut n2 = self.list[lp_first as usize];
        let mut n3 = self.list[self.lptr[lp_first as usize] as usize];

        for _ in 0..self.points.len() {
            let p1 = self.points[n1 as usize];
            let p2 = self.points[n2 as usize];
            let p3 = self.points[n3 as usize];

            let b1 = p.dot(p2.cross(p3));
            let b2 = p.dot(p3.cross(p1));
            let b3 = p.dot(p1.cross(p2));

            if b1 >= CONTAINMENT_EPS && b2 >= CONTAINMENT_EPS && b3 >= CONTAINMENT_EPS {
                return (n1, n2, n3);
            }

            // Cross the edge with the most negative barycentric coordinate.
            if b1 <= b2 && b1 <= b3 {
                // Cross edge n2-n3 (opposite n1).
                // Adjacent CCW triangle: (n4, n3, n2).
                let lp = self.list_find(self.lend[n3 as usize], n2);
                let n4 = self.list[self.lptr[lp as usize] as usize];
                n1 = n4;
                std::mem::swap(&mut n2, &mut n3);
            } else if b2 <= b3 {
                // Cross edge n3-n1 (opposite n2).
                // Adjacent CCW triangle: (n1, n3, n4).
                let lp = self.list_find(self.lend[n1 as usize], n3);
                let n4 = self.list[self.lptr[lp as usize] as usize];
                n2 = n3;
                n3 = n4;
            } else {
                // Cross edge n1-n2 (opposite n3).
                // Adjacent CCW triangle: (n4, n2, n1).
                let lp = self.list_find(self.lend[n2 as usize], n1);
                let n4 = self.list[self.lptr[lp as usize] as usize];
                n3 = n1;
                n1 = n4;
            }
        }

        self.find_brute_force(p)
    }

    /// O(n) exhaustive search. Used only as fallback when the walk cycles.
    /// Returns the closest triangle even when no exact containment is found
    /// (floating-point edge proximity on a closed sphere).
    fn find_brute_force(&self, p: DVec3) -> (u32, u32, u32) {
        let n = self.points.len();
        let mut best_tri = (0u32, 0u32, 0u32);
        let mut best_min = f64::NEG_INFINITY;

        for i in 0..n {
            if self.lend[i] == UNSET { continue; }
            let mut lp = self.lptr[self.lend[i] as usize];
            let first = lp;
            loop {
                let j = self.list[lp as usize];
                let k = self.list[self.lptr[lp as usize] as usize];
                let pi = self.points[i];
                let pj = self.points[j as usize];
                let pk = self.points[k as usize];
                let o1 = p.dot(pj.cross(pk));
                let o2 = p.dot(pk.cross(pi));
                let o3 = p.dot(pi.cross(pj));
                if o1 >= CONTAINMENT_EPS && o2 >= CONTAINMENT_EPS && o3 >= CONTAINMENT_EPS {
                    return (i as u32, j, k);
                }
                let min_s = o1.min(o2).min(o3);
                if min_s > best_min {
                    best_min = min_s;
                    best_tri = (i as u32, j, k);
                }
                lp = self.lptr[lp as usize];
                if lp == first { break; }
            }
        }
        // On a closed sphere every point is inside some triangle.
        // If we reach here, the point is on a triangle edge within float precision.
        debug_assert!(best_min > -1e-2,
            "point {p:?} far from all triangles (min={best_min:.4e})");
        best_tri
    }

    // ── Interior insertion ──────────────────────────────────────────────────

    fn insert_interior(&mut self, k: u32, i1: u32, i2: u32, i3: u32) {
        // Insert k into each vertex's neighbor list at the correct position.
        let lp = self.list_find(self.lend[i1 as usize], i2);
        self.list_insert(lp, k);

        let lp = self.list_find(self.lend[i2 as usize], i3);
        self.list_insert(lp, k);

        let lp = self.list_find(self.lend[i3 as usize], i1);
        self.list_insert(lp, k);

        // Create k's neighbor list: [i1, i2, i3] in CCW order.
        let base = self.lnew;
        self.list[base] = i1;
        self.list[base + 1] = i2;
        self.list[base + 2] = i3;
        self.lptr[base] = base as u32 + 1;
        self.lptr[base + 1] = base as u32 + 2;
        self.lptr[base + 2] = base as u32;
        self.lend[k as usize] = base as u32 + 2;
        self.lnew = base + 3;
    }

    // ── Lawson edge flipping ────────────────────────────────────────────────

    /// STRIPACK ADDNOD lines 148-181: cascade Lawson flips around K.
    fn enforce_delaunay(&mut self, k: u32) {
        // IO2 = first neighbor of K, IO1 = second (STRIPACK naming).
        let lpf = self.lptr[self.lend[k as usize] as usize];
        let mut io2 = self.list[lpf as usize];
        let mut lpo1 = self.lptr[lpf as usize];
        let mut io1 = self.list[lpo1 as usize];

        loop {
            // Find IN1: at IO1, find IO2, take next entry.
            let lp = self.list_find(self.lend[io1 as usize], io2);
            let in1 = self.list[self.lptr[lp as usize] as usize];

            // SWPTST(IN1, K, IO1, IO2) = orient3d(IO1, K, IN1, IO2) > 0
            let orient = orient3d(
                self.points[io1 as usize],
                self.points[k as usize],
                self.points[in1 as usize],
                self.points[io2 as usize],
            );

            #[cfg(test)]
            eprintln!("  EDGE: k={k} io1={io1} io2={io2} in1={in1} orient={orient:.6e}");

            if orient > 0.0 {
                #[cfg(test)]
                eprintln!("  SWAP: in1={in1} k={k} io1={io1} io2={io2}");
                lpo1 = self.swap(in1, k, io1, io2);
                io1 = in1;
                continue;
            }

            // No swap. Test for termination and advance.
            if lpo1 == lpf { break; }
            io2 = io1;
            lpo1 = self.lptr[lpo1 as usize];
            io1 = self.list[lpo1 as usize];
        }
    }

    /// STRIPACK SWAP: replace diagonal IO1-IO2 with IN1-IN2.
    /// Returns LP21 = pointer to IN1 in IN2's neighbor list after the swap.
    fn swap(&mut self, in1: u32, in2: u32, io1: u32, io2: u32) -> u32 {
        // Delete io2 from io1's neighbor list.
        let lp = self.list_find(self.lend[io1 as usize], in2);
        let lph = self.lptr[lp as usize];
        self.lptr[lp as usize] = self.lptr[lph as usize];
        if self.lend[io1 as usize] == lph { self.lend[io1 as usize] = lp; }

        // Insert in2 into in1's list after io1, reusing the freed slot.
        let lp = self.list_find(self.lend[in1 as usize], io1);
        let lpsav = self.lptr[lp as usize];
        self.lptr[lp as usize] = lph;
        self.list[lph as usize] = in2;
        self.lptr[lph as usize] = lpsav;

        // Delete io1 from io2's neighbor list.
        let lp = self.list_find(self.lend[io2 as usize], in1);
        let lph = self.lptr[lp as usize];
        self.lptr[lp as usize] = self.lptr[lph as usize];
        if self.lend[io2 as usize] == lph { self.lend[io2 as usize] = lp; }

        // Insert in1 into in2's list after io2, reusing the freed slot.
        let lp = self.list_find(self.lend[in2 as usize], io2);
        let lpsav = self.lptr[lp as usize];
        self.lptr[lp as usize] = lph;
        self.list[lph as usize] = in1;
        self.lptr[lph as usize] = lpsav;

        lph
    }

    // ── Linked list primitives ──────────────────────────────────────────────

    /// Find `target` in the circular neighbor list, starting from lptr[start].
    /// Returns the pointer TO the entry containing `target` (matches STRIPACK's LSTPTR).
    fn list_find(&self, start: u32, target: u32) -> u32 {
        let mut lp = self.lptr[start as usize];
        let mut visited = Vec::new();
        loop {
            if self.list[lp as usize] == target { return lp; }
            visited.push(self.list[lp as usize]);
            lp = self.lptr[lp as usize];
            if lp == self.lptr[start as usize] {
                panic!("neighbor {target} not found in list (visited: {visited:?}, start_ptr={start}, list[start]={})", self.list[start as usize]);
            }
        }
    }

    /// Insert `neighbor` into the circular list after position `pos`.
    fn list_insert(&mut self, pos: u32, neighbor: u32) {
        let slot = self.lnew;
        self.list[slot] = neighbor;
        self.lptr[slot] = self.lptr[pos as usize];
        self.lptr[pos as usize] = slot as u32;
        self.lnew = slot + 1;
    }

    // ── Output conversion ───────────────────────────────────────────────────

    fn into_delaunay(self) -> SphericalDelaunay {
        let n = self.points.len();
        let mut triangles = Vec::new();

        // Extract triangles: for each node i, walk consecutive neighbor pairs (j, k).
        // Emit triangle (i, j, k) only when i < j && i < k (canonical ordering).
        // This ensures each triangle is emitted exactly once.
        for i in 0..n {
            if self.lend[i] == UNSET { continue; }
            let first_lp = self.lptr[self.lend[i] as usize];
            let mut lp = first_lp;
            loop {
                let j = self.list[lp as usize];
                let k = self.list[self.lptr[lp as usize] as usize];
                if (i as u32) < j && (i as u32) < k {
                    triangles.push(i as u32);
                    triangles.push(j);
                    triangles.push(k);
                }
                lp = self.lptr[lp as usize];
                if lp == first_lp { break; }
            }
        }

        // Build halfedge adjacency via edge map.
        let tri_count = triangles.len() / 3;
        let mut halfedges = vec![UNSET; triangles.len()];
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

    // ── orient3d tests ──────────────────────────────────────────────────────

    #[test]
    fn orient3d_positive_tetrahedron() {
        let a = DVec3::new(1.0, 0.0, 0.0);
        let b = DVec3::new(0.0, 1.0, 0.0);
        let c = DVec3::new(0.0, 0.0, 1.0);
        let d = DVec3::ZERO;
        assert!(orient3d(a, b, c, d) > 0.0);
    }

    #[test]
    fn orient3d_negative_when_swapped() {
        let a = DVec3::new(1.0, 0.0, 0.0);
        let b = DVec3::new(0.0, 1.0, 0.0);
        let c = DVec3::new(0.0, 0.0, 1.0);
        let d = DVec3::ZERO;
        assert!(orient3d(a, c, b, d) < 0.0);
    }

    #[test]
    fn orient3d_zero_for_coplanar() {
        let a = DVec3::new(1.0, 0.0, 0.0);
        let b = DVec3::new(0.0, 1.0, 0.0);
        let c = DVec3::new(-1.0, -1.0, 0.0);
        let d = DVec3::new(0.5, 0.5, 0.0);
        assert!(orient3d(a, b, c, d).abs() < 1e-15);
    }

    // ── List primitive tests ────────────────────────────────────────────────

    #[test]
    fn list_insert_and_find() {
        let points = (0..100).map(|_| DVec3::X).collect::<Vec<_>>();
        let mut adj = AdjacencyLists::new(&points);
        // Manually create a 3-entry circular list for node 0: [1, 2, 3]
        adj.list[0] = 1; adj.list[1] = 2; adj.list[2] = 3;
        adj.lptr[0] = 1; adj.lptr[1] = 2; adj.lptr[2] = 0;
        adj.lend[0] = 2;
        adj.lnew = 3;

        // list_find returns the pointer TO the target.
        let lp = adj.list_find(adj.lend[0], 2);
        assert_eq!(adj.list[lp as usize], 2, "list_find should return pointer TO target");

        // Insert 99 after the position holding node 1.
        let lp1 = adj.list_find(adj.lend[0], 1);
        adj.list_insert(lp1, 99);
        // List should be: 1 -> 99 -> 2 -> 3 -> (back to 1)
        let neighbors = adj.dump_node(0);
        assert_eq!(neighbors, vec![1, 99, 2, 3]);
    }

    // ── Initial triangle tests ──────────────────────────────────────────────

    #[test]
    fn initial_tetrahedron_has_correct_neighbors() {
        let points = vec![
            DVec3::new(1.0, 1.0, 1.0).normalize(),
            DVec3::new(-1.0, -1.0, 1.0).normalize(),
            DVec3::new(-1.0, 1.0, -1.0).normalize(),
            DVec3::new(1.0, -1.0, -1.0).normalize(),
        ];
        let mut adj = AdjacencyLists::new(&points);
        adj.create_initial_tetrahedron([0, 1, 2, 3]);

        // Each of the 4 nodes should have exactly 3 neighbors.
        for node in 0..4u32 {
            let neighbors = adj.dump_node(node);
            assert_eq!(neighbors.len(), 3, "node {node} should have 3 neighbors, got {:?}", neighbors);
        }
        // Neighbor symmetry: if a has b, then b has a.
        for i in 0..4u32 {
            for &j in &adj.dump_node(i) {
                assert!(adj.dump_node(j).contains(&i),
                    "node {i} has neighbor {j}, but node {j} = {:?} doesn't have {i}", adj.dump_node(j));
            }
        }
    }

    // ── Integration tests ───────────────────────────────────────────────────

    #[test]
    fn tetrahedron_four_points() {
        let points = vec![
            DVec3::new(1.0, 1.0, 1.0).normalize(),
            DVec3::new(-1.0, -1.0, 1.0).normalize(),
            DVec3::new(-1.0, 1.0, -1.0).normalize(),
            DVec3::new(1.0, -1.0, -1.0).normalize(),
        ];
        // Use debug build to trace what happens.
        let mut adj = AdjacencyLists::new(&points);
        adj.build();
        let del = adj.into_delaunay();
        assert_eq!(del.triangle_count(), 4);
    }

    #[test]
    fn correct_triangle_count() {
        // Debug the n=8 case first.
        {
            let sf = SphericalFibonacci::new(8);
            let points = sf.all_points();
            let mut adj = AdjacencyLists::new(&points);
            adj.build();
            let del = adj.into_delaunay();
            assert_eq!(del.triangle_count(), 2 * 8 - 4, "n=8");
        }
        for n in [20, 100, 500] {
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
    fn five_point_manual_trace() {
        let points = vec![
            DVec3::new(1.0, 0.0, 0.0),   // 0: +X
            DVec3::new(-1.0, 0.0, 0.0),  // 1: -X
            DVec3::new(0.0, 1.0, 0.0),   // 2: +Y
            DVec3::new(0.0, 0.0, 1.0),   // 3: +Z
            DVec3::new(0.0, 0.0, -1.0),  // 4: -Z
        ];
        let mut adj = AdjacencyLists::new(&points);
        adj.create_initial_tetrahedron([0, 1, 2, 3]);

        eprintln!("INITIAL TETRAHEDRON:");
        for i in 0..4u32 { eprintln!("  node {i}: {:?}", adj.dump_node(i)); }

        // Expected: node 0: [1,2,3], node 1: [0,3,2], node 2: [0,1,3], node 3: [0,2,1]
        assert_eq!(adj.dump_node(0), vec![1, 2, 3], "node 0 wrong");
        assert_eq!(adj.dump_node(1), vec![0, 3, 2], "node 1 wrong");
        assert_eq!(adj.dump_node(2), vec![0, 1, 3], "node 2 wrong");
        assert_eq!(adj.dump_node(3), vec![0, 2, 1], "node 3 wrong");

        // Locate point 4
        let (i1, i2, i3) = adj.find(adj.points[4], 0);
        eprintln!("LOCATE 4: triangle ({i1},{i2},{i3})");
        assert!((i1, i2, i3) == (0, 1, 2) || (i1, i2, i3) == (1, 2, 0) || (i1, i2, i3) == (2, 0, 1),
            "expected triangle (0,1,2) in some rotation, got ({i1},{i2},{i3})");

        // Insert
        adj.insert_interior(4, i1, i2, i3);
        eprintln!("AFTER INSERT:");
        for i in 0..5u32 { eprintln!("  node {i}: {:?}", adj.dump_node(i)); }

        // Verify triangles are edge-consistent
        adj.validate_triangles("after insert 4");

        // Enforce Delaunay
        adj.enforce_delaunay(4);
        eprintln!("AFTER ENFORCE:");
        for i in 0..5u32 { eprintln!("  node {i}: {:?}", adj.dump_node(i)); }

        adj.validate_triangles("after enforce 4");

        // Extract and check
        let del = adj.into_delaunay();
        assert_eq!(del.triangle_count(), 6, "expected 6 triangles");

        // Check all normals face outward (>= 0 because axis-aligned octahedron
        // vertices produce great-circle triangles where normal·centroid = exactly 0).
        for tri in 0..del.triangle_count() {
            let a = points[del.triangles[tri * 3] as usize];
            let b = points[del.triangles[tri * 3 + 1] as usize];
            let c = points[del.triangles[tri * 3 + 2] as usize];
            let normal = (b - a).cross(c - a);
            assert!(normal.dot(a + b + c) >= 0.0, "triangle {tri} faces inward");
        }

        // Check all halfedges paired
        for (i, &twin) in del.halfedges.iter().enumerate() {
            assert_ne!(twin, UNSET, "halfedge {i} has no twin");
        }
    }

    #[test]
    fn find_first_hull_violation() {
        // Build incrementally and check convex hull after each insertion.
        let points_raw = SphericalFibonacci::new(500).all_points();
        let mut adj = AdjacencyLists::new(&points_raw);
        let (initial, order) = adj.plan_insertion();
        adj.create_initial_tetrahedron(initial);

        let mut last_start = initial[0];
        for (step, &k) in order.iter().enumerate() {
            let (i1, i2, i3) = adj.find(adj.points[k as usize], last_start);
            adj.insert_interior(k, i1, i2, i3);
            adj.enforce_delaunay(k);
            last_start = k;

            // Check convex hull property.
            let del = {
                // Clone-ish: extract triangles without consuming.
                let mut tris = Vec::new();
                for i in 0..adj.points.len() {
                    if adj.lend[i] == UNSET { continue; }
                    let first_lp = adj.lptr[adj.lend[i] as usize];
                    let mut lp = first_lp;
                    loop {
                        let j = adj.list[lp as usize];
                        let kk = adj.list[adj.lptr[lp as usize] as usize];
                        if (i as u32) < j && (i as u32) < kk {
                            tris.push((i as u32, j, kk));
                        }
                        lp = adj.lptr[lp as usize];
                        if lp == first_lp { break; }
                    }
                }
                tris
            };

            for &(a, b, c) in &del {
                let pa = adj.points[a as usize];
                let pb = adj.points[b as usize];
                let pc = adj.points[c as usize];
                let normal = (pb - pa).cross(pc - pa).normalize();
                let d = normal.dot(pa);
                for (pi, p) in adj.points.iter().enumerate() {
                    if adj.lend[pi] == UNSET { continue; } // not inserted yet
                    let violation = normal.dot(*p) - d;
                    if violation > 1e-10 {
                        eprintln!("HULL VIOLATION at step {step} (inserted k={k}):");
                        eprintln!("  point {pi} is {violation:.6e} above triangle ({a},{b},{c})");
                        eprintln!("  k={k} neighbors: {:?}", adj.dump_node(k));
                        panic!("first hull violation found");
                    }
                }
            }
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
            for (pi, p) in points.iter().enumerate() {
                let violation = normal.dot(*p) - d;
                if violation > 1e-10 {
                    eprintln!("HULL VIOLATION: point {pi} is {violation:.6e} above triangle {tri} = ({},{},{})",
                        del.triangles[tri * 3], del.triangles[tri * 3 + 1], del.triangles[tri * 3 + 2]);
                }
                assert!(violation <= 1e-10, "point {pi} outside hull by {violation:.6e}");
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
    }

    #[test]
    fn rotated_fibonacci_points() {
        use glam::DQuat;
        let sf = SphericalFibonacci::new(500);
        let mut points = sf.all_points();
        let rotation = DQuat::from_axis_angle(DVec3::new(0.3, 0.7, 0.5).normalize(), 0.15);
        for p in &mut points { *p = rotation.mul_vec3(*p); }
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
    fn walk_matches_brute_force() {
        let points = SphericalFibonacci::new(500).all_points();
        let mut adj = AdjacencyLists::new(&points);
        adj.build();

        // Query 50 random points; verify walk and brute force agree.
        let mut state = 98765u64;
        let next = |s: &mut u64| -> f64 {
            *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*s >> 11) as f64 / ((1u64 << 53) as f64)
        };
        for _ in 0..50 {
            let z = next(&mut state) * 2.0 - 1.0;
            let theta = next(&mut state) * std::f64::consts::TAU;
            let r = (1.0 - z * z).sqrt();
            let p = DVec3::new(r * theta.cos(), r * theta.sin(), z).normalize();

            let walk = adj.find(p, 0);
            let brute = adj.find_brute_force(p);

            // Both should return a valid triangle containing p.
            let contains = |tri: (u32, u32, u32)| {
                let p1 = adj.points[tri.0 as usize];
                let p2 = adj.points[tri.1 as usize];
                let p3 = adj.points[tri.2 as usize];
                p.dot(p2.cross(p3)) >= 0.0
                    && p.dot(p3.cross(p1)) >= 0.0
                    && p.dot(p1.cross(p2)) >= 0.0
            };
            assert!(contains(walk), "walk result doesn't contain point");
            assert!(contains(brute), "brute result doesn't contain point");
        }
    }

    #[test]
    fn enforce_delaunay_cascades() {
        // Verify that swap cascading happens: after insertion,
        // some nodes should have degree > 3 (their initial fan).
        let points = SphericalFibonacci::new(100).all_points();
        let mut adj = AdjacencyLists::new(&points);
        let (initial, order) = adj.plan_insertion();
        adj.create_initial_tetrahedron(initial);

        let mut max_degree = 3usize;
        let mut last_start = initial[0];
        for &k in &order {
            let (i1, i2, i3) = adj.find(adj.points[k as usize], last_start);
            adj.insert_interior(k, i1, i2, i3);
            adj.enforce_delaunay(k);
            let degree = adj.dump_node(k).len();
            if degree > max_degree { max_degree = degree; }
            last_start = k;
        }
        // With 100 fibonacci points, cascading must produce nodes with degree > 3.
        assert!(max_degree > 3, "no cascading observed (max degree = {max_degree})");
    }
}
