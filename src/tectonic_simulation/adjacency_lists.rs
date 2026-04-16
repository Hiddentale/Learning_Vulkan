use std::collections::HashMap;

use glam::DVec3;

use super::orient3d::orient3d;

const UNSET: u32 = u32::MAX;

/// Tolerance for spherical triangle containment tests.
/// Prevents walk cycling at triangle edges due to floating-point error.
const CONTAINMENT_EPS: f64 = -1e-12;

/// Pre-allocated neighbor-list slots per point.
/// Each insertion adds ~3 entries; 10x provides headroom for flips.
const NEIGHBOR_SLOTS_PER_POINT: usize = 10;

/// Minimum barycentric coordinate for brute-force fallback acceptance.
/// If no triangle strictly contains the point, the closest must be within this tolerance.
const BRUTE_FORCE_TOLERANCE: f64 = -1e-2;

/// Maximum neighbor-list length before the debug dump aborts (cycle guard).
const MAX_DUMP_NEIGHBORS: usize = 20;

/// Adjacency structure where each node stores its neighbors in CCW order
/// as a circular linked list. Triangles are implicit — no explicit triangle
/// storage means no winding invariant to maintain through flips.
pub(super) struct AdjacencyLists {
    /// Neighbor node indices. Each node's neighbors form a circular linked list.
    list: Vec<u32>,
    /// Next-pointers: lptr[i] = index in `list` of the neighbor after list[i].
    lptr: Vec<u32>,
    /// lend[node] = index in `list` of the last neighbor entry for `node`.
    lend: Vec<u32>,
    /// Next free slot in list/lptr.
    lnew: usize,
    pub(super) points: Vec<DVec3>,
}

impl AdjacencyLists {
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
            if neighbors.len() > MAX_DUMP_NEIGHBORS { neighbors.push(UNSET); break; }
        }
        neighbors
    }

    pub(super) fn new(points: &[DVec3]) -> Self {
        let n = points.len();
        let capacity = NEIGHBOR_SLOTS_PER_POINT * n;
        let normalized: Vec<DVec3> = points.iter().map(|p| p.normalize()).collect();
        Self {
            list: vec![0; capacity],
            lptr: vec![0; capacity],
            lend: vec![UNSET; n],
            lnew: 0,
            points: normalized,
        }
    }

    pub(super) fn build(&mut self) {
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
        for i in 0..self.points.len() {
            if self.lend[i] == UNSET { continue; }
            let first_lp = self.lptr[self.lend[i] as usize];
            let mut lp = first_lp;
            loop {
                let j = self.list[lp as usize];
                let k = self.list[self.lptr[lp as usize] as usize];
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

    // ── Seed selection and insertion order ───────────────────────────────

    fn plan_insertion(&self) -> ([u32; 4], Vec<u32>) {
        let pts = &self.points;
        let n = pts.len();

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

        let ab = (pts[b] - pts[a]).normalize();
        let mut c = 0;
        best = 0.0;
        for i in 0..n {
            if i == a || i == b { continue; }
            let v = pts[i] - pts[a];
            let dist = (v - ab * v.dot(ab)).length_squared();
            if dist > best { best = dist; c = i; }
        }

        let normal = (pts[b] - pts[a]).cross(pts[c] - pts[a]).normalize();
        let mut d = 0;
        best = 0.0;
        for i in 0..n {
            if i == a || i == b || i == c { continue; }
            let dist = (pts[i] - pts[a]).dot(normal).abs();
            if dist > best { best = dist; d = i; }
        }

        let initial = [a as u32, b as u32, c as u32, d as u32];

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

    /// Initialize the 4-node circular neighbor lists for a tetrahedron.
    ///
    /// Vertices are oriented so (a,b,c) is CCW from outside (d inside).
    /// The four faces are: (a,b,c), (a,c,d), (a,d,b), (b,d,c).
    /// Each node's CCW neighbor fan follows from walking those faces:
    ///   a: [b, c, d],  b: [a, d, c],  c: [a, b, d],  d: [a, c, b]
    fn create_initial_tetrahedron(&mut self, vertices: [u32; 4]) {
        let [a, b, c, d] = vertices;

        let (a, b, c) = if orient3d(
            self.points[a as usize], self.points[b as usize],
            self.points[c as usize], self.points[d as usize],
        ) > 0.0 {
            (a, b, c)
        } else {
            (a, c, b)
        };

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

    // ── Point location ──────────────────────────────────────────────────

    /// Visibility walk: start at a triangle near `start`, hop toward `p`.
    /// Expected O(sqrt(n)) steps. Falls back to brute force on cycle detection.
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

            if b1 <= b2 && b1 <= b3 {
                let lp = self.list_find(self.lend[n3 as usize], n2);
                let n4 = self.list[self.lptr[lp as usize] as usize];
                n1 = n4;
                std::mem::swap(&mut n2, &mut n3);
            } else if b2 <= b3 {
                let lp = self.list_find(self.lend[n1 as usize], n3);
                let n4 = self.list[self.lptr[lp as usize] as usize];
                n2 = n3;
                n3 = n4;
            } else {
                let lp = self.list_find(self.lend[n2 as usize], n1);
                let n4 = self.list[self.lptr[lp as usize] as usize];
                n3 = n1;
                n1 = n4;
            }
        }

        self.find_brute_force(p)
    }

    /// O(n) exhaustive search. Used only as fallback when the walk cycles.
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
        debug_assert!(best_min > BRUTE_FORCE_TOLERANCE,
            "point {p:?} far from all triangles (min={best_min:.4e})");
        best_tri
    }

    // ── Interior insertion ──────────────────────────────────────────────

    fn insert_interior(&mut self, k: u32, i1: u32, i2: u32, i3: u32) {
        let lp = self.list_find(self.lend[i1 as usize], i2);
        self.list_insert(lp, k);

        let lp = self.list_find(self.lend[i2 as usize], i3);
        self.list_insert(lp, k);

        let lp = self.list_find(self.lend[i3 as usize], i1);
        self.list_insert(lp, k);

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

    // ── Lawson edge flipping ────────────────────────────────────────────

    /// Cascade Lawson flips around newly inserted node K (STRIPACK ADDNOD).
    fn enforce_delaunay(&mut self, k: u32) {
        let lpf = self.lptr[self.lend[k as usize] as usize];
        let mut io2 = self.list[lpf as usize];
        let mut lpo1 = self.lptr[lpf as usize];
        let mut io1 = self.list[lpo1 as usize];

        loop {
            let lp = self.list_find(self.lend[io1 as usize], io2);
            let in1 = self.list[self.lptr[lp as usize] as usize];

            let orient = orient3d(
                self.points[io1 as usize],
                self.points[k as usize],
                self.points[in1 as usize],
                self.points[io2 as usize],
            );

            if orient > 0.0 {
                lpo1 = self.swap(in1, k, io1, io2);
                io1 = in1;
                continue;
            }

            if lpo1 == lpf { break; }
            io2 = io1;
            lpo1 = self.lptr[lpo1 as usize];
            io1 = self.list[lpo1 as usize];
        }
    }

    /// Replace diagonal IO1-IO2 with IN1-IN2 (STRIPACK SWAP).
    /// Returns LP21 = pointer to IN1 in IN2's neighbor list after the swap.
    fn swap(&mut self, in1: u32, in2: u32, io1: u32, io2: u32) -> u32 {
        let lp = self.list_find(self.lend[io1 as usize], in2);
        let lph = self.lptr[lp as usize];
        self.lptr[lp as usize] = self.lptr[lph as usize];
        if self.lend[io1 as usize] == lph { self.lend[io1 as usize] = lp; }

        let lp = self.list_find(self.lend[in1 as usize], io1);
        let lpsav = self.lptr[lp as usize];
        self.lptr[lp as usize] = lph;
        self.list[lph as usize] = in2;
        self.lptr[lph as usize] = lpsav;

        let lp = self.list_find(self.lend[io2 as usize], in1);
        let lph = self.lptr[lp as usize];
        self.lptr[lp as usize] = self.lptr[lph as usize];
        if self.lend[io2 as usize] == lph { self.lend[io2 as usize] = lp; }

        let lp = self.list_find(self.lend[in2 as usize], io2);
        let lpsav = self.lptr[lp as usize];
        self.lptr[lp as usize] = lph;
        self.list[lph as usize] = in1;
        self.lptr[lph as usize] = lpsav;

        lph
    }

    // ── Linked list primitives ──────────────────────────────────────────

    /// Find `target` in the circular neighbor list, starting from lptr[start].
    fn list_find(&self, start: u32, target: u32) -> u32 {
        let first = self.lptr[start as usize];
        let mut lp = first;
        loop {
            if self.list[lp as usize] == target { return lp; }
            lp = self.lptr[lp as usize];
            if lp == first {
                panic!(
                    "neighbor {target} not found in list of node with lend={start}, \
                     first entry={}",
                    self.list[start as usize]
                );
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

    // ── Output conversion ───────────────────────────────────────────────

    pub(super) fn into_halfedge(self) -> (Vec<u32>, Vec<u32>) {
        let n = self.points.len();
        let mut triangles = Vec::new();

        // Emit triangle (i, j, k) only when i < j && i < k (canonical ordering).
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

        (triangles, halfedges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_insert_and_find() {
        let points = (0..100).map(|_| DVec3::X).collect::<Vec<_>>();
        let mut adj = AdjacencyLists::new(&points);
        adj.list[0] = 1; adj.list[1] = 2; adj.list[2] = 3;
        adj.lptr[0] = 1; adj.lptr[1] = 2; adj.lptr[2] = 0;
        adj.lend[0] = 2;
        adj.lnew = 3;

        let lp = adj.list_find(adj.lend[0], 2);
        assert_eq!(adj.list[lp as usize], 2);

        let lp1 = adj.list_find(adj.lend[0], 1);
        adj.list_insert(lp1, 99);
        let neighbors = adj.dump_node(0);
        assert_eq!(neighbors, vec![1, 99, 2, 3]);
    }

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

        for node in 0..4u32 {
            let neighbors = adj.dump_node(node);
            assert_eq!(neighbors.len(), 3, "node {node} should have 3 neighbors, got {:?}", neighbors);
        }
        for i in 0..4u32 {
            for &j in &adj.dump_node(i) {
                assert!(adj.dump_node(j).contains(&i),
                    "node {i} has neighbor {j}, but node {j} = {:?} doesn't have {i}", adj.dump_node(j));
            }
        }
    }

    #[test]
    fn five_point_manual_trace() {
        let points = vec![
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(0.0, 0.0, -1.0),
        ];
        let mut adj = AdjacencyLists::new(&points);
        adj.create_initial_tetrahedron([0, 1, 2, 3]);

        assert_eq!(adj.dump_node(0), vec![1, 2, 3]);
        assert_eq!(adj.dump_node(1), vec![0, 3, 2]);
        assert_eq!(adj.dump_node(2), vec![0, 1, 3]);
        assert_eq!(adj.dump_node(3), vec![0, 2, 1]);

        let (i1, i2, i3) = adj.find(adj.points[4], 0);
        assert!((i1, i2, i3) == (0, 1, 2) || (i1, i2, i3) == (1, 2, 0) || (i1, i2, i3) == (2, 0, 1),
            "expected triangle (0,1,2) in some rotation, got ({i1},{i2},{i3})");

        adj.insert_interior(4, i1, i2, i3);
        adj.validate_triangles("after insert 4");

        adj.enforce_delaunay(4);
        adj.validate_triangles("after enforce 4");

        let (triangles, halfedges) = adj.into_halfedge();
        assert_eq!(triangles.len() / 3, 6, "expected 6 triangles");

        for tri in 0..triangles.len() / 3 {
            let a = points[triangles[tri * 3] as usize];
            let b = points[triangles[tri * 3 + 1] as usize];
            let c = points[triangles[tri * 3 + 2] as usize];
            let normal = (b - a).cross(c - a);
            assert!(normal.dot(a + b + c) >= 0.0, "triangle {tri} faces inward");
        }

        for (i, &twin) in halfedges.iter().enumerate() {
            assert_ne!(twin, UNSET, "halfedge {i} has no twin");
        }
    }

    #[test]
    fn walk_matches_brute_force() {
        use crate::tectonic_simulation::fibonnaci_spiral::SphericalFibonacci;
        let points = SphericalFibonacci::new(500).all_points();
        let mut adj = AdjacencyLists::new(&points);
        adj.build();

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
        use crate::tectonic_simulation::fibonnaci_spiral::SphericalFibonacci;
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
        assert!(max_degree > 3, "no cascading observed (max degree = {max_degree})");
    }

    #[test]
    fn find_first_hull_violation() {
        use crate::tectonic_simulation::fibonnaci_spiral::SphericalFibonacci;
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

            let tris = {
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

            for &(a, b, c) in &tris {
                let pa = adj.points[a as usize];
                let pb = adj.points[b as usize];
                let pc = adj.points[c as usize];
                let normal = (pb - pa).cross(pc - pa).normalize();
                let d = normal.dot(pa);
                for (pi, p) in adj.points.iter().enumerate() {
                    if adj.lend[pi] == UNSET { continue; }
                    let violation = normal.dot(*p) - d;
                    assert!(
                        violation <= 1e-10,
                        "HULL VIOLATION at step {step} (inserted k={k}): \
                         point {pi} is {violation:.6e} above triangle ({a},{b},{c})"
                    );
                }
            }
        }
    }
}
