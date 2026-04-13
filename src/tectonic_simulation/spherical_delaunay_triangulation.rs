use std::collections::{BinaryHeap, HashMap};

use glam::DVec3;

const UNSET: u32 = u32::MAX;

fn orient3d(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> f64 {
    (a - d).dot((b - d).cross(c - d))
}

/// Outward-facing normal for a triangle on a convex hull that contains the origin.
fn outward_normal(a: DVec3, b: DVec3, c: DVec3) -> DVec3 {
    let n = (b - a).cross(c - a);
    if n.dot(a) < 0.0 { -n } else { n }
}

struct Facet {
    vertices: [u32; 3],
    neighbors: [u32; 3],
    normal: DVec3,
    offset: f64,
    farthest_dist: f64,
    farthest_point: u32,
    conflict: Vec<u32>,
    alive: bool,
}

impl Facet {
    fn new(v0: u32, v1: u32, v2: u32, points: &[DVec3]) -> Self {
        let normal = outward_normal(points[v0 as usize], points[v1 as usize], points[v2 as usize]);
        Self {
            vertices: [v0, v1, v2],
            neighbors: [UNSET; 3],
            offset: normal.dot(points[v0 as usize]),
            normal,
            farthest_dist: f64::NEG_INFINITY,
            farthest_point: UNSET,
            conflict: Vec::new(),
            alive: true,
        }
    }

    fn distance(&self, point: DVec3) -> f64 {
        self.normal.dot(point) - self.offset
    }
}

/// Spherical Delaunay triangulation computed via 3D convex hull.
///
/// For points on a sphere, the convex hull facets are exactly the Delaunay triangles.
/// Points must span the full sphere (origin inside their convex hull).
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
        let mut hull = QuickHull::new(points);
        hull.build();
        hull.into_delaunay()
    }
}

#[derive(Clone, Copy)]
struct FacetPriority {
    dist: f64,
    index: u32,
}

impl PartialEq for FacetPriority {
    fn eq(&self, other: &Self) -> bool {
        self.dist.total_cmp(&other.dist) == std::cmp::Ordering::Equal
    }
}

impl Eq for FacetPriority {}

impl PartialOrd for FacetPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FacetPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist.total_cmp(&other.dist)
    }
}

struct QuickHull<'a> {
    points: &'a [DVec3],
    facets: Vec<Facet>,
    queue: BinaryHeap<FacetPriority>,
}

impl<'a> QuickHull<'a> {
    fn new(points: &'a [DVec3]) -> Self {
        Self { points, facets: Vec::new(), queue: BinaryHeap::new() }
    }

    fn build(&mut self) {
        self.create_initial_tetrahedron();
        self.assign_all_points();
        self.seed_queue();
        self.expand_hull();
    }

    fn seed_queue(&mut self) {
        for fi in 0..self.facets.len() {
            if !self.facets[fi].conflict.is_empty() {
                self.queue.push(FacetPriority {
                    dist: self.facets[fi].farthest_dist,
                    index: fi as u32,
                });
            }
        }
    }

    fn create_initial_tetrahedron(&mut self) {
        let pts = self.points;
        let n = pts.len();

        // Find axis-aligned extremes and pick the two most distant.
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

        // Point farthest from edge ab.
        let ab = (pts[b] - pts[a]).normalize();
        let mut c = 0;
        best = 0.0;
        for i in 0..n {
            if i == a || i == b { continue; }
            let v = pts[i] - pts[a];
            let dist = (v - ab * v.dot(ab)).length_squared();
            if dist > best { best = dist; c = i; }
        }

        // Point farthest from plane abc.
        let normal = (pts[b] - pts[a]).cross(pts[c] - pts[a]).normalize();
        let mut d = 0;
        best = 0.0;
        for i in 0..n {
            if i == a || i == b || i == c { continue; }
            let dist = (pts[i] - pts[a]).dot(normal).abs();
            if dist > best { best = dist; d = i; }
        }

        // Orient so d is on the negative side of (a,b,c).
        let (a, b, c, d) = (a as u32, b as u32, c as u32, d as u32);
        let (a, b, c) = if orient3d(pts[a as usize], pts[b as usize], pts[c as usize], pts[d as usize]) > 0.0 {
            (a, b, c)
        } else {
            (a, c, b)
        };

        self.facets = vec![
            Facet::new(a, b, c, self.points), // f0, opposite d
            Facet::new(a, c, d, self.points), // f1, opposite b
            Facet::new(a, d, b, self.points), // f2, opposite c
            Facet::new(b, d, c, self.points), // f3, opposite a
        ];
        self.facets[0].neighbors = [3, 1, 2];
        self.facets[1].neighbors = [3, 2, 0];
        self.facets[2].neighbors = [3, 0, 1];
        self.facets[3].neighbors = [1, 0, 2];
    }

    fn assign_all_points(&mut self) {
        let initial: Vec<u32> = self.facets.iter()
            .flat_map(|f| f.vertices.iter().copied()).collect();
        for i in 0..self.points.len() {
            let idx = i as u32;
            if !initial.contains(&idx) { self.assign_point(idx); }
        }
    }

    fn assign_point(&mut self, point_idx: u32) {
        let p = self.points[point_idx as usize];
        let mut best_facet = UNSET;
        let mut best_dist = 0.0_f64;
        for (fi, facet) in self.facets.iter().enumerate() {
            if !facet.alive { continue; }
            let dist = facet.distance(p);
            if dist > best_dist { best_dist = dist; best_facet = fi as u32; }
        }
        if best_facet != UNSET {
            let f = &mut self.facets[best_facet as usize];
            f.conflict.push(point_idx);
            if best_dist > f.farthest_dist {
                f.farthest_dist = best_dist;
                f.farthest_point = point_idx;
            }
        }
    }

    fn expand_hull(&mut self) {
        let mut visited = Vec::new();

        while let Some(entry) = self.queue.pop() {
            let fi = entry.index as usize;
            if !self.facets[fi].alive || self.facets[fi].conflict.is_empty() {
                continue;
            }

            let eye = self.facets[fi].farthest_point;
            let eye_point = self.points[eye as usize];

            let mut visible = Vec::new();
            visited.resize(self.facets.len(), false);
            let mut horizon = Vec::new();
            self.find_visible(fi, eye_point, &mut visible, &mut visited, &mut horizon);

            // Reset visited flags for reuse.
            for &vi in &visible { visited[vi] = false; }

            let mut orphans = Vec::new();
            for &vi in &visible {
                for pid in std::mem::take(&mut self.facets[vi].conflict) {
                    if pid != eye { orphans.push(pid); }
                }
                self.facets[vi].alive = false;
            }

            // Sort horizon into cyclic order: each edge's v1 == next edge's v0.
            for i in 0..horizon.len() - 1 {
                let target = horizon[i].1;
                if let Some(j) = (i + 1..horizon.len()).find(|&j| horizon[j].0 == target) {
                    horizon.swap(i + 1, j);
                }
            }

            let new_start = self.facets.len();
            let horizon_len = horizon.len();
            for &(v0, v1, neighbor) in &horizon {
                let mut f = Facet::new(v0, v1, eye, self.points);
                f.neighbors[2] = neighbor as u32;
                self.facets.push(f);
            }
            for i in 0..horizon_len {
                let nfi = new_start + i;
                self.facets[nfi].neighbors[0] = (new_start + (i + 1) % horizon_len) as u32;
                self.facets[nfi].neighbors[1] = (new_start + (i + horizon_len - 1) % horizon_len) as u32;
                let old_neighbor = horizon[i].2;
                let (v0, v1) = (horizon[i].0, horizon[i].1);
                patch_neighbor(&mut self.facets, old_neighbor, v0, v1, nfi as u32);
            }

            // Conflict graph: orphans only tested against new cone facets.
            for pid in orphans {
                self.assign_point_to_range(pid, new_start);
            }

            // Enqueue new facets that received conflict points.
            for nfi in new_start..self.facets.len() {
                if !self.facets[nfi].conflict.is_empty() {
                    self.queue.push(FacetPriority {
                        dist: self.facets[nfi].farthest_dist,
                        index: nfi as u32,
                    });
                }
            }
        }
    }

    /// Assign a point to the first visible facet in `facets[start..]`.
    fn assign_point_to_range(&mut self, point_idx: u32, start: usize) {
        let p = self.points[point_idx as usize];
        let mut best_facet = UNSET;
        let mut best_dist = 0.0_f64;
        for fi in start..self.facets.len() {
            let facet = &self.facets[fi];
            if !facet.alive { continue; }
            let dist = facet.distance(p);
            if dist > best_dist {
                best_dist = dist;
                best_facet = fi as u32;
            }
        }
        if best_facet != UNSET {
            let f = &mut self.facets[best_facet as usize];
            f.conflict.push(point_idx);
            if best_dist > f.farthest_dist {
                f.farthest_dist = best_dist;
                f.farthest_point = point_idx;
            }
        }
    }

    fn find_visible(
        &self, facet_idx: usize, eye: DVec3,
        visible: &mut Vec<usize>, visited: &mut Vec<bool>,
        horizon: &mut Vec<(u32, u32, usize)>,
    ) {
        visited[facet_idx] = true;
        visible.push(facet_idx);
        let verts = self.facets[facet_idx].vertices;
        let neighbors = self.facets[facet_idx].neighbors;

        for edge in 0..3 {
            let ni = neighbors[edge] as usize;
            if visited.get(ni).copied().unwrap_or(true) { continue; }
            let neighbor = &self.facets[ni];
            if !neighbor.alive { continue; }
            if neighbor.distance(eye) > 0.0 {
                self.find_visible(ni, eye, visible, visited, horizon);
            } else {
                let v0 = verts[(edge + 1) % 3];
                let v1 = verts[(edge + 2) % 3];
                horizon.push((v0, v1, ni));
            }
        }
    }

    fn into_delaunay(self) -> SphericalDelaunay {
        let tri_count = self.facets.iter().filter(|f| f.alive).count();
        let mut triangles = Vec::with_capacity(tri_count * 3);
        let mut halfedges = vec![UNSET; tri_count * 3];

        for f in &self.facets {
            if f.alive { triangles.extend_from_slice(&f.vertices); }
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

fn patch_neighbor(facets: &mut [Facet], old_neighbor: usize, v0: u32, v1: u32, new_facet: u32) {
    let f = &mut facets[old_neighbor];
    for edge in 0..3 {
        let ev0 = f.vertices[(edge + 1) % 3];
        let ev1 = f.vertices[(edge + 2) % 3];
        if (ev0 == v1 && ev1 == v0) || (ev0 == v0 && ev1 == v1) {
            f.neighbors[edge] = new_facet;
            return;
        }
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
        // Deterministic pseudo-random points spanning the full unit sphere.
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
}
