use std::sync::atomic::{AtomicU64, Ordering};

use glam::DVec3;

use super::adjacency_lists::AdjacencyLists;

const UNSET: u32 = u32::MAX;

/// Tolerance for spherical triangle containment tests.
const CONTAINMENT_EPS: f64 = -1e-12;

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
    /// Number of times `locate` fell back to brute force.
    brute_force_count: AtomicU64,
}

impl SphericalDelaunay {
    pub fn triangle_count(&self) -> usize {
        self.triangles.len() / 3
    }

    pub fn from_points(points: &[DVec3]) -> Self {
        assert!(points.len() >= 4, "need at least 4 points");
        let mut adj = AdjacencyLists::new(points);
        adj.build();
        let (triangles, halfedges) = adj.into_halfedge();
        Self {
            triangles,
            halfedges,
            brute_force_count: AtomicU64::new(0),
        }
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

        self.brute_force_count.fetch_add(1, Ordering::Relaxed);
        self.locate_brute(p, points)
    }

    pub fn brute_force_count(&self) -> u64 {
        self.brute_force_count.load(Ordering::Relaxed)
    }

    pub fn reset_locate_stats(&self) {
        self.brute_force_count.store(0, Ordering::Relaxed);
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
                assert!(violation <= 1e-10, "point {pi} outside hull by {violation:.6e}");
            }
        }
    }
}
