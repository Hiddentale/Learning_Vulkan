use std::io::Write;

use glam::{DQuat, DVec3};

use super::plate_seed_placement::Adjacency;
use super::continental_collision;
use super::oceanic_crust_generation;
use super::plates::{CrustData, CrustType, OrogenyType, Plate, SampleCache, WalkResult};
use super::resample;
use super::subduction;
use super::util::plate_centroid;

/// Timestep in Myr.
const DT: f64 = 2.0;
/// Planet radius in km.
const PLANET_RADIUS: f64 = 6370.0;
/// Maximum triangle-walk steps before giving up.
const MAX_WALK_STEPS: u32 = 256;
/// Sentinel: sample not covered by any plate (gap region).
pub const NO_PLATE: u32 = u32::MAX;
const NO_TRIANGLE: u32 = u32::MAX;
/// Angular speed below which rotation is skipped.
const MIN_ANGULAR_SPEED: f64 = 1e-15;
/// Dot product threshold for 1800km subduction radius.
const SUBDUCTION_DOT_THRESHOLD: f64 = 0.9603;
/// Minimum subduction uplift to classify as active Andean orogeny.
const MIN_OROGENY_UPLIFT: f64 = 0.06;
/// Interpenetration threshold for continental collision (km).
const COLLISION_THRESHOLD: f64 = 300.0;

/// Boundary between two adjacent samples assigned to different plates.
pub struct BoundarySample {
    /// World-space midpoint of the two sample points.
    pub position: DVec3,
    pub sample_a: u32,
    pub sample_b: u32,
    pub plate_a: u32,
    pub plate_b: u32,
    pub crust_type_a: CrustType,
    pub crust_type_b: CrustType,
    pub age_a: f64,
    pub age_b: f64,
}

/// Step-range diagnostic logger.
pub struct DiagnosticLog {
    pub(super) file: std::io::BufWriter<std::fs::File>,
    start_step: usize,
    end_step: usize,
}

impl DiagnosticLog {
    pub fn new(path: &std::path::Path, start_step: usize, end_step: usize) -> Self {
        let file = std::fs::File::create(path).expect("failed to create diagnostic log");
        Self {
            file: std::io::BufWriter::new(file),
            start_step,
            end_step,
        }
    }

    pub(super) fn active(&self, step: usize) -> bool {
        step >= self.start_step && step <= self.end_step
    }
}

/// Full simulation state.
///
/// Architecture: plates own reference sub-meshes (T_k) with accumulated
/// rotations (R_k). The global sample grid is a fixed Fibonacci lattice
/// queried against plates each step via warm-start triangle walks.
/// Only R_k changes between resamples; meshes are stable.
pub struct Simulation {
    /// Global sample grid (Fibonacci points, fixed between resamples).
    pub sample_points: Vec<DVec3>,
    /// Per-sample warm-start: plate + triangle + barycentric coords.
    pub sample_cache: Vec<SampleCache>,
    /// Sample grid adjacency for boundary detection.
    pub sample_adjacency: Adjacency,
    pub plates: Vec<Plate>,
    pub time: f64,
    pub target_sample_count: u32,
    pub(super) step_count: usize,
    steps_since_resample: usize,
    steps_since_crust_generation: usize,
    steps_since_rift_check: usize,
    rift_seed: u64,
    pub(super) diagnostics: Option<DiagnosticLog>,
}

impl Simulation {
    pub fn new(
        sample_points: Vec<DVec3>,
        sample_cache: Vec<SampleCache>,
        sample_adjacency: Adjacency,
        plates: Vec<Plate>,
    ) -> Self {
        let target = sample_points.len() as u32;
        Self {
            sample_points,
            sample_cache,
            sample_adjacency,
            plates,
            time: 0.0,
            target_sample_count: target,
            step_count: 0,
            steps_since_resample: 0,
            steps_since_crust_generation: 0,
            steps_since_rift_check: 0,
            rift_seed: 0,
            diagnostics: None,
        }
    }

    pub fn enable_diagnostics(
        &mut self,
        path: &std::path::Path,
        start_step: usize,
        end_step: usize,
    ) {
        self.diagnostics = Some(DiagnosticLog::new(path, start_step, end_step));
    }

    /// Advance the simulation by one timestep.
    pub fn step(&mut self) {
        self.step_count += 1;

        self.advance_rotations();
        self.assign_samples();
        self.log_assignment_stats("after assign_samples");
        self.log_crust_summary("  crust");

        let boundary = self.detect_boundaries();
        self.phase_subduction(&boundary);
        self.phase_collision(&boundary);

        let plates_before = self.plates.len();
        self.remove_empty_plates();
        if self.plates.len() != plates_before {
            self.log_msg(&format!(
                "  remove_empty_plates: {} → {} plates",
                plates_before,
                self.plates.len()
            ));
        }

        let boundary = self.detect_boundaries();
        self.phase_rift(&boundary);
        self.phase_oceanic_gen(&boundary);

        self.time += DT;
        self.steps_since_resample += 1;
        self.maybe_resample();
    }

    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    // ── Rotation ──────────────────────────────────────────────────────

    /// Update each plate's accumulated rotation R_k. No points move.
    fn advance_rotations(&mut self) {
        for plate in &mut self.plates {
            if plate.angular_speed.abs() < MIN_ANGULAR_SPEED {
                continue;
            }
            let step_rot =
                DQuat::from_axis_angle(plate.rotation_axis, plate.angular_speed * DT);
            plate.rotation = (step_rot * plate.rotation).normalize();
        }
    }

    // ── Sample assignment (steps 1–8) ─────────────────────────────────

    /// For every sample point, find which plate covers it using the
    /// warm-start triangle walk. Most samples stay in the same plate;
    /// only boundary crossings trigger the candidate search.
    fn assign_samples(&mut self) {
        for i in 0..self.sample_points.len() {
            let p = self.sample_points[i];
            let cache = self.sample_cache[i];

            // Step 1–2: try previous plate.
            if cache.plate != NO_PLATE && (cache.plate as usize) < self.plates.len() {
                let plate = &self.plates[cache.plate as usize];
                let q = plate.to_reference(p);
                if let WalkResult::Found { triangle, bary } =
                    plate.walk_to_point(q, cache.triangle, MAX_WALK_STEPS)
                {
                    self.sample_cache[i] = SampleCache {
                        plate: cache.plate,
                        triangle,
                        bary,
                    };
                    continue;
                }
            }

            // Steps 4–6: search candidate plates.
            self.sample_cache[i] = self.search_candidates(p);
        }
    }

    /// Steps 4–6: test candidate plates via bounding caps, walk in each,
    /// resolve gaps and overlaps.
    fn search_candidates(&self, world_p: DVec3) -> SampleCache {
        let mut hits: Vec<SampleCache> = Vec::new();

        for (k, plate) in self.plates.iter().enumerate() {
            if !plate.world_point_in_cap(world_p) {
                continue;
            }
            let q = plate.to_reference(world_p);
            if let WalkResult::Found { triangle, bary } =
                plate.walk_to_point(q, 0, MAX_WALK_STEPS)
            {
                hits.push(SampleCache {
                    plate: k as u32,
                    triangle,
                    bary,
                });
            }
        }

        match hits.len() {
            0 => SampleCache {
                plate: NO_PLATE,
                triangle: NO_TRIANGLE,
                bary: [0.0; 3],
            },
            1 => hits[0],
            _ => self.resolve_overlap(&hits),
        }
    }

    /// Step 6, overlap: multiple plates claim the point. Lower-density wins
    /// (continental over oceanic; younger oceanic over older).
    fn resolve_overlap(&self, hits: &[SampleCache]) -> SampleCache {
        let mut best = 0;
        for i in 1..hits.len() {
            let ci = self.dominant_crust(&hits[i]);
            let cb = self.dominant_crust(&hits[best]);
            let i_wins = match (ci.crust_type, cb.crust_type) {
                (CrustType::Continental, CrustType::Oceanic) => true,
                (CrustType::Oceanic, CrustType::Continental) => false,
                _ => ci.age < cb.age,
            };
            if i_wins {
                best = i;
            }
        }
        hits[best]
    }

    // ── Crust access ──────────────────────────────────────────────────

    /// Barycentric-blended crust at a sample's cached location (step 7).
    pub fn interpolated_crust(&self, cache: &SampleCache) -> CrustData {
        if cache.plate == NO_PLATE {
            return CrustData::oceanic(7.0, -4.0, 0.0, DVec3::X);
        }
        let plate = &self.plates[cache.plate as usize];
        let [vi, vj, vk] = plate.triangles[cache.triangle as usize];
        CrustData::barycentric_blend(
            &plate.crust[vi as usize],
            &plate.crust[vj as usize],
            &plate.crust[vk as usize],
            cache.bary,
        )
    }

    /// Dominant-vertex crust (no blending) — for discrete fields like crust_type.
    pub fn dominant_crust(&self, cache: &SampleCache) -> &CrustData {
        let plate = &self.plates[cache.plate as usize];
        let [vi, vj, vk] = plate.triangles[cache.triangle as usize];
        let dom = if cache.bary[0] >= cache.bary[1] && cache.bary[0] >= cache.bary[2] {
            vi
        } else if cache.bary[1] >= cache.bary[2] {
            vj
        } else {
            vk
        };
        &plate.crust[dom as usize]
    }

    // ── Boundary detection ────────────────────────────────────────────

    /// Scan sample adjacency for edges crossing plate boundaries.
    fn detect_boundaries(&self) -> Vec<BoundarySample> {
        let mut boundaries = Vec::new();

        for i in 0..self.sample_points.len() {
            let ca = &self.sample_cache[i];
            if ca.plate == NO_PLATE {
                continue;
            }

            for &j in self.sample_adjacency.neighbors_of(i as u32) {
                if (j as usize) <= i {
                    continue;
                }
                let cb = &self.sample_cache[j as usize];
                if cb.plate == NO_PLATE || cb.plate == ca.plate {
                    continue;
                }

                let crust_a = self.dominant_crust(ca);
                let crust_b = self.dominant_crust(cb);
                let midpoint =
                    (self.sample_points[i] + self.sample_points[j as usize]).normalize();

                boundaries.push(BoundarySample {
                    position: midpoint,
                    sample_a: i as u32,
                    sample_b: j,
                    plate_a: ca.plate,
                    plate_b: cb.plate,
                    crust_type_a: crust_a.crust_type,
                    crust_type_b: crust_b.crust_type,
                    age_a: crust_a.age,
                    age_b: crust_b.age,
                });
            }
        }

        boundaries
    }

    // ── Physics phases (stubs — sub-modules need adapting) ────────────
    //
    // Each phase will be implemented by adapting its existing sub-module
    // to the per-plate mesh model. Key difference from the old code:
    // world-space positions come from `plate.to_world(plate.reference_points[i])`
    // instead of a global `sim.points[global_idx]` array.

    fn phase_subduction(&mut self, boundary: &[BoundarySample]) {
        let mut sources: Vec<Vec<SubductionSource>> = vec![Vec::new(); self.plates.len()];
        let mut slab_pull_points: Vec<Vec<DVec3>> = vec![Vec::new(); self.plates.len()];

        for edge in boundary {
            let result = subduction::resolve_subduction(
                edge.plate_a,
                edge.plate_b,
                edge.crust_type_a,
                edge.age_a,
                edge.crust_type_b,
                edge.age_b,
            );

            let (sub_idx, over_idx) = match result {
                subduction::SubductionResult::PlateSubducts(s) => {
                    let o = if s == edge.plate_a {
                        edge.plate_b
                    } else {
                        edge.plate_a
                    };
                    (s, o)
                }
                subduction::SubductionResult::ContinentalCollision => continue,
            };

            // Check convergence: plates must be closing.
            let pos_a = self.sample_points[edge.sample_a as usize];
            let pos_b = self.sample_points[edge.sample_b as usize];
            let vel_a = self.plates[edge.plate_a as usize].surface_velocity(edge.position);
            let vel_b = self.plates[edge.plate_b as usize].surface_velocity(edge.position);
            let gap_dir = (pos_b - pos_a).normalize_or_zero();
            if (vel_a - vel_b).dot(gap_dir) <= 0.0 {
                continue;
            }

            slab_pull_points[sub_idx as usize].push(edge.position);

            // Subducting plate's elevation at the boundary.
            let sub_cache = if sub_idx == edge.plate_a {
                &self.sample_cache[edge.sample_a as usize]
            } else {
                &self.sample_cache[edge.sample_b as usize]
            };
            let sub_elevation = if sub_cache.plate != NO_PLATE {
                self.interpolated_crust(sub_cache).elevation
            } else {
                -4.0
            };

            let vel_sub = self.plates[sub_idx as usize].surface_velocity(edge.position);
            let vel_over = self.plates[over_idx as usize].surface_velocity(edge.position);

            sources[over_idx as usize].push(SubductionSource {
                position: edge.position,
                vel_sub,
                vel_over,
                subducting_elevation: sub_elevation,
            });

            // Advance subducted_distance on subducting plate vertices near boundary.
            let relative_speed = (vel_sub - vel_over).length() * PLANET_RADIUS;
            let delta = relative_speed * DT;
            advance_nearest_vertex(&mut self.plates[sub_idx as usize], edge.position, delta);
        }

        self.apply_uplift(&sources);
        self.apply_slab_pull(&slab_pull_points);
    }

    fn phase_collision(&mut self, boundary: &[BoundarySample]) {
        // Collect continental-continental convergent boundary points by plate pair.
        let mut collision_boundaries: std::collections::HashMap<(u32, u32), Vec<DVec3>> =
            std::collections::HashMap::new();

        for edge in boundary {
            if edge.crust_type_a != CrustType::Continental
                || edge.crust_type_b != CrustType::Continental
            {
                continue;
            }

            // Check convergence.
            let pos_a = self.sample_points[edge.sample_a as usize];
            let pos_b = self.sample_points[edge.sample_b as usize];
            let vel_a = self.plates[edge.plate_a as usize].surface_velocity(edge.position);
            let vel_b = self.plates[edge.plate_b as usize].surface_velocity(edge.position);
            let gap_dir = (pos_b - pos_a).normalize_or_zero();
            if (vel_a - vel_b).dot(gap_dir) <= 0.0 {
                continue;
            }

            let pair = (edge.plate_a.min(edge.plate_b), edge.plate_a.max(edge.plate_b));
            collision_boundaries
                .entry(pair)
                .or_default()
                .push(edge.position);
        }

        for ((plate_a, plate_b), boundary_points) in &collision_boundaries {
            self.try_collision(*plate_a, *plate_b, boundary_points);
        }
    }

    fn try_collision(&mut self, plate_a: u32, plate_b: u32, boundary_points: &[DVec3]) {
        let terranes_a = continental_collision::find_terranes(&self.plates[plate_a as usize]);
        if terranes_a.is_empty() {
            return;
        }

        // Find the terrane closest to the collision boundary.
        let nearest = terranes_a
            .iter()
            .min_by(|a, b| {
                let da = min_arc_distance(a.centroid, boundary_points);
                let db = min_arc_distance(b.centroid, boundary_points);
                da.partial_cmp(&db).unwrap()
            })
            .unwrap();

        let dist_to_boundary = min_arc_distance(nearest.centroid, boundary_points);
        if dist_to_boundary > COLLISION_THRESHOLD {
            return;
        }

        // Estimate terrane area from vertex count.
        let total_verts: usize = self.plates.iter().map(|p| p.point_count()).sum();
        let terrane_area = estimate_area(nearest.vertices.len(), total_verts);

        let vel_a = self.plates[plate_a as usize].surface_velocity(nearest.centroid);
        let vel_b = self.plates[plate_b as usize].surface_velocity(nearest.centroid);
        let relative_speed = (vel_a - vel_b).length();

        let radius =
            continental_collision::influence_radius(relative_speed, terrane_area, self.plates.len());
        let dot_threshold = (radius / PLANET_RADIUS).cos();

        // Apply uplift surge to plate_b vertices within influence radius.
        let plate_b_ref = &mut self.plates[plate_b as usize];
        for i in 0..plate_b_ref.reference_points.len() {
            let world_p = plate_b_ref.to_world(plate_b_ref.reference_points[i]);
            if world_p.dot(nearest.centroid) < dot_threshold {
                continue;
            }
            let d = world_p.dot(nearest.centroid).clamp(-1.0, 1.0).acos() * PLANET_RADIUS;
            let (new_z, new_fold) = continental_collision::apply(
                plate_b_ref.crust[i].elevation,
                world_p,
                nearest.centroid,
                terrane_area,
                d,
                radius,
            );
            plate_b_ref.crust[i].elevation = new_z;
            plate_b_ref.crust[i].local_direction = new_fold;
            plate_b_ref.crust[i].orogeny_type = Some(OrogenyType::Himalayan);
        }

        // TODO: transfer_terrane — mesh surgery to move terrane from plate_a to plate_b.
    }

    fn phase_rift(&mut self, _boundary: &[BoundarySample]) {
        // TODO: adapt plate rifting module.
        // - Detect rift conditions on plate meshes
        // - Split one plate's mesh into two new plates
        // - Rebuild adjacency for the new plates
    }

    fn phase_oceanic_gen(&mut self, boundary: &[BoundarySample]) {
        self.steps_since_crust_generation += 1;
        if self.steps_since_crust_generation
            < oceanic_crust_generation::generation_interval(&self.plates)
        {
            return;
        }
        self.steps_since_crust_generation = 0;
        oceanic_crust_generation::generate_oceanic_crust(&mut self.plates, boundary);
    }

    fn apply_uplift(&mut self, sources: &[Vec<SubductionSource>]) {
        for (plate_idx, plate_sources) in sources.iter().enumerate() {
            if plate_sources.is_empty() {
                continue;
            }
            let plate = &mut self.plates[plate_idx];
            for i in 0..plate.reference_points.len() {
                let world_p = plate.to_world(plate.reference_points[i]);
                for src in plate_sources {
                    if world_p.dot(src.position) < SUBDUCTION_DOT_THRESHOLD {
                        continue;
                    }
                    let d = world_p.dot(src.position).clamp(-1.0, 1.0).acos() * PLANET_RADIUS;
                    let (new_z, new_fold, new_age) = subduction::apply_subduction_step(
                        plate.crust[i].elevation,
                        plate.crust[i].local_direction,
                        plate.crust[i].age,
                        d,
                        src.vel_sub,
                        src.vel_over,
                        src.subducting_elevation,
                        DT,
                    );
                    let uplift = subduction::subduction_uplift(
                        d,
                        (src.vel_sub - src.vel_over).length(),
                        src.subducting_elevation,
                    );
                    plate.crust[i].elevation = new_z;
                    plate.crust[i].local_direction = new_fold;
                    plate.crust[i].age = new_age;
                    if uplift > MIN_OROGENY_UPLIFT {
                        if plate.crust[i].crust_type == CrustType::Oceanic
                            && plate.crust[i].elevation > 0.0
                        {
                            plate.crust[i].crust_type = CrustType::Continental;
                            plate.crust[i].orogeny_type = Some(OrogenyType::Andean);
                        } else if plate.crust[i].crust_type == CrustType::Continental {
                            plate.crust[i].orogeny_type = Some(OrogenyType::Andean);
                        }
                    }
                }
            }
        }
    }

    fn apply_slab_pull(&mut self, slab_pull_points: &[Vec<DVec3>]) {
        for (plate_idx, pull_points) in slab_pull_points.iter().enumerate() {
            if pull_points.is_empty() {
                continue;
            }
            let center = plate_centroid(&self.plates[plate_idx]);
            self.plates[plate_idx].rotation_axis = subduction::apply_slab_pull(
                self.plates[plate_idx].rotation_axis,
                center,
                pull_points,
                DT,
            );
        }
    }

    fn remove_empty_plates(&mut self) {
        let old_len = self.plates.len();
        let mut old_to_new = vec![NO_PLATE; old_len];
        let mut new_idx = 0u32;
        for i in 0..old_len {
            if !self.plates[i].reference_points.is_empty() {
                old_to_new[i] = new_idx;
                new_idx += 1;
            }
        }
        self.plates.retain(|p| !p.reference_points.is_empty());

        for cache in &mut self.sample_cache {
            if cache.plate != NO_PLATE && (cache.plate as usize) < old_len {
                cache.plate = old_to_new[cache.plate as usize];
            }
        }
    }

    fn maybe_resample(&mut self) {
        if self.steps_since_resample < resample::resample_interval(&self.plates) {
            return;
        }
        resample::resample(self);
        self.steps_since_resample = 0;
    }

    // ── Diagnostics ───────────────────────────────────────────────────

    fn logging_active(&self) -> bool {
        self.diagnostics
            .as_ref()
            .map_or(false, |d| d.active(self.step_count))
    }

    fn log_msg(&mut self, msg: &str) {
        if let Some(diag) = &mut self.diagnostics {
            let _ = writeln!(diag.file, "{}", msg);
        }
    }

    fn log_crust_summary(&mut self, label: &str) {
        if !self.logging_active() {
            return;
        }
        let mut continental = 0usize;
        let mut oceanic = 0usize;
        let mut cont_below_sea = 0usize;
        let mut min_cont = f64::MAX;
        let mut max_cont = f64::MIN;
        let mut avg_cont = 0.0f64;
        let mut andean = 0usize;
        let mut himalayan = 0usize;

        for plate in &self.plates {
            for c in &plate.crust {
                match c.crust_type {
                    CrustType::Continental => {
                        continental += 1;
                        avg_cont += c.elevation;
                        if c.elevation < 0.0 {
                            cont_below_sea += 1;
                        }
                        min_cont = min_cont.min(c.elevation);
                        max_cont = max_cont.max(c.elevation);
                        match c.orogeny_type {
                            Some(OrogenyType::Andean) => andean += 1,
                            Some(OrogenyType::Himalayan) => himalayan += 1,
                            _ => {}
                        }
                    }
                    CrustType::Oceanic => oceanic += 1,
                }
            }
        }

        let total = continental + oceanic;
        if total == 0 {
            return;
        }
        if continental > 0 {
            avg_cont /= continental as f64;
        }

        if let Some(diag) = &mut self.diagnostics {
            let _ = writeln!(
                diag.file,
                "{}: cont={} ({:.1}%) ocean={} below_sea={} elev=[{:.3},{:.3}] avg={:.3} andean={} himalayan={}",
                label,
                continental,
                continental as f64 / total as f64 * 100.0,
                oceanic,
                cont_below_sea,
                if min_cont == f64::MAX { 0.0 } else { min_cont },
                if max_cont == f64::MIN { 0.0 } else { max_cont },
                avg_cont,
                andean,
                himalayan,
            );
        }
    }

    fn log_assignment_stats(&mut self, label: &str) {
        if !self.logging_active() {
            return;
        }
        let total = self.sample_cache.len();
        let assigned = self.sample_cache.iter().filter(|c| c.plate != NO_PLATE).count();
        let gap = total - assigned;

        let mut per_plate = vec![0usize; self.plates.len()];
        for c in &self.sample_cache {
            if c.plate != NO_PLATE && (c.plate as usize) < self.plates.len() {
                per_plate[c.plate as usize] += 1;
            }
        }
        let plate_sizes: Vec<usize> = self.plates.iter().map(|p| p.point_count()).collect();

        if let Some(diag) = &mut self.diagnostics {
            let _ = writeln!(
                diag.file,
                "  {} step={}: assigned={}/{} gap={} plates={}",
                label, self.step_count, assigned, total, gap, self.plates.len()
            );
            for (i, (&samples, &verts)) in per_plate.iter().zip(plate_sizes.iter()).enumerate() {
                let _ = writeln!(
                    diag.file,
                    "    plate[{}]: {} samples, {} mesh verts, {} tris",
                    i, samples, verts, self.plates[i].triangle_count()
                );
            }
        }
    }
}

fn arc_distance(a: DVec3, b: DVec3) -> f64 {
    a.normalize().dot(b.normalize()).clamp(-1.0, 1.0).acos() * PLANET_RADIUS
}

fn min_arc_distance(point: DVec3, targets: &[DVec3]) -> f64 {
    targets
        .iter()
        .map(|&t| arc_distance(point, t))
        .fold(f64::MAX, f64::min)
}

fn estimate_area(terrane_points: usize, total_points: usize) -> f64 {
    let total_area = 4.0 * std::f64::consts::PI * PLANET_RADIUS * PLANET_RADIUS;
    total_area * terrane_points as f64 / total_points as f64
}

#[derive(Clone)]
struct SubductionSource {
    position: DVec3,
    vel_sub: DVec3,
    vel_over: DVec3,
    subducting_elevation: f64,
}

/// Advance `subducted_distance` on the plate vertex nearest to a boundary point.
fn advance_nearest_vertex(plate: &mut Plate, world_boundary: DVec3, delta: f64) {
    let ref_p = plate.to_reference(world_boundary);
    let mut best_dot = f64::NEG_INFINITY;
    let mut best_i = 0;
    for (i, &v) in plate.reference_points.iter().enumerate() {
        let dot = v.dot(ref_p);
        if dot > best_dot {
            best_dot = dot;
            best_i = i;
        }
    }
    plate.crust[best_i].subducted_distance += delta;
}

#[cfg(test)]
mod tests {
    use super::super::fibonnaci_spiral::SphericalFibonacci;
    use super::super::plate_initializer::{initialize_plates, InitParams};
    use super::super::plate_seed_placement::{assign_plates, Adjacency};
    use super::super::spherical_delaunay_triangulation::SphericalDelaunay;
    use super::*;

    fn make_sim(point_count: u32, plate_count: u32) -> Simulation {
        let fib = SphericalFibonacci::new(point_count);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, plate_count, 42);
        let (plates, cache) = initialize_plates(&points, &del, &assignment, &InitParams::default());
        let adj = Adjacency::from_delaunay(points.len(), &del);
        Simulation::new(points, cache, adj, plates)
    }

    #[test]
    fn time_advances_each_step() {
        let mut sim = make_sim(500, 8);
        assert_eq!(sim.time, 0.0);
        sim.step();
        assert!((sim.time - DT).abs() < 1e-10);
        sim.run(5);
        assert!((sim.time - 6.0 * DT).abs() < 1e-10);
    }

    #[test]
    fn rotation_accumulates_over_steps() {
        let mut sim = make_sim(500, 8);
        let initial_rots: Vec<DQuat> = sim.plates.iter().map(|p| p.rotation).collect();
        sim.step();
        for (i, plate) in sim.plates.iter().enumerate() {
            if plate.angular_speed.abs() > MIN_ANGULAR_SPEED {
                let diff = (plate.rotation - initial_rots[i]).length();
                assert!(diff > 1e-10, "plate {i} rotation didn't change");
            }
        }
    }

    #[test]
    fn advance_rotations_skips_zero_speed() {
        let mut sim = make_sim(500, 4);
        for plate in &mut sim.plates {
            plate.angular_speed = 0.0;
        }
        let rots_before: Vec<DQuat> = sim.plates.iter().map(|p| p.rotation).collect();
        sim.step();
        for (i, plate) in sim.plates.iter().enumerate() {
            let diff = (plate.rotation - rots_before[i]).length();
            assert!(diff < 1e-10, "plate {i} rotated despite zero speed");
        }
    }

    #[test]
    fn all_samples_assigned_after_init() {
        let sim = make_sim(500, 8);
        let assigned = sim.sample_cache.iter().filter(|c| c.plate != NO_PLATE).count();
        assert!(
            assigned >= 490,
            "only {assigned}/500 assigned after init"
        );
    }

    #[test]
    fn samples_stay_assigned_after_step() {
        let mut sim = make_sim(500, 8);
        sim.step();
        let assigned = sim.sample_cache.iter().filter(|c| c.plate != NO_PLATE).count();
        // Boundary gaps between plates leave ~10-20% unassigned.
        assert!(
            assigned >= 350,
            "only {assigned}/500 assigned after one step"
        );
    }

    #[test]
    fn detect_boundaries_finds_cross_plate_edges() {
        let sim = make_sim(500, 8);
        let boundaries = sim.detect_boundaries();
        assert!(!boundaries.is_empty(), "should find plate boundaries");
        for b in &boundaries {
            assert_ne!(b.plate_a, b.plate_b);
        }
    }

    #[test]
    fn interpolated_crust_returns_valid_data() {
        let sim = make_sim(500, 8);
        for cache in &sim.sample_cache {
            if cache.plate == NO_PLATE {
                continue;
            }
            let crust = sim.interpolated_crust(cache);
            assert!(!crust.elevation.is_nan());
            assert!(!crust.thickness.is_nan());
            assert!(crust.thickness > 0.0);
        }
    }

    #[test]
    fn dominant_crust_matches_highest_bary_weight() {
        let sim = make_sim(500, 8);
        for cache in &sim.sample_cache {
            if cache.plate == NO_PLATE {
                continue;
            }
            let crust = sim.dominant_crust(cache);
            let plate = &sim.plates[cache.plate as usize];
            let [vi, vj, vk] = plate.triangles[cache.triangle as usize];
            let dom = if cache.bary[0] >= cache.bary[1] && cache.bary[0] >= cache.bary[2] {
                vi
            } else if cache.bary[1] >= cache.bary[2] {
                vj
            } else {
                vk
            };
            assert_eq!(
                crust.crust_type,
                plate.crust[dom as usize].crust_type
            );
            break;
        }
    }

    #[test]
    fn run_ten_steps_without_panic() {
        let mut sim = make_sim(1000, 12);
        sim.run(10);
        assert!(sim.plates.len() > 0);
        assert!((sim.time - 10.0 * DT).abs() < 1e-10);
    }

    #[test]
    fn rotation_axes_remain_normalized() {
        let mut sim = make_sim(500, 8);
        sim.run(5);
        for plate in &sim.plates {
            let len = plate.rotation_axis.length();
            assert!((len - 1.0).abs() < 1e-6, "axis not normalized: {len}");
        }
    }

    #[test]
    fn rotation_quaternions_remain_normalized() {
        let mut sim = make_sim(500, 8);
        sim.run(5);
        for plate in &sim.plates {
            let len = plate.rotation.length();
            assert!((len - 1.0).abs() < 1e-6, "quat not normalized: {len}");
        }
    }

    #[test]
    fn original_reference_points_unchanged_after_steps() {
        let mut sim = make_sim(500, 4);
        let before: Vec<Vec<DVec3>> = sim
            .plates
            .iter()
            .map(|p| p.reference_points.clone())
            .collect();
        let before_counts: Vec<usize> = before.iter().map(|v| v.len()).collect();
        sim.run(5);
        // Oceanic generation may append new vertices, but original ones must not move.
        for (i, plate) in sim.plates.iter().enumerate() {
            for j in 0..before_counts[i] {
                let diff = (plate.reference_points[j] - before[i][j]).length();
                assert!(
                    diff < 1e-15,
                    "reference point {j} of plate {i} moved by {diff}"
                );
            }
        }
    }

    #[test]
    fn sample_points_unchanged_between_resamples() {
        let mut sim = make_sim(500, 8);
        let before = sim.sample_points.clone();
        sim.step();
        for (i, &p) in sim.sample_points.iter().enumerate() {
            assert!(
                (p - before[i]).length() < 1e-15,
                "sample point {i} changed"
            );
        }
    }
}
