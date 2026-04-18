use std::collections::HashMap;
use std::io::Write;

use glam::{DQuat, DVec3};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use super::boundary::{find_boundary_edges, BoundaryEdge};
use super::continental_collision;
use super::oceanic_crust_generation;
use super::plate_rifting;
use super::plate_seed_placement::Adjacency;
use super::plates::{CrustType, OrogenyType, Plate};
use super::resample;
use super::spherical_delaunay_triangulation::SphericalDelaunay;
use super::subduction;
use super::util::plate_centroid;

/// Timestep δt in Myr.
const DT: f64 = 2.0;
/// Planet radius in km, for arc distance conversion.
const PLANET_RADIUS: f64 = 6370.0;
/// Continental erosion rate ε_c in km/Myr (converted from 0.03 mm/yr).
const CONTINENTAL_EROSION: f64 = 0.03e-3;
/// Oceanic elevation damping rate ε_o in km/Myr (converted from 0.04 mm/yr).
const OCEANIC_DAMPING: f64 = 0.04e-3;
/// Sediment accretion rate ε_t in km/Myr (converted from 0.3 mm/yr).
const SEDIMENT_ACCRETION: f64 = 0.3e-3;
/// Highest continental altitude z_c in km, normalizes erosion.
const MAX_CONTINENTAL_ALTITUDE: f64 = 10.0;
/// Oceanic trench depth z_t in km, normalizes damping.
const OCEANIC_TRENCH_DEPTH: f64 = -10.0;
/// Interpenetration threshold for triggering a continental collision, in km.
const COLLISION_THRESHOLD: f64 = 300.0;
/// Dot product threshold for the 1800km subduction radius.
/// cos(1800 / 6370) — points with dot > this are within 1800km.
const SUBDUCTION_DOT_THRESHOLD: f64 = 0.9603;
/// Low-frequency noise amplitude for spatial erosion/damping variation.
/// A value of 0.4 means rates vary between 60%–140% of their base value.
const EROSION_NOISE_AMPLITUDE: f64 = 0.4;
/// Noise frequency — low values produce continent-scale variation.
const EROSION_NOISE_FREQUENCY: f64 = 1.5;
/// FBM octaves for erosion noise.
const EROSION_NOISE_OCTAVES: usize = 3;
/// Noise seed for erosion spatial variation.
const EROSION_NOISE_SEED: u32 = 7;
/// Angular speed below which plate rotation is skipped (effectively zero).
const MIN_ANGULAR_SPEED: f64 = 1e-15;
/// Floor for noise-modulated erosion multiplier. Prevents sign flip.
const EROSION_NOISE_FLOOR: f64 = 0.1;
/// Minimum subduction uplift (mm/yr) to classify a point as active Andean orogeny.
/// ~10% of BASE_UPLIFT — gates orogeny to the zone near the subduction front
/// where deformation is strong enough to build mountains.
const MIN_OROGENY_UPLIFT: f64 = 0.06;

/// Number of latitude bands for subduction source clustering.
const CLUSTER_LAT_BINS: usize = 64;
/// Number of longitude bands for subduction source clustering.
const CLUSTER_LON_BINS: usize = 128;

/// A subduction source on the overriding plate's boundary.
#[derive(Clone)]
struct SubductionSource {
    boundary_pos: DVec3,
    vel_sub: DVec3,
    vel_over: DVec3,
    /// Elevation of the subducting plate at the boundary (for height transfer h(z̃_i)).
    subducting_elevation: f64,
}

/// Step-range diagnostic logger. Writes per-phase crust statistics to a file.
pub struct DiagnosticLog {
    pub(super) file: std::io::BufWriter<std::fs::File>,
    start_step: usize,
    end_step: usize,
}

impl DiagnosticLog {
    pub fn new(path: &std::path::Path, start_step: usize, end_step: usize) -> Self {
        let file = std::fs::File::create(path).expect("failed to create diagnostic log");
        Self { file: std::io::BufWriter::new(file), start_step, end_step }
    }

    pub(super) fn active(&self, step: usize) -> bool {
        step >= self.start_step && step <= self.end_step
    }
}

/// Full simulation state.
pub struct Simulation {
    pub points: Vec<DVec3>,
    pub plates: Vec<Plate>,
    pub adjacency: Adjacency,
    pub time: f64,
    pub point_count: u32,
    pub(super) step_count: usize,
    steps_since_resample: usize,
    steps_since_crust_generation: usize,
    steps_since_rift_check: usize,
    rift_seed: u64,
    pub(super) diagnostics: Option<DiagnosticLog>,
}

impl Simulation {
    pub fn new(
        points: Vec<DVec3>,
        plates: Vec<Plate>,
        delaunay: &SphericalDelaunay,
    ) -> Self {
        let point_count = points.len() as u32;
        let adjacency = Adjacency::from_delaunay(points.len(), delaunay);
        Self {
            points, plates, adjacency, time: 0.0, point_count,
            step_count: 0,
            steps_since_resample: 0, steps_since_crust_generation: 0,
            steps_since_rift_check: 0, rift_seed: 0,
            diagnostics: None,
        }
    }

    pub fn enable_diagnostics(&mut self, path: &std::path::Path, start_step: usize, end_step: usize) {
        self.diagnostics = Some(DiagnosticLog::new(path, start_step, end_step));
    }

    /// Look up the elevation of a global point within a plate (by plate index).
    fn elevation_at_point(&self, plate_idx: u32, global_point: u32) -> f64 {
        let plate = &self.plates[plate_idx as usize];
        for (local, &global) in plate.point_indices.iter().enumerate() {
            if global == global_point {
                return plate.crust[local].elevation;
            }
        }
        0.0
    }

    /// Advance the simulation by one timestep.
    pub fn step(&mut self) {
        self.step_count += 1;
        let logging = self.diagnostics.as_ref().map_or(false, |d| d.active(self.step_count));

        if logging {
            self.log_header("STEP START");
            self.log_crust_summary("  initial");
        }

        self.move_plates();

        let boundary = find_boundary_edges(&self.plates, &self.points, &self.adjacency);

        let snap_before_sub = if logging { Some(self.snapshot_crust()) } else { None };
        self.process_subduction(&boundary);
        if let Some(before) = snap_before_sub {
            self.log_crust_diff("  after subduction", &before);
        }

        self.consume_subducted_vertices();

        // Boundary list is indexed into self.points; rebuild after consumption.
        let boundary = find_boundary_edges(&self.plates, &self.points, &self.adjacency);

        let snap_before_col = if logging { Some(self.snapshot_crust()) } else { None };
        let plates_before_col = if logging { self.plates.len() } else { 0 };
        self.process_collisions(&boundary);
        if let Some(before) = snap_before_col {
            self.log_crust_diff("  after collision", &before);
            if self.plates.len() != plates_before_col {
                self.log_msg(&format!("  collision transferred terrane: {} -> {} plates",
                    plates_before_col, self.plates.len()));
            }
        }

        self.remove_empty_plates();

        let snap_before_rift = if logging { Some(self.snapshot_crust()) } else { None };
        let boundary = self.maybe_rift(boundary);
        if let Some(before) = snap_before_rift {
            self.log_crust_diff("  after rift", &before);
        }

        let snap_before_ocean = if logging { Some(self.snapshot_crust()) } else { None };
        self.maybe_generate_oceanic_crust(&boundary);
        if let Some(before) = snap_before_ocean {
            self.log_crust_diff("  after oceanic gen", &before);
        }

        let snap_before_erosion = if logging { Some(self.snapshot_crust()) } else { None };
        self.apply_erosion_and_damping();
        if let Some(before) = snap_before_erosion {
            self.log_elevation_changes("  after erosion", &before);
        }

        self.time += DT;
        self.steps_since_resample += 1;

        let snap_before_resample = if logging { Some(self.snapshot_crust()) } else { None };
        self.maybe_resample();
        if let Some(before) = snap_before_resample {
            self.log_crust_diff("  after resample", &before);
            self.log_crust_summary("  final");
        }
    }

    /// Run the simulation for a number of timesteps.
    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    /// Rotate each plate's points around its Euler pole by angular_speed * dt.
    fn move_plates(&mut self) {
        for plate in &self.plates {
            let angle = plate.angular_speed * DT;
            if angle.abs() < MIN_ANGULAR_SPEED {
                continue;
            }
            let rotation = DQuat::from_axis_angle(plate.rotation_axis, angle);
            for &global in &plate.point_indices {
                self.points[global as usize] =
                    rotation.mul_vec3(self.points[global as usize]).normalize();
            }
        }
    }

    fn process_subduction(&mut self, boundary: &[BoundaryEdge]) {
        let mut slab_pull_points: Vec<Vec<DVec3>> = vec![Vec::new(); self.plates.len()];
        let mut advancements: Vec<(u32, u32, f64)> = Vec::new();
        let sources = self.collect_subduction_sources(boundary, &mut slab_pull_points, &mut advancements);
        let clustered = cluster_sources(&sources);
        self.apply_uplift(&clustered);
        self.apply_slab_pull(&slab_pull_points);
        self.apply_subducted_advancements(&advancements);
    }

    fn collect_subduction_sources(
        &self,
        boundary: &[BoundaryEdge],
        slab_pull_points: &mut [Vec<DVec3>],
        advancements: &mut Vec<(u32, u32, f64)>,
    ) -> Vec<Vec<SubductionSource>> {
        let mut sources: Vec<Vec<SubductionSource>> = vec![Vec::new(); self.plates.len()];

        for edge in boundary {
            let result = subduction::resolve_subduction(
                edge.plate_a, edge.plate_b,
                edge.crust_a, edge.age_a,
                edge.crust_b, edge.age_b,
            );

            let (subducting_idx, overriding_idx) = match result {
                subduction::SubductionResult::PlateSubducts(s) => {
                    let o = if s == edge.plate_a { edge.plate_b } else { edge.plate_a };
                    (s, o)
                }
                subduction::SubductionResult::ContinentalCollision => continue,
            };

            let boundary_pos = self.points[edge.point as usize];

            // Only process convergent boundaries: plates must be closing.
            let vel_sub = self.plates[subducting_idx as usize].surface_velocity(boundary_pos);
            let vel_over = self.plates[overriding_idx as usize].surface_velocity(boundary_pos);
            let neighbor_pos = self.points[edge.neighbor as usize];
            let edge_dir = (neighbor_pos - boundary_pos).normalize_or_zero();
            let relative = vel_sub - vel_over;
            // Negative component along edge direction means plates are closing.
            if relative.dot(edge_dir) >= 0.0 {
                continue;
            }

            slab_pull_points[subducting_idx as usize].push(boundary_pos);

            // Paper: d(p, t+δt) = d(p, t) + ||s_sub - s_over|| δt. Advance the
            // subducting vertex's distance-past-front so it can be removed once
            // it has fully subducted.
            let subducting_vertex = if subducting_idx == edge.plate_a {
                edge.point
            } else {
                edge.neighbor
            };
            let delta = relative.length() * DT;
            advancements.push((subducting_idx, subducting_vertex, delta));

            // Look up the subducting plate's elevation at the boundary for h(z̃_i).
            let subducting_elevation = if subducting_idx == edge.plate_a {
                self.elevation_at_point(edge.plate_a, edge.point)
            } else {
                self.elevation_at_point(edge.plate_b, edge.neighbor)
            };

            sources[overriding_idx as usize].push(SubductionSource {
                boundary_pos,
                vel_sub,
                vel_over,
                subducting_elevation,
            });
        }

        sources
    }

    /// Apply advancements to `subducted_distance` for convergent boundary vertices.
    fn apply_subducted_advancements(&mut self, advancements: &[(u32, u32, f64)]) {
        for &(plate_idx, global_vertex, delta) in advancements {
            let plate = &mut self.plates[plate_idx as usize];
            if let Some(local) = plate.point_indices.iter().position(|&g| g == global_vertex) {
                plate.crust[local].subducted_distance += delta;
            }
        }
    }

    fn apply_uplift(&mut self, clustered: &[Vec<SubductionSource>]) {
        for plate_idx in 0..self.plates.len() {
            let sources = &clustered[plate_idx];
            if sources.is_empty() {
                continue;
            }
            let plate = &mut self.plates[plate_idx];
            for (local, &global) in plate.point_indices.iter().enumerate() {
                let p = self.points[global as usize];
                for src in sources {
                    if p.dot(src.boundary_pos) < SUBDUCTION_DOT_THRESHOLD {
                        continue;
                    }
                    let d = p.dot(src.boundary_pos).clamp(-1.0, 1.0).acos() * PLANET_RADIUS;

                    let crust = &plate.crust[local];
                    let relative_speed = (src.vel_sub - src.vel_over).length();
                    // Paper: h(z̃_i) uses the subducting plate's elevation, not the overriding plate's.
                    let uplift = subduction::subduction_uplift(d, relative_speed, src.subducting_elevation);
                    let (new_z, new_fold, new_age) = subduction::apply_subduction_step(
                        crust.elevation, crust.local_direction, crust.age,
                        d, src.vel_sub, src.vel_over, src.subducting_elevation, DT,
                    );
                    let crust = &mut plate.crust[local];
                    crust.elevation = new_z;
                    crust.local_direction = new_fold;
                    crust.age = new_age;
                    if uplift > MIN_OROGENY_UPLIFT {
                        if crust.crust_type == CrustType::Oceanic && crust.elevation > 0.0 {
                            crust.crust_type = CrustType::Continental;
                            crust.orogeny_type = Some(OrogenyType::Andean);
                        } else if crust.crust_type == CrustType::Continental {
                            crust.orogeny_type = Some(OrogenyType::Andean);
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
            let center = plate_centroid(&self.plates[plate_idx], &self.points);
            self.plates[plate_idx].rotation_axis = subduction::apply_slab_pull(
                self.plates[plate_idx].rotation_axis,
                center,
                pull_points,
                DT,
            );
        }
    }

    fn process_collisions(&mut self, boundary: &[BoundaryEdge]) {
        let mut collision_boundaries: HashMap<(u32, u32), Vec<DVec3>> = HashMap::new();
        for edge in boundary {
            if edge.crust_a != CrustType::Continental || edge.crust_b != CrustType::Continental {
                continue;
            }
            let boundary_pos = self.points[edge.point as usize];
            let neighbor_pos = self.points[edge.neighbor as usize];
            let vel_a = self.plates[edge.plate_a as usize].surface_velocity(boundary_pos);
            let vel_b = self.plates[edge.plate_b as usize].surface_velocity(boundary_pos);
            let relative = vel_a - vel_b;
            // Check convergence: relative velocity should close the gap across the boundary.
            // edge_dir points from this plate's point toward the neighboring plate's point.
            let edge_dir = (neighbor_pos - boundary_pos).normalize_or_zero();
            if relative.dot(edge_dir) >= 0.0 {
                continue;
            }
            let pair = (edge.plate_a.min(edge.plate_b), edge.plate_a.max(edge.plate_b));
            collision_boundaries.entry(pair).or_default().push(boundary_pos);
        }

        for ((plate_a, plate_b), boundary_points) in collision_boundaries {
            self.try_collision(plate_a, plate_b, &boundary_points);
        }
    }

    fn try_collision(&mut self, plate_a: u32, plate_b: u32, boundary_points: &[DVec3]) {
        let terranes_a = continental_collision::find_terranes(
            &self.plates[plate_a as usize], &self.points, &self.adjacency,
        );
        if terranes_a.is_empty() {
            return;
        }

        let nearest = terranes_a.iter().min_by(|a, b| {
            let da = min_arc_distance(a.centroid, boundary_points);
            let db = min_arc_distance(b.centroid, boundary_points);
            da.partial_cmp(&db).unwrap()
        }).unwrap();

        let dist_to_boundary = min_arc_distance(nearest.centroid, boundary_points);
        if dist_to_boundary > COLLISION_THRESHOLD {
            return;
        }

        let terrane_area = estimate_area(nearest.points.len(), self.points.len());
        let vel_a = self.plates[plate_a as usize].surface_velocity(nearest.centroid);
        let vel_b = self.plates[plate_b as usize].surface_velocity(nearest.centroid);
        let relative_speed = (vel_a - vel_b).length();

        let radius = continental_collision::influence_radius(
            relative_speed, terrane_area, self.plates.len(),
        );

        let dot_threshold = (radius / PLANET_RADIUS).cos();
        let plate_b_ref = &mut self.plates[plate_b as usize];
        for (local, &global) in plate_b_ref.point_indices.iter().enumerate() {
            let p = self.points[global as usize];
            let dot = p.dot(nearest.centroid);
            if dot < dot_threshold {
                continue;
            }
            let d = dot.clamp(-1.0, 1.0).acos() * PLANET_RADIUS;
            let (new_z, new_fold) = continental_collision::apply(
                plate_b_ref.crust[local].elevation,
                p, nearest.centroid, terrane_area, d, radius,
            );
            plate_b_ref.crust[local].elevation = new_z;
            plate_b_ref.crust[local].local_direction = new_fold;
            plate_b_ref.crust[local].orogeny_type = Some(OrogenyType::Himalayan);
        }

        let (src, dst) = borrow_two_mut(&mut self.plates, plate_a as usize, plate_b as usize);
        continental_collision::transfer_terrane(nearest, src, dst);
    }

    fn remove_empty_plates(&mut self) {
        self.plates.retain(|p| !p.point_indices.is_empty());
    }

    /// Remove vertices that have fully subducted (traveled past the convergence
    /// front by more than `SUBDUCTION_DISTANCE`). Compacts `sim.points`, remaps
    /// every plate's `point_indices`, and rebuilds `sim.adjacency`.
    fn consume_subducted_vertices(&mut self) {
        let mut is_consumed = vec![false; self.points.len()];
        let mut consumed_count = 0usize;

        for plate in &self.plates {
            for (local, &global) in plate.point_indices.iter().enumerate() {
                if plate.crust[local].subducted_distance > subduction::SUBDUCTION_DISTANCE {
                    is_consumed[global as usize] = true;
                    consumed_count += 1;
                }
            }
        }

        if consumed_count == 0 {
            return;
        }

        let original_len = self.points.len();
        let mut old_to_new = vec![u32::MAX; self.points.len()];
        let mut new_points = Vec::with_capacity(original_len - consumed_count);
        for (i, &p) in self.points.iter().enumerate() {
            if is_consumed[i] {
                continue;
            }
            old_to_new[i] = new_points.len() as u32;
            new_points.push(p);
        }

        for plate in self.plates.iter_mut() {
            let mut new_indices = Vec::with_capacity(plate.point_indices.len());
            let mut new_crust = Vec::with_capacity(plate.crust.len());
            for (local, &global) in plate.point_indices.iter().enumerate() {
                if is_consumed[global as usize] {
                    continue;
                }
                new_indices.push(old_to_new[global as usize]);
                new_crust.push(plate.crust[local].clone());
            }
            plate.point_indices = new_indices;
            plate.crust = new_crust;
        }

        self.points = new_points;
        let delaunay = SphericalDelaunay::from_points(&self.points);
        self.adjacency = Adjacency::from_delaunay(self.points.len(), &delaunay);

        if let Some(diag) = self.diagnostics.as_mut() {
            let _ = writeln!(
                diag.file,
                "=== SUBDUCTION CONSUMPTION step={} t={:.0} Myr consumed={} of {} remaining={} ===",
                self.step_count, self.time, consumed_count, original_len, self.points.len(),
            );
            let _ = diag.file.flush();
        }
    }

    // --- Diagnostic logging helpers ---

    /// Per-point crust snapshot: (crust_type, elevation, orogeny_type, plate_index).
    fn snapshot_crust(&self) -> Vec<(CrustType, f64, Option<OrogenyType>, usize)> {
        let mut snap = vec![(CrustType::Oceanic, 0.0, None, 0usize); self.points.len()];
        for (plate_idx, plate) in self.plates.iter().enumerate() {
            for (local, &global) in plate.point_indices.iter().enumerate() {
                let c = &plate.crust[local];
                snap[global as usize] = (c.crust_type, c.elevation, c.orogeny_type, plate_idx);
            }
        }
        snap
    }

    fn log_msg(&mut self, msg: &str) {
        if let Some(diag) = &mut self.diagnostics {
            let _ = writeln!(diag.file, "{}", msg);
        }
    }

    fn log_header(&mut self, label: &str) {
        let step = self.step_count;
        let time = self.time + DT; // time after this step completes
        if let Some(diag) = &mut self.diagnostics {
            let _ = writeln!(diag.file, "\n=== {} step={} t={:.0} Myr plates={} ===",
                label, step, time, self.plates.len());
        }
    }

    fn log_crust_summary(&mut self, label: &str) {
        let mut continental = 0usize;
        let mut oceanic = 0usize;
        let mut andean = 0usize;
        let mut himalayan = 0usize;
        let mut no_orogeny = 0usize;
        let mut cont_below_sea = 0usize;
        let mut min_cont_elev = f64::MAX;
        let mut max_cont_elev = f64::MIN;

        for plate in &self.plates {
            for c in &plate.crust {
                match c.crust_type {
                    CrustType::Continental => {
                        continental += 1;
                        if c.elevation < 0.0 { cont_below_sea += 1; }
                        min_cont_elev = min_cont_elev.min(c.elevation);
                        max_cont_elev = max_cont_elev.max(c.elevation);
                        match c.orogeny_type {
                            Some(OrogenyType::Andean) => andean += 1,
                            Some(OrogenyType::Himalayan) => himalayan += 1,
                            None => no_orogeny += 1,
                        }
                    }
                    CrustType::Oceanic => oceanic += 1,
                }
            }
        }

        let total = continental + oceanic;
        if let Some(diag) = &mut self.diagnostics {
            let _ = writeln!(diag.file,
                "{}: total={} cont={} ({:.1}%) ocean={} | andean={} himalayan={} plain={} | cont_below_sea={} elev=[{:.3},{:.3}]",
                label, total, continental, continental as f64 / total as f64 * 100.0, oceanic,
                andean, himalayan, no_orogeny, cont_below_sea,
                if min_cont_elev == f64::MAX { 0.0 } else { min_cont_elev },
                if max_cont_elev == f64::MIN { 0.0 } else { max_cont_elev },
            );
            // Per-plate breakdown
            for (i, plate) in self.plates.iter().enumerate() {
                let pc: usize = plate.crust.iter().filter(|c| c.crust_type == CrustType::Continental).count();
                let po: usize = plate.crust.iter().filter(|c| c.crust_type == CrustType::Oceanic).count();
                if pc > 0 || plate.crust.len() < 100 {
                    let _ = writeln!(diag.file, "    plate[{}]: {} pts (cont={} ocean={})", i, plate.crust.len(), pc, po);
                }
            }
        }
    }

    fn log_crust_diff(&mut self, label: &str,
        before: &[(CrustType, f64, Option<OrogenyType>, usize)])
    {
        let after = self.snapshot_crust();
        let mut cont_to_ocean = 0usize;
        let mut ocean_to_cont = 0usize;
        let mut orogeny_changes = 0usize;
        let mut plate_changes = 0usize;
        let mut biggest_elev_drop = 0.0f64;
        let mut biggest_drop_idx = 0usize;

        for i in 0..before.len().min(after.len()) {
            let (bt, be, bo, bp) = before[i];
            let (at, ae, ao, ap) = after[i];
            if bt == CrustType::Continental && at == CrustType::Oceanic {
                cont_to_ocean += 1;
            }
            if bt == CrustType::Oceanic && at == CrustType::Continental {
                ocean_to_cont += 1;
            }
            if bo != ao { orogeny_changes += 1; }
            if bp != ap { plate_changes += 1; }
            let drop = be - ae;
            if bt == CrustType::Continental && drop > biggest_elev_drop {
                biggest_elev_drop = drop;
                biggest_drop_idx = i;
            }
        }

        // Only log if something happened
        if cont_to_ocean == 0 && ocean_to_cont == 0 && orogeny_changes == 0
            && plate_changes == 0 && biggest_elev_drop < 0.001
        {
            return;
        }

        if let Some(diag) = &mut self.diagnostics {
            let _ = writeln!(diag.file,
                "{}: cont->ocean={} ocean->cont={} orogeny_changed={} plate_changed={} max_cont_elev_drop={:.4}km (pt {})",
                label, cont_to_ocean, ocean_to_cont, orogeny_changes, plate_changes,
                biggest_elev_drop, biggest_drop_idx,
            );
            // Detail the first few continental->oceanic transitions
            if cont_to_ocean > 0 {
                let mut shown = 0;
                for i in 0..before.len().min(after.len()) {
                    if before[i].0 == CrustType::Continental && after[i].0 == CrustType::Oceanic {
                        let pos = self.points[i];
                        let _ = writeln!(diag.file,
                            "    cont->ocean pt={} pos=({:.3},{:.3},{:.3}) elev_before={:.4} elev_after={:.4} plate {}->{}",
                            i, pos.x, pos.y, pos.z,
                            before[i].1, after[i].1, before[i].3, after[i].3,
                        );
                        shown += 1;
                        if shown >= 20 { break; }
                    }
                }
            }
        }
    }

    fn log_elevation_changes(&mut self, label: &str,
        before: &[(CrustType, f64, Option<OrogenyType>, usize)])
    {
        let after = self.snapshot_crust();
        let mut cont_lost_elev = 0usize;
        let mut cont_sank_below_sea = 0usize;

        for i in 0..before.len().min(after.len()) {
            if before[i].0 == CrustType::Continental && after[i].0 == CrustType::Continental {
                if after[i].1 < before[i].1 { cont_lost_elev += 1; }
                if before[i].1 >= 0.0 && after[i].1 < 0.0 { cont_sank_below_sea += 1; }
            }
        }

        if cont_sank_below_sea > 0 {
            if let Some(diag) = &mut self.diagnostics {
                let _ = writeln!(diag.file,
                    "{}: continental_lost_elevation={} continental_sank_below_sea={}",
                    label, cont_lost_elev, cont_sank_below_sea);
            }
        }
    }

    fn maybe_rift(&mut self, boundary: Vec<BoundaryEdge>) -> Vec<BoundaryEdge> {
        // TEMP: rifting disabled to isolate whether it causes runaway fragmentation.
        return boundary;

        #[allow(unreachable_code)]
        {
            self.steps_since_rift_check += 1;
            if self.steps_since_rift_check < plate_rifting::RIFT_CHECK_INTERVAL {
                return boundary;
            }
            self.steps_since_rift_check = 0;

            if self.process_rifting() {
                find_boundary_edges(&self.plates, &self.points, &self.adjacency)
            } else {
                boundary
            }
        }
    }

    fn process_rifting(&mut self) -> bool {
        let mut rifted = false;
        let mut new_plates: Vec<Plate> = Vec::new();
        let mut emptied: Vec<usize> = Vec::new();

        let total_points = self.points.len();
        let plate_count = self.plates.len();

        for i in 0..plate_count {
            if !plate_rifting::should_rift(&self.plates[i], i, total_points, plate_count, self.time, self.rift_seed) {
                continue;
            }
            let sub_plates =
                plate_rifting::rift_plate(&self.plates[i], &self.points, &self.adjacency, self.rift_seed);
            if sub_plates.len() < 2 {
                continue;
            }
            emptied.push(i);
            new_plates.extend(sub_plates);
            rifted = true;
        }

        if !rifted {
            self.rift_seed = self.rift_seed.wrapping_add(1);
            return false;
        }

        emptied.sort_unstable();
        for &i in emptied.iter().rev() {
            self.plates.swap_remove(i);
        }
        self.plates.extend(new_plates);

        let delaunay = SphericalDelaunay::from_points(&self.points);
        self.adjacency = Adjacency::from_delaunay(self.points.len(), &delaunay);
        self.rift_seed = self.rift_seed.wrapping_add(1);
        true
    }

    fn maybe_generate_oceanic_crust(&mut self, boundary: &[BoundaryEdge]) {
        self.steps_since_crust_generation += 1;
        if self.steps_since_crust_generation
            < oceanic_crust_generation::generation_interval(&self.plates)
        {
            return;
        }
        self.generate_oceanic_crust(boundary);
        self.steps_since_crust_generation = 0;
    }

    fn generate_oceanic_crust(&mut self, boundary: &[BoundaryEdge]) {
        let divergent = oceanic_crust_generation::find_divergent_edges(
            boundary, &self.plates, &self.points,
        );
        if divergent.is_empty() {
            return;
        }

        let new_points = oceanic_crust_generation::generate_ridge_points(
            &divergent, &self.plates, &self.points,
        );
        if new_points.is_empty() {
            return;
        }

        for np in new_points {
            let global_idx = self.points.len() as u32;
            self.points.push(np.position);
            self.plates[np.plate_index as usize].point_indices.push(global_idx);
            self.plates[np.plate_index as usize].crust.push(np.crust);
        }

        let delaunay = SphericalDelaunay::from_points(&self.points);
        self.adjacency = Adjacency::from_delaunay(self.points.len(), &delaunay);
    }

    fn maybe_resample(&mut self) {
        if self.steps_since_resample < resample::RESAMPLE_INTERVAL {
            return;
        }
        resample::resample(self, self.point_count);
        self.steps_since_resample = 0;
    }

    fn apply_erosion_and_damping(&mut self) {
        let fbm: Fbm<Perlin> = Fbm::new(EROSION_NOISE_SEED)
            .set_octaves(EROSION_NOISE_OCTAVES)
            .set_frequency(EROSION_NOISE_FREQUENCY);

        for plate in &mut self.plates {
            for (local, &global) in plate.point_indices.iter().enumerate() {
                let p = self.points[global as usize];
                let noise = 1.0 + EROSION_NOISE_AMPLITUDE * fbm.get([p.x, p.y, p.z]);
                let scale = noise.max(EROSION_NOISE_FLOOR);
                let crust = &mut plate.crust[local];
                match crust.crust_type {
                    CrustType::Continental => {
                        crust.elevation -= (crust.elevation / MAX_CONTINENTAL_ALTITUDE)
                            * CONTINENTAL_EROSION
                            * scale
                            * DT;
                    }
                    CrustType::Oceanic => {
                        crust.elevation -= (1.0 - crust.elevation / OCEANIC_TRENCH_DEPTH)
                            * OCEANIC_DAMPING
                            * scale
                            * DT;
                        crust.elevation += SEDIMENT_ACCRETION * DT;
                        crust.age += DT;
                    }
                }
            }
        }
    }
}

/// Cluster boundary sources by grid cell to reduce per-point work.
/// Many boundary edges are geographically adjacent — averaging them per cell
/// preserves physical accuracy while dramatically cutting source count.
fn cluster_sources(
    raw_sources: &[Vec<SubductionSource>],
) -> Vec<Vec<SubductionSource>> {
    let bin_count = CLUSTER_LAT_BINS * CLUSTER_LON_BINS;
    let mut clustered = Vec::with_capacity(raw_sources.len());

    for sources in raw_sources {
        if sources.is_empty() {
            clustered.push(Vec::new());
            continue;
        }
        let mut grid: HashMap<usize, Vec<&SubductionSource>> = HashMap::new();
        for src in sources {
            let bin = cluster_bin(src.boundary_pos, bin_count);
            grid.entry(bin).or_default().push(src);
        }
        let mut plate_clustered = Vec::with_capacity(grid.len());
        for (_, cell) in &grid {
            let n = cell.len() as f64;
            let avg_pos: DVec3 = cell.iter().map(|s| s.boundary_pos).sum::<DVec3>() / n;
            let avg_vel_sub: DVec3 = cell.iter().map(|s| s.vel_sub).sum::<DVec3>() / n;
            let avg_vel_over: DVec3 = cell.iter().map(|s| s.vel_over).sum::<DVec3>() / n;
            let avg_elev: f64 = cell.iter().map(|s| s.subducting_elevation).sum::<f64>() / n;
            plate_clustered.push(SubductionSource {
                boundary_pos: avg_pos.normalize(),
                vel_sub: avg_vel_sub,
                vel_over: avg_vel_over,
                subducting_elevation: avg_elev,
            });
        }
        clustered.push(plate_clustered);
    }

    clustered
}

fn cluster_bin(p: DVec3, _bin_count: usize) -> usize {
    let lat = p.y.clamp(-1.0, 1.0).asin();
    let lon = p.z.atan2(p.x);
    let lat_bin = ((lat / std::f64::consts::PI + 0.5) * CLUSTER_LAT_BINS as f64)
        .max(0.0)
        .min(CLUSTER_LAT_BINS as f64 - 1.0) as usize;
    let lon_bin = ((lon / std::f64::consts::TAU + 0.5) * CLUSTER_LON_BINS as f64)
        .max(0.0)
        .min(CLUSTER_LON_BINS as f64 - 1.0) as usize;
    lat_bin * CLUSTER_LON_BINS + lon_bin
}

fn arc_distance(a: DVec3, b: DVec3) -> f64 {
    a.normalize().dot(b.normalize()).clamp(-1.0, 1.0).acos() * PLANET_RADIUS
}

fn min_arc_distance(point: DVec3, targets: &[DVec3]) -> f64 {
    targets.iter()
        .map(|&t| arc_distance(point, t))
        .fold(f64::MAX, f64::min)
}

/// Rough terrane area estimate from point count, assuming uniform density.
fn estimate_area(terrane_points: usize, total_points: usize) -> f64 {
    let total_area = 4.0 * std::f64::consts::PI * PLANET_RADIUS * PLANET_RADIUS;
    total_area * terrane_points as f64 / total_points as f64
}

/// Borrow two distinct elements of a slice mutably.
fn borrow_two_mut(plates: &mut [Plate], a: usize, b: usize) -> (&mut Plate, &mut Plate) {
    assert_ne!(a, b);
    if a < b {
        let (left, right) = plates.split_at_mut(b);
        (&mut left[a], &mut right[0])
    } else {
        let (left, right) = plates.split_at_mut(a);
        (&mut right[0], &mut left[b])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::fibonnaci_spiral::SphericalFibonacci;
    use super::super::plate_initializer::{initialize_plates, InitParams};
    use super::super::plate_seed_placement::assign_plates;

    fn setup(point_count: u32, plate_count: u32) -> Simulation {
        let fib = SphericalFibonacci::new(point_count);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, plate_count, 42);
        let plates = initialize_plates(&points, &del, &assignment, &InitParams::default());
        Simulation::new(points, plates, &del)
    }

    #[test]
    fn single_step_preserves_point_count() {
        let mut sim = setup(500, 8);
        let total_before: usize = sim.plates.iter().map(|p| p.point_count()).sum();
        sim.step();
        let total_after: usize = sim.plates.iter().map(|p| p.point_count()).sum();
        assert_eq!(total_before, total_after);
    }

    #[test]
    fn time_advances() {
        let mut sim = setup(500, 8);
        assert_eq!(sim.time, 0.0);
        sim.step();
        assert!((sim.time - DT).abs() < 1e-10);
        sim.run(5);
        assert!((sim.time - 6.0 * DT).abs() < 1e-10);
    }

    #[test]
    fn erosion_reduces_average_continental_elevation() {
        let mut sim = setup(500, 8);
        let avg_before = avg_continental_elevation(&sim);
        for plate in &mut sim.plates {
            plate.angular_speed = 0.0;
        }
        sim.run(10);
        let avg_after = avg_continental_elevation(&sim);
        assert!(avg_after < avg_before, "erosion should reduce average: {avg_before} -> {avg_after}");
    }

    fn avg_continental_elevation(sim: &Simulation) -> f64 {
        let (mut sum, mut count) = (0.0, 0usize);
        for plate in &sim.plates {
            for crust in &plate.crust {
                if crust.crust_type == CrustType::Continental {
                    sum += crust.elevation;
                    count += 1;
                }
            }
        }
        sum / count as f64
    }

    #[test]
    fn oceanic_crust_ages() {
        let mut sim = setup(500, 8);
        sim.step();
        let any_aged = sim.plates.iter().any(|p| {
            p.crust.iter().any(|c| c.crust_type == CrustType::Oceanic && c.age > 0.0)
        });
        assert!(any_aged, "oceanic crust should age each step");
    }

    #[test]
    fn run_ten_steps_without_panic() {
        let mut sim = setup(1000, 12);
        sim.run(10);
        let total: usize = sim.plates.iter().map(|p| p.point_count()).sum();
        assert_eq!(total, 1000);
    }

    #[test]
    fn boundary_edges_found() {
        let sim = setup(500, 8);
        let edges = find_boundary_edges(&sim.plates, &sim.points, &sim.adjacency);
        assert!(!edges.is_empty(), "should find plate boundaries");
    }

    #[test]
    fn rotation_axes_remain_normalized() {
        let mut sim = setup(500, 8);
        sim.run(5);
        for plate in &sim.plates {
            let len = plate.rotation_axis.length();
            assert!((len - 1.0).abs() < 1e-6, "axis not normalized: {len}");
        }
    }
}
