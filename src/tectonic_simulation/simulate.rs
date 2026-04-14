use std::collections::HashSet;
use std::io::Write;
use std::time::Instant;

use glam::{DQuat, DVec3};
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};

use super::continental_collision;
use super::oceanic_crust_generation;
use super::plate_rifting;
use super::plate_seed_placement::Adjacency;
use super::plates::{CrustType, OrogenyType, Plate};
use super::resample;
use super::spherical_delaunay_triangulation::SphericalDelaunay;
use super::subduction;

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

/// Full simulation state.
pub struct Simulation {
    pub points: Vec<DVec3>,
    pub plates: Vec<Plate>,
    pub adjacency: Adjacency,
    pub time: f64,
    pub point_count: u32,
    steps_since_resample: usize,
    steps_since_crust_generation: usize,
    steps_since_rift_check: usize,
    rift_seed: u64,
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
            steps_since_resample: 0, steps_since_crust_generation: 0,
            steps_since_rift_check: 0, rift_seed: 0,
        }
    }

    /// Advance the simulation by one timestep.
    pub fn step(&mut self) {
        let step_start = Instant::now();
        let mut log = std::fs::OpenOptions::new()
            .create(true).append(true)
            .open("sim_profile.log").ok();

        let t0 = Instant::now();
        self.move_plates();
        let move_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let boundary = find_boundary_edges(&self.plates, &self.points, &self.adjacency);
        let boundary_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        self.process_subduction(&boundary);
        let subduction_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        self.process_collisions(&boundary);
        let collision_ms = t0.elapsed().as_secs_f64() * 1000.0;

        self.steps_since_rift_check += 1;
        let t0 = Instant::now();
        let rifted = if self.steps_since_rift_check >= plate_rifting::RIFT_CHECK_INTERVAL {
            self.steps_since_rift_check = 0;
            self.process_rifting()
        } else {
            false
        };
        let rift_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        let boundary = if rifted {
            find_boundary_edges(&self.plates, &self.points, &self.adjacency)
        } else {
            boundary
        };
        let rift_boundary_ms = t0.elapsed().as_secs_f64() * 1000.0;

        self.steps_since_crust_generation += 1;
        let t0 = Instant::now();
        if self.steps_since_crust_generation
            >= oceanic_crust_generation::generation_interval(&self.plates)
        {
            self.generate_oceanic_crust(&boundary);
            self.steps_since_crust_generation = 0;
        }
        let oceanic_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t0 = Instant::now();
        self.apply_erosion_and_damping();
        let erosion_ms = t0.elapsed().as_secs_f64() * 1000.0;

        self.time += DT;
        self.steps_since_resample += 1;

        let t0 = Instant::now();
        if self.steps_since_resample >= resample::RESAMPLE_INTERVAL {
            resample::resample(self, self.point_count);
            self.steps_since_resample = 0;
        }
        let resample_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let total_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        if let Some(ref mut f) = log {
            let _ = writeln!(f,
                "step={:.0} pts={} plates={} total={:.1}ms | move={:.1} boundary={:.1} subduction={:.1} collision={:.1} rift={:.1} rift_bnd={:.1} oceanic={:.1} erosion={:.1} resample={:.1}",
                self.time, self.points.len(), self.plates.len(), total_ms,
                move_ms, boundary_ms, subduction_ms, collision_ms, rift_ms, rift_boundary_ms, oceanic_ms, erosion_ms, resample_ms
            );
        }
    }

    /// Rotate each plate's points around its Euler pole by angular_speed * dt.
    fn move_plates(&mut self) {
        for plate in &self.plates {
            let angle = plate.angular_speed * DT;
            if angle.abs() < 1e-15 {
                continue;
            }
            let rotation = DQuat::from_axis_angle(plate.rotation_axis, angle);
            for &global in &plate.point_indices {
                self.points[global as usize] =
                    rotation.mul_vec3(self.points[global as usize]).normalize();
            }
        }
    }

    /// Run the simulation for a number of timesteps.
    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }

    fn process_subduction(&mut self, boundary: &[BoundaryEdge]) {
        let mut slab_pull_points: Vec<Vec<DVec3>> = vec![Vec::new(); self.plates.len()];

        // Collect raw uplift sources per overriding plate.
        let mut raw_sources: Vec<Vec<(DVec3, DVec3, DVec3)>> = vec![Vec::new(); self.plates.len()];

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
            slab_pull_points[subducting_idx as usize].push(boundary_pos);

            let vel_sub = self.plates[subducting_idx as usize].surface_velocity(boundary_pos);
            let vel_over = self.plates[overriding_idx as usize].surface_velocity(boundary_pos);
            raw_sources[overriding_idx as usize].push((boundary_pos, vel_sub, vel_over));
        }

        // Cluster boundary sources by grid cell to reduce per-point work.
        // Many boundary edges are geographically adjacent — averaging them per cell
        // preserves physical accuracy while dramatically cutting source count.
        let mut clustered_sources: Vec<Vec<(DVec3, DVec3, DVec3)>> = vec![Vec::new(); self.plates.len()];
        for plate_idx in 0..self.plates.len() {
            if raw_sources[plate_idx].is_empty() {
                continue;
            }
            let mut grid: SphereGrid<(DVec3, DVec3, DVec3)> = SphereGrid::new();
            for &src in &raw_sources[plate_idx] {
                grid.insert(src.0, src);
            }
            for cell in &grid.cells {
                if cell.is_empty() {
                    continue;
                }
                let n = cell.len() as f64;
                let avg_pos: DVec3 = cell.iter().map(|s| s.0).sum::<DVec3>() / n;
                let avg_vel_sub: DVec3 = cell.iter().map(|s| s.1).sum::<DVec3>() / n;
                let avg_vel_over: DVec3 = cell.iter().map(|s| s.2).sum::<DVec3>() / n;
                clustered_sources[plate_idx].push((avg_pos.normalize(), avg_vel_sub, avg_vel_over));
            }
        }

        // Single pass per plate: each point checks clustered boundary sources.
        for plate_idx in 0..self.plates.len() {
            let sources = &clustered_sources[plate_idx];
            if sources.is_empty() {
                continue;
            }
            let plate = &mut self.plates[plate_idx];
            for (local, &global) in plate.point_indices.iter().enumerate() {
                let p = self.points[global as usize];
                for &(boundary_pos, vel_sub, vel_over) in sources {
                    if p.dot(boundary_pos) < SUBDUCTION_DOT_THRESHOLD {
                        continue;
                    }
                    let d = p.dot(boundary_pos).clamp(-1.0, 1.0).acos() * PLANET_RADIUS;

                    let crust = &plate.crust[local];
                    let (new_z, new_fold, new_age) = subduction::apply_subduction_step(
                        crust.elevation, crust.local_direction, crust.age,
                        d, vel_sub, vel_over, DT,
                    );
                    let crust = &mut plate.crust[local];
                    crust.elevation = new_z;
                    crust.local_direction = new_fold;
                    crust.age = new_age;
                    if crust.elevation > 0.0 && crust.crust_type == CrustType::Oceanic {
                        crust.crust_type = CrustType::Continental;
                        crust.orogeny_type = Some(OrogenyType::Andean);
                    }
                }
            }
        }

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
        // Find continental-continental boundary pairs with enough interpenetration.
        let mut collision_pairs: HashSet<(u32, u32)> = HashSet::new();
        for edge in boundary {
            if edge.crust_a != CrustType::Continental || edge.crust_b != CrustType::Continental {
                continue;
            }
            let vel_a = self.plates[edge.plate_a as usize]
                .surface_velocity(self.points[edge.point as usize]);
            let vel_b = self.plates[edge.plate_b as usize]
                .surface_velocity(self.points[edge.point as usize]);
            // Only converging plates.
            let relative = vel_a - vel_b;
            let toward = self.points[edge.point as usize];
            if relative.dot(toward) >= 0.0 {
                continue;
            }
            let pair = (edge.plate_a.min(edge.plate_b), edge.plate_a.max(edge.plate_b));
            collision_pairs.insert(pair);
        }

        for (plate_a, plate_b) in collision_pairs {
            self.try_collision(plate_a, plate_b);
        }
    }

    fn try_collision(&mut self, plate_a: u32, plate_b: u32) {
        let terranes_a = continental_collision::find_terranes(
            &self.plates[plate_a as usize], &self.points, &self.adjacency,
        );
        if terranes_a.is_empty() {
            return;
        }

        let center_b = plate_centroid(&self.plates[plate_b as usize], &self.points);

        // Pick the terrane closest to plate_b.
        let nearest = terranes_a.iter().min_by(|a, b| {
            let da = arc_distance(a.centroid, center_b);
            let db = arc_distance(b.centroid, center_b);
            da.partial_cmp(&db).unwrap()
        }).unwrap();

        let dist_to_boundary = arc_distance(nearest.centroid, center_b);
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

        // Apply elevation surge to points within influence radius on plate_b.
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

        // Transfer the terrane from plate_a to plate_b.
        let (src, dst) = borrow_two_mut(&mut self.plates, plate_a as usize, plate_b as usize);
        continental_collision::transfer_terrane(nearest, src, dst);
    }

    fn process_rifting(&mut self) -> bool {
        let mut rifted = false;
        let mut new_plates: Vec<Plate> = Vec::new();
        let mut emptied: Vec<usize> = Vec::new();

        let total_points = self.points.len();
        let plate_count = self.plates.len();

        for i in 0..plate_count {
            let p = &self.plates[i];
            let cont = p.crust.iter().filter(|c| c.crust_type == CrustType::Continental).count();
            let frac = cont as f64 / p.point_count() as f64;
            let eligible = p.point_count() >= 50 && frac >= 0.3;
            if eligible {
                let avg = total_points as f64 / plate_count as f64;
                let area_ratio = p.point_count() as f64 / avg;
                let lambda = 0.02 * frac * area_ratio;
                let prob = lambda * (-lambda).exp();
                eprintln!("[RIFT] t={:.0} plate[{i}]: {} pts, cont={:.0}%, area_ratio={:.2}, λ={:.4}, P={:.4}",
                    self.time, p.point_count(), frac*100.0, area_ratio, lambda, prob);
            }
            if !plate_rifting::should_rift(&self.plates[i], i, total_points, plate_count, self.time, self.rift_seed) {
                continue;
            }
            eprintln!("[RIFT] >>> RIFTED plate[{i}]!");
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

        // Remove emptied plates in reverse order to preserve indices.
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

    fn generate_oceanic_crust(&mut self, boundary: &[BoundaryEdge]) {
        let divergent = oceanic_crust_generation::find_divergent_edges(
            boundary, &self.plates, &self.points,
        );
        if divergent.is_empty() {
            return;
        }

        let dt_since = self.steps_since_crust_generation as f64 * DT;
        let new_points = oceanic_crust_generation::generate_ridge_points(
            &divergent, &self.plates, &self.points, dt_since,
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

    fn apply_erosion_and_damping(&mut self) {
        let fbm: Fbm<Perlin> = Fbm::new(EROSION_NOISE_SEED)
            .set_octaves(EROSION_NOISE_OCTAVES)
            .set_frequency(EROSION_NOISE_FREQUENCY);

        for plate in &mut self.plates {
            for (local, &global) in plate.point_indices.iter().enumerate() {
                let p = self.points[global as usize];
                let noise = 1.0 + EROSION_NOISE_AMPLITUDE * fbm.get([p.x, p.y, p.z]);
                let scale = noise.max(0.1);
                let crust = &mut plate.crust[local];
                match crust.crust_type {
                    CrustType::Continental => {
                        // z(t+dt) = z(t) - (z/zc) * εc * δt, modulated by spatial noise.
                        crust.elevation -= (crust.elevation / MAX_CONTINENTAL_ALTITUDE)
                            * CONTINENTAL_EROSION
                            * scale
                            * DT;
                    }
                    CrustType::Oceanic => {
                        // z(t+dt) = z(t) - (1 - z/zt) * εo * δt, modulated by spatial noise.
                        crust.elevation -= (1.0 - crust.elevation / OCEANIC_TRENCH_DEPTH)
                            * OCEANIC_DAMPING
                            * scale
                            * DT;
                        // Trench sediment fill.
                        crust.elevation += SEDIMENT_ACCRETION * DT;
                        // Age oceanic crust.
                        crust.age += DT;
                    }
                }
            }
        }
    }
}

/// Spatial hash grid on the unit sphere for O(1) proximity lookups.
///
/// Discretizes the sphere into latitude/longitude bins. Queries expand to
/// neighboring bins that could overlap the angular search radius.
struct SphereGrid<T> {
    /// Bins indexed by `lat_bin * LON_BINS + lon_bin`.
    cells: Vec<Vec<T>>,
}

/// Number of latitude bands.
const LAT_BINS: usize = 64;
/// Number of longitude bands.
const LON_BINS: usize = 128;

impl<T> SphereGrid<T> {
    fn new() -> Self {
        let cells: Vec<Vec<T>> = (0..LAT_BINS * LON_BINS).map(|_| Vec::new()).collect();
        Self { cells }
    }

    fn bin(p: DVec3) -> (usize, usize) {
        let lat = p.y.clamp(-1.0, 1.0).asin(); // -π/2..π/2
        let lon = p.z.atan2(p.x);               // -π..π
        let lat_bin = ((lat / std::f64::consts::PI + 0.5) * LAT_BINS as f64)
            .max(0.0).min(LAT_BINS as f64 - 1.0) as usize;
        let lon_bin = ((lon / std::f64::consts::TAU + 0.5) * LON_BINS as f64)
            .max(0.0).min(LON_BINS as f64 - 1.0) as usize;
        (lat_bin, lon_bin)
    }

    fn insert(&mut self, p: DVec3, item: T) {
        let (lat, lon) = Self::bin(p);
        self.cells[lat * LON_BINS + lon].push(item);
    }

    /// Iterate all items in bins that could contain points within `dot_threshold`
    /// of `p`. The angular radius is `acos(dot_threshold)`.
    fn query(&self, p: DVec3, dot_threshold: f64) -> SphereGridIter<'_, T> {
        let radius = dot_threshold.clamp(-1.0, 1.0).acos();
        // Add one bin width as margin.
        let margin = radius + std::f64::consts::PI / LAT_BINS as f64;

        let lat = p.y.clamp(-1.0, 1.0).asin();
        let lon = p.z.atan2(p.x);

        let lat_half = margin / (std::f64::consts::PI / LAT_BINS as f64);
        let (plat, _) = Self::bin(p);
        let lat_lo = (plat as isize - lat_half.ceil() as isize).max(0) as usize;
        let lat_hi = ((plat as isize + lat_half.ceil() as isize) as usize).min(LAT_BINS - 1);

        // Longitude range expands near poles due to convergence.
        let cos_lat = lat.cos().max(0.01);
        let lon_half = (margin / cos_lat) / (std::f64::consts::TAU / LON_BINS as f64);
        let (_, plon) = Self::bin(p);
        let lon_span = lon_half.ceil() as usize;

        SphereGridIter {
            cells: &self.cells,
            lat: lat_lo,
            lat_hi,
            lon_center: plon,
            lon_span,
            lon_offset: 0,
            item_idx: 0,
        }
    }
}

struct SphereGridIter<'a, T> {
    cells: &'a [Vec<T>],
    lat: usize,
    lat_hi: usize,
    lon_center: usize,
    lon_span: usize,
    lon_offset: usize,  // 0..=2*lon_span
    item_idx: usize,
}

impl<'a, T> Iterator for SphereGridIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.lat > self.lat_hi {
                return None;
            }
            let lon_count = 2 * self.lon_span + 1;
            if self.lon_offset >= lon_count {
                self.lat += 1;
                self.lon_offset = 0;
                self.item_idx = 0;
                continue;
            }
            let lon = (self.lon_center + LON_BINS + self.lon_offset - self.lon_span) % LON_BINS;
            let cell = &self.cells[self.lat * LON_BINS + lon];
            if self.item_idx < cell.len() {
                let item = &cell[self.item_idx];
                self.item_idx += 1;
                return Some(item);
            }
            self.lon_offset += 1;
            self.item_idx = 0;
        }
    }
}

/// A boundary edge: a point where two plates meet.
pub(super) struct BoundaryEdge {
    pub(super) point: u32,
    pub(super) neighbor: u32,
    pub(super) plate_a: u32,
    pub(super) plate_b: u32,
    pub(super) crust_a: CrustType,
    pub(super) crust_b: CrustType,
    pub(super) age_a: f64,
    pub(super) age_b: f64,
}

/// Scan the adjacency graph for edges that cross plate boundaries.
fn find_boundary_edges(
    plates: &[Plate],
    points: &[DVec3],
    adjacency: &Adjacency,
) -> Vec<BoundaryEdge> {
    let mut point_to_plate = vec![0u32; points.len()];
    let mut point_to_local = vec![0usize; points.len()];
    for (plate_idx, plate) in plates.iter().enumerate() {
        for (local, &global) in plate.point_indices.iter().enumerate() {
            point_to_plate[global as usize] = plate_idx as u32;
            point_to_local[global as usize] = local;
        }
    }

    let mut edges = Vec::new();
    let mut seen: HashSet<(u32, u32)> = HashSet::new();

    for point in 0..points.len() as u32 {
        let plate_a = point_to_plate[point as usize];
        for &neighbor in adjacency.neighbors_of(point) {
            let plate_b = point_to_plate[neighbor as usize];
            if plate_a == plate_b {
                continue;
            }
            let pair = (point.min(neighbor), point.max(neighbor));
            if !seen.insert(pair) {
                continue;
            }
            let local_a = point_to_local[point as usize];
            let local_b = point_to_local[neighbor as usize];
            edges.push(BoundaryEdge {
                point,
                neighbor,
                plate_a,
                plate_b,
                crust_a: plates[plate_a as usize].crust[local_a].crust_type,
                crust_b: plates[plate_b as usize].crust[local_b].crust_type,
                age_a: plates[plate_a as usize].crust[local_a].age,
                age_b: plates[plate_b as usize].crust[local_b].age,
            });
        }
    }

    edges
}

fn arc_distance(a: DVec3, b: DVec3) -> f64 {
    a.normalize().dot(b.normalize()).clamp(-1.0, 1.0).acos() * PLANET_RADIUS
}

fn plate_centroid(plate: &Plate, points: &[DVec3]) -> DVec3 {
    let sum: DVec3 = plate.point_indices.iter().map(|&i| points[i as usize]).sum();
    sum.normalize_or_zero()
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
    use super::super::plate_seed_placement::{assign_plates, WarpParams};

    fn setup(point_count: u32, plate_count: u32) -> Simulation {
        let fib = SphericalFibonacci::new(point_count);
        let points = fib.all_points();
        let del = SphericalDelaunay::from_points(&points);
        let assignment = assign_plates(&points, &fib, &del, plate_count, 42, &WarpParams::default());
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
        // Disable plate motion so only erosion/damping acts.
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
