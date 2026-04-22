use std::f64::consts::PI;

use glam::DVec3;

use super::elevation::smoothstep;

const EQUATOR_TEMP: f64 = 30.0;
const POLE_TEMP_DROP: f64 = 55.0;
const LAPSE_RATE: f64 = 6.5;
const PRECIP_MAX: f64 = 2500.0;
const PRECIP_MIN: f64 = 50.0;

pub(super) fn temperature(p: DVec3, elevation_km: f64) -> f64 {
    let latitude = p.y.clamp(-1.0, 1.0).asin();
    let lat_frac = latitude / (PI / 2.0);
    let base_temp = EQUATOR_TEMP - POLE_TEMP_DROP * lat_frac * lat_frac;
    base_temp - LAPSE_RATE * elevation_km.max(0.0)
}

pub(super) fn precipitation(p: DVec3, dist_coast: f64) -> f64 {
    let latitude = p.y.clamp(-1.0, 1.0).asin();
    let abs_lat = latitude.abs();

    let itcz = (-8.0 * abs_lat * abs_lat).exp();
    let midlat = 0.6 * (-8.0 * (abs_lat - 1.05).powi(2)).exp();
    let lat_factor = itcz + midlat + 0.15;

    let continental_drying = 1.0 - 0.6 * smoothstep(0.0, 20.0, dist_coast);

    (PRECIP_MAX * lat_factor * continental_drying).max(PRECIP_MIN)
}
