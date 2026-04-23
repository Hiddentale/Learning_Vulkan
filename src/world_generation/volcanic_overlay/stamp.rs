use glam::DVec3;

use super::atlas::VolcanicAtlas;
use super::Placement;
use crate::world_generation::terrain_amplification::cross_layout::CrossLayout;

const PATCH_SIZE: usize = 256;
const FEATHER_FRAC: f64 = 0.2;
const MIN_FEATURE_HEIGHT: f32 = 200.0;
const MAX_ISLAND_SUMMIT: f32 = 1500.0;

/// Stamp all volcanic features into the cross-layout elevation grid.
/// Builds a zero-initialized overlay, stamps features into it, then adds
/// to the cross elevation. Only raises ocean pixels.
pub(super) fn stamp_all(
    cross_elevation: &mut [f32],
    cross_width: u32,
    cross_height: u32,
    placements: &[Placement],
    atlas: &VolcanicAtlas,
) {
    let w = cross_width as usize;
    let h = cross_height as usize;
    let n = w * h;

    let cross = CrossLayout::new(cross_width / 4);

    // Zero-initialized overlay
    let mut overlay = vec![0.0f32; n];

    for placement in placements {
        stamp_into_overlay(&mut overlay, &cross, w, h, placement, atlas);
    }

    // Additive merge: only raise ocean pixels
    for px in 0..n {
        let add = overlay[px];
        if add > 0.0 && cross_elevation[px] < 0.0 {
            cross_elevation[px] += add;
        }
    }
}

fn stamp_into_overlay(
    overlay: &mut [f32],
    cross: &CrossLayout,
    w: usize,
    h: usize,
    placement: &Placement,
    atlas: &VolcanicAtlas,
) {
    let meta = atlas.patch_meta(placement.category, placement.patch_idx);
    if !meta.valid {
        return;
    }

    let pixels = atlas.patch_pixels(placement.category, placement.patch_idx);
    let base = meta.base_level as f32;
    let peak = meta.peak_height as f32;
    let ang_r = placement.angular_radius;
    let feather_start = ang_r * (1.0 - FEATHER_FRAC);

    let height_normalize = if peak > 1.0 {
        (MAX_ISLAND_SUMMIT - base.min(0.0)) / peak
    } else {
        1.0
    };

    let center = placement.center;
    let (tangent_u, tangent_v) = tangent_frame(center);

    for i in 0..h as u32 {
        for j in 0..w as u32 {
            let dir = match cross.pixel_to_sphere(i, j) {
                Some(d) => d,
                None => continue, // inactive cross pixel
            };

            let ang_dist = center.dot(dir).clamp(-1.0, 1.0).acos();
            if ang_dist > ang_r {
                continue;
            }

            let offset = dir - center * center.dot(dir);
            let u = offset.dot(tangent_u);
            let v = offset.dot(tangent_v);
            let nu = (u / ang_r * 0.5 + 0.5).clamp(0.0, 1.0);
            let nv = (v / ang_r * 0.5 + 0.5).clamp(0.0, 1.0);
            let (pu, pv) = apply_transform(nu, nv, placement.rotation, placement.flip);

            let px = pu * (PATCH_SIZE - 1) as f64;
            let py = pv * (PATCH_SIZE - 1) as f64;
            let patch_elev = bilinear_sample(pixels, px, py);

            let feature_height = patch_elev - base;
            if feature_height < MIN_FEATURE_HEIGHT {
                continue;
            }

            let weight = if ang_dist < feather_start {
                1.0f32
            } else {
                let t = ((ang_dist - feather_start) / (ang_r - feather_start)).clamp(0.0, 1.0);
                1.0 - t as f32 * t as f32 * (3.0 - 2.0 * t as f32)
            };

            let value = feature_height * height_normalize * placement.height_scale * weight;
            let idx = i as usize * w + j as usize;

            if value > overlay[idx] {
                overlay[idx] = value;
            }
        }
    }
}

fn tangent_frame(center: DVec3) -> (DVec3, DVec3) {
    let up = if center.y.abs() < 0.9 { DVec3::Y } else { DVec3::X };
    let u = center.cross(up).normalize();
    let v = center.cross(u).normalize();
    (u, v)
}

fn apply_transform(u: f64, v: f64, rotation: u8, flip: bool) -> (f64, f64) {
    let (mut u, v) = match rotation {
        0 => (u, v),
        1 => (1.0 - v, u),
        2 => (1.0 - u, 1.0 - v),
        3 => (v, 1.0 - u),
        _ => (u, v),
    };
    if flip {
        u = 1.0 - u;
    }
    (u, v)
}

fn bilinear_sample(pixels: &[i16], x: f64, y: f64) -> f32 {
    let x0 = (x as usize).min(PATCH_SIZE - 2);
    let y0 = (y as usize).min(PATCH_SIZE - 2);
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let fx = (x - x0 as f64) as f32;
    let fy = (y - y0 as f64) as f32;

    let v00 = pixels[y0 * PATCH_SIZE + x0] as f32;
    let v10 = pixels[y0 * PATCH_SIZE + x1] as f32;
    let v01 = pixels[y1 * PATCH_SIZE + x0] as f32;
    let v11 = pixels[y1 * PATCH_SIZE + x1] as f32;

    let top = v00 + (v10 - v00) * fx;
    let bot = v01 + (v11 - v01) * fx;
    top + (bot - top) * fy
}
