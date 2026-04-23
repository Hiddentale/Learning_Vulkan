use glam::DVec3;

use super::atlas::VolcanicAtlas;
use super::Placement;

const PATCH_SIZE: usize = 256;
const FEATHER_FRAC: f64 = 0.2;

/// Minimum feature height above base level (meters) to stamp.
const MIN_FEATURE_HEIGHT: f32 = 200.0;

/// Build a zero-initialized overlay grid for each face, stamp all volcanic
/// features into it, then add the overlay onto the terrain.
/// The original terrain is never read during stamping — only at the final
/// additive step — so no side effects are possible.
pub(super) fn stamp_all(
    terrain: &mut super::super::terrain_amplification::AmplifiedTerrain,
    placements: &[Placement],
    atlas: &VolcanicAtlas,
) {
    let res = terrain.faces[0].resolution;
    let face_px = (res * res) as usize;

    // Zero-initialized overlay: only volcanic features go here
    let mut overlay: Vec<Vec<f32>> = (0..6).map(|_| vec![0.0f32; face_px]).collect();

    for placement in placements {
        stamp_into_overlay(&mut overlay, res, placement, atlas);
    }

    // Additive merge: only raise ocean pixels, never touch land or lower anything
    for fi in 0..6 {
        for px in 0..face_px {
            let add = overlay[fi][px];
            if add > 0.0 && terrain.faces[fi].elevation[px] < 0.0 {
                terrain.faces[fi].elevation[px] += add;
            }
        }
    }
}

fn stamp_into_overlay(
    overlay: &mut [Vec<f32>],
    res: u32,
    placement: &Placement,
    atlas: &VolcanicAtlas,
) {
    let meta = atlas.patch_meta(placement.category, placement.patch_idx);
    if !meta.valid {
        return;
    }

    let pixels = atlas.patch_pixels(placement.category, placement.patch_idx);
    let base = meta.base_level as f32;
    let ang_r = placement.angular_radius;
    let feather_start = ang_r * (1.0 - FEATHER_FRAC);

    let center = placement.center;
    let (tangent_u, tangent_v) = tangent_frame(center);

    for fi in 0..6usize {
        let face_center = face_center_dir(fi);
        let face_half_angle = std::f64::consts::FRAC_PI_4 * 1.5;
        if center.dot(face_center).acos() > ang_r + face_half_angle {
            continue;
        }

        for r in 0..res {
            for c in 0..res {
                let dir = pixel_to_sphere(fi, r, c, res);
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

                let value = feature_height * placement.height_scale * weight;
                let idx = (r * res + c) as usize;

                // Max with existing overlay (multiple placements can overlap)
                if value > overlay[fi][idx] {
                    overlay[fi][idx] = value;
                }
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

fn face_center_dir(fi: usize) -> DVec3 {
    match fi {
        0 => DVec3::X,
        1 => DVec3::NEG_X,
        2 => DVec3::Y,
        3 => DVec3::NEG_Y,
        4 => DVec3::Z,
        _ => DVec3::NEG_Z,
    }
}

fn pixel_to_sphere(fi: usize, r: u32, c: u32, res: u32) -> DVec3 {
    let vr = r as f64 / (res - 1).max(1) as f64;
    let uc = c as f64 / (res - 1).max(1) as f64;
    let (v_param, u_param) = if fi == 0 || fi == 1 || fi == 4 || fi == 5 {
        (1.0 - vr, uc)
    } else {
        (vr, uc)
    };
    let (x, y, z) = match fi {
        0 => (1.0, 2.0 * v_param - 1.0, 1.0 - 2.0 * u_param),
        1 => (-1.0, 2.0 * v_param - 1.0, 2.0 * u_param - 1.0),
        2 => (2.0 * u_param - 1.0, 1.0, 2.0 * v_param - 1.0),
        3 => (2.0 * u_param - 1.0, -1.0, -(2.0 * v_param - 1.0)),
        4 => (2.0 * u_param - 1.0, 2.0 * v_param - 1.0, 1.0),
        _ => (1.0 - 2.0 * u_param, 2.0 * v_param - 1.0, -1.0),
    };
    DVec3::new(x, y, z).normalize()
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
