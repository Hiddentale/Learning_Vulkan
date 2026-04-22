use glam::DVec3;

#[derive(Copy, Clone, Debug, PartialEq)]
struct Coord3D {
    x: f64,
    y: f64,
    z: f64,
}

/// Adaptive-precision orient3d using Shewchuk's four-level predicate.
///
/// Returns a positive value if d lies below the plane (a, b, c),
/// negative if above, zero if coplanar. Uses an error-bounded fast path
/// and progressively exact fallbacks.
pub fn orient3d(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> f64 {
    orient3d_exact(
        Coord3D { x: a.x, y: a.y, z: a.z },
        Coord3D { x: b.x, y: b.y, z: b.z },
        Coord3D { x: c.x, y: c.y, z: c.z },
        Coord3D { x: d.x, y: d.y, z: d.z },
    )
}

const DEKKER_SPLITTER: f64 = 134_217_729f64;
const MACHINE_EPSILON: f64 = 0.000_000_000_000_000_111_022_302_462_515_65;
const CORRECTION_ERROR_BOUND: f64 = (3.0 + 8.0 * MACHINE_EPSILON) * MACHINE_EPSILON;
const FAST_PATH_ERROR_BOUND: f64 = (7.0 + 56.0 * MACHINE_EPSILON) * MACHINE_EPSILON;
const EXACT_SUBDET_ERROR_BOUND: f64 = (3.0 + 28.0 * MACHINE_EPSILON) * MACHINE_EPSILON;
const TAIL_CORRECTION_ERROR_BOUND: f64 = (26.0 + 288.0 * MACHINE_EPSILON) * MACHINE_EPSILON * MACHINE_EPSILON;

pub fn orient3d_exact(
    a: Coord3D,
    b: Coord3D,
    c: Coord3D,
    d: Coord3D,
) -> f64 {

    let adx = a.x - d.x;
    let bdx = b.x - d.x;
    let cdx = c.x - d.x;
    let ady = a.y - d.y;
    let bdy = b.y - d.y;
    let cdy = c.y - d.y;
    let adz = a.z - d.z;
    let bdz = b.z - d.z;
    let cdz = c.z - d.z;

    let bdxcdy = bdx * cdy;
    let cdxbdy = cdx * bdy;
    let cdxady = cdx * ady;
    let adxcdy = adx * cdy;
    let adxbdy = adx * bdy;
    let bdxady = bdx * ady;

    let determinant = adz * (bdxcdy - cdxbdy) + bdz * (cdxady - adxcdy) + cdz * (adxbdy - bdxady);

    let worst_case_magnitude = (bdxcdy.abs() + cdxbdy.abs()) * adz.abs()
        + (cdxady.abs() + adxcdy.abs()) * bdz.abs()
        + (adxbdy.abs() + bdxady.abs()) * cdz.abs();

    let error_bound = FAST_PATH_ERROR_BOUND * worst_case_magnitude;
    if determinant > error_bound || -determinant > error_bound {
        return determinant;
    }

    orient3dadapt(a, b, c, d, worst_case_magnitude)
}

fn orient3dadapt(
    a: Coord3D,
    b: Coord3D,
    c: Coord3D,
    d: Coord3D,
    worst_case_magnitude: f64,
) -> f64 {
    let adx = a.x - d.x;
    let bdx = b.x - d.x;
    let cdx = c.x - d.x;
    let ady = a.y - d.y;
    let bdy = b.y - d.y;
    let cdy = c.y - d.y;
    let adz = a.z - d.z;
    let bdz = b.z - d.z;
    let cdz = c.z - d.z;

    let (bdxcdy1, bdxcdy0) = two_product(bdx, cdy);
    let (cdxbdy1, cdxbdy0) = two_product(cdx, bdy);
    let (bc3, bc2, bc1, bc0) = two_two_diff(bdxcdy1, bdxcdy0, cdxbdy1, cdxbdy0);
    let bc = [bc0, bc1, bc2, bc3];
    let mut adet = [0f64; 8];
    let alen = scale_expansion_zeroelim(&bc[..4], adz, &mut adet);

    let (cdxady1, cdxady0) = two_product(cdx, ady);
    let (adxcdy1, adxcdy0) = two_product(adx, cdy);
    let (ca3, ca2, ca1, ca0) = two_two_diff(cdxady1, cdxady0, adxcdy1, adxcdy0);
    let ca = [ca0, ca1, ca2, ca3];
    let mut bdet = [0f64; 8];
    let blen = scale_expansion_zeroelim(&ca[..4], bdz, &mut bdet);

    let (adxbdy1, adxbdy0) = two_product(adx, bdy);
    let (bdxady1, bdxady0) = two_product(bdx, ady);
    let (ab3, ab2, ab1, ab0) = two_two_diff(adxbdy1, adxbdy0, bdxady1, bdxady0);
    let ab = [ab0, ab1, ab2, ab3];
    let mut cdet = [0f64; 8];
    let clen = scale_expansion_zeroelim(&ab[..4], cdz, &mut cdet);

    let mut abdet = [0f64; 16];
    let ablen = fast_expansion_sum_zeroelim(&adet[..alen], &bdet[..blen], &mut abdet);
    let mut fin1 = [0f64; 192];
    let finlength = fast_expansion_sum_zeroelim(&abdet[..ablen], &cdet[..clen], &mut fin1);

    let mut det = estimate(&fin1[..finlength]);
    let mut errbound = EXACT_SUBDET_ERROR_BOUND * worst_case_magnitude;
    if det >= errbound || -det >= errbound {
        return det;
    }

    let adxtail = two_diff_tail(a.x, d.x, adx);
    let bdxtail = two_diff_tail(b.x, d.x, bdx);
    let cdxtail = two_diff_tail(c.x, d.x, cdx);
    let adytail = two_diff_tail(a.y, d.y, ady);
    let bdytail = two_diff_tail(b.y, d.y, bdy);
    let cdytail = two_diff_tail(c.y, d.y, cdy);
    let adztail = two_diff_tail(a.z, d.z, adz);
    let bdztail = two_diff_tail(b.z, d.z, bdz);
    let cdztail = two_diff_tail(c.z, d.z, cdz);

    if (adxtail == 0.0)
        && (bdxtail == 0.0)
        && (cdxtail == 0.0)
        && (adytail == 0.0)
        && (bdytail == 0.0)
        && (cdytail == 0.0)
        && (adztail == 0.0)
        && (bdztail == 0.0)
        && (cdztail == 0.0)
    {
        return det;
    }

    errbound = TAIL_CORRECTION_ERROR_BOUND * worst_case_magnitude + CORRECTION_ERROR_BOUND * det.abs();
    det += (adz * ((bdx * cdytail + cdy * bdxtail) - (bdy * cdxtail + cdx * bdytail))
        + adztail * (bdx * cdy - bdy * cdx))
        + (bdz * ((cdx * adytail + ady * cdxtail) - (cdy * adxtail + adx * cdytail))
            + bdztail * (cdx * ady - cdy * adx))
        + (cdz * ((adx * bdytail + bdy * adxtail) - (ady * bdxtail + bdx * adytail))
            + cdztail * (adx * bdy - ady * bdx));
    if det >= errbound || -det >= errbound {
        return det;
    }

    let mut finnow = fin1;
    let mut finother = [0f64; 192];

    let mut at_b = [0f64; 4];
    let mut at_c = [0f64; 4];
    let mut bt_c = [0f64; 4];
    let mut bt_a = [0f64; 4];
    let mut ct_a = [0f64; 4];
    let mut ct_b = [0f64; 4];
    let at_blen: usize;
    let at_clen: usize;
    let bt_clen: usize;
    let bt_alen: usize;
    let ct_alen: usize;
    let ct_blen: usize;
    if adxtail == 0.0 {
        if adytail == 0.0 {
            at_b[0] = 0.0;
            at_blen = 1;
            at_c[0] = 0.0;
            at_clen = 1;
        } else {
            let negate = -adytail;
            (at_b[1], at_b[0]) = two_product(negate, bdx);
            at_blen = 2;
            (at_c[1], at_c[0]) = two_product(adytail, cdx);
            at_clen = 2;
        }
    } else if adytail == 0.0 {
        (at_b[1], at_b[0]) = two_product(adxtail, bdy);
        at_blen = 2;
        let negate = -adxtail;
        (at_c[1], at_c[0]) = two_product(negate, cdy);
        at_clen = 2;
    } else {
        let (adxt_bdy1, adxt_bdy0) = two_product(adxtail, bdy);
        let (adyt_bdx1, adyt_bdx0) = two_product(adytail, bdx);
        (at_b[3], at_b[2], at_b[1], at_b[0]) =
            two_two_diff(adxt_bdy1, adxt_bdy0, adyt_bdx1, adyt_bdx0);
        at_blen = 4;
        let (adyt_cdx1, adyt_cdx0) = two_product(adytail, cdx);
        let (adxt_cdy1, adxt_cdy0) = two_product(adxtail, cdy);
        (at_c[3], at_c[2], at_c[1], at_c[0]) =
            two_two_diff(adyt_cdx1, adyt_cdx0, adxt_cdy1, adxt_cdy0);
        at_clen = 4;
    }
    if bdxtail == 0.0 {
        if bdytail == 0.0 {
            bt_c[0] = 0.0;
            bt_clen = 1;
            bt_a[0] = 0.0;
            bt_alen = 1;
        } else {
            let negate = -bdytail;
            (bt_c[1], bt_c[0]) = two_product(negate, cdx);
            bt_clen = 2;
            (bt_a[1], bt_a[0]) = two_product(bdytail, adx);
            bt_alen = 2;
        }
    } else if bdytail == 0.0 {
        (bt_c[1], bt_c[0]) = two_product(bdxtail, cdy);
        bt_clen = 2;
        let negate = -bdxtail;
        (bt_a[1], bt_a[0]) = two_product(negate, ady);
        bt_alen = 2;
    } else {
        let (bdxt_cdy1, bdxt_cdy0) = two_product(bdxtail, cdy);
        let (bdyt_cdx1, bdyt_cdx0) = two_product(bdytail, cdx);
        (bt_c[3], bt_c[2], bt_c[1], bt_c[0]) =
            two_two_diff(bdxt_cdy1, bdxt_cdy0, bdyt_cdx1, bdyt_cdx0);
        bt_clen = 4;
        let (bdyt_adx1, bdyt_adx0) = two_product(bdytail, adx);
        let (bdxt_ady1, bdxt_ady0) = two_product(bdxtail, ady);
        (bt_a[3], bt_a[2], bt_a[1], bt_a[0]) =
            two_two_diff(bdyt_adx1, bdyt_adx0, bdxt_ady1, bdxt_ady0);
        bt_alen = 4;
    }
    if cdxtail == 0.0 {
        if cdytail == 0.0 {
            ct_a[0] = 0.0;
            ct_alen = 1;
            ct_b[0] = 0.0;
            ct_blen = 1;
        } else {
            let negate = -cdytail;
            (ct_a[1], ct_a[0]) = two_product(negate, adx);
            ct_alen = 2;
            (ct_b[1], ct_b[0]) = two_product(cdytail, bdx);
            ct_blen = 2;
        }
    } else if cdytail == 0.0 {
        (ct_a[1], ct_a[0]) = two_product(cdxtail, ady);
        ct_alen = 2;
        let negate = -cdxtail;
        (ct_b[1], ct_b[0]) = two_product(negate, bdy);
        ct_blen = 2;
    } else {
        let (cdxt_ady1, cdxt_ady0) = two_product(cdxtail, ady);
        let (cdyt_adx1, cdyt_adx0) = two_product(cdytail, adx);
        (ct_a[3], ct_a[2], ct_a[1], ct_a[0]) =
            two_two_diff(cdxt_ady1, cdxt_ady0, cdyt_adx1, cdyt_adx0);
        ct_alen = 4;
        let (cdyt_bdx1, cdyt_bdx0) = two_product(cdytail, bdx);
        let (cdxt_bdy1, cdxt_bdy0) = two_product(cdxtail, bdy);
        (ct_b[3], ct_b[2], ct_b[1], ct_b[0]) =
            two_two_diff(cdyt_bdx1, cdyt_bdx0, cdxt_bdy1, cdxt_bdy0);
        ct_blen = 4;
    }

    let mut bct = [0f64; 8];
    let mut cat = [0f64; 8];
    let mut abt = [0f64; 8];
    let mut u = [0f64; 4];
    let mut v = [0f64; 12];
    let mut w = [0f64; 16];
    let mut vlength: usize;
    let mut wlength: usize;

    let bctlen = fast_expansion_sum_zeroelim(&bt_c[..bt_clen], &ct_b[..ct_blen], &mut bct);
    wlength = scale_expansion_zeroelim(&bct[..bctlen], adz, &mut w);
    let mut finlength =
        fast_expansion_sum_zeroelim(&finnow[..finlength], &w[..wlength], &mut finother);
    ::core::mem::swap(&mut finnow, &mut finother);

    let catlen = fast_expansion_sum_zeroelim(&ct_a[..ct_alen], &at_c[..at_clen], &mut cat);
    wlength = scale_expansion_zeroelim(&cat[..catlen], bdz, &mut w);
    finlength = fast_expansion_sum_zeroelim(&finnow[..finlength], &w[..wlength], &mut finother);
    ::core::mem::swap(&mut finnow, &mut finother);

    let abtlen = fast_expansion_sum_zeroelim(&at_b[..at_blen], &bt_a[..bt_alen], &mut abt);
    wlength = scale_expansion_zeroelim(&abt[..abtlen], cdz, &mut w);
    finlength = fast_expansion_sum_zeroelim(&finnow[..finlength], &w[..wlength], &mut finother);
    ::core::mem::swap(&mut finnow, &mut finother);

    if adztail != 0.0 {
        vlength = scale_expansion_zeroelim(&bc[..4], adztail, &mut v);
        finlength = fast_expansion_sum_zeroelim(&finnow[..finlength], &v[..vlength], &mut finother);
        ::core::mem::swap(&mut finnow, &mut finother);
    }
    if bdztail != 0.0 {
        vlength = scale_expansion_zeroelim(&ca[..4], bdztail, &mut v);
        finlength = fast_expansion_sum_zeroelim(&finnow[..finlength], &v[..vlength], &mut finother);
        ::core::mem::swap(&mut finnow, &mut finother);
    }
    if cdztail != 0.0 {
        vlength = scale_expansion_zeroelim(&ab[..4], cdztail, &mut v);
        finlength = fast_expansion_sum_zeroelim(&finnow[..finlength], &v[..vlength], &mut finother);
        ::core::mem::swap(&mut finnow, &mut finother);
    }

    if adxtail != 0.0 {
        if bdytail != 0.0 {
            let (adxt_bdyt1, adxt_bdyt0) = two_product(adxtail, bdytail);
            (u[3], u[2], u[1], u[0]) = two_one_product(adxt_bdyt1, adxt_bdyt0, cdz);
            finlength = fast_expansion_sum_zeroelim(&finnow[..finlength], &u[..4], &mut finother);
            ::core::mem::swap(&mut finnow, &mut finother);
            if cdztail != 0.0 {
                (u[3], u[2], u[1], u[0]) = two_one_product(adxt_bdyt1, adxt_bdyt0, cdztail);
                finlength =
                    fast_expansion_sum_zeroelim(&finnow[..finlength], &u[..4], &mut finother);
                ::core::mem::swap(&mut finnow, &mut finother);
            }
        }
        if cdytail != 0.0 {
            let negate = -adxtail;
            let (adxt_cdyt1, adxt_cdyt0) = two_product(negate, cdytail);
            (u[3], u[2], u[1], u[0]) = two_one_product(adxt_cdyt1, adxt_cdyt0, bdz);
            finlength = fast_expansion_sum_zeroelim(&finnow[..finlength], &u[..4], &mut finother);
            ::core::mem::swap(&mut finnow, &mut finother);
            if bdztail != 0.0 {
                (u[3], u[2], u[1], u[0]) = two_one_product(adxt_cdyt1, adxt_cdyt0, bdztail);
                finlength =
                    fast_expansion_sum_zeroelim(&finnow[..finlength], &u[..4], &mut finother);
                ::core::mem::swap(&mut finnow, &mut finother);
            }
        }
    }
    if bdxtail != 0.0 {
        if cdytail != 0.0 {
            let (bdxt_cdyt1, bdxt_cdyt0) = two_product(bdxtail, cdytail);
            (u[3], u[2], u[1], u[0]) = two_one_product(bdxt_cdyt1, bdxt_cdyt0, adz);
            finlength = fast_expansion_sum_zeroelim(&finnow[..finlength], &u[..4], &mut finother);
            ::core::mem::swap(&mut finnow, &mut finother);
            if adztail != 0.0 {
                (u[3], u[2], u[1], u[0]) = two_one_product(bdxt_cdyt1, bdxt_cdyt0, adztail);
                finlength =
                    fast_expansion_sum_zeroelim(&finnow[..finlength], &u[..4], &mut finother);
                ::core::mem::swap(&mut finnow, &mut finother);
            }
        }
        if adytail != 0.0 {
            let negate = -bdxtail;
            let (bdxt_adyt1, bdxt_adyt0) = two_product(negate, adytail);
            (u[3], u[2], u[1], u[0]) = two_one_product(bdxt_adyt1, bdxt_adyt0, cdz);
            finlength = fast_expansion_sum_zeroelim(&finnow[..finlength], &u[..4], &mut finother);
            ::core::mem::swap(&mut finnow, &mut finother);
            if cdztail != 0.0 {
                (u[3], u[2], u[1], u[0]) = two_one_product(bdxt_adyt1, bdxt_adyt0, cdztail);
                finlength =
                    fast_expansion_sum_zeroelim(&finnow[..finlength], &u[..4], &mut finother);
                ::core::mem::swap(&mut finnow, &mut finother);
            }
        }
    }
    if cdxtail != 0.0 {
        if adytail != 0.0 {
            let (cdxt_adyt1, cdxt_adyt0) = two_product(cdxtail, adytail);
            (u[3], u[2], u[1], u[0]) = two_one_product(cdxt_adyt1, cdxt_adyt0, bdz);
            finlength = fast_expansion_sum_zeroelim(&finnow[..finlength], &u[..4], &mut finother);
            ::core::mem::swap(&mut finnow, &mut finother);
            if bdztail != 0.0 {
                (u[3], u[2], u[1], u[0]) = two_one_product(cdxt_adyt1, cdxt_adyt0, bdztail);
                finlength =
                    fast_expansion_sum_zeroelim(&finnow[..finlength], &u[..4], &mut finother);
                ::core::mem::swap(&mut finnow, &mut finother);
            }
        }
        if bdytail != 0.0 {
            let negate = -cdxtail;
            let (cdxt_bdyt1, cdxt_bdyt0) = two_product(negate, bdytail);
            (u[3], u[2], u[1], u[0]) = two_one_product(cdxt_bdyt1, cdxt_bdyt0, adz);
            finlength = fast_expansion_sum_zeroelim(&finnow[..finlength], &u[..4], &mut finother);
            ::core::mem::swap(&mut finnow, &mut finother);
            if adztail != 0.0 {
                (u[3], u[2], u[1], u[0]) = two_one_product(cdxt_bdyt1, cdxt_bdyt0, adztail);
                finlength =
                    fast_expansion_sum_zeroelim(&finnow[..finlength], &u[..4], &mut finother);
                ::core::mem::swap(&mut finnow, &mut finother);
            }
        }
    }

    if adztail != 0.0 {
        wlength = scale_expansion_zeroelim(&bct[..bctlen], adztail, &mut w);
        finlength = fast_expansion_sum_zeroelim(&finnow[..finlength], &w[..wlength], &mut finother);
        ::core::mem::swap(&mut finnow, &mut finother);
    }
    if bdztail != 0.0 {
        wlength = scale_expansion_zeroelim(&cat[..catlen], bdztail, &mut w);
        finlength = fast_expansion_sum_zeroelim(&finnow[..finlength], &w[..wlength], &mut finother);
        ::core::mem::swap(&mut finnow, &mut finother);
    }
    if cdztail != 0.0 {
        wlength = scale_expansion_zeroelim(&abt[..abtlen], cdztail, &mut w);
        finlength = fast_expansion_sum_zeroelim(&finnow[..finlength], &w[..wlength], &mut finother);
        ::core::mem::swap(&mut finnow, &mut finother);
    }

    finnow[finlength - 1]
}

fn scale_expansion_zeroelim(e: &[f64], b: f64, h: &mut [f64]) -> usize {
    let (bhi, blo) = split(b);
    let (mut Q, hh) = two_product_presplit(e[0], b, bhi, blo);
    let mut hindex = 0;
    if hh != 0.0 {
        h[hindex] = hh;
        hindex += 1;
    }
    for eindex in 1..e.len() {
        let enow = e[eindex];
        let (product1, product0) = two_product_presplit(enow, b, bhi, blo);
        let (sum, hh) = two_sum(Q, product0);
        if hh != 0.0 {
            h[hindex] = hh;
            hindex += 1;
        }
        let (new_q, hh) = fast_two_sum(product1, sum);
        Q = new_q;
        if hh != 0.0 {
            h[hindex] = hh;
            hindex += 1;
        }
    }
    if Q != 0.0 || hindex == 0 {
        h[hindex] = Q;
        hindex += 1;
    }
    hindex
}

#[inline]
fn two_product(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    (x, two_product_tail(a, b, x))
}

#[inline]
fn two_product_tail(a: f64, b: f64, x: f64) -> f64 {
    let (ahi, alo) = split(a);
    let (bhi, blo) = split(b);
    let err1 = x - (ahi * bhi);
    let err2 = err1 - (alo * bhi);
    let err3 = err2 - (ahi * blo);
    (alo * blo) - err3
}

#[inline]
fn split(a: f64) -> (f64, f64) {
    let c = DEKKER_SPLITTER * a;
    let abig = c - a;
    let ahi = c - abig;
    let alo = a - ahi;
    (ahi, alo)
}

#[inline]
fn two_product_presplit(a: f64, b: f64, bhi: f64, blo: f64) -> (f64, f64) {
    let x = a * b;
    let (ahi, alo) = split(a);
    let err1 = x - ahi * bhi;
    let err2 = err1 - alo * bhi;
    let err3 = err2 - ahi * blo;
    let y = alo * blo - err3;
    (x, y)
}

#[inline]
fn two_two_diff(a1: f64, a0: f64, b1: f64, b0: f64) -> (f64, f64, f64, f64) {
    let (j, _r0, x0) = two_one_diff(a1, a0, b0);
    let (x3, x2, x1) = two_one_diff(j, _r0, b1);
    (x3, x2, x1, x0)
}

#[inline]
fn two_one_diff(a1: f64, a0: f64, b: f64) -> (f64, f64, f64) {
    let (i, x0) = two_diff(a0, b);
    let (x2, x1) = two_sum(a1, i);
    (x2, x1, x0)
}

#[inline]
fn two_diff(a: f64, b: f64) -> (f64, f64) {
    let x = a - b;
    (x, two_diff_tail(a, b, x))
}

#[inline]
fn two_diff_tail(a: f64, b: f64, x: f64) -> f64 {
    let bvirt = a - x;
    let avirt = x + bvirt;
    let bround = bvirt - b;
    let around = a - avirt;
    around + bround
}

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    (x, two_sum_tail(a, b, x))
}

#[inline]
fn two_sum_tail(a: f64, b: f64, x: f64) -> f64 {
    let bvirt = x - a;
    let avirt = x - bvirt;
    let bround = b - bvirt;
    let around = a - avirt;
    around + bround
}

fn estimate(e: &[f64]) -> f64 {
    let mut q = e[0];
    for cur in &e[1..] {
        q += *cur;
    }
    q
}

fn fast_expansion_sum_zeroelim(e: &[f64], f: &[f64], h: &mut [f64]) -> usize {
    let mut enow = e[0];
    let mut fnow = f[0];
    let mut eindex = 0;
    let mut findex = 0;
    let mut Qnew;
    let mut hh;
    let mut Q;
    if (fnow > enow) == (fnow > -enow) {
        Q = enow;
        eindex += 1;
    } else {
        Q = fnow;
        findex += 1;
    }

    let mut hindex = 0;
    if eindex < e.len() && findex < f.len() {
        enow = e[eindex];
        fnow = f[findex];
        if (fnow > enow) == (fnow > -enow) {
            let r = fast_two_sum(enow, Q);
            Qnew = r.0;
            hh = r.1;
            eindex += 1;
        } else {
            let r = fast_two_sum(fnow, Q);
            Qnew = r.0;
            hh = r.1;
            findex += 1;
        }
        Q = Qnew;
        if hh != 0.0 {
            h[hindex] = hh;
            hindex += 1;
        }

        while eindex < e.len() && findex < f.len() {
            enow = e[eindex];
            fnow = f[findex];
            if (fnow > enow) == (fnow > -enow) {
                let r = two_sum(Q, enow);
                Qnew = r.0;
                hh = r.1;
                eindex += 1;
            } else {
                let r = two_sum(Q, fnow);
                Qnew = r.0;
                hh = r.1;
                findex += 1;
            };
            Q = Qnew;
            if hh != 0.0 {
                h[hindex] = hh;
                hindex += 1;
            }
        }
    }

    while eindex < e.len() {
        enow = e[eindex];
        let r = two_sum(Q, enow);
        Qnew = r.0;
        hh = r.1;
        Q = Qnew;
        eindex += 1;
        if hh != 0.0 {
            h[hindex] = hh;
            hindex += 1
        }
    }

    while findex < f.len() {
        fnow = f[findex];
        let r = two_sum(Q, fnow);
        Qnew = r.0;
        hh = r.1;
        Q = Qnew;
        findex += 1;
        if hh != 0.0 {
            h[hindex] = hh;
            hindex += 1
        }
    }

    if Q != 0.0 || hindex == 0 {
        h[hindex] = Q;
        hindex += 1;
    }
    hindex
}

#[inline]
fn fast_two_sum_tail(a: f64, b: f64, x: f64) -> f64 {
    let bvirt = x - a;
    b - bvirt
}

#[inline]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    (x, fast_two_sum_tail(a, b, x))
}

#[inline]
fn two_one_product(a1: f64, a0: f64, b: f64) -> (f64, f64, f64, f64) {
    let (bhi, blo) = split(b);
    let (mut _i, x0) = two_product_presplit(a0, b, bhi, blo);
    let (mut _j, _0) = two_product_presplit(a1, b, bhi, blo);
    let (_k, x1) = two_sum(_i, _0);
    let (x3, x2) = fast_two_sum(_j, _k);
    (x3, x2, x1, x0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orient3d_fixtures() {
        let content = std::fs::read_to_string("src/world_generation/sphere_geometry/orient3d.txt")
            .expect("fixture file missing");
        for (idx, line) in content.lines().enumerate() {
            let f: Vec<f64> = line.split_whitespace()
                .skip(1)
                .map(|s| s.parse::<f64>().unwrap())
                .collect();
            let a = DVec3::new(f[0], f[1], f[2]);
            let b = DVec3::new(f[3], f[4], f[5]);
            let c = DVec3::new(f[6], f[7], f[8]);
            let d = DVec3::new(f[9], f[10], f[11]);
            let expected_sign = f[12];

            let result = orient3d(a, b, c, d);
            assert!(
                result.signum() == expected_sign.signum(),
                "Line {}: orient3d returned {result} (sign {}), expected sign {expected_sign}",
                idx + 1, result.signum(),
            );
        }
    }

    #[test]
    fn test_orient3d_basic() {
        let plane_a = DVec3::new(1., 0., 1.);
        let plane_b = DVec3::new(-1., 0., -1.);
        let plane_c = DVec3::new(-1., 0., 0.);

        let above = DVec3::splat(f64::MIN_POSITIVE);
        let below = DVec3::splat(-f64::MIN_POSITIVE);
        let coplanar = DVec3::ZERO;

        assert!(orient3d(plane_a, plane_b, plane_c, above) < 0.0);
        assert!(orient3d(plane_a, plane_b, plane_c, below) > 0.0);
        assert!(orient3d(plane_a, plane_b, plane_c, coplanar) == 0.0);
    }

    #[test]
    fn positive_tetrahedron() {
        let a = DVec3::new(1.0, 0.0, 0.0);
        let b = DVec3::new(0.0, 1.0, 0.0);
        let c = DVec3::new(0.0, 0.0, 1.0);
        let d = DVec3::ZERO;
        assert!(orient3d(a, b, c, d) > 0.0);
    }

    #[test]
    fn negative_when_swapped() {
        let a = DVec3::new(1.0, 0.0, 0.0);
        let b = DVec3::new(0.0, 1.0, 0.0);
        let c = DVec3::new(0.0, 0.0, 1.0);
        let d = DVec3::ZERO;
        assert!(orient3d(a, c, b, d) < 0.0);
    }

    #[test]
    fn zero_for_coplanar() {
        let a = DVec3::new(1.0, 0.0, 0.0);
        let b = DVec3::new(0.0, 1.0, 0.0);
        let c = DVec3::new(-1.0, -1.0, 0.0);
        let d = DVec3::new(0.5, 0.5, 0.0);
        assert_eq!(orient3d(a, b, c, d), 0.0);
    }
}
