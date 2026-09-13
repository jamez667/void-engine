//! Narrow-phase overlap probes: pure geometry, no spatial state.
//!
//! These are the functions a caller runs over the candidate list that
//! [`SpatialGrid`](super::SpatialGrid) hands back. Each takes plain shape
//! parameters and returns `Some((normal, overlap))` when the two shapes
//! penetrate, where `normal` points from B toward A — push A along
//! `+normal` to separate them.
//!
//! Nothing here touches the grid, so they are usable standalone by a
//! caller with its own broad phase or none at all.

use glam::DVec2;

/// Swept segment `p0 → p1` vs a static circle at `center` / `radius`,
/// returning the **near-hit parameter** `t ∈ [0, 1]` — the fraction along
/// the segment at which it first touches the disc, so the contact point is
/// `p0 + (p1 - p0) * t`.
///
/// `Some(0.0)` means `p0` is already inside the disc (no entry crossing to
/// find). `None` means the segment misses entirely, or the disc lies
/// wholly behind `p0` / beyond `p1`.
///
/// Callers wanting only a yes/no should use [`segment_vs_circle`], which
/// is this function's `.is_some()`.
pub fn segment_vs_circle_t(p0: DVec2, p1: DVec2, center: DVec2, radius: f64) -> Option<f64> {
    let d = p1 - p0;
    let f = p0 - center;
    let a = d.dot(d);

    // Degenerate: stationary segment collapses to a point-in-disc test.
    if a == 0.0 {
        return if f.length() < radius { Some(0.0) } else { None };
    }

    let b = 2.0 * f.dot(d);
    let c = f.dot(f) - radius * radius;
    let disc = b * b - 4.0 * a * c;

    if disc < 0.0 { return None; }

    let sq = disc.sqrt();
    let t1 = (-b - sq) / (2.0 * a);
    let t2 = (-b + sq) / (2.0 * a);

    // Solution range [t1, t2] must overlap the segment [0, 1].
    if t1 > 1.0 || t2 < 0.0 { return None; }

    // Clamp the entry root into the segment: a start already inside the
    // disc has t1 < 0, and its first "contact" is t = 0.
    Some(t1.max(0.0))
}

/// Swept segment `p0 → p1` vs a static circle at `center` / `radius`.
/// `true` iff the segment (or its degenerate stationary point) intersects
/// the disc. Natural narrow-phase pair for
/// [`SpatialGrid::query_segment`](super::SpatialGrid::query_segment):
/// query yields candidates, this checks each one.
///
/// Thin wrapper over [`segment_vs_circle_t`] for callers that don't need
/// the contact parameter.
#[inline]
pub fn segment_vs_circle(p0: DVec2, p1: DVec2, center: DVec2, radius: f64) -> bool {
    segment_vs_circle_t(p0, p1, center, radius).is_some()
}

/// Local axes of an OBB with rotation `rot` (rad). `[0]` is the box's
/// local +X in world space, `[1]` its local +Y.
#[inline]
pub fn obb_axes(rot: f64) -> [DVec2; 2] {
    let (s, c) = rot.sin_cos();
    [DVec2::new(c, s), DVec2::new(-s, c)]
}

/// Circle-vs-circle overlap probe. Returns `(normal, overlap)` where
/// `normal` points from B toward A (i.e. push A along `+normal`).
/// `None` when separated or coincident (< 0.001 m apart — degenerate
/// direction).
pub fn circle_vs_circle(
    pos_a: DVec2, rad_a: f64,
    pos_b: DVec2, rad_b: f64,
) -> Option<(DVec2, f64)> {
    let diff = pos_a - pos_b;
    let dist = diff.length();
    let min_dist = rad_a + rad_b;
    if dist >= min_dist || dist < 0.001 { return None; }
    Some((diff / dist, min_dist - dist))
}

/// OBB-vs-OBB overlap via SAT. Projects both boxes onto each of the 4
/// candidate axes (both boxes' local X and Y), takes the axis of MIN
/// penetration as the separating normal. Reduces to AABB when both
/// rotations are 0. Normal points from B toward A.
pub fn obb_vs_obb(
    pos_a: DVec2, half_a: [f64; 2], rot_a: f64,
    pos_b: DVec2, half_b: [f64; 2], rot_b: f64,
) -> Option<(DVec2, f64)> {
    let axes_a = obb_axes(rot_a);
    let axes_b = obb_axes(rot_b);
    let d = pos_a - pos_b;
    let axes = [axes_a[0], axes_a[1], axes_b[0], axes_b[1]];
    let mut min_overlap = f64::INFINITY;
    let mut best_axis = DVec2::new(1.0, 0.0);
    for axis in axes.iter() {
        let proj_a = half_a[0] * (axis.dot(axes_a[0])).abs()
                   + half_a[1] * (axis.dot(axes_a[1])).abs();
        let proj_b = half_b[0] * (axis.dot(axes_b[0])).abs()
                   + half_b[1] * (axis.dot(axes_b[1])).abs();
        let dist = d.dot(*axis).abs();
        let ov = (proj_a + proj_b) - dist;
        if ov <= 0.0 { return None; }
        if ov < min_overlap {
            min_overlap = ov;
            let sign = if d.dot(*axis) < 0.0 { -1.0 } else { 1.0 };
            best_axis = *axis * sign;
        }
    }
    Some((best_axis, min_overlap))
}

/// Circle-vs-OBB overlap via clamp-to-half-extents. Transforms the
/// circle centre into the box's local frame, clamps to `box_half`,
/// then either uses the offset from clamped-to-circle for a normal or
/// (when the circle centre sits INSIDE the box) picks the shallowest
/// local axis. Returned `normal` points from box toward circle by
/// default; pass `circle_is_a = false` to invert (matches the
/// caller-side "normal points from B toward A" convention regardless
/// of which side is the circle).
pub fn circle_vs_obb(
    circle_pos: DVec2, circle_r: f64,
    box_pos:    DVec2, box_half:  [f64; 2], box_rot: f64,
    circle_is_a: bool,
) -> Option<(DVec2, f64)> {
    let d = circle_pos - box_pos;
    let (s, c) = box_rot.sin_cos();
    let local = DVec2::new( c * d.x + s * d.y,
                           -s * d.x + c * d.y);
    let closest_local = DVec2::new(
        local.x.clamp(-box_half[0], box_half[0]),
        local.y.clamp(-box_half[1], box_half[1]),
    );
    let dv_local = local - closest_local;
    let dist = dv_local.length();
    if dist >= circle_r {
        return None;
    }
    if dist < 0.001 {
        // Circle centre sits inside box — push along the shallowest
        // local axis, then rotate back to world.
        let dx_out = box_half[0] - local.x.abs();
        let dy_out = box_half[1] - local.y.abs();
        let n_local = if dx_out < dy_out {
            DVec2::new(local.x.signum(), 0.0)
        } else {
            DVec2::new(0.0, local.y.signum())
        };
        let n_world = DVec2::new(c * n_local.x - s * n_local.y,
                                 s * n_local.x + c * n_local.y);
        let n_world = if circle_is_a { n_world } else { -n_world };
        Some((n_world, circle_r + dx_out.min(dy_out)))
    } else {
        let n_local = dv_local / dist;
        let n_world = DVec2::new(c * n_local.x - s * n_local.y,
                                 s * n_local.x + c * n_local.y);
        let n_world = if circle_is_a { n_world } else { -n_world };
        Some((n_world, circle_r - dist))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn circle_pair_overlap() {
        let (n, ov) = circle_vs_circle(
            DVec2::new(5.0, 0.0), 3.0,
            DVec2::new(0.0, 0.0), 3.0,
        ).expect("overlap");
        assert!((ov - 1.0).abs() < 1e-9);
        assert!((n - DVec2::new(1.0, 0.0)).length() < 1e-9);
    }

    #[test]
    fn circle_pair_separated() {
        assert!(circle_vs_circle(
            DVec2::new(10.0, 0.0), 1.0,
            DVec2::new(0.0, 0.0), 1.0,
        ).is_none());
    }

    #[test]
    fn axis_aligned_boxes_overlap() {
        let (_n, ov) = obb_vs_obb(
            DVec2::new(0.0, 0.0), [2.0, 2.0], 0.0,
            DVec2::new(3.0, 0.0), [2.0, 2.0], 0.0,
        ).expect("overlap");
        assert!((ov - 1.0).abs() < 1e-6);
    }

    #[test]
    fn separated_boxes_no_overlap() {
        assert!(obb_vs_obb(
            DVec2::new(0.0, 0.0), [1.0, 1.0], 0.0,
            DVec2::new(10.0, 0.0), [1.0, 1.0], 0.0,
        ).is_none());
    }

    #[test]
    fn segment_vs_circle_hits_and_misses() {
        assert!(segment_vs_circle(
            DVec2::new(-5.0, 0.0), DVec2::new(5.0, 0.0),
            DVec2::new(0.0, 0.0), 1.0,
        ));
        assert!(!segment_vs_circle(
            DVec2::new(-5.0, 10.0), DVec2::new(5.0, 10.0),
            DVec2::new(0.0, 0.0), 1.0,
        ));
        // Stationary segment inside the disc.
        assert!(segment_vs_circle(
            DVec2::new(0.5, 0.0), DVec2::new(0.5, 0.0),
            DVec2::new(0.0, 0.0), 1.0,
        ));
    }

    #[test]
    fn segment_vs_circle_t_reports_near_hit() {
        // Unit disc at origin, segment from x=-5 to x=+5 (length 10): the
        // entry crossing is at x = -1, i.e. 4/10 along.
        let t = segment_vs_circle_t(
            DVec2::new(-5.0, 0.0), DVec2::new(5.0, 0.0),
            DVec2::new(0.0, 0.0), 1.0,
        ).expect("segment crosses the disc");
        assert!((t - 0.4).abs() < 1e-9, "t = {t}");

        // Start already inside → first contact is t = 0, not a back-solve.
        let t = segment_vs_circle_t(
            DVec2::new(0.5, 0.0), DVec2::new(5.0, 0.0),
            DVec2::new(0.0, 0.0), 1.0,
        ).expect("start is inside the disc");
        assert_eq!(t, 0.0);

        // Disc entirely beyond the segment end → no hit.
        assert!(segment_vs_circle_t(
            DVec2::new(-5.0, 0.0), DVec2::new(-3.0, 0.0),
            DVec2::new(0.0, 0.0), 1.0,
        ).is_none());

        // Miss.
        assert!(segment_vs_circle_t(
            DVec2::new(-5.0, 10.0), DVec2::new(5.0, 10.0),
            DVec2::new(0.0, 0.0), 1.0,
        ).is_none());
    }

    #[test]
    fn circle_vs_box_edge_penetration() {
        let (n, ov) = circle_vs_obb(
            DVec2::new(3.0, 0.0), 2.0,
            DVec2::new(0.0, 0.0), [2.0, 2.0], 0.0,
            true,
        ).expect("overlap");
        assert!((ov - 1.0).abs() < 1e-6);
        assert!(n.x > 0.0, "normal should push circle further from box");
    }
}
