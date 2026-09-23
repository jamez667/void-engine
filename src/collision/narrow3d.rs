//! Stateless 3D narrowphase: sphere and oriented-box overlap tests.
//!
//! The 3D counterpart to [`crate::collision::narrow`], and like it a pure-geometry
//! half with no state and no knowledge of the grid. Every function returns
//! `Option<(normal, overlap)>` with the **normal pointing from B toward
//! A**, matching the 2D convention so a caller porting between them does
//! not have to relearn the sign.
//!
//! # Why this is not the 2D code generalised
//!
//! `circle_vs_circle` genuinely is the same formula a dimension wider, and
//! is written that way. `obb_vs_obb` is not: 2D SAT tests **4** axes (each
//! box's two face normals), while 3D needs **15** — three face normals
//! each, plus the nine pairwise cross products of their edge directions.
//! Those nine are what catch an edge-on-edge overlap, where no face normal
//! separates the boxes but a diagonal does. Omitting them yields a test
//! that reports a collision correctly most of the time and misses exactly
//! the configurations a player notices.

use glam::{DVec3, Quat};

/// The three local axes of a box with the given orientation.
///
/// The 3D analogue of `narrow::obb_axes`, which builds two axes from a
/// scalar angle. A quaternion cannot be decomposed that way, so the axes
/// come from rotating the basis vectors.
#[inline]
pub fn obb_axes(rot: Quat) -> [DVec3; 3] {
    let r = rot.normalize();
    [
        r.mul_vec3(glam::Vec3::X).as_dvec3(),
        r.mul_vec3(glam::Vec3::Y).as_dvec3(),
        r.mul_vec3(glam::Vec3::Z).as_dvec3(),
    ]
}

/// Sphere-vs-sphere overlap. Normal points from B toward A.
///
/// The `dist < 0.001` guard matches the 2D version: two centres on top of
/// each other have no defined separating direction, and normalising the
/// zero vector would hand the caller a NaN normal to push along.
pub fn sphere_vs_sphere(
    pos_a: DVec3,
    rad_a: f64,
    pos_b: DVec3,
    rad_b: f64,
) -> Option<(DVec3, f64)> {
    let diff = pos_a - pos_b;
    let dist = diff.length();
    let min_dist = rad_a + rad_b;
    if dist >= min_dist || dist < 0.001 {
        return None;
    }
    Some((diff / dist, min_dist - dist))
}

/// Project a box's half-extents onto an axis.
#[inline]
fn project(half: [f64; 3], axes: &[DVec3; 3], axis: DVec3) -> f64 {
    half[0] * axis.dot(axes[0]).abs()
        + half[1] * axis.dot(axes[1]).abs()
        + half[2] * axis.dot(axes[2]).abs()
}

/// OBB-vs-OBB overlap via SAT over all 15 axes. Normal points from B
/// toward A, along the axis of least penetration.
///
/// # The 15 axes
///
/// Three face normals from A, three from B, and the nine cross products
/// `a_i × b_j`. The cross products are what detect an edge-on-edge
/// overlap: two boxes can be tilted so that neither one's faces separate
/// them, yet a plane spanned by one edge from each does.
///
/// # The degenerate cross product
///
/// When two edge directions are parallel their cross product is the zero
/// vector, which normalises to NaN. A NaN axis makes every comparison
/// false, so the overlap test silently passes it and the pair is reported
/// as colliding on a meaningless axis — or, worse, `min_overlap` is left
/// at a NaN that propagates into the caller's resolution. Parallel edges
/// are not exotic: two axis-aligned boxes have *six* such pairs. They are
/// skipped, which is correct because a parallel pair spans no plane the
/// face normals have not already covered.
pub fn obb_vs_obb(
    pos_a: DVec3,
    half_a: [f64; 3],
    rot_a: Quat,
    pos_b: DVec3,
    half_b: [f64; 3],
    rot_b: Quat,
) -> Option<(DVec3, f64)> {
    let axes_a = obb_axes(rot_a);
    let axes_b = obb_axes(rot_b);
    let d = pos_a - pos_b;

    let mut min_overlap = f64::INFINITY;
    let mut best_axis = DVec3::X;

    // Face normals: 3 from each box.
    let mut candidates: Vec<DVec3> = Vec::with_capacity(15);
    candidates.extend_from_slice(&axes_a);
    candidates.extend_from_slice(&axes_b);

    // Edge-pair cross products: 9 more.
    for a in &axes_a {
        for b in &axes_b {
            let c = a.cross(*b);
            // Squared length rather than length: cheaper, and the
            // threshold is on the *area* the two edges span, which is
            // what "parallel enough to be degenerate" means.
            if c.length_squared() > 1e-12 {
                candidates.push(c.normalize());
            }
        }
    }

    for axis in candidates {
        let proj_a = project(half_a, &axes_a, axis);
        let proj_b = project(half_b, &axes_b, axis);
        let dist = d.dot(axis).abs();
        let ov = (proj_a + proj_b) - dist;
        if ov <= 0.0 {
            // A separating axis: the boxes cannot overlap.
            return None;
        }
        if ov < min_overlap {
            min_overlap = ov;
            let sign = if d.dot(axis) < 0.0 { -1.0 } else { 1.0 };
            best_axis = axis * sign;
        }
    }

    Some((best_axis, min_overlap))
}

/// Sphere-vs-OBB overlap by clamping the centre into the box's local
/// frame.
///
/// `sphere_is_a` follows the 2D `circle_vs_obb`: the returned normal
/// points from box toward sphere by default, and is inverted when the
/// sphere is the B side, so the caller's "B toward A" convention holds
/// regardless of which argument is which shape.
pub fn sphere_vs_obb(
    sphere_pos: DVec3,
    sphere_r: f64,
    box_pos: DVec3,
    box_half: [f64; 3],
    box_rot: Quat,
    sphere_is_a: bool,
) -> Option<(DVec3, f64)> {
    let axes = obb_axes(box_rot);
    let d = sphere_pos - box_pos;

    // Into the box's local frame.
    let local = DVec3::new(d.dot(axes[0]), d.dot(axes[1]), d.dot(axes[2]));
    let clamped = DVec3::new(
        local.x.clamp(-box_half[0], box_half[0]),
        local.y.clamp(-box_half[1], box_half[1]),
        local.z.clamp(-box_half[2], box_half[2]),
    );

    let offset = local - clamped;
    let dist_sq = offset.length_squared();

    if dist_sq > sphere_r * sphere_r {
        return None;
    }

    let (local_normal, overlap) = if dist_sq > 1e-12 {
        // Centre outside the box: push along the line to the nearest
        // point on its surface.
        let dist = dist_sq.sqrt();
        (offset / dist, sphere_r - dist)
    } else {
        // Centre *inside* the box. There is no nearest-surface direction
        // to use, so pick the face it is shallowest against — the same
        // rule the 2D version applies, one axis wider.
        let dx = box_half[0] - local.x.abs();
        let dy = box_half[1] - local.y.abs();
        let dz = box_half[2] - local.z.abs();
        if dx <= dy && dx <= dz {
            let s = if local.x < 0.0 { -1.0 } else { 1.0 };
            (DVec3::new(s, 0.0, 0.0), dx + sphere_r)
        } else if dy <= dz {
            let s = if local.y < 0.0 { -1.0 } else { 1.0 };
            (DVec3::new(0.0, s, 0.0), dy + sphere_r)
        } else {
            let s = if local.z < 0.0 { -1.0 } else { 1.0 };
            (DVec3::new(0.0, 0.0, s), dz + sphere_r)
        }
    };

    // Back to world axes.
    let world_normal =
        axes[0] * local_normal.x + axes[1] * local_normal.y + axes[2] * local_normal.z;
    let normal = if sphere_is_a { world_normal } else { -world_normal };
    Some((normal, overlap))
}

/// Segment-vs-sphere, returning the entry parameter `t` along `p0..p1`.
///
/// The 3D counterpart to `narrow::segment_vs_circle_t`, and genuinely the
/// same quadratic one dimension wider — hit-scan weapons and ray picks
/// want it.
pub fn segment_vs_sphere_t(
    p0: DVec3,
    p1: DVec3,
    center: DVec3,
    radius: f64,
) -> Option<f64> {
    let d = p1 - p0;
    let f = p0 - center;
    let a = d.dot(d);
    if a < 1e-12 {
        // A degenerate segment is a point: inside or not, but there is no
        // direction to parameterise along.
        return if f.length_squared() <= radius * radius { Some(0.0) } else { None };
    }
    let b = 2.0 * f.dot(d);
    let c = f.dot(f) - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t1 = (-b - sq) / (2.0 * a);
    let t2 = (-b + sq) / (2.0 * a);
    // Nearest hit within the segment. t1 <= t2 always.
    if (0.0..=1.0).contains(&t1) {
        Some(t1)
    } else if (0.0..=1.0).contains(&t2) {
        Some(t2)
    } else {
        None
    }
}

/// Whether a segment touches a sphere at all.
pub fn segment_vs_sphere(p0: DVec3, p1: DVec3, center: DVec3, radius: f64) -> bool {
    segment_vs_sphere_t(p0, p1, center, radius).is_some()
}

/// The deepest point of box A into box B, for a contact along `normal`.
///
/// # Why a contact needs this, and what goes wrong without it
///
/// A solver applies its impulse *at a point*, and the lever arm from
/// each body's centre to that point is what decides how much of the
/// impulse becomes spin rather than stopping the approach — see
/// `effective_mass` in [`crate::physics3d::solver`].
///
/// An approximate point does not degrade the result gracefully; it
/// breaks it. Using the midpoint between centres puts the contact
/// metres away from the surfaces when one body is large — a crate on a
/// wide floor is the ordinary case — which inflates the lever arm,
/// inflates the effective mass, and shrinks the impulse to a fraction of
/// what is needed. The crate then sinks a little further every tick,
/// penetration grows monotonically, and it falls through. Measured on
/// `examples/crates3d`: penetration climbed 0.05 -> 0.88 over three
/// seconds and the crates passed through the floor.
///
/// # What this returns
///
/// The support point of A in the direction of `-normal`: the corner (or
/// face centre, when the box is square-on) that reaches furthest into B.
///
/// Returned as-is, with no offset along the normal. For a resting box
/// that point already *is* the contact surface — a crate at z = 0.5 with
/// half-extent 0.5 supports at z = 0, which is exactly the floor's top.
/// An earlier revision nudged it by half the penetration and had the
/// sign backwards, moving the point away from the surface and leaving
/// the crates sinking exactly as before; the nudge bought nothing even
/// with the sign right.
///
/// This is a *single* point, not a manifold. One point per pair is enough
/// to stop bodies interpenetrating and is what makes a stack stand up;
/// four points per face pair is what makes it stand up *steadily*, and is
/// the obvious next step.
pub fn obb_contact_point(
    pos_a: DVec3,
    half_a: [f64; 3],
    rot_a: Quat,
    normal: DVec3,
    penetration: f64,
) -> DVec3 {
    let axes = obb_axes(rot_a);
    // Walk from A's centre to the corner furthest along -normal. On each
    // local axis, step to whichever face the normal points away from.
    let mut p = pos_a;
    for i in 0..3 {
        let d = axes[i].dot(normal);
        // A near-zero dot means the normal lies in this face's plane, so
        // neither direction is "deeper" — the support is the face centre
        // on that axis, which is what contributing nothing gives.
        if d > 0.0 {
            p -= axes[i] * half_a[i].abs();
        } else if d < 0.0 {
            p += axes[i] * half_a[i].abs();
        }
    }
    let _ = penetration;
    p
}

/// Segment-vs-oriented-box, returning the entry parameter `t` along
/// `p0..p1`.
///
/// The counterpart to [`segment_vs_sphere_t`], and what picking needs to
/// stop treating boxes as their bounding spheres — which over-reports
/// near the corners by up to the difference between a cube and the sphere
/// around it, so a click just off a crate's edge still registers.
///
/// # The slab method
///
/// Transform the segment into the box's local frame, where the box is
/// axis-aligned, then treat it as three pairs of parallel planes — the
/// "slabs". For each axis, work out the interval of `t` during which the
/// segment is between that axis's two planes. The segment is inside the
/// box exactly while it is inside *all three* intervals at once, so the
/// answer is the intersection: the largest entry and the smallest exit.
/// If the largest entry is past the smallest exit, the segment misses.
///
/// # The axis-parallel case
///
/// A segment with no motion along an axis never crosses that axis's
/// planes, so `1/d` is infinite and the usual arithmetic gives NaN. Such
/// a segment is either inside that slab for its whole length or outside
/// for all of it, which is a containment test rather than an
/// intersection — and getting it wrong means a ray fired exactly along an
/// axis, which is the common case for a top-down or side-on camera,
/// silently misses everything.
pub fn segment_vs_obb_t(
    p0: DVec3,
    p1: DVec3,
    box_pos: DVec3,
    box_half: [f64; 3],
    box_rot: Quat,
) -> Option<f64> {
    let axes = obb_axes(box_rot);
    let d = p1 - p0;

    // Into the box's local frame, where it is axis-aligned.
    let origin = p0 - box_pos;
    let local_o = DVec3::new(origin.dot(axes[0]), origin.dot(axes[1]), origin.dot(axes[2]));
    let local_d = DVec3::new(d.dot(axes[0]), d.dot(axes[1]), d.dot(axes[2]));

    let half = DVec3::new(box_half[0].abs(), box_half[1].abs(), box_half[2].abs());

    // The interval of `t` for which the segment is inside every slab.
    let mut t_enter = 0.0f64;
    let mut t_exit = 1.0f64;

    for i in 0..3 {
        let (o, dir, h) = (local_o[i], local_d[i], half[i]);
        if dir.abs() < 1e-12 {
            // Parallel to this slab: inside for the whole segment, or
            // outside for all of it. No interval to intersect.
            if o < -h || o > h {
                return None;
            }
            continue;
        }
        let inv = 1.0 / dir;
        let mut t0 = (-h - o) * inv;
        let mut t1 = (h - o) * inv;
        if t0 > t1 {
            std::mem::swap(&mut t0, &mut t1);
        }
        t_enter = t_enter.max(t0);
        t_exit = t_exit.min(t1);
        if t_enter > t_exit {
            return None;
        }
    }

    // `t_enter` is clamped at 0, so a segment starting *inside* the box
    // reports an entry of 0 rather than a negative value behind its
    // origin — which is what a caller wants when the camera is already
    // within something.
    Some(t_enter)
}

/// Whether a segment touches an oriented box at all.
pub fn segment_vs_obb(
    p0: DVec3,
    p1: DVec3,
    box_pos: DVec3,
    box_half: [f64; 3],
    box_rot: Quat,
) -> bool {
    segment_vs_obb_t(p0, p1, box_pos, box_half, box_rot).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q() -> Quat {
        Quat::IDENTITY
    }

    #[test]
    fn spheres_overlap_and_the_normal_points_from_b_to_a() {
        let hit = sphere_vs_sphere(DVec3::new(1.0, 0.0, 0.0), 1.0, DVec3::ZERO, 1.0);
        let (n, ov) = hit.expect("unit spheres 1 apart overlap");
        assert!((n - DVec3::X).length() < 1e-9, "normal should point +X, got {n:?}");
        assert!((ov - 1.0).abs() < 1e-9, "overlap should be 1.0, got {ov}");
    }

    #[test]
    fn distant_spheres_do_not_overlap() {
        assert!(sphere_vs_sphere(DVec3::new(5.0, 0.0, 0.0), 1.0, DVec3::ZERO, 1.0).is_none());
    }

    /// Coincident centres have no separating direction; returning a
    /// normalised zero vector would hand the caller NaN to push along.
    #[test]
    fn coincident_spheres_report_no_hit_rather_than_a_nan_normal() {
        assert!(sphere_vs_sphere(DVec3::ZERO, 1.0, DVec3::ZERO, 1.0).is_none());
    }

    #[test]
    fn axis_aligned_boxes_overlap_on_the_shallowest_axis() {
        // Overlapping by 0.5 in x, by 2.0 in y and z.
        let hit = obb_vs_obb(
            DVec3::new(1.5, 0.0, 0.0),
            [1.0; 3],
            q(),
            DVec3::ZERO,
            [1.0; 3],
            q(),
        );
        let (n, ov) = hit.expect("boxes 1.5 apart with half-extent 1 overlap");
        assert!((ov - 0.5).abs() < 1e-9, "shallowest overlap is 0.5, got {ov}");
        assert!((n - DVec3::X).length() < 1e-9, "normal should be +X, got {n:?}");
    }

    #[test]
    fn separated_boxes_report_no_hit() {
        assert!(obb_vs_obb(
            DVec3::new(3.0, 0.0, 0.0),
            [1.0; 3],
            q(),
            DVec3::ZERO,
            [1.0; 3],
            q()
        )
        .is_none());
    }

    /// Two axis-aligned boxes have six parallel edge pairs, whose cross
    /// products are zero. Normalising those gives NaN, and a NaN axis
    /// makes `ov <= 0.0` false — so the box would be reported as
    /// overlapping on a meaningless axis, with a NaN normal reaching the
    /// caller. This is the most likely way to get 3D SAT wrong.
    #[test]
    fn parallel_edge_axes_do_not_produce_a_nan_normal() {
        let (n, ov) = obb_vs_obb(
            DVec3::new(1.5, 0.0, 0.0),
            [1.0; 3],
            q(),
            DVec3::ZERO,
            [1.0; 3],
            q(),
        )
        .expect("overlapping");
        assert!(n.is_finite(), "normal was {n:?}");
        assert!(ov.is_finite(), "overlap was {ov}");
    }

    /// The case the nine cross-product axes exist for.
    ///
    /// These two boxes are separated by a plane spanned by one edge from
    /// each — and by **no face normal of either**. A 6-axis test (face
    /// normals only) therefore finds no separating axis and wrongly
    /// reports a collision; the full 15-axis test correctly reports none.
    ///
    /// The configuration is not hand-picked by intuition: an earlier
    /// revision of this test used two boxes crossed at 45°, which *looks*
    /// like the edge-on-edge case but is separated by a face normal too,
    /// so it passed with the cross-product axes deleted and proved
    /// nothing. These values come from a search for a configuration the
    /// 6-axis test actually gets wrong, and were verified load-bearing by
    /// deleting the cross products and watching this fail.
    #[test]
    fn an_edge_on_edge_gap_is_found_only_by_the_cross_product_axes() {
        let a_rot = Quat::from_axis_angle(
            glam::Vec3::new(0.27446, -1.526525, 1.6507).normalize(),
            1.97114,
        );
        let b_rot = Quat::from_axis_angle(
            glam::Vec3::new(0.154336, -0.38714, 2.029072).normalize(),
            2.977315,
        );

        let hit = obb_vs_obb(
            DVec3::new(-0.090772, -2.901357, -0.810456),
            [0.950234, 0.715685, 1.469132],
            a_rot,
            DVec3::ZERO,
            [0.260557, 1.316009, 0.576492],
            b_rot,
        );
        assert!(
            hit.is_none(),
            "these boxes are separated by an edge-edge plane and by no \
             face normal; reporting a hit means the 9 cross-product axes \
             are missing or wrong",
        );
    }

    #[test]
    fn a_sphere_outside_a_box_is_pushed_along_the_nearest_face() {
        let hit = sphere_vs_obb(
            DVec3::new(1.5, 0.0, 0.0),
            1.0,
            DVec3::ZERO,
            [1.0; 3],
            q(),
            true,
        );
        let (n, ov) = hit.expect("sphere at 1.5 with r=1 touches a box of half-extent 1");
        assert!((n - DVec3::X).length() < 1e-9, "normal should be +X, got {n:?}");
        assert!((ov - 0.5).abs() < 1e-9, "overlap should be 0.5, got {ov}");
    }

    /// A sphere whose centre is inside the box has no nearest-surface
    /// direction, so the shallowest-face rule has to pick one. Getting
    /// this wrong pushes the sphere out through the *long* axis, which
    /// looks like teleporting.
    #[test]
    fn a_sphere_inside_a_box_leaves_by_the_shallowest_face() {
        // Box is thin in z, so that is the way out.
        let hit = sphere_vs_obb(
            DVec3::new(0.1, 0.1, 0.1),
            0.2,
            DVec3::ZERO,
            [5.0, 5.0, 0.5],
            q(),
            true,
        );
        let (n, _) = hit.expect("a centre inside the box always overlaps");
        assert!(
            (n - DVec3::Z).length() < 1e-9,
            "should exit through the thin +Z face, got {n:?}",
        );
    }

    #[test]
    fn sphere_is_a_flips_the_normal() {
        let p = DVec3::new(1.5, 0.0, 0.0);
        let (a, _) = sphere_vs_obb(p, 1.0, DVec3::ZERO, [1.0; 3], q(), true).unwrap();
        let (b, _) = sphere_vs_obb(p, 1.0, DVec3::ZERO, [1.0; 3], q(), false).unwrap();
        assert!((a + b).length() < 1e-9, "the two should be opposites");
    }

    #[test]
    fn a_segment_through_a_sphere_reports_the_entry_point() {
        let t = segment_vs_sphere_t(
            DVec3::new(-5.0, 0.0, 0.0),
            DVec3::new(5.0, 0.0, 0.0),
            DVec3::ZERO,
            1.0,
        )
        .expect("a segment through the origin hits a unit sphere there");
        // Entry at x = -1, i.e. 4/10 of the way along.
        assert!((t - 0.4).abs() < 1e-9, "entry t should be 0.4, got {t}");
    }

    #[test]
    fn a_segment_missing_the_sphere_reports_nothing() {
        assert!(!segment_vs_sphere(
            DVec3::new(-5.0, 3.0, 0.0),
            DVec3::new(5.0, 3.0, 0.0),
            DVec3::ZERO,
            1.0
        ));
    }

    // ---- segment vs box -------------------------------------------------

    /// A unit box at the origin, axis-aligned.
    fn unit_box() -> ([f64; 3], Quat) {
        ([1.0, 1.0, 1.0], Quat::IDENTITY)
    }

    #[test]
    fn a_segment_through_a_box_reports_the_entry_point() {
        let (half, rot) = unit_box();
        let t = segment_vs_obb_t(
            DVec3::new(-5.0, 0.0, 0.0),
            DVec3::new(5.0, 0.0, 0.0),
            DVec3::ZERO,
            half,
            rot,
        )
        .expect("a segment through the origin hits a unit box");
        // Enters at x = -1, i.e. 4/10 along.
        assert!((t - 0.4).abs() < 1e-9, "entry t should be 0.4, got {t}");
    }

    #[test]
    fn a_segment_missing_the_box_reports_nothing() {
        let (half, rot) = unit_box();
        assert!(!segment_vs_obb(
            DVec3::new(-5.0, 3.0, 0.0),
            DVec3::new(5.0, 3.0, 0.0),
            DVec3::ZERO,
            half,
            rot,
        ));
    }

    /// The corner case this whole function exists for. A segment passing
    /// diagonally past a box's corner misses the box but *hits its
    /// bounding sphere*, which has radius sqrt(3) here. Picking with the
    /// sphere fallback therefore registers a click on empty space just
    /// off the crate's edge.
    #[test]
    fn a_segment_past_a_corner_misses_the_box_but_hits_its_bounding_sphere() {
        let (half, rot) = unit_box();
        // A segment along y at x = 1.2. Its closest approach to the
        // origin is 1.2 — outside the box's half-extent of 1, inside the
        // bounding sphere's radius of sqrt(3) = 1.73. That gap is the
        // over-reporting, and it is widest at the corners.
        let a = DVec3::new(1.2, -5.0, 0.0);
        let b = DVec3::new(1.2, 5.0, 0.0);

        assert!(
            !segment_vs_obb(a, b, DVec3::ZERO, half, rot),
            "x = 1.2 is outside a half-extent of 1, so the box is missed",
        );
        assert!(
            segment_vs_sphere(a, b, DVec3::ZERO, 3.0f64.sqrt()),
            "precondition: the bounding sphere *is* hit, which is exactly \
             the over-reporting the box test removes",
        );
    }

    /// A segment fired exactly along an axis has zero motion on the other
    /// two, where `1/d` is infinite. Getting that wrong makes a top-down
    /// or side-on camera — the common case — silently pick nothing.
    #[test]
    fn an_axis_parallel_segment_is_handled_without_nan() {
        let (half, rot) = unit_box();
        // Straight down +Z through the middle: no motion on x or y.
        let t = segment_vs_obb_t(
            DVec3::new(0.0, 0.0, -5.0),
            DVec3::new(0.0, 0.0, 5.0),
            DVec3::ZERO,
            half,
            rot,
        )
        .expect("an axis-parallel segment through the box should hit");
        assert!(t.is_finite(), "got {t}");
        assert!((t - 0.4).abs() < 1e-9, "entry t should be 0.4, got {t}");

        // And one that is parallel but outside must miss rather than
        // sneak through on a NaN comparison.
        assert!(!segment_vs_obb(
            DVec3::new(9.0, 0.0, -5.0),
            DVec3::new(9.0, 0.0, 5.0),
            DVec3::ZERO,
            half,
            rot,
        ));
    }

    /// Rotation must be respected: a segment that misses an axis-aligned
    /// box can hit the same box turned 45 degrees, and vice versa. A slab
    /// test that forgot to transform into the box's frame would give the
    /// same answer either way.
    #[test]
    fn a_rotated_box_is_hit_where_an_axis_aligned_one_is_not() {
        // A thin slab, long in x and narrow in y.
        let half = [3.0, 0.2, 3.0];
        // A segment along y, offset well out in x: misses the thin slab
        // when it lies along x...
        let a = DVec3::new(2.5, -5.0, 0.0);
        let b = DVec3::new(2.5, 5.0, 0.0);
        assert!(
            segment_vs_obb(a, b, DVec3::ZERO, half, Quat::IDENTITY),
            "precondition: the segment crosses the unrotated slab",
        );

        // ...and misses once the slab is turned a quarter turn about z,
        // which swaps its long and short axes.
        let turned = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        assert!(
            !segment_vs_obb(a, b, DVec3::ZERO, half, turned),
            "the rotated slab is narrow where the segment passes, so it \
             should be missed — the rotation is not being applied",
        );
    }

    /// A segment starting inside the box reports an entry of 0 rather
    /// than a negative value behind its own origin. That is what a caller
    /// wants when the camera is already within something.
    #[test]
    fn a_segment_starting_inside_the_box_enters_at_zero() {
        let (half, rot) = unit_box();
        let t = segment_vs_obb_t(DVec3::ZERO, DVec3::new(5.0, 0.0, 0.0), DVec3::ZERO, half, rot)
            .expect("a segment from inside the box hits it");
        assert_eq!(t, 0.0);
    }

    /// A box entirely beyond the segment's far end is not hit. The slab
    /// test works in the segment's own 0..1 parameter, so this is the
    /// check that it is a *segment* and not an infinite ray.
    #[test]
    fn a_box_beyond_the_end_of_the_segment_is_not_hit() {
        let (half, rot) = unit_box();
        assert!(!segment_vs_obb(
            DVec3::ZERO,
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(50.0, 0.0, 0.0),
            half,
            rot,
        ));
    }

    /// And one behind the start is not hit either — the failure that
    /// would let a click select something behind the camera.
    #[test]
    fn a_box_behind_the_segment_is_not_hit() {
        let (half, rot) = unit_box();
        assert!(!segment_vs_obb(
            DVec3::ZERO,
            DVec3::new(0.0, 10.0, 0.0),
            DVec3::new(0.0, -20.0, 0.0),
            half,
            rot,
        ));
    }

    /// A zero-length segment is a point. Dividing by its squared length
    /// would be a divide by zero, so it is answered as containment.
    #[test]
    fn a_degenerate_segment_is_treated_as_a_point() {
        assert!(segment_vs_sphere(DVec3::ZERO, DVec3::ZERO, DVec3::ZERO, 1.0));
        assert!(!segment_vs_sphere(
            DVec3::new(9.0, 0.0, 0.0),
            DVec3::new(9.0, 0.0, 0.0),
            DVec3::ZERO,
            1.0
        ));
    }
}
