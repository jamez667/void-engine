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
