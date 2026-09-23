//! Turning a mouse position into "what did the player click on".
//!
//! # Why this is not trivial
//!
//! In 2D the mouse is at a pixel and UI rects are in pixels, so the
//! question is a coordinate comparison — `ui::UiRect::contains` already
//! answers it. In 3D a pixel does not name a position: it names a *ray*
//! from the eye through that pixel and off into the scene, and everything
//! along that ray is a candidate.
//!
//! So picking is two steps, and this module is both:
//!
//! 1. **Unproject** a screen pixel into a world-space
//!    [`crate::pick3d::Ray3D`], by
//!    pushing it back through the inverse of the camera's `view_proj`.
//! 2. **Cast** that ray against the scene, nearest hit first.
//!
//! # Coordinate frame
//!
//! Rays come out **camera-relative**, like everything else the 3D path
//! consumes — see `renderer::camera::Camera3D`, which builds its view
//! matrix with the eye at the origin. (Not a link: that type is behind
//! the `client` feature, and this module is not — a server validating a
//! claimed hit runs the same cast.) A ray's origin is therefore the zero
//! vector for a perspective camera, and collider positions fed to
//! [`crate::pick3d::cast`] must be in the same frame.

use glam::{DVec3, Mat4, Quat, Vec2};

use crate::collision::grid3d::SpatialGrid3D;
use crate::collision::narrow3d;

/// A ray: where it starts and which way it goes.
///
/// `dir` is kept normalised, so the `t` values [`cast`] returns are
/// distances in metres rather than multiples of an arbitrary length.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Ray3D {
    pub origin: DVec3,
    dir: DVec3,
}

impl Ray3D {
    /// Build a ray, normalising `dir`.
    ///
    /// A zero or non-finite direction has no ray to describe; `None`
    /// rather than a NaN direction that would silently make every
    /// intersection test fail in a way nobody could trace.
    pub fn new(origin: DVec3, dir: DVec3) -> Option<Self> {
        if !dir.is_finite() || !origin.is_finite() || dir.length_squared() <= 0.0 {
            return None;
        }
        Some(Self { origin, dir: dir.normalize() })
    }

    /// The unit direction.
    pub fn dir(&self) -> DVec3 {
        self.dir
    }

    /// The point `t` metres along the ray.
    pub fn at(&self, t: f64) -> DVec3 {
        self.origin + self.dir * t
    }
}

/// Unproject a screen pixel into a camera-relative ray.
///
/// `screen` is in pixels with the origin at the **top-left** and +Y down,
/// which is what winit reports and what `Camera2D::screen_to_world`
/// already assumes; `viewport` is the surface size in the same units.
///
/// # How
///
/// The pixel becomes two points in clip space — one on the near plane,
/// one on the far — which the inverse `view_proj` maps back to world
/// space. The ray is the line between them. Taking two points rather than
/// a point and a direction is what makes this work for an *orthographic*
/// camera too, where every ray is parallel and the eye position is not a
/// single point.
///
/// Returns `None` when the matrix cannot be inverted (a degenerate
/// camera) or the viewport has no area, rather than handing back a ray
/// full of NaN.
pub fn ray_from_screen(view_proj: Mat4, viewport: Vec2, screen: Vec2) -> Option<Ray3D> {
    if viewport.x <= 0.0 || viewport.y <= 0.0 || !view_proj.is_finite() {
        return None;
    }
    let inv = view_proj.inverse();
    if !inv.is_finite() {
        return None;
    }

    // Pixels -> normalised device coordinates. X maps -1..1 left to
    // right; Y is *flipped*, because clip space has +Y up and the screen
    // has +Y down.
    let ndc_x = (screen.x / viewport.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (screen.y / viewport.y) * 2.0;

    // wgpu's clip space runs z from 0 at the near plane to 1 at the far
    // one — the same convention `Camera3D` picks `perspective_rh` for.
    let near = inv.project_point3(glam::Vec3::new(ndc_x, ndc_y, 0.0));
    let far = inv.project_point3(glam::Vec3::new(ndc_x, ndc_y, 1.0));
    if !near.is_finite() || !far.is_finite() {
        return None;
    }

    Ray3D::new(near.as_dvec3(), (far - near).as_dvec3())
}

/// What a ray hit.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Hit {
    /// The collider's slot index in the grid it was queried from.
    pub index: u32,
    /// Distance along the ray, in metres.
    pub distance: f64,
    /// Where the ray entered, in the ray's own frame.
    pub point: DVec3,
}

/// What shape a slot has, for the purposes of a ray test.
///
/// Mirrors [`crate::components::Collider3D`]'s two cases. A caller that
/// only has spheres can build these with [`PickShape::sphere`] and never
/// think about it.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum PickShape {
    Sphere { centre: DVec3, radius: f64 },
    /// An oriented box. Tested exactly rather than through its bounding
    /// sphere, which over-reports near the corners — a click just off a
    /// crate's edge would otherwise register.
    Obb { centre: DVec3, half_extents: [f64; 3], rot: Quat },
}

impl PickShape {
    pub fn sphere(centre: DVec3, radius: f64) -> Self {
        Self::Sphere { centre, radius }
    }

    pub fn obb(centre: DVec3, half_extents: [f64; 3], rot: Quat) -> Self {
        Self::Obb { centre, half_extents, rot }
    }

    /// Build from a [`crate::components::Collider3D`] and a pose, using
    /// the same sphere-or-box discriminator the collider itself uses.
    pub fn from_collider(
        collider: &crate::components::Collider3D,
        centre: DVec3,
        rot: Quat,
    ) -> Self {
        if collider.is_box() {
            Self::Obb {
                centre,
                half_extents: [
                    collider.half_extents[0] as f64,
                    collider.half_extents[1] as f64,
                    collider.half_extents[2] as f64,
                ],
                rot,
            }
        } else {
            Self::Sphere { centre, radius: collider.radius as f64 }
        }
    }

    /// Entry parameter along `p0..p1`, or `None` for a miss.
    fn segment_t(&self, p0: DVec3, p1: DVec3) -> Option<f64> {
        match *self {
            PickShape::Sphere { centre, radius } => {
                narrow3d::segment_vs_sphere_t(p0, p1, centre, radius)
            }
            PickShape::Obb { centre, half_extents, rot } => {
                narrow3d::segment_vs_obb_t(p0, p1, centre, half_extents, rot)
            }
        }
    }
}

/// Cast a ray through a grid and return every hit, **nearest first**.
///
/// `max_distance` bounds the search: a ray is infinite, the grid is not,
/// and traversing to the edge of representable space would visit every
/// cell in between. Pick something like the camera's far plane.
///
/// `shape_of` maps a slot index to its collider. Returning `None` skips
/// that slot, which is how a caller filters — the player's own body, a
/// trigger volume, a corpse. Filtering here rather than after the fact
/// means a skipped collider cannot occlude one behind it.
///
/// Boxes are tested exactly, not through their bounding sphere. The grid
/// still *finds* candidates by bounding sphere — that is what it stores —
/// so a box near the ray is considered and then correctly rejected.
pub fn cast(
    grid: &SpatialGrid3D,
    ray: Ray3D,
    max_distance: f64,
    partition: u32,
    shape_of: impl Fn(u32) -> Option<PickShape>,
) -> Vec<Hit> {
    if !max_distance.is_finite() || max_distance <= 0.0 {
        return Vec::new();
    }

    let end = ray.at(max_distance);
    let candidates = grid.query_segment_in(ray.origin, end, partition);

    let mut hits: Vec<Hit> = Vec::new();
    for index in candidates {
        let Some(shape) = shape_of(index) else {
            continue;
        };
        // Both narrowphase tests return the *entry* parameter along the
        // segment, which is exactly what sorting by nearest needs.
        let Some(t) = shape.segment_t(ray.origin, end) else {
            continue;
        };
        // `t` is a fraction of the segment; scale to metres.
        let distance = t * max_distance;
        hits.push(Hit { index, distance, point: ray.at(distance) });
    }

    // Nearest first. `total_cmp` rather than `partial_cmp().unwrap()`
    // because a NaN would panic the sort, and one bad collider should not
    // take the frame with it.
    hits.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    hits
}

/// The nearest thing a ray hits, or `None`.
///
/// The common case — "what is under the cursor" — so it is worth not
/// making every caller index `[0]` and handle the empty case.
pub fn cast_nearest(
    grid: &SpatialGrid3D,
    ray: Ray3D,
    max_distance: f64,
    partition: u32,
    shape_of: impl Fn(u32) -> Option<PickShape>,
) -> Option<Hit> {
    cast(grid, ray, max_distance, partition, shape_of)
        .into_iter()
        .next()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A grid holding spheres, with the lookup `cast` needs.
    fn grid_of(spheres: &[(DVec3, f64)]) -> SpatialGrid3D {
        let mut g = SpatialGrid3D::new(4.0);
        for &(c, r) in spheres {
            g.insert(c, r);
        }
        g
    }

    fn lookup(spheres: Vec<(DVec3, f64)>) -> impl Fn(u32) -> Option<PickShape> {
        move |i| spheres.get(i as usize).map(|&(c, r)| PickShape::sphere(c, r))
    }

    #[test]
    fn a_ray_normalises_its_direction_so_t_is_in_metres() {
        let r = Ray3D::new(DVec3::ZERO, DVec3::new(0.0, 10.0, 0.0)).unwrap();
        assert!((r.dir().length() - 1.0).abs() < 1e-12);
        assert!((r.at(3.0) - DVec3::new(0.0, 3.0, 0.0)).length() < 1e-12);
    }

    /// A zero direction has no ray to describe. Normalising it yields
    /// NaN, which makes every later intersection test quietly fail.
    #[test]
    fn a_degenerate_ray_is_rejected_rather_than_producing_nan() {
        assert!(Ray3D::new(DVec3::ZERO, DVec3::ZERO).is_none());
        assert!(Ray3D::new(DVec3::ZERO, DVec3::new(f64::NAN, 0.0, 0.0)).is_none());
        assert!(Ray3D::new(DVec3::new(f64::INFINITY, 0.0, 0.0), DVec3::Y).is_none());
    }

    /// The basic job: a sphere in front of the ray is hit, at the right
    /// distance.
    #[test]
    fn a_ray_hits_a_sphere_in_front_of_it() {
        let spheres = vec![(DVec3::new(0.0, 10.0, 0.0), 1.0)];
        let g = grid_of(&spheres);
        let ray = Ray3D::new(DVec3::ZERO, DVec3::Y).unwrap();

        let hit = cast_nearest(&g, ray, 100.0, 0, lookup(spheres)).expect("should hit");
        assert_eq!(hit.index, 0);
        // Enters at y = 9, since the sphere spans 9..11.
        assert!(
            (hit.distance - 9.0).abs() < 0.01,
            "expected entry at 9 m, got {}",
            hit.distance,
        );
    }

    /// A sphere *behind* the ray must not be hit. Getting this wrong
    /// means clicking selects things behind the camera.
    #[test]
    fn a_ray_does_not_hit_what_is_behind_it() {
        let spheres = vec![(DVec3::new(0.0, -10.0, 0.0), 1.0)];
        let g = grid_of(&spheres);
        let ray = Ray3D::new(DVec3::ZERO, DVec3::Y).unwrap();
        assert!(cast_nearest(&g, ray, 100.0, 0, lookup(spheres)).is_none());
    }

    /// The whole point of sorting: the *nearest* thing is what the player
    /// clicked, not whichever the grid happened to yield first.
    #[test]
    fn hits_come_back_nearest_first() {
        let spheres = vec![
            (DVec3::new(0.0, 30.0, 0.0), 1.0),
            (DVec3::new(0.0, 10.0, 0.0), 1.0),
            (DVec3::new(0.0, 20.0, 0.0), 1.0),
        ];
        let g = grid_of(&spheres);
        let ray = Ray3D::new(DVec3::ZERO, DVec3::Y).unwrap();

        let hits = cast(&g, ray, 100.0, 0, lookup(spheres));
        assert_eq!(hits.len(), 3);
        let order: Vec<u32> = hits.iter().map(|h| h.index).collect();
        assert_eq!(order, vec![1, 2, 0], "should be sorted by distance");
        assert!(hits[0].distance < hits[1].distance);
        assert!(hits[1].distance < hits[2].distance);
    }

    /// A ray that misses hits nothing, however close it passes.
    #[test]
    fn a_ray_passing_beside_a_sphere_misses_it() {
        let spheres = vec![(DVec3::new(5.0, 10.0, 0.0), 1.0)];
        let g = grid_of(&spheres);
        let ray = Ray3D::new(DVec3::ZERO, DVec3::Y).unwrap();
        assert!(cast_nearest(&g, ray, 100.0, 0, lookup(spheres)).is_none());
    }

    /// `max_distance` bounds the search. A target beyond it is not hit,
    /// which is what stops a click selecting something on the far side of
    /// the level.
    #[test]
    fn a_target_beyond_the_max_distance_is_not_hit() {
        let spheres = vec![(DVec3::new(0.0, 50.0, 0.0), 1.0)];
        let g = grid_of(&spheres);
        let ray = Ray3D::new(DVec3::ZERO, DVec3::Y).unwrap();

        assert!(cast_nearest(&g, ray, 10.0, 0, lookup(spheres.clone())).is_none());
        assert!(cast_nearest(&g, ray, 100.0, 0, lookup(spheres)).is_some());
    }

    /// Returning `None` from `shape_of` skips a collider — how a caller
    /// excludes the player's own body. Crucially the thing *behind* it
    /// must then become visible, which is why filtering happens during
    /// the cast rather than after.
    #[test]
    fn a_filtered_collider_does_not_occlude_what_is_behind_it() {
        let spheres = vec![
            (DVec3::new(0.0, 5.0, 0.0), 1.0),  // nearest, filtered out
            (DVec3::new(0.0, 15.0, 0.0), 1.0), // what should be picked
        ];
        let g = grid_of(&spheres);
        let ray = Ray3D::new(DVec3::ZERO, DVec3::Y).unwrap();

        let hit = cast_nearest(&g, ray, 100.0, 0, |i| {
            if i == 0 { None } else { spheres.get(i as usize).map(|&(c, r)| PickShape::sphere(c, r)) }
        })
        .expect("the far sphere should be picked");
        assert_eq!(hit.index, 1);
    }

    /// A long ray must find a target many cells away. This is what the
    /// grid traversal is for — a sphere query would need a radius
    /// covering the whole path.
    #[test]
    fn a_ray_finds_a_target_many_cells_away() {
        // Cell size 4, target at 200 m: fifty cells along.
        let spheres = vec![(DVec3::new(0.0, 200.0, 0.0), 2.0)];
        let g = grid_of(&spheres);
        let ray = Ray3D::new(DVec3::ZERO, DVec3::Y).unwrap();

        let hit = cast_nearest(&g, ray, 500.0, 0, lookup(spheres))
            .expect("a distant target should still be found");
        assert!((hit.distance - 198.0).abs() < 0.1);
    }

    /// A diagonal ray steps on all three axes, which is the case the
    /// third DDA branch exists for. An axis-aligned ray would never
    /// exercise it.
    #[test]
    fn a_diagonal_ray_finds_a_target_off_every_axis() {
        let target = DVec3::new(30.0, 30.0, 30.0);
        let spheres = vec![(target, 2.0)];
        let g = grid_of(&spheres);
        let ray = Ray3D::new(DVec3::ZERO, target).unwrap();

        assert!(
            cast_nearest(&g, ray, 200.0, 0, lookup(spheres)).is_some(),
            "a diagonal ray should traverse cells on all three axes",
        );
    }

    /// Partitions do not see each other, matching the grid's own rule.
    #[test]
    fn a_ray_does_not_pick_across_partitions() {
        let mut g = SpatialGrid3D::new(4.0);
        g.insert_partitioned(DVec3::new(0.0, 10.0, 0.0), 1.0, 1);
        let ray = Ray3D::new(DVec3::ZERO, DVec3::Y).unwrap();

        let shape = |_| Some(PickShape::sphere(DVec3::new(0.0, 10.0, 0.0), 1.0));
        assert!(cast_nearest(&g, ray, 100.0, 0, shape).is_none());
        assert!(cast_nearest(&g, ray, 100.0, 1, shape).is_some());
    }

    /// A nonsense distance is answered with nothing rather than a
    /// traversal to the edge of representable space.
    #[test]
    fn a_nonsense_max_distance_returns_nothing() {
        let spheres = vec![(DVec3::new(0.0, 10.0, 0.0), 1.0)];
        let g = grid_of(&spheres);
        let ray = Ray3D::new(DVec3::ZERO, DVec3::Y).unwrap();
        for d in [0.0, -5.0, f64::NAN, f64::INFINITY] {
            assert!(cast(&g, ray, d, 0, lookup(spheres.clone())).is_empty(), "d = {d}");
        }
    }

    // ---- boxes ----------------------------------------------------------

    /// The reason box support exists. A ray passing just outside a box's
    /// face hits its bounding sphere but not the box, so picking by
    /// sphere registers a click on empty space beside a crate.
    #[test]
    fn a_box_is_tested_exactly_rather_than_by_its_bounding_sphere() {
        // A unit box: half-extent 1, bounding sphere radius sqrt(3).
        let centre = DVec3::new(0.0, 10.0, 0.0);
        let mut g = SpatialGrid3D::new(4.0);
        // Inserted by bounding sphere, which is what the grid stores —
        // so the candidate is found either way and the difference is
        // entirely in the narrowphase.
        g.insert(centre, 3.0f64.sqrt());

        // Offset 1.2 in x: outside the box, inside the sphere.
        let ray = Ray3D::new(DVec3::new(1.2, 0.0, 0.0), DVec3::Y).unwrap();

        let as_sphere = cast_nearest(&g, ray, 100.0, 0, |_| {
            Some(PickShape::sphere(centre, 3.0f64.sqrt()))
        });
        assert!(
            as_sphere.is_some(),
            "precondition: the bounding sphere is hit, which is the \
             over-reporting being removed",
        );

        let as_box = cast_nearest(&g, ray, 100.0, 0, |_| {
            Some(PickShape::obb(centre, [1.0, 1.0, 1.0], Quat::IDENTITY))
        });
        assert!(
            as_box.is_none(),
            "the box itself is missed at x = 1.2, so picking it as a box \
             must report nothing — got {as_box:?}",
        );
    }

    /// And a ray that does hit the box reports a sensible entry distance,
    /// not just a boolean.
    #[test]
    fn a_box_hit_reports_its_entry_distance() {
        let centre = DVec3::new(0.0, 10.0, 0.0);
        let mut g = SpatialGrid3D::new(4.0);
        g.insert(centre, 3.0f64.sqrt());
        let ray = Ray3D::new(DVec3::ZERO, DVec3::Y).unwrap();

        let hit = cast_nearest(&g, ray, 100.0, 0, |_| {
            Some(PickShape::obb(centre, [1.0, 1.0, 1.0], Quat::IDENTITY))
        })
        .expect("a ray through the middle should hit the box");
        // Enters the near face at y = 9.
        assert!(
            (hit.distance - 9.0).abs() < 0.01,
            "expected entry at 9 m, got {}",
            hit.distance,
        );
    }

    /// A rotated box must be picked in its rotated position. Picking that
    /// ignored orientation would select a crate by where it used to be.
    #[test]
    fn a_rotated_box_is_picked_in_its_rotated_orientation() {
        // A slab: long in x, thin in y.
        let centre = DVec3::new(0.0, 10.0, 0.0);
        let half = [4.0, 0.2, 4.0];
        let mut g = SpatialGrid3D::new(8.0);
        g.insert(centre, 6.0);

        // A ray straight up +Z through x = 3: inside the slab's long
        // axis while it lies along x.
        let ray = Ray3D::new(DVec3::new(3.0, 10.0, -20.0), DVec3::Z).unwrap();

        let flat = cast_nearest(&g, ray, 100.0, 0, |_| {
            Some(PickShape::obb(centre, half, Quat::IDENTITY))
        });
        assert!(flat.is_some(), "precondition: the unrotated slab is hit");

        // Turned a quarter turn about z, its long axis becomes y and it
        // is now thin where the ray passes.
        let turned = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let spun = cast_nearest(&g, ray, 100.0, 0, |_| {
            Some(PickShape::obb(centre, half, turned))
        });
        assert!(
            spun.is_none(),
            "the rotated slab no longer covers the ray, so it should be \
             missed — orientation is being ignored",
        );
    }

    /// `PickShape::from_collider` must pick the same sphere-or-box branch
    /// the collider itself reports, or a box would be picked as a sphere
    /// and the exact test silently skipped.
    #[test]
    fn from_collider_follows_the_colliders_own_shape() {
        use crate::components::Collider3D;

        let s = PickShape::from_collider(&Collider3D::sphere(2.0), DVec3::ZERO, Quat::IDENTITY);
        assert!(matches!(s, PickShape::Sphere { radius, .. } if (radius - 2.0).abs() < 1e-9));

        let b = PickShape::from_collider(
            &Collider3D::box3d(1.0, 2.0, 3.0),
            DVec3::ZERO,
            Quat::IDENTITY,
        );
        match b {
            PickShape::Obb { half_extents, .. } => {
                assert_eq!(half_extents, [1.0, 2.0, 3.0]);
            }
            other => panic!("a box collider should pick as a box, got {other:?}"),
        }
    }

    // ---- unprojection -------------------------------------------------

    /// The viewport the unprojection tests use.
    const VIEWPORT: Vec2 = Vec2::new(800.0, 600.0);

    /// A camera at the origin looking down +Y, with +Z up.
    ///
    /// Built from `glam` directly rather than through `Camera3D`, which
    /// lives behind the `client` feature — `pick3d` itself is headless
    /// (a server validating a claimed hit runs the same cast), so its
    /// tests must be too. The construction mirrors
    /// `Camera3D::build_uniform` exactly: `perspective_rh` for wgpu's
    /// 0..1 depth range, and a camera-relative `look_at_rh` with the eye
    /// at the origin.
    fn view_proj() -> Mat4 {
        let view = Mat4::look_at_rh(
            glam::Vec3::ZERO,
            glam::Vec3::new(0.0, 1.0, 0.0),
            glam::Vec3::Z,
        );
        let proj = Mat4::perspective_rh(
            std::f32::consts::FRAC_PI_3,
            VIEWPORT.x / VIEWPORT.y,
            0.1,
            10_000.0,
        );
        proj * view
    }

    /// A ray through the centre of the screen must go straight down the
    /// camera's gaze.
    #[test]
    fn the_centre_pixel_unprojects_along_the_view_axis() {
        let ray = ray_from_screen(view_proj(), VIEWPORT, Vec2::new(400.0, 300.0))
            .expect("centre pixel should unproject");

        assert!(
            (ray.dir() - DVec3::Y).length() < 1e-3,
            "expected a ray down +Y, got {:?}",
            ray.dir(),
        );
    }

    /// Screen Y runs *down* and clip Y runs *up*. Forgetting the flip is
    /// the classic unprojection bug: picking then works but is mirrored
    /// vertically, which feels like the cursor fighting the player.
    #[test]
    fn the_screen_y_axis_is_flipped_into_clip_space() {
        let vp = view_proj();

        // Up-screen is a small y pixel.
        let upper = ray_from_screen(vp, VIEWPORT, Vec2::new(400.0, 100.0)).unwrap();
        let lower = ray_from_screen(vp, VIEWPORT, Vec2::new(400.0, 500.0)).unwrap();

        // +Z is the engine's up, so the upper pixel must aim higher.
        assert!(
            upper.dir().z > lower.dir().z,
            "a pixel nearer the top of the screen should aim upward: \
             upper {:?} vs lower {:?} — the Y flip is missing or doubled",
            upper.dir(),
            lower.dir(),
        );
    }

    /// And the horizontal axis is *not* flipped.
    ///
    /// Looking down +Y with +Z up, a right-handed camera's right is
    /// `cross(forward, up)` = **+X**. (An earlier revision of this test
    /// asserted -X and failed; the code was right and the assertion was
    /// backwards, which is worth recording because the sign is easy to
    /// talk yourself into either way.)
    #[test]
    fn a_pixel_right_of_centre_aims_right() {
        let vp = view_proj();
        let right = ray_from_screen(vp, VIEWPORT, Vec2::new(700.0, 300.0)).unwrap();
        assert!(
            right.dir().x > 0.0,
            "a pixel right of centre should aim toward +X, got {:?}",
            right.dir(),
        );
        // And symmetrically on the other side, so this cannot pass by a
        // ray that simply always points +X.
        let left = ray_from_screen(vp, VIEWPORT, Vec2::new(100.0, 300.0)).unwrap();
        assert!(left.dir().x < 0.0, "a pixel left of centre should aim -X, got {:?}", left.dir());
    }

    /// End to end: click the centre of the screen, hit the thing in front
    /// of the camera. This is the whole feature in one assertion.
    #[test]
    fn clicking_the_centre_of_the_screen_picks_what_is_in_front() {
        let spheres = vec![(DVec3::new(0.0, 20.0, 0.0), 3.0)];
        let g = grid_of(&spheres);

        let ray = ray_from_screen(view_proj(), VIEWPORT, Vec2::new(400.0, 300.0))
            .expect("unproject");
        let hit = cast_nearest(&g, ray, 100.0, 0, lookup(spheres))
            .expect("the sphere in front of the camera should be picked");
        assert_eq!(hit.index, 0);
    }

    /// A zero-area viewport is what a minimised window reports, and
    /// dividing by it yields NaN.
    #[test]
    fn a_zero_area_viewport_is_rejected() {
        let vp = view_proj();
        assert!(ray_from_screen(vp, Vec2::new(0.0, 600.0), Vec2::ZERO).is_none());
        assert!(ray_from_screen(vp, Vec2::new(800.0, 0.0), Vec2::ZERO).is_none());
    }

    /// A singular matrix cannot be inverted, and `Mat4::inverse` returns
    /// one full of NaN rather than failing.
    #[test]
    fn a_degenerate_camera_matrix_is_rejected() {
        assert!(
            ray_from_screen(Mat4::ZERO, Vec2::new(800.0, 600.0), Vec2::new(400.0, 300.0))
                .is_none()
        );
    }
}
