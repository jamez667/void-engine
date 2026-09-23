//! Generic 2D physics integrator. Semi-implicit Euler + per-tick drag.
//!
//! Drag constants are exposed as parameters because the game-friendly
//! defaults (0.999 linear, 0.995 angular) are the void_sim baseline;
//! other consumers may want cleaner conservation or heavier damping.
//! The zero-arg convenience wrappers in `void_sim::physics` supply the
//! sim's canonical values.

use crate::{EntityId, World};
use crate::components::{Transform2D, Velocity};

/// Integrate one entity for `dt`. `lin_drag` and `ang_drag` are per-tick
/// multipliers on `Velocity.linear` / `Velocity.angular` (1.0 = no drag,
/// 0.0 = instant stop).
pub fn integrate_entity(world: &mut World, id: EntityId, dt: f32, lin_drag: f64, ang_drag: f32) {
    let vel = match world.get::<Velocity>(id) { Some(v) => v.clone(), None => return };
    if let Some(t) = world.get_mut::<Transform2D>(id) {
        t.pos += vel.linear * dt as f64;
        t.rot += vel.angular * dt;
    }
    if let Some(v) = world.get_mut::<Velocity>(id) {
        v.linear *= lin_drag;
        v.angular *= ang_drag;
    }
}

/// Integrate every entity that has a `Velocity` for `dt`. Typed iteration
/// avoids the O(N_total) miss-heavy full-world walk — only real movers
/// (ships, projectiles, particles) get touched.
///
/// The collect-into-Vec dance is intentional: we drop the immutable
/// `iter::<Velocity>` borrow before taking `get_mut` twice per mover.
pub fn integrate(world: &mut World, dt: f32, lin_drag: f64, ang_drag: f32) {
    let movers: Vec<(EntityId, Velocity)> = world.iter::<Velocity>()
        .map(|(id, v)| (id, v.clone()))
        .collect();
    for (id, vel) in movers {
        if let Some(t) = world.get_mut::<Transform2D>(id) {
            t.pos += vel.linear * dt as f64;
            t.rot += vel.angular * dt;
        }
        if let Some(v) = world.get_mut::<Velocity>(id) {
            v.linear *= lin_drag;
            v.angular *= ang_drag;
        }
    }
}

// ── 3D ──────────────────────────────────────────────────────────────────
//
// Beside the 2D integrator, not replacing it. The linear half is the same
// arithmetic one axis wider; the angular half is genuinely different, and
// that difference is why this is a separate function rather than a
// generic one.

use crate::components::{Transform3D, Velocity3D};

/// Advance a pose by an axis-angle angular velocity over `dt`.
///
/// `Velocity3D.angular` is an axis-angle vector: direction = axis,
/// magnitude = radians per second. Converting it to a quaternion and
/// multiplying is exact for a constant angular velocity, where the 2D
/// path's `rot += angular * dt` has no 3D equivalent — adding angles
/// componentwise is not composition of rotations.
///
/// Order matters: `delta * rot` applies the spin in **world** axes, which
/// is what a torque-free tumble does. `rot * delta` would spin about the
/// body's own axes, a different and usually wrong motion.
///
/// A zero angular velocity short-circuits, because normalising a
/// zero-length axis yields NaN and would poison the pose permanently.
#[inline]
pub fn integrate_rotation(rot: glam::Quat, angular: glam::DVec3, dt: f32) -> glam::Quat {
    let theta = angular.length();
    if theta <= 0.0 || !theta.is_finite() {
        return rot;
    }
    let axis = (angular / theta).as_vec3();
    let delta = glam::Quat::from_axis_angle(axis, (theta * dt as f64) as f32);
    // Renormalise: repeated multiplication accumulates float error, and a
    // drifting non-unit quaternion scales geometry as well as rotating it.
    (delta * rot).normalize()
}

/// Integrate one entity's [`Transform3D`] for `dt`.
///
/// `ang_drag` is `f32` for symmetry with the 2D signature even though the
/// velocity it scales is `f64`, so a caller porting from 2D passes the
/// same constants.
pub fn integrate_entity_3d(
    world: &mut World,
    id: EntityId,
    dt: f32,
    lin_drag: f64,
    ang_drag: f32,
) {
    let vel = match world.get::<Velocity3D>(id) {
        Some(v) => v.clone(),
        None => return,
    };
    if let Some(t) = world.get_mut::<Transform3D>(id) {
        t.pos += vel.linear * dt as f64;
        t.rot = integrate_rotation(t.rot, vel.angular, dt);
    }
    if let Some(v) = world.get_mut::<Velocity3D>(id) {
        v.linear *= lin_drag;
        v.angular *= ang_drag as f64;
    }
}

/// Integrate every entity with a [`Velocity3D`]. The 3D counterpart to
/// [`integrate`], including the same collect-first borrow dance.
pub fn integrate_3d(world: &mut World, dt: f32, lin_drag: f64, ang_drag: f32) {
    let movers: Vec<(EntityId, Velocity3D)> = world
        .iter::<Velocity3D>()
        .map(|(id, v)| (id, v.clone()))
        .collect();
    for (id, vel) in movers {
        if let Some(t) = world.get_mut::<Transform3D>(id) {
            t.pos += vel.linear * dt as f64;
            t.rot = integrate_rotation(t.rot, vel.angular, dt);
        }
        if let Some(v) = world.get_mut::<Velocity3D>(id) {
            v.linear *= lin_drag;
            v.angular *= ang_drag as f64;
        }
    }
}

#[cfg(test)]
mod tests_3d {
    use super::*;
    use glam::{DVec3, Quat, Vec3};

    /// A quarter turn about Z, integrated in one step and in a hundred,
    /// must land in the same place. This is what makes the axis-angle
    /// representation worth using: naive angle addition does not compose.
    #[test]
    fn rotation_integration_composes_over_substeps() {
        let w = DVec3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2);

        let one = integrate_rotation(Quat::IDENTITY, w, 1.0);
        let mut many = Quat::IDENTITY;
        for _ in 0..100 {
            many = integrate_rotation(many, w, 0.01);
        }

        let a = one * Vec3::X;
        let b = many * Vec3::X;
        assert!(
            (a - b).length() < 1e-3,
            "one step gave {a:?}, a hundred gave {b:?} — the integration \
             does not compose, so the result depends on the tick rate",
        );
        // And it really is a quarter turn: +X goes to +Y.
        assert!(
            (a - Vec3::Y).length() < 1e-3,
            "a quarter turn about Z should carry +X to +Y, got {a:?}",
        );
    }

    /// Zero angular velocity is the common case — most entities never
    /// spin — and normalising a zero-length axis yields NaN, which would
    /// poison the pose for the rest of the session.
    #[test]
    fn zero_angular_velocity_leaves_the_pose_untouched_and_finite() {
        let start = Quat::from_rotation_x(0.7);
        let out = integrate_rotation(start, DVec3::ZERO, 1.0 / 60.0);
        assert_eq!(out, start);
        assert!(out.is_finite());
    }

    /// Quaternions drift from unit length under repeated multiplication,
    /// and a non-unit quaternion scales geometry as well as rotating it —
    /// so a long-lived spinning object would slowly inflate or shrink.
    #[test]
    fn long_integration_keeps_the_quaternion_normalised() {
        let w = DVec3::new(0.3, -0.7, 1.1);
        let mut q = Quat::IDENTITY;
        for _ in 0..10_000 {
            q = integrate_rotation(q, w, 1.0 / 60.0);
        }
        assert!(
            (q.length() - 1.0).abs() < 1e-3,
            "quaternion drifted to length {} after 10k steps — geometry \
             rotated by it would be scaled, not just turned",
            q.length(),
        );
    }

    /// The linear half must behave exactly like the 2D one, one axis
    /// wider: position advances by velocity * dt, and drag applies after
    /// the step.
    #[test]
    fn linear_integration_matches_the_2d_semantics() {
        let mut world = World::new();
        let id = world.spawn();
        world.insert(id, Transform3D::at(DVec3::ZERO));
        world.insert(
            id,
            Velocity3D { linear: DVec3::new(1.0, 2.0, 3.0), angular: DVec3::ZERO },
        );

        integrate_3d(&mut world, 0.5, 0.5, 1.0);

        let t = world.get::<Transform3D>(id).unwrap();
        assert!((t.pos - DVec3::new(0.5, 1.0, 1.5)).length() < 1e-9);
        let v = world.get::<Velocity3D>(id).unwrap();
        assert!(
            (v.linear - DVec3::new(0.5, 1.0, 1.5)).length() < 1e-9,
            "drag should apply after the step, as it does in 2D",
        );
    }

    /// An entity with a 3D velocity but a 2D transform must be left alone
    /// rather than panicking — the two component sets coexist, and a game
    /// mixing them is doing something odd but not illegal.
    #[test]
    fn a_mover_without_a_3d_transform_is_skipped() {
        let mut world = World::new();
        let id = world.spawn();
        world.insert(id, Velocity3D { linear: DVec3::ONE, angular: DVec3::ZERO });
        integrate_3d(&mut world, 1.0, 1.0, 1.0);
        assert!(world.get::<Transform3D>(id).is_none());
    }

    /// World-axis spin, not body-axis. `delta * rot` and `rot * delta`
    /// differ as soon as the body is already rotated, and picking the
    /// wrong one gives a tumble that looks plausible but is wrong.
    #[test]
    fn spin_is_applied_in_world_axes() {
        // Body already turned 90 degrees about Z, then spun about world X.
        let start = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let w = DVec3::new(std::f64::consts::FRAC_PI_2, 0.0, 0.0);
        let got = integrate_rotation(start, w, 1.0);

        let want = Quat::from_rotation_x(std::f32::consts::FRAC_PI_2) * start;
        assert!(
            (got * Vec3::X - want * Vec3::X).length() < 1e-4
                && (got * Vec3::Z - want * Vec3::Z).length() < 1e-4,
            "expected a world-axis spin (delta * rot); got something else, \
             which means the multiplication order is reversed",
        );
    }
}
