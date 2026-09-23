//! Rigid-body dynamics for the 3D path.
//!
//! # What this adds that the engine never had
//!
//! [`crate::collision`] says outright that "the engine gives you the pair
//! list, not the resolution", and until now that was true in both
//! dimensions: every consumer game wrote its own collision response. This
//! module is that response — gravity, contact resolution, friction,
//! restitution and sleeping — for 3D.
//!
//! The 2D path is deliberately untouched. Two shipped games rely on
//! resolving collisions themselves, and giving them a solver they did not
//! ask for would change how they behave.
//!
//! # The shape of a step
//!
//! ```text
//! gravity + damping  -> integrate -> broadphase (SpatialGrid3D)
//!   -> narrowphase (narrow3d) -> solve contacts -> sleep
//! ```
//!
//! [`crate::physics3d::step`] runs the first two and the last; the middle
//! is the caller's, because which pairs are worth testing and which shapes
//! they are is a game's business.
//! [`crate::physics3d::solver::solve`] takes the contact list that comes
//! out.
//!
//! # What it is not
//!
//! No continuous collision detection: a fast enough body still tunnels
//! through a thin wall in one step. No joints or constraints. No
//! broadphase of its own — that is [`crate::collision::grid3d`]. Each is
//! a real gap rather than an oversight, and each is its own piece of work.

pub mod body;
pub mod solver;

pub use body::{apply_impulse, apply_impulse_at, BodyKind, Material3D, RigidBody};
pub use solver::{BodyRef, Contact};

use glam::DVec3;

/// Earth gravity, in metres per second squared, with +Z up.
///
/// +Z because that is the engine's up everywhere else — see
/// `terrain::field::hillshade`, whose surface normal is
/// `(-dz/dx, -dz/dy, 1)`, and `Camera3D`, whose default up is `Vec3::Z`.
pub const GRAVITY: DVec3 = DVec3::new(0.0, 0.0, -9.81);

/// Linear speed below which a body counts as still, in m/s.
pub const SLEEP_LINEAR_THRESHOLD: f64 = 0.05;
/// Angular speed below which a body counts as still, in rad/s.
pub const SLEEP_ANGULAR_THRESHOLD: f64 = 0.05;
/// How long a body must stay below both thresholds before it sleeps.
///
/// Half a second, so a body that is genuinely settling is not woken by
/// the next frame's jitter, and one that is merely slow at the top of an
/// arc does not fall asleep mid-flight.
pub const TIME_TO_SLEEP: f32 = 0.5;

/// Whether a body is moving slowly enough to be a sleep candidate.
pub fn is_still(velocity: &crate::components::Velocity3D) -> bool {
    velocity.linear.length_squared() < SLEEP_LINEAR_THRESHOLD * SLEEP_LINEAR_THRESHOLD
        && velocity.angular.length_squared()
            < SLEEP_ANGULAR_THRESHOLD * SLEEP_ANGULAR_THRESHOLD
}

/// Update one body's sleep state for a step that has just run.
///
/// A body must be still for [`TIME_TO_SLEEP`] continuously; any motion
/// resets the clock. Sleeping zeroes the velocity outright — a body that
/// kept its last sub-threshold velocity would drift imperceptibly
/// forever, and over a long session that adds up.
pub fn update_sleep(
    rb: &mut RigidBody,
    velocity: &mut crate::components::Velocity3D,
    dt: f32,
) {
    if !rb.kind.is_dynamic() || rb.sleeping {
        return;
    }
    if is_still(velocity) {
        rb.time_below_threshold += dt;
        if rb.time_below_threshold >= TIME_TO_SLEEP {
            rb.sleeping = true;
            velocity.linear = DVec3::ZERO;
            velocity.angular = DVec3::ZERO;
        }
    } else {
        rb.time_below_threshold = 0.0;
    }
}

/// Apply gravity and damping, then integrate, then update sleep state.
///
/// The half of a physics step that does not need to know about contacts.
/// A caller runs this, builds its contact list from
/// [`crate::collision::grid3d`] and [`crate::collision::narrow3d`], and
/// calls [`solver::solve`].
///
/// `gravity` is a parameter rather than [`GRAVITY`] directly because a
/// game in space, underwater, or on a small body wants its own — and
/// because a test wants to turn it off.
pub fn step(
    bodies: &mut [BodyRef<'_>],
    gravity: DVec3,
    dt: f32,
) {
    for r in bodies.iter_mut() {
        if !r.body.is_movable() {
            continue;
        }

        // Gravity is an acceleration, so it is mass-independent — a
        // feather and an anvil fall at the same rate, and multiplying by
        // inv_mass here would be the classic error that makes them not.
        r.velocity.linear += gravity * dt as f64;

        r.velocity.linear *= r.body.linear_damping;
        r.velocity.angular *= r.body.angular_damping;

        r.transform.pos += r.velocity.linear * dt as f64;
        r.transform.rot = crate::physics::integrate_rotation(
            r.transform.rot,
            r.velocity.angular,
            dt,
        );

        update_sleep(r.body, r.velocity, dt);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::components::{Transform3D, Velocity3D};

    struct Scene {
        bodies: Vec<RigidBody>,
        transforms: Vec<Transform3D>,
        velocities: Vec<Velocity3D>,
    }

    impl Scene {
        fn new() -> Self {
            Self { bodies: Vec::new(), transforms: Vec::new(), velocities: Vec::new() }
        }

        fn push(&mut self, b: RigidBody, pos: DVec3, vel: DVec3) -> usize {
            self.bodies.push(b);
            self.transforms.push(Transform3D::at(pos));
            self.velocities.push(Velocity3D { linear: vel, angular: DVec3::ZERO });
            self.bodies.len() - 1
        }

        fn step(&mut self, gravity: DVec3, dt: f32) {
            let mut refs: Vec<BodyRef<'_>> = self
                .bodies
                .iter_mut()
                .zip(self.transforms.iter_mut())
                .zip(self.velocities.iter_mut())
                .map(|((body, transform), velocity)| BodyRef { body, transform, velocity })
                .collect();
            super::step(&mut refs, gravity, dt);
        }
    }

    #[test]
    fn gravity_pulls_a_body_down_the_up_axis() {
        let mut s = Scene::new();
        let a = s.push(RigidBody::sphere(1.0, 0.5), DVec3::ZERO, DVec3::ZERO);
        s.step(GRAVITY, 1.0);
        assert!(
            s.velocities[a].linear.z < 0.0 && s.transforms[a].pos.z < 0.0,
            "a body under gravity should move down -Z, got {:?}",
            s.transforms[a].pos,
        );
    }

    /// Gravity is an acceleration, so mass must not affect it. Scaling by
    /// inverse mass here is the classic error that makes a heavy object
    /// fall faster.
    #[test]
    fn a_heavy_and_a_light_body_fall_at_the_same_rate() {
        let mut s = Scene::new();
        let light = s.push(RigidBody::sphere(0.1, 0.5), DVec3::ZERO, DVec3::ZERO);
        let heavy = s.push(RigidBody::sphere(1000.0, 0.5), DVec3::ZERO, DVec3::ZERO);
        s.step(GRAVITY, 0.5);
        assert!(
            (s.transforms[light].pos.z - s.transforms[heavy].pos.z).abs() < 1e-12,
            "mass must not affect free fall: {} vs {}",
            s.transforms[light].pos.z,
            s.transforms[heavy].pos.z,
        );
    }

    #[test]
    fn a_static_body_ignores_gravity() {
        let mut s = Scene::new();
        let a = s.push(RigidBody::static_body(), DVec3::ZERO, DVec3::ZERO);
        s.step(GRAVITY, 1.0);
        assert_eq!(s.transforms[a].pos, DVec3::ZERO);
        assert_eq!(s.velocities[a].linear, DVec3::ZERO);
    }

    /// A kinematic body is moved by whatever sets its transform, not by
    /// the solver — so gravity must leave it alone too.
    #[test]
    fn a_kinematic_body_ignores_gravity() {
        let mut s = Scene::new();
        let a = s.push(RigidBody::kinematic(), DVec3::ZERO, DVec3::ZERO);
        s.step(GRAVITY, 1.0);
        assert_eq!(s.transforms[a].pos, DVec3::ZERO);
    }

    /// The point of sleeping: a settled scene must stop costing anything
    /// and stop drifting. Without it a stack jitters forever at the
    /// solver's error floor.
    #[test]
    fn a_still_body_falls_asleep_and_stops_moving() {
        let mut s = Scene::new();
        let a = s.push(RigidBody::sphere(1.0, 0.5), DVec3::ZERO, DVec3::ZERO);
        // No gravity, so nothing keeps it awake.
        for _ in 0..40 {
            s.step(DVec3::ZERO, 1.0 / 60.0);
        }
        assert!(s.bodies[a].sleeping, "a still body should have fallen asleep");
        assert_eq!(s.velocities[a].linear, DVec3::ZERO);
    }

    /// A body in flight must not fall asleep just because it is slow at
    /// the top of its arc.
    #[test]
    fn a_moving_body_stays_awake() {
        let mut s = Scene::new();
        let a = s.push(
            RigidBody::sphere(1.0, 0.5),
            DVec3::ZERO,
            DVec3::new(5.0, 0.0, 0.0),
        );
        for _ in 0..120 {
            s.step(DVec3::ZERO, 1.0 / 60.0);
        }
        assert!(!s.bodies[a].sleeping, "a moving body must stay awake");
    }

    /// The clock must reset on motion, or a body that is briefly still
    /// several times accumulates its way to sleep while still moving.
    #[test]
    fn motion_resets_the_sleep_clock() {
        let mut s = Scene::new();
        let a = s.push(RigidBody::sphere(1.0, 0.5), DVec3::ZERO, DVec3::ZERO);

        // Almost asleep...
        for _ in 0..25 {
            s.step(DVec3::ZERO, 1.0 / 60.0);
        }
        assert!(!s.bodies[a].sleeping);

        // ...then shoved.
        s.velocities[a].linear = DVec3::new(5.0, 0.0, 0.0);
        s.step(DVec3::ZERO, 1.0 / 60.0);
        assert_eq!(
            s.bodies[a].time_below_threshold, 0.0,
            "the sleep clock should have reset",
        );
    }

    /// A sleeping body must stay put: it is skipped by `step`, so gravity
    /// does not pull it through the floor it settled on.
    #[test]
    fn a_sleeping_body_is_not_moved_by_gravity() {
        let mut s = Scene::new();
        let a = s.push(RigidBody::sphere(1.0, 0.5), DVec3::ZERO, DVec3::ZERO);
        s.bodies[a].sleeping = true;

        s.step(GRAVITY, 1.0);
        assert_eq!(
            s.transforms[a].pos, DVec3::ZERO,
            "a sleeping body must not be dragged down by gravity",
        );
    }

    #[test]
    fn damping_bleeds_off_velocity() {
        let mut s = Scene::new();
        let a = s.push(
            RigidBody { linear_damping: 0.5, ..RigidBody::sphere(1.0, 0.5) },
            DVec3::ZERO,
            DVec3::new(10.0, 0.0, 0.0),
        );
        s.step(DVec3::ZERO, 1.0 / 60.0);
        assert!(
            (s.velocities[a].linear.x - 5.0).abs() < 1e-9,
            "0.5 damping should halve the velocity, got {}",
            s.velocities[a].linear.x,
        );
    }

    /// Gravity is a parameter, not a constant, so a game in space or
    /// underwater can set its own — and a test can turn it off.
    #[test]
    fn gravity_is_caller_supplied() {
        let mut s = Scene::new();
        let a = s.push(RigidBody::sphere(1.0, 0.5), DVec3::ZERO, DVec3::ZERO);
        // Upward gravity, because nothing stops a caller choosing it.
        s.step(DVec3::new(0.0, 0.0, 20.0), 1.0);
        assert!(s.transforms[a].pos.z > 0.0);
    }
}
