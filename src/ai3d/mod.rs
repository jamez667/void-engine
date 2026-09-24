//! Point-to-point walking for a 3D agent.
//!
//! An agent here is an ordinary dynamic rigid body that pushes itself
//! around. It is not scripted onto a path: it applies a force, the
//! integrator turns that into motion, and the contact solver has the
//! final say. So it collides with crates, gets shoved by them, and can be
//! blocked by a wall it walked into — all of which fall out of using the
//! same physics as everything else, rather than being special-cased.
//!
//! ```text
//!   drive_agent   <- steering force, applied as an impulse
//!   physics3d::step          <- gravity and integration consume it
//!   ...build contacts...
//!   solver::solve            <- contacts get the final say
//!   update_sleep_all
//! ```
//!
//! # Where this runs in a tick
//!
//! [`crate::ai3d::drive_agent`] runs **before** [`crate::physics3d::step`], in the
//! same place a player's input would be read. The force it applies is an
//! acceleration the integrator then consumes. Running it after
//! integration delays every agent by a tick; running it after the solve
//! fights contact resolution outright — the agent would push itself back
//! into the wall the solver just pushed it out of.
//!
//! # What this is not
//!
//! There is no behaviour tree, no task system and no notion of *why* an
//! agent wants to be somewhere. This module answers "walk to that point"
//! and stops there. Deciding where to go — stacking crates, following a
//! player, fleeing — is the caller's, and belongs in a layer above.
//!
//! There is also no path smoothing. The route is a sequence of tile
//! centres and the agent walks them literally, so it takes stair-step
//! corners rather than diagonals. See [`crate::ai3d::nav::plan_path`] for the cheap
//! mitigation and why smoothing is not built here.

pub mod agent;
pub mod nav;
pub mod steer;
pub mod task;

use std::collections::HashSet;

use glam::DVec3;

use crate::components::{Transform3D, Velocity3D};
use crate::pathfind::TileSource;
use crate::physics3d::body::{self, RigidBody};

pub use agent::{Agent3D, AgentState, PROGRESS_EPSILON, STALL_TIMEOUT};
pub use nav::{plan_path, NavPath, NavPlane};
pub use steer::{desired_speed, steering_force, WalkTuning3D};
pub use task::{
    drive_stacker, fork_height_for_layer, CarryPose, CrateId, CrateInfo, Forks, StackAction,
    StackState, StackTask, StackTuning, FORK_THICKNESS,
};

/// Advance one agent by one tick, applying its walk force to `body`.
///
/// Returns the force that was applied, for a caller that wants to draw or
/// log it. [`DVec3::ZERO`] means the agent applied nothing this tick —
/// it is idle, arrived, blocked, airborne, or not a dynamic body.
///
/// # Why this takes three borrows instead of a `BodyRef`
///
/// [`crate::physics3d::BodyRef`] holds `&mut Transform3D`, and an agent
/// must never move the transform. Placing a body directly is what the
/// integrator does; an AI that did it would teleport through walls and
/// skip the contact solver entirely. Taking `&Transform3D` makes that a
/// compile error rather than a code-review note.
pub fn drive_agent(
    agent: &mut Agent3D,
    body: &mut RigidBody,
    transform: &Transform3D,
    velocity: &mut Velocity3D,
    dt: f32,
) -> DVec3 {
    // A static or kinematic body cannot be pushed, and its `inv_mass` is
    // zero — so this guard also keeps the mass division below finite.
    if !body.kind.is_dynamic() {
        return DVec3::ZERO;
    }

    let Some(path) = agent.path.as_mut() else {
        // No route: nothing to do, and deliberately no `wake()`. See the
        // note on sleeping below.
        if agent.state == AgentState::Walking {
            agent.state = AgentState::Idle;
        }
        return DVec3::ZERO;
    };

    let plane = path.plane;
    let pos = transform.pos;

    // An airborne agent has no traction. Steering one would let it fly to
    // its goal, and would also apply a force the floor is not there to
    // react against.
    if (pos.z - plane.floor_z).abs() > agent.tuning.ground_tolerance {
        return DVec3::ZERO;
    }

    // Advance past every waypoint already reached — a `while`, not an
    // `if`, because a fast agent on a small grid can cross more than one
    // in a single tick, and stopping after one leaves it turning back
    // toward a tile it has passed.
    while !path.on_final_leg() {
        let Some(next) = path.next_world() else { break };
        // The test is planar. An agent's centre rides a half-height above
        // the floor, so a 3D distance to a waypoint it is standing
        // directly on never falls below the acceptance radius, and the
        // agent would grind against its own feet forever.
        let to_next = plane.flatten(next - pos);
        if to_next.length() <= agent.tuning.arrive_radius {
            path.advance();
            continue;
        }

        // A radius alone is not enough. It only retires a waypoint the
        // agent is *near*, so one it has been carried well past — shoved
        // by a crate, or simply moving fast — stays the target, and the
        // agent turns round and walks back to it. Measured before this:
        // an agent displaced four tiles along its own route kept the
        // first waypoint as its target and walked ten metres backwards.
        //
        // So also retire a waypoint the agent is *past*, which is a
        // question of direction rather than distance. The leg from this
        // waypoint toward the next one is the direction of travel; if the
        // agent has already gone further along it than this waypoint is,
        // the waypoint is behind and there is nothing to walk back for.
        let Some(after) = path.peek_world(1) else { break };
        let leg = plane.flatten(after - next);
        if leg.length_squared() > 0.0 && plane.flatten(pos - next).dot(leg) > 0.0 {
            path.advance();
            continue;
        }
        break;
    }

    let Some(target) = path.next_world() else {
        agent.state = AgentState::Arrived;
        agent.reset_progress();
        return DVec3::ZERO;
    };

    let is_final = path.on_final_leg();
    let distance = plane.flatten(target - pos).length();

    let force = steering_force(
        pos,
        velocity.linear,
        target,
        mass_of(body),
        is_final,
        agent.tuning,
    );

    if force == DVec3::ZERO {
        // Inside the goal's dead zone: arrived. Applying no force is what
        // lets friction and the sleep system actually stop the agent.
        if is_final {
            agent.state = AgentState::Arrived;
            path.advance();
            agent.reset_progress();
        }
        return DVec3::ZERO;
    }

    agent.state = AgentState::Walking;
    agent.record_progress(distance, dt);

    // **Wake only when actually steering.**
    //
    // Waking unconditionally recreates the bug `solver::solve` documents
    // at length: a body woken every tick can never accumulate
    // `TIME_TO_SLEEP`, so an idle agent standing on the floor costs the
    // solver full work forever and never sleeps. Every early return above
    // therefore returns *before* reaching this line.
    //
    // Conversely this must fire even when the force achieves nothing — an
    // agent walking into a wall is below the 0.05 m/s stillness threshold
    // and would otherwise fall asleep mid-push and stay stuck until
    // something hit it.
    body.wake();

    // At the centre of mass, never `apply_impulse_at`.
    //
    // This is the line that keeps the agent upright. An off-centre
    // self-propulsion force applies torque every tick, and a box that
    // torques itself face-plants within a second. Only *external*
    // contacts are allowed to spin an agent.
    body::apply_impulse(body, velocity, force * dt as f64);
    force
}

/// Plan or re-plan an agent's route to its goal.
///
/// Returns whether a route was found; a failure leaves the agent in
/// [`AgentState::Blocked`] with no path, so it stands still rather than
/// walking into a wall.
///
/// # Why this is separate from [`drive_agent`]
///
/// Pathfinding needs the caller's [`TileSource`] and `drive_agent`
/// deliberately does not take one. An agent walks every tick and plans
/// rarely, so making the hot path generic over the tile source would push
/// that type parameter onto every caller of it for the benefit of the
/// cold one.
///
/// It is also why a stuck agent is not re-planned automatically: this
/// module cannot see the world. A caller polls [`Agent3D::stuck`] and
/// decides.
pub fn replan<T: TileSource>(
    agent: &mut Agent3D,
    plane: NavPlane,
    src: &T,
    from: DVec3,
    extra_blocked: &HashSet<(i32, i32)>,
) -> bool {
    let Some(goal) = agent.goal else {
        agent.path = None;
        agent.state = AgentState::Idle;
        return false;
    };

    match plan_path(plane, src, from, goal, extra_blocked) {
        Some(path) => {
            // An empty route means the agent is already standing on the
            // goal tile. That is arrival, not a failure to plan.
            let empty = path.waypoints.is_empty();
            agent.path = Some(path);
            agent.state = if empty { AgentState::Arrived } else { AgentState::Walking };
            agent.reset_progress();
            true
        }
        None => {
            agent.path = None;
            agent.state = AgentState::Blocked;
            agent.reset_progress();
            false
        }
    }
}

/// An agent's mass in kilograms.
///
/// `inv_mass` is clamped to at least `1.0 / MIN_MASS` for every dynamic
/// body by `RigidBody`'s constructors, and `drive_agent` has already
/// rejected the non-dynamic bodies whose `inv_mass` is zero — so this
/// division is finite by construction.
fn mass_of(body: &RigidBody) -> f64 {
    1.0 / body.inv_mass as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics3d::{self, BodyRef};

    /// A floor plane and an open grid, matching `crates3d`'s scale.
    fn plane() -> NavPlane {
        NavPlane::new((9, 7), 2.0, 0.0)
    }

    struct OpenFloor;
    impl TileSource for OpenFloor {
        fn dims(&self) -> (u32, u32) {
            (9, 7)
        }
        fn blocks(&self, c: i32, r: i32) -> bool {
            c < 0 || r < 0 || c >= 9 || r >= 7
        }
    }

    /// A world with one agent and a static floor, stepped through the
    /// real physics sequence — the same shape a game's tick has.
    struct World {
        bodies: Vec<RigidBody>,
        transforms: Vec<Transform3D>,
        velocities: Vec<Velocity3D>,
        half: [f64; 3],
    }

    /// A short, wide agent: half-width 0.5 m, half-height 0.25 m.
    ///
    /// Deliberately not a cube. Friction reacts at the feet, a
    /// half-height below the centre of mass, so a walker's aspect ratio
    /// decides whether it can accelerate without face-planting — see
    /// [`WalkTuning3D::for_body`]. Measured across the same walk: this
    /// shape peaks at 0.10 degrees off vertical, a cube of the same mass
    /// at 87 degrees.
    const AGENT_HW: f64 = 0.5;
    const AGENT_HH: f64 = 0.25;
    const AGENT_MASS: f32 = 80.0;
    const FLOOR_FRICTION: f64 = 0.6;

    impl World {
        /// An agent of a given mass and shape.
        fn with_shape(start: DVec3, mass: f32, half: [f64; 3]) -> Self {
            let agent = RigidBody::box3d(
                mass,
                [half[0] as f32, half[1] as f32, half[2] as f32],
            )
            .with_material(physics3d::Material3D {
                restitution: 0.0,
                friction: FLOOR_FRICTION as f32,
            });
            let floor = RigidBody::static_body().with_material(physics3d::Material3D {
                restitution: 0.0,
                friction: FLOOR_FRICTION as f32,
            });
            Self {
                bodies: vec![agent, floor],
                transforms: vec![
                    Transform3D::at(start),
                    Transform3D::at(DVec3::new(0.0, 0.0, -0.5)),
                ],
                velocities: vec![Velocity3D::default(), Velocity3D::default()],
                half,
            }
        }

        /// An agent standing on a floor whose top face is at `floor_z`.
        fn new(start: DVec3) -> Self {
            Self::with_shape(start, AGENT_MASS, [AGENT_HW, AGENT_HW, AGENT_HH])
        }

        fn pos(&self) -> DVec3 {
            self.transforms[0].pos
        }

        /// One full tick: drive, step, contacts, solve, sleep.
        fn tick(&mut self, agent: &mut Agent3D, dt: f32) -> DVec3 {
            let force = {
                let (body, rest) = self.bodies.split_at_mut(1);
                let _ = rest;
                drive_agent(
                    agent,
                    &mut body[0],
                    &self.transforms[0],
                    &mut self.velocities[0],
                    dt,
                )
            };

            {
                let mut refs = self.refs();
                physics3d::step(&mut refs, physics3d::GRAVITY, dt);
            }

            let contacts = self.contacts();

            {
                let mut refs = self.refs();
                physics3d::solver::solve(&mut refs, &contacts, dt as f64);
                physics3d::update_sleep_all(&mut refs, dt);
            }

            force
        }

        fn refs(&mut self) -> Vec<BodyRef<'_>> {
            self.bodies
                .iter_mut()
                .zip(self.transforms.iter_mut())
                .zip(self.velocities.iter_mut())
                .map(|((body, transform), velocity)| BodyRef { body, transform, velocity })
                .collect()
        }

        /// The agent-vs-floor contact, built the way `crates3d` builds
        /// its own: dynamic body first, full manifold.
        fn contacts(&self) -> Vec<physics3d::Contact> {
            let floor_half = [12.0, 12.0, 0.5];
            let hit = crate::collision::narrow3d::obb_vs_obb(
                self.transforms[0].pos,
                self.half,
                self.transforms[0].rot,
                self.transforms[1].pos,
                floor_half,
                self.transforms[1].rot,
            );
            let Some((normal, penetration)) = hit else {
                return Vec::new();
            };
            crate::collision::narrow3d::obb_contact_manifold(
                self.transforms[0].pos,
                self.half,
                self.transforms[0].rot,
                self.transforms[1].pos,
                floor_half,
                self.transforms[1].rot,
                normal,
                penetration,
            )
            .into_iter()
            .map(|(point, depth)| physics3d::Contact {
                a: 0,
                b: 1,
                normal,
                penetration: depth,
                point,
            })
            .collect()
        }
    }

    /// Where an agent's centre sits when resting on the plane.
    fn standing_at(p: NavPlane, col: i32, row: i32) -> DVec3 {
        p.tile_center(col, row) + DVec3::new(0.0, 0.0, AGENT_HH)
    }

    /// Tuning sized for the harness agent, rather than the generic
    /// default: `max_force` depends on mass and shape.
    fn tuning() -> WalkTuning3D {
        WalkTuning3D::for_body(AGENT_MASS as f64, [AGENT_HW, AGENT_HW, AGENT_HH], FLOOR_FRICTION)
    }

    /// The headline test: an agent given a goal walks to it and stops
    /// there.
    #[test]
    fn an_agent_walks_across_the_floor_to_its_goal() {
        let p = plane();
        let start = standing_at(p, 1, 3);
        let goal = p.tile_center(7, 3);

        let mut w = World::new(start);
        let mut a = Agent3D::new(tuning());
        a.set_goal(goal);
        assert!(replan(&mut a, p, &OpenFloor, start, &HashSet::new()));

        for _ in 0..900 {
            w.tick(&mut a, 1.0 / 60.0);
            if a.state == AgentState::Arrived {
                break;
            }
        }

        assert_eq!(a.state, AgentState::Arrived, "the agent never arrived");
        let offset = p.flatten(w.pos() - goal).length();
        assert!(
            offset <= a.tuning.goal_radius,
            "stopped {offset:.3} m from the goal, outside the {:.2} m radius",
            a.tuning.goal_radius,
        );
    }

    /// An arrived agent must be *still*, not hovering around the goal.
    /// A controller that keeps correcting near its setpoint oscillates
    /// forever; this is the test that catches the limit cycle.
    #[test]
    fn an_arrived_agent_does_not_jitter() {
        let p = plane();
        let start = standing_at(p, 2, 3);
        let goal = p.tile_center(5, 3);

        let mut w = World::new(start);
        let mut a = Agent3D::new(tuning());
        a.set_goal(goal);
        replan(&mut a, p, &OpenFloor, start, &HashSet::new());

        for _ in 0..900 {
            w.tick(&mut a, 1.0 / 60.0);
            if a.state == AgentState::Arrived {
                break;
            }
        }
        assert_eq!(a.state, AgentState::Arrived);

        // Now measure how far it wanders over the next ten seconds.
        let settled = w.pos();
        let mut travelled = 0.0;
        let mut prev = settled;
        for _ in 0..600 {
            w.tick(&mut a, 1.0 / 60.0);
            travelled += p.flatten(w.pos() - prev).length();
            prev = w.pos();
        }
        assert!(
            travelled < 0.01,
            "an arrived agent wandered {travelled:.4} m over 600 ticks",
        );
    }

    /// Arriving must let the body sleep. An agent that keeps waking
    /// itself costs the solver full work forever, which is exactly the
    /// bug the solver's own wake rule documents.
    #[test]
    fn an_arrived_agent_falls_asleep() {
        let p = plane();
        let start = standing_at(p, 3, 3);
        let goal = p.tile_center(5, 3);

        let mut w = World::new(start);
        let mut a = Agent3D::new(tuning());
        a.set_goal(goal);
        replan(&mut a, p, &OpenFloor, start, &HashSet::new());

        for _ in 0..1200 {
            w.tick(&mut a, 1.0 / 60.0);
        }

        assert_eq!(a.state, AgentState::Arrived);
        assert!(
            w.bodies[0].sleeping,
            "an agent that has arrived and stopped should fall asleep",
        );
    }

    /// A walking agent must never be put to sleep, or it stops mid-stride
    /// and the solver skips it entirely.
    #[test]
    fn a_walking_agent_is_never_put_to_sleep() {
        let p = plane();
        let start = standing_at(p, 0, 3);
        let goal = p.tile_center(8, 3);

        let mut w = World::new(start);
        let mut a = Agent3D::new(tuning());
        a.set_goal(goal);
        replan(&mut a, p, &OpenFloor, start, &HashSet::new());

        for _ in 0..900 {
            w.tick(&mut a, 1.0 / 60.0);
            if a.state == AgentState::Arrived {
                break;
            }
            assert!(
                !w.bodies[0].sleeping,
                "the agent fell asleep mid-walk at {:?}",
                w.pos(),
            );
        }
    }

    /// Walking must not tip the agent over. The force goes through the
    /// centre of mass precisely so that it cannot.
    #[test]
    fn an_agent_does_not_tip_over_while_walking() {
        let p = plane();
        let start = standing_at(p, 1, 3);
        let goal = p.tile_center(7, 3);

        let mut w = World::new(start);
        let mut a = Agent3D::new(tuning());
        a.set_goal(goal);
        replan(&mut a, p, &OpenFloor, start, &HashSet::new());

        // Two degrees, not "not fallen over". A correctly-proportioned
        // agent driven through its centre of mass measures 0.10 degrees
        // across this walk; anything approaching a degree means the walk
        // force has acquired a lever arm. A loose bound here passes
        // happily with a quarter-metre of lever applied every tick.
        const MAX_TILT_DEG: f64 = 2.0;
        let mut worst: f64 = 0.0;
        for _ in 0..900 {
            w.tick(&mut a, 1.0 / 60.0);
            let up = (w.transforms[0].rot * glam::Vec3::Z).as_dvec3();
            worst = worst.max(up.dot(DVec3::Z).clamp(-1.0, 1.0).acos().to_degrees());
            assert!(
                worst < MAX_TILT_DEG,
                "the agent leaned {worst:.2} degrees off vertical",
            );
            if a.state == AgentState::Arrived {
                break;
            }
        }
    }

    /// The walk force must impart **no spin at all**.
    ///
    /// The tilt test above measures the consequence; this measures the
    /// cause, and catches a lever arm too small to topple the agent but
    /// large enough to keep it rocking — which is enough to stop it ever
    /// sleeping. Only external contacts may spin an agent.
    #[test]
    fn the_walk_force_imparts_no_angular_velocity() {
        let p = plane();
        let start = standing_at(p, 1, 3);
        let mut w = World::new(start);
        let mut a = Agent3D::new(tuning());
        a.set_goal(p.tile_center(7, 3));
        replan(&mut a, p, &OpenFloor, start, &HashSet::new());

        // Straight from `drive_agent`, with no physics in between, so
        // nothing but the walk force can have touched the spin.
        let before = w.velocities[0].angular;
        let force = drive_agent(
            &mut a,
            &mut w.bodies[0],
            &w.transforms[0],
            &mut w.velocities[0],
            1.0 / 60.0,
        );

        assert!(force.length() > 0.0, "the agent should be pushing");
        assert_eq!(
            w.velocities[0].angular, before,
            "a walk force applied through the centre of mass cannot spin the body",
        );
    }

    /// The AI drives the body with forces; it must never place it. A
    /// future refactor that widened the borrow would break this.
    #[test]
    fn driving_an_agent_never_moves_the_transform() {
        let p = plane();
        let start = standing_at(p, 1, 3);
        let mut w = World::new(start);
        let mut a = Agent3D::new(tuning());
        a.set_goal(p.tile_center(7, 3));
        replan(&mut a, p, &OpenFloor, start, &HashSet::new());

        let before = w.transforms[0].clone();
        let force = drive_agent(
            &mut a,
            &mut w.bodies[0],
            &w.transforms[0],
            &mut w.velocities[0],
            1.0 / 60.0,
        );

        assert!(force.length() > 0.0, "the agent should be pushing");
        assert_eq!(w.transforms[0].pos, before.pos, "the AI moved the body itself");
        assert_eq!(w.transforms[0].rot, before.rot);
    }

    /// An agent with nowhere to go applies nothing and, critically, does
    /// not wake the body — otherwise an idle agent keeps the whole scene
    /// awake.
    #[test]
    fn an_idle_agent_applies_no_force_and_leaves_the_body_asleep() {
        let p = plane();
        let mut w = World::new(standing_at(p, 4, 3));
        let mut a = Agent3D::new(tuning());

        w.bodies[0].sleeping = true;
        let force = drive_agent(
            &mut a,
            &mut w.bodies[0],
            &w.transforms[0],
            &mut w.velocities[0],
            1.0 / 60.0,
        );

        assert_eq!(force, DVec3::ZERO);
        assert!(w.bodies[0].sleeping, "an idle agent must not wake the body");
    }

    /// An airborne agent has no traction, so steering it would be pure
    /// fantasy — and would let it fly to its goal.
    #[test]
    fn an_airborne_agent_applies_no_walk_force() {
        let p = plane();
        let start = standing_at(p, 1, 3);
        let mut a = Agent3D::new(tuning());
        a.set_goal(p.tile_center(7, 3));
        replan(&mut a, p, &OpenFloor, start, &HashSet::new());

        // Well above the floor, past `ground_tolerance`.
        let airborne = start + DVec3::new(0.0, 0.0, 6.0);
        let mut w = World::new(airborne);
        let force = drive_agent(
            &mut a,
            &mut w.bodies[0],
            &w.transforms[0],
            &mut w.velocities[0],
            1.0 / 60.0,
        );
        assert_eq!(force, DVec3::ZERO, "an agent in mid-air cannot walk");
    }

    /// A static or kinematic body cannot be driven, and its zero
    /// `inv_mass` must not reach the mass division.
    #[test]
    fn a_non_dynamic_body_is_left_alone() {
        let p = plane();
        let start = standing_at(p, 1, 3);
        let mut w = World::new(start);
        let mut a = Agent3D::new(tuning());
        a.set_goal(p.tile_center(7, 3));
        replan(&mut a, p, &OpenFloor, start, &HashSet::new());

        w.bodies[0] = RigidBody::static_body();
        let force = drive_agent(
            &mut a,
            &mut w.bodies[0],
            &w.transforms[0],
            &mut w.velocities[0],
            1.0 / 60.0,
        );
        assert_eq!(force, DVec3::ZERO);
        assert_eq!(w.velocities[0].linear, DVec3::ZERO, "a static body must not move");
    }

    /// No route means the agent stands still and says so, rather than
    /// walking into the wall it cannot get past.
    #[test]
    fn an_unreachable_goal_leaves_the_agent_blocked() {
        struct Sealed;
        impl TileSource for Sealed {
            fn dims(&self) -> (u32, u32) {
                (9, 7)
            }
            fn blocks(&self, c: i32, r: i32) -> bool {
                // Everything past column 3 is walled off.
                c < 0 || r < 0 || c >= 9 || r >= 7 || c == 4
            }
        }

        let p = plane();
        let start = standing_at(p, 1, 3);
        let mut a = Agent3D::new(tuning());
        a.set_goal(p.tile_center(7, 3));

        assert!(!replan(&mut a, p, &Sealed, start, &HashSet::new()));
        assert_eq!(a.state, AgentState::Blocked);
        assert!(a.path.is_none());

        let mut w = World::new(start);
        let force = drive_agent(
            &mut a,
            &mut w.bodies[0],
            &w.transforms[0],
            &mut w.velocities[0],
            1.0 / 60.0,
        );
        assert_eq!(force, DVec3::ZERO, "a blocked agent should not push");
    }

    /// Standing on the goal tile already is arrival, not a planning
    /// failure — A* returns an empty route for start == goal.
    #[test]
    fn planning_to_the_tile_underfoot_counts_as_arrived() {
        let p = plane();
        let start = standing_at(p, 4, 3);
        let mut a = Agent3D::new(tuning());
        a.set_goal(p.tile_center(4, 3));

        assert!(replan(&mut a, p, &OpenFloor, start, &HashSet::new()));
        assert_eq!(a.state, AgentState::Arrived);
    }

    /// An agent shoved into a wall gets nowhere, and must notice — and
    /// must stay awake while it pushes, or it sleeps mid-shove and is
    /// stuck until something hits it.
    #[test]
    fn an_agent_pushing_against_a_wall_stays_awake_and_reports_stuck() {
        let p = plane();
        let start = standing_at(p, 4, 3);
        let mut w = World::new(start);
        let mut a = Agent3D::new(tuning());
        a.set_goal(p.tile_center(8, 3));
        replan(&mut a, p, &OpenFloor, start, &HashSet::new());

        // Pin the agent in place: it keeps steering, but never moves.
        for _ in 0..180 {
            let force = drive_agent(
                &mut a,
                &mut w.bodies[0],
                &w.transforms[0],
                &mut w.velocities[0],
                1.0 / 60.0,
            );
            assert!(force.length() > 0.0, "a blocked agent should keep pushing");
            // Whatever the push did, the wall undoes.
            w.velocities[0] = Velocity3D::default();
            let mut refs = w.refs();
            physics3d::update_sleep_all(&mut refs, 1.0 / 60.0);
        }

        assert!(!w.bodies[0].sleeping, "an agent pushing a wall must stay awake");
        assert!(a.stuck(), "an agent getting nowhere should report stuck");
    }

    /// A fast agent can cross more than one waypoint in a tick; the
    /// cursor must skip all of them, or it turns back toward a tile it
    /// has already passed.
    #[test]
    fn crossing_several_waypoints_in_one_tick_advances_past_all_of_them() {
        let p = plane();
        let start = standing_at(p, 0, 3);
        let mut a = Agent3D::new(tuning());
        a.set_goal(p.tile_center(8, 3));
        replan(&mut a, p, &OpenFloor, start, &HashSet::new());
        let before = a.path.as_ref().unwrap().cursor;

        // Teleport most of the way there, as a shove might.
        let mut w = World::new(standing_at(p, 6, 3));
        drive_agent(
            &mut a,
            &mut w.bodies[0],
            &w.transforms[0],
            &mut w.velocities[0],
            1.0 / 60.0,
        );

        // Every waypoint the agent has gone past must be retired in the
        // one call, and the new target must be ahead of it rather than
        // behind — walking backwards to a tile it already crossed is the
        // bug this guards.
        let path = a.path.as_ref().unwrap();
        let after = path.cursor;
        assert!(
            after >= before + 4,
            "cursor only moved {before} -> {after}; it should skip every passed waypoint",
        );
        let target = path.next_world().expect("still walking");
        assert!(
            target.x > w.transforms[0].pos.x,
            "the new target at x={:.2} is behind the agent at x={:.2}",
            target.x,
            w.transforms[0].pos.x,
        );
    }




}
