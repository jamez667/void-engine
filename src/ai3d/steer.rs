//! Turning "the waypoint is over there" into a force, and nothing else.
//!
//! Every function here is pure: it takes numbers and returns a force.
//! Nothing borrows a body, nothing mutates, nothing knows what a tick is.
//! That is deliberate — steering is the half of an agent that is worth
//! testing exhaustively, and a pure function is testable without building
//! a world.

use glam::DVec3;

/// Tuning for one agent's gait.
///
/// # The invariant between the radii
///
/// `goal_radius > arrive_radius`, and **both** must be comfortably larger
/// than the distance the agent covers in one tick (`speed * dt`). An
/// acceptance radius smaller than a single step lets the agent jump clean
/// over it, miss the test on both sides, and orbit the point forever.
/// At the 30 Hz headless rate a 3 m/s agent moves 0.1 m per tick, so the
/// defaults leave roughly a 3x margin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WalkTuning3D {
    /// Cruising speed, m/s.
    ///
    /// Bounded above by tunnelling, not by taste: `physics3d` has no
    /// continuous collision detection, so an agent that moves further in
    /// one tick than its own half-extent can step straight through a thin
    /// wall the narrowphase never sees. At 30 Hz, 3 m/s is 0.1 m/tick
    /// against a half-metre agent — a 5x margin.
    pub speed: f64,
    /// Largest force the agent may apply, in newtons.
    ///
    /// Caps acceleration, so a heavy agent starts heavily instead of
    /// teleporting to speed. It must also be large enough to **break
    /// static friction**, which is easy to get wrong by an order of
    /// magnitude.
    ///
    /// It is also bounded *above* by tipping. Both bounds, and what a
    /// value outside either one looks like, are set out on
    /// [`WalkTuning3D::for_body`] — which is how a caller should pick
    /// this number rather than choosing one by eye.
    pub max_force: f64,
    /// How hard the agent corrects toward [`WalkTuning3D::speed`], in
    /// 1/s. Effectively a proportional gain on velocity error.
    pub responsiveness: f64,
    /// Planar distance at which a waypoint counts as reached, in metres.
    pub arrive_radius: f64,
    /// Distance from the *goal* at which the agent starts slowing down.
    pub brake_radius: f64,
    /// Distance from the goal at which the agent is done and stops
    /// steering entirely.
    pub goal_radius: f64,
    /// How far above the floor plane the agent may be and still walk.
    ///
    /// An airborne agent has no traction, and a horizontal force on one
    /// is pure fantasy — it would let an agent steer itself mid-fall.
    pub ground_tolerance: f64,
}

/// How much of the tipping limit an agent is allowed to use.
///
/// Not 1.0: the limit is where toppling torque *equals* righting torque,
/// and a body sitting exactly there tips over on the first bump. Nine
/// tenths leaves a margin without making the agent sluggish.
pub const TIP_MARGIN: f64 = 0.9;

impl WalkTuning3D {
    /// Tuning for an agent of a given mass and shape on a floor of a
    /// given friction.
    ///
    /// Use this rather than setting `max_force` by eye. It is squeezed
    /// between two constraints that a guessed number lands outside of,
    /// and both failures look like bugs elsewhere.
    ///
    /// # The lower bound: friction
    ///
    /// An agent presses down with `m*g` and friction opposes it with up
    /// to `mu*m*g`. Below that the agent does not move **at all**.
    /// Measured with a hand-picked 60 N: an 80 kg agent on a mu = 0.6
    /// floor needed 471 N, crawled 0.2 m in fifteen seconds, and looked
    /// exactly like a broken path — while `steering_force` returned its
    /// full maximum every single tick.
    ///
    /// # The upper bound: tipping
    ///
    /// The walk force acts at the centre of mass but friction reacts at
    /// the *feet*, a half-height below, so the pair is a couple that
    /// pitches the agent forward. Gravity rights it with a lever of a
    /// half-width. So the agent stays upright only while
    /// `F * half_height < m*g * half_width`. Measured with three times
    /// the friction force — 1413 N against a 785 N limit — the agent
    /// face-planted to 26° off vertical within a second of setting off.
    ///
    /// # The window
    ///
    /// For a cube the two bounds are `mu*m*g` and `m*g`, so the whole
    /// window is a factor of `1/mu` wide and a *short, wide* agent has
    /// much more room than a tall thin one. This returns the largest
    /// force that respects the upper bound, and **clamps up** to clear
    /// the lower one if the shape leaves no room — a tall agent on a
    /// grippy floor has no feasible force, and being unable to walk is a
    /// worse failure than leaning.
    pub fn for_body(mass_kg: f64, half_extents: [f64; 3], floor_friction: f64) -> Self {
        let weight = mass_kg * crate::physics3d::GRAVITY.z.abs();
        // The narrower horizontal axis is the one it tips over.
        let half_width = half_extents[0].abs().min(half_extents[1].abs()).max(1e-6);
        let half_height = half_extents[2].abs().max(1e-6);

        let tip_limit = weight * (half_width / half_height) * TIP_MARGIN;
        let to_move = weight * floor_friction.max(0.0);

        Self {
            // `max` rather than `min`: if the shape cannot support enough
            // force to move, an agent that leans is still better than one
            // that is glued down.
            max_force: tip_limit.max(to_move * 1.05),
            ..Self::default()
        }
    }
}

impl Default for WalkTuning3D {
    /// Human-scale numbers for a roughly 1 m, roughly 1 tile-per-second
    /// walker on a 2 m grid.
    ///
    /// `max_force` assumes an 80 kg cube-shaped agent on a mu = 0.6
    /// floor — see [`WalkTuning3D::for_body`], which is what a caller of
    /// any other weight or shape should use.
    fn default() -> Self {
        Self {
            speed: 3.0,
            max_force: 80.0 * 9.81 * TIP_MARGIN,
            responsiveness: 8.0,
            arrive_radius: 0.45,
            brake_radius: 1.5,
            goal_radius: 0.30,
            ground_tolerance: 1.2,
        }
    }
}

/// The speed the agent wants right now.
///
/// Full cruising speed everywhere except the final approach, where it
/// tapers linearly to zero across `brake_radius`. Intermediate waypoints
/// deliberately do *not* brake: an agent that slows into every tile
/// corner shuffles rather than walks, and the corner is not a place it
/// needs to stop.
pub fn desired_speed(distance: f64, is_final: bool, tuning: WalkTuning3D) -> f64 {
    if !is_final || distance >= tuning.brake_radius || tuning.brake_radius <= 0.0 {
        return tuning.speed;
    }
    tuning.speed * (distance / tuning.brake_radius)
}

/// The horizontal force that moves an agent toward `target` this tick.
///
/// Returns exactly [`DVec3::ZERO`] when the agent is within
/// `goal_radius` of a final target. **Exactly** zero, not a small force:
/// a controller that keeps nudging near its setpoint oscillates around
/// it at the amplitude of one tick's overshoot and never settles. That
/// limit cycle is visually indistinguishable from solver jitter, and it
/// also keeps the body above the sleep threshold forever.
///
/// # Why this matches velocity rather than chasing position
///
/// The force is proportional to `desired_velocity - current_velocity`,
/// not to the offset to the target. An agent already at cruising speed
/// therefore applies almost no force, and one being shoved sideways by a
/// crate applies a *corrective* one. Pushing along the offset instead
/// accelerates without bound and overshoots every waypoint.
///
/// It is also why no friction-compensation term is needed: friction
/// shows up as velocity error like anything else, so the agent
/// automatically pushes harder on a grippy floor. A feed-forward term
/// would have to know the floor's material, which this function has no
/// business knowing.
pub fn steering_force(
    pos: DVec3,
    velocity: DVec3,
    target: DVec3,
    mass: f64,
    is_final: bool,
    tuning: WalkTuning3D,
) -> DVec3 {
    // Flatten first: the floor holds the agent up, so a walk force never
    // has a vertical component. Without this an agent whose target is
    // below it tries to fly down to it.
    let to_target = DVec3::new(target.x - pos.x, target.y - pos.y, 0.0);
    let distance = to_target.length();

    // The dead zone. Only final targets have one — stopping short of an
    // intermediate waypoint would strand the agent between tiles.
    if is_final && distance <= tuning.goal_radius {
        return DVec3::ZERO;
    }
    if distance <= 0.0 {
        return DVec3::ZERO;
    }

    let desired = to_target / distance * desired_speed(distance, is_final, tuning);
    let planar_velocity = DVec3::new(velocity.x, velocity.y, 0.0);

    // F = m*a, with a = k * (v_wanted - v_actual). Scaling by mass is
    // what makes `responsiveness` mean the same thing for a 10 kg agent
    // and a 100 kg one.
    let force = (desired - planar_velocity) * tuning.responsiveness * mass;

    // Clamp the *vector*, never per-axis: clamping each axis separately
    // lets a diagonal push exceed a cardinal one by root two, which is
    // the same bug `walk.rs` normalises away for its input directions.
    clamp_length(force, tuning.max_force)
}

/// Shorten `v` to at most `max`, leaving its direction alone.
fn clamp_length(v: DVec3, max: f64) -> DVec3 {
    let len = v.length();
    if len > max && len > 0.0 {
        v / len * max
    } else {
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tuning() -> WalkTuning3D {
        WalkTuning3D::default()
    }

    /// The basic job.
    #[test]
    fn an_agent_at_rest_is_pushed_toward_its_target() {
        let f = steering_force(
            DVec3::ZERO,
            DVec3::ZERO,
            DVec3::new(10.0, 0.0, 0.0),
            1.0,
            false,
            tuning(),
        );
        assert!(f.x > 0.0, "should push toward +X, got {f:?}");
        assert_eq!(f.y, 0.0);
    }

    /// An agent already travelling at cruising speed has nothing to
    /// correct. Without the velocity term the force stays at maximum
    /// forever and the agent accelerates without bound.
    #[test]
    fn an_agent_at_cruising_speed_applies_almost_no_force() {
        let t = tuning();
        let cruising = DVec3::new(t.speed, 0.0, 0.0);
        let f = steering_force(
            DVec3::ZERO,
            cruising,
            DVec3::new(100.0, 0.0, 0.0),
            1.0,
            false,
            t,
        );
        assert!(
            f.length() < 1e-9,
            "a correctly-moving agent should coast, got {f:?}",
        );
    }

    /// An agent shoved sideways corrects back, which is what makes it
    /// robust to being hit by a crate.
    #[test]
    fn a_sideways_shove_produces_a_corrective_force() {
        let f = steering_force(
            DVec3::ZERO,
            DVec3::new(0.0, 2.0, 0.0),
            DVec3::new(10.0, 0.0, 0.0),
            1.0,
            false,
            tuning(),
        );
        assert!(f.y < 0.0, "should push back against the sideways drift, got {f:?}");
    }

    /// However far the target, the force is bounded — otherwise a distant
    /// goal launches the agent.
    #[test]
    fn the_force_is_capped_however_far_away_the_target_is() {
        let t = tuning();
        for distance in [10.0, 1_000.0, 10_000.0] {
            let f = steering_force(
                DVec3::ZERO,
                DVec3::ZERO,
                DVec3::new(distance, 0.0, 0.0),
                80.0,
                false,
                t,
            );
            assert!(
                f.length() <= t.max_force + 1e-9,
                "force {} exceeded the cap at distance {distance}",
                f.length(),
            );
        }
    }

    /// The clamp must shorten the vector, not each axis. Per-axis
    /// clamping would let a diagonal push be root-two faster than a
    /// cardinal one.
    #[test]
    fn a_diagonal_push_is_no_stronger_than_a_cardinal_one() {
        let t = tuning();
        let cardinal = steering_force(
            DVec3::ZERO,
            DVec3::ZERO,
            DVec3::new(500.0, 0.0, 0.0),
            80.0,
            false,
            t,
        );
        let diagonal = steering_force(
            DVec3::ZERO,
            DVec3::ZERO,
            DVec3::new(500.0, 500.0, 0.0),
            80.0,
            false,
            t,
        );
        assert!(
            (diagonal.length() - cardinal.length()).abs() < 1e-9,
            "diagonal {} vs cardinal {}",
            diagonal.length(),
            cardinal.length(),
        );
    }

    /// A walk force is never vertical. Without flattening, an agent whose
    /// goal is below it tries to fly.
    #[test]
    fn the_force_is_purely_horizontal() {
        let f = steering_force(
            DVec3::ZERO,
            DVec3::new(0.0, 0.0, -9.0),
            DVec3::new(10.0, 0.0, -50.0),
            1.0,
            false,
            tuning(),
        );
        assert_eq!(f.z, 0.0, "a walk force must have no vertical part, got {f:?}");
    }

    /// Exactly zero inside the goal radius. A scaled-down force here
    /// would oscillate around the goal forever and keep the body awake.
    #[test]
    fn an_agent_inside_the_goal_radius_applies_exactly_zero_force() {
        let t = tuning();
        let f = steering_force(
            DVec3::ZERO,
            DVec3::new(0.2, 0.0, 0.0),
            DVec3::new(t.goal_radius * 0.5, 0.0, 0.0),
            80.0,
            true,
            t,
        );
        assert_eq!(f, DVec3::ZERO, "the dead zone must be exactly zero");
    }

    /// The dead zone is for goals only. Stopping short of an intermediate
    /// waypoint would strand the agent between tiles.
    #[test]
    fn there_is_no_dead_zone_at_an_intermediate_waypoint() {
        let t = tuning();
        let f = steering_force(
            DVec3::ZERO,
            DVec3::ZERO,
            DVec3::new(t.goal_radius * 0.5, 0.0, 0.0),
            80.0,
            false,
            t,
        );
        assert!(f.length() > 0.0, "an intermediate waypoint should still pull");
    }

    /// Braking applies to the final approach only.
    #[test]
    fn the_agent_slows_only_on_the_final_leg() {
        let t = tuning();
        let near = t.brake_radius * 0.25;
        assert_eq!(
            desired_speed(near, false, t),
            t.speed,
            "an intermediate waypoint should not slow the agent",
        );
        assert!(
            desired_speed(near, true, t) < t.speed,
            "the final approach should slow down",
        );
    }

    /// Braking tapers to zero at the goal rather than stopping dead.
    #[test]
    fn the_braking_taper_reaches_zero_at_the_goal() {
        let t = tuning();
        assert_eq!(desired_speed(0.0, true, t), 0.0);
        assert!(desired_speed(t.brake_radius, true, t) >= t.speed - 1e-9);
    }

    /// `responsiveness` must mean the same thing whatever the agent
    /// weighs, which is what scaling by mass buys.
    #[test]
    fn the_force_scales_with_mass_so_acceleration_does_not() {
        let t = WalkTuning3D { max_force: f64::MAX, ..tuning() };
        let light = steering_force(
            DVec3::ZERO,
            DVec3::ZERO,
            DVec3::new(10.0, 0.0, 0.0),
            10.0,
            false,
            t,
        );
        let heavy = steering_force(
            DVec3::ZERO,
            DVec3::ZERO,
            DVec3::new(10.0, 0.0, 0.0),
            100.0,
            false,
            t,
        );
        // Ten times the mass, ten times the force: the same acceleration.
        assert!(
            (heavy.length() / light.length() - 10.0).abs() < 1e-9,
            "force should scale linearly with mass",
        );
    }
}
