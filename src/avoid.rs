//! Steering around things, for anything that moves.
//!
//! One question, asked by every agent that walks anywhere: *given where I
//! am, where I want to go, and what is near me — which way should I
//! actually steer?* This answers it and nothing else.
//!
//! # Why this is not part of the walker, or of the pathfinder
//!
//! The engine has two movers and they work in opposite ways.
//! [`crate::walk::integrate_walker`] is **position-based**: it moves a
//! `DVec2` directly and hands the result to a collision closure.
//! [`crate::ai3d::drive_agent`] is **force-based**: it applies an impulse
//! and lets the contact solver decide what actually happens. Neither can
//! use the other's movement code.
//!
//! But both need the same steering decision, so that decision is what
//! lives here — as a direction, not a displacement and not a force. The
//! 2D walker folds the result into its walk vector; the 3D agent folds it
//! into its steering force; a game with its own movement code folds it
//! into whatever it has. None of them has to agree about anything else.
//!
//! [`crate::pathfind`] is the other half of the same job and deliberately
//! separate. Planning routes *around* known obstacles and steering away
//! from whatever turns up are different problems with different failure
//! modes — a plan cannot react to something that moved after it was made,
//! and a reactive steer walks into dead ends a plan would have avoided.
//! Most agents want both.
//!
//! # Dimension-free
//!
//! Everything takes [`DVec3`]. A 2D caller passes `z = 0` and gets `z = 0`
//! back; the maths never introduces a vertical component of its own. That
//! follows [`crate::pathfind::TileSource`], which is reused unchanged by
//! the 3D navigator because it never mentioned a dimension in the first
//! place — rather than the alternative of widening a 2D type, which the
//! project deliberately does not do.

use glam::DVec3;

/// Something to steer around.
///
/// A position and a radius, by value. Deliberately *not* a reference to
/// anything the engine owns: an agent avoiding things should not need the
/// physics bodies, the broadphase, or an ECS — the caller knows what is
/// near it and says so. That is what lets a 2D tile game, a 3D rigid-body
/// game and a game with its own movement code all use this.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obstacle {
    pub pos: DVec3,
    /// How far out to treat as solid. A bounding radius is enough —
    /// steering is a nudge, not a contact resolution, and the difference
    /// between a circle and the box inside it is smaller than the margin
    /// a sane agent leaves anyway.
    pub radius: f64,
}

impl Obstacle {
    pub fn new(pos: DVec3, radius: f64) -> Self {
        Self { pos, radius }
    }
}

/// How an agent steers around things.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AvoidTuning {
    /// The agent's own radius. Obstacle clearances are measured between
    /// surfaces, not centres, so a wide agent keeps a wide berth without
    /// any of the other numbers changing.
    pub self_radius: f64,
    /// How far beyond touching to start steering away, in metres.
    ///
    /// The whole avoidance range is `self_radius + obstacle.radius +
    /// margin`, so this is the only part a caller tunes: how much daylight
    /// the agent tries to keep.
    ///
    /// # This is a *time* budget in disguise, and guessing it fails
    ///
    /// Avoidance can only move the mover sideways while it is inside the
    /// range, so the range divided by the closing speed is all the time
    /// there is — and a fast mover in a narrow range has no time at all.
    /// The failure is quiet: the steering is computed perfectly, the
    /// mover leans, and it clips the obstacle anyway.
    ///
    /// Measured on a 3 m/s walker with this margin at 0.4 m: a 1.5 m
    /// range, a 0.5 s window, and 0.14 m of lateral movement against the
    /// 0.6 m it needed to clear the crate. It drove through it.
    ///
    /// [`AvoidTuning::for_speed`] picks this from the numbers that
    /// actually decide it. Prefer it to setting this by eye.
    pub margin: f64,
    /// How hard it steers away, relative to how hard it steers toward its
    /// goal.
    ///
    /// Above 1.0 avoidance wins outright when the two are opposed, which
    /// is how an agent ends up refusing to approach anything — including
    /// the thing it was sent to. Below 1.0 the goal always eventually
    /// wins, and the agent squeezes past rather than giving up. That is
    /// the behaviour almost every caller wants, so the default is 0.8 and
    /// the field is documented as a ratio rather than a force.
    pub strength: f64,
}

impl AvoidTuning {
    /// Tuning for a mover of a given size and speed.
    ///
    /// Use this rather than setting [`AvoidTuning::margin`] by eye, for
    /// the reason that field documents: the margin is really a *time*
    /// budget, and the time a mover needs does not depend on distance —
    /// it depends on how fast it is closing and how far sideways it has
    /// to get.
    ///
    /// # Where the number comes from
    ///
    /// To miss an obstacle the mover must move sideways by roughly its
    /// own radius plus the obstacle's before it arrives. Avoidance
    /// reaches full strength at contact and zero at the edge of the
    /// range, so the *average* sideways push over the approach is about
    /// half of `strength`, giving a usable lateral speed of roughly
    /// `speed * strength / 2`.
    ///
    /// Requiring `sideways_needed / lateral_speed <= range / speed` and
    /// solving for the range gives `2 * sideways_needed / strength`,
    /// which is what this returns — with the obstacle's own radius left
    /// out, because it is added at the point of use and is not known
    /// here. `clearance` stands in for it: the radius of the largest
    /// thing the mover expects to have to get round.
    pub fn for_speed(self_radius: f64, clearance: f64, strength: f64) -> Self {
        let strength = strength.clamp(0.05, 1.0);
        let sideways = (self_radius + clearance).max(0.0);
        Self {
            self_radius,
            // The mover's own radius is already in the range at the point
            // of use, so only the shortfall goes in the margin.
            margin: (2.0 * sideways / strength - self_radius - clearance).max(0.0),
            strength,
        }
    }
}

impl Default for AvoidTuning {
    fn default() -> Self {
        Self {
            self_radius: 0.5,
            margin: 0.4,
            strength: 0.8,
        }
    }
}

/// Which way to steer, as a unit vector, or `None` when nothing is close.
///
/// `desired` is the direction the agent wanted to go — usually toward its
/// goal. The result is that direction bent away from whatever is nearby,
/// renormalised. Feed it back into whatever moves the agent: scale it by
/// speed for a position-based walker, by force for a force-based one.
///
/// # Why a direction rather than a force
///
/// A force would have to respect the caller's own limits — a walking
/// agent has a maximum it can push without tipping over, a tile walker
/// has a step size — and this function does not know them. Returning a
/// direction leaves the magnitude where the knowledge is, and means the
/// result cannot break a caller's force budget by construction.
///
/// # Why it bends rather than sums
///
/// The naive version adds a repulsion vector to the goal vector and
/// normalises. That stalls: an obstacle directly between the agent and
/// its goal produces a repulsion exactly opposite the goal direction, the
/// two cancel, and the agent stands still — the classic local minimum.
///
/// Bending sideways instead has no such point. The push is applied
/// perpendicular to the line to the obstacle, on whichever side the agent
/// is already leaning, so an obstacle dead ahead pushes the agent *round*
/// it rather than back down its own path.
pub fn steer_around(
    pos: DVec3,
    desired: DVec3,
    obstacles: &[Obstacle],
    tuning: AvoidTuning,
) -> Option<DVec3> {
    let want = desired.normalize_or_zero();
    if want.length_squared() < 0.5 {
        // No direction to bend. An agent that wants to stand still is not
        // avoiding anything by standing still somewhere else.
        return None;
    }

    // A zero `want.z` means the caller's problem is planar, so the
    // obstacles are treated as planar too.
    //
    // Without this a walker gets nonsense from an obstacle whose centre
    // is a few millimetres above its own: the bend is computed
    // perpendicular to a line that tilts out of the plane, so it points
    // mostly at the *sky*, and the caller flattens that away to nothing.
    // Measured: an agent walking at a crate 5 mm above its centre
    // deviated by 3 mm over a 12 m walk and drove straight through it.
    //
    // Keyed off `want` rather than a flag because it is the caller's own
    // direction — a mover that can climb hands us a `want` with a `z`,
    // and gets the full 3D behaviour with nothing to configure.
    let planar = want.z == 0.0;
    let flatten = |v: DVec3| if planar { DVec3::new(v.x, v.y, 0.0) } else { v };

    let mut bend = DVec3::ZERO;
    let mut any = false;

    for o in obstacles {
        let to_obstacle = flatten(o.pos - pos);
        let dist = to_obstacle.length();

        // Anything the mover is *inside* is scenery, not an obstacle.
        //
        // A bounding radius is a poor stand-in for a very large flat
        // thing, and the floor is the case that proves it: in
        // `examples/crates3d` the floor's bounding sphere has a radius of
        // 17 m, so an agent anywhere on it is deep inside that sphere.
        // Without this the agent is shoved away from the world origin
        // every tick by a force that means nothing — which does not look
        // like a bug, it looks like the agent is merely drifty.
        //
        // Ground, rooms and arenas are all like this: you do not steer
        // around the thing you are standing on.
        if dist <= o.radius {
            continue;
        }

        let clear = tuning.self_radius + o.radius + tuning.margin;
        if dist >= clear || dist <= 1e-9 {
            continue;
        }

        // Only things roughly ahead. Steering away from what is already
        // behind makes an agent flee its own wake, and the obstacle it has
        // just squeezed past would shove it back into the one in front.
        let toward = to_obstacle / dist;
        let ahead = toward.dot(want);
        if ahead <= 0.0 {
            continue;
        }

        // Sideways, not backwards: the component of "away" that is
        // perpendicular to where the agent wants to go. This is the part
        // that has no local minimum — see the function docs.
        let mut side = want * ahead - toward;
        if side.length_squared() < 1e-12 {
            // Dead ahead, exactly. There is no preferred side, so pick one
            // deterministically rather than leaving it to float noise:
            // an agent that dithers between left and right walks straight
            // into the thing.
            side = DVec3::new(-want.y, want.x, want.z * 0.0);
            if side.length_squared() < 1e-12 {
                side = DVec3::new(0.0, -want.z, want.y);
            }
        }

        // Closer is stronger, reaching full strength at touching and zero
        // at the edge of the margin. Linear rather than inverse-square:
        // an inverse law is unbounded at contact, and a steering nudge
        // that goes to infinity is one that flings the agent.
        let urgency = ((clear - dist) / clear.max(1e-9)).clamp(0.0, 1.0);
        bend += side.normalize_or_zero() * urgency * ahead;
        any = true;
    }

    if !any {
        return None;
    }
    Some((want + bend * tuning.strength).normalize_or_zero())
}

/// Whether anything is close enough to be worth steering around.
///
/// Cheaper than [`steer_around`] and answers a different question: a
/// caller that wants to slow down near obstacles, or to re-plan, needs to
/// know *that* something is near without caring which way to go.
pub fn nearest_clearance(pos: DVec3, obstacles: &[Obstacle], self_radius: f64) -> Option<f64> {
    obstacles
        .iter()
        .map(|o| (o.pos - pos).length() - o.radius - self_radius)
        .fold(None, |best: Option<f64>, d| {
            Some(best.map_or(d, |b| b.min(d)))
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tuning() -> AvoidTuning {
        AvoidTuning { self_radius: 0.5, margin: 0.4, strength: 0.8 }
    }

    /// Nothing nearby means nothing to say. A caller that gets `None`
    /// keeps its own direction, which is cheaper than being handed back
    /// the same vector it passed in.
    #[test]
    fn an_empty_field_needs_no_steering() {
        let got = steer_around(DVec3::ZERO, DVec3::X, &[], tuning());
        assert_eq!(got, None);
    }

    /// And neither does one far away.
    #[test]
    fn a_distant_obstacle_needs_no_steering() {
        let far = [Obstacle::new(DVec3::new(50.0, 0.0, 0.0), 0.5)];
        assert_eq!(steer_around(DVec3::ZERO, DVec3::X, &far, tuning()), None);
    }

    /// An obstacle dead ahead must push the agent **round** it, not back
    /// down its own path.
    ///
    /// This is the property the whole design turns on. Summing a
    /// repulsion vector into the goal vector gives exactly the opposite:
    /// the two cancel, the agent stops, and it never gets anywhere. That
    /// is the classic local minimum, and it is why the push here is
    /// applied sideways.
    #[test]
    fn an_obstacle_dead_ahead_is_steered_around_not_backed_away_from() {
        let blocking = [Obstacle::new(DVec3::new(1.0, 0.0, 0.0), 0.5)];
        let got = steer_around(DVec3::ZERO, DVec3::X, &blocking, tuning())
            .expect("something is in the way");

        assert!(
            got.x > 0.0,
            "still heading generally forward, got {got:?} — a negative x \
             means the agent has turned back rather than going round",
        );
        assert!(
            got.y.abs() > 0.1,
            "steering sideways to get past, got {got:?}",
        );
    }

    /// The result is always a direction, whatever the input magnitudes.
    #[test]
    fn the_result_is_a_unit_direction() {
        let blocking = [Obstacle::new(DVec3::new(1.0, 0.2, 0.0), 0.5)];
        for scale in [0.01, 1.0, 1000.0] {
            let got = steer_around(DVec3::ZERO, DVec3::X * scale, &blocking, tuning())
                .expect("something is in the way");
            assert!(
                (got.length() - 1.0).abs() < 1e-9,
                "input scaled by {scale} gave a vector of length {}",
                got.length(),
            );
        }
    }

    /// Something behind the agent is not its problem.
    ///
    /// Steering away from what has already been passed makes an agent flee
    /// its own wake — and worse, the obstacle it has just squeezed past
    /// shoves it back into whatever is in front.
    #[test]
    fn an_obstacle_behind_is_ignored() {
        let behind = [Obstacle::new(DVec3::new(-0.6, 0.0, 0.0), 0.5)];
        assert_eq!(steer_around(DVec3::ZERO, DVec3::X, &behind, tuning()), None);
    }

    /// Closer pushes harder, so an agent threading a gap leans away from
    /// whichever side it is closest to.
    #[test]
    fn a_closer_obstacle_pushes_harder() {
        let near = [Obstacle::new(DVec3::new(1.0, 0.25, 0.0), 0.5)];
        let far = [Obstacle::new(DVec3::new(1.3, 0.25, 0.0), 0.5)];
        let a = steer_around(DVec3::ZERO, DVec3::X, &near, tuning()).unwrap();
        let b = steer_around(DVec3::ZERO, DVec3::X, &far, tuning()).unwrap();
        assert!(
            a.y.abs() > b.y.abs(),
            "the nearer obstacle should bend more: {:.4} vs {:.4}",
            a.y.abs(),
            b.y.abs(),
        );
    }

    /// The bend goes *away* from the obstacle, not into it.
    #[test]
    fn the_bend_is_away_from_the_obstacle() {
        let on_the_left = [Obstacle::new(DVec3::new(1.0, 0.3, 0.0), 0.5)];
        let got = steer_around(DVec3::ZERO, DVec3::X, &on_the_left, tuning()).unwrap();
        assert!(
            got.y < 0.0,
            "an obstacle to the left should push the agent right, got {got:?}",
        );
    }

    /// A 2D caller gets a 2D answer. The maths never invents a vertical
    /// component of its own, which is what lets a top-down game use this
    /// without thinking about z at all.
    #[test]
    fn a_flat_problem_stays_flat() {
        let blocking = [Obstacle::new(DVec3::new(1.0, 0.1, 0.0), 0.5)];
        let got = steer_around(DVec3::ZERO, DVec3::X, &blocking, tuning()).unwrap();
        assert_eq!(got.z, 0.0, "a flat field produced a vertical steer: {got:?}");
    }

    /// An agent with nowhere to go is not avoiding anything.
    #[test]
    fn no_desired_direction_gives_no_steering() {
        let blocking = [Obstacle::new(DVec3::new(0.5, 0.0, 0.0), 0.5)];
        assert_eq!(
            steer_around(DVec3::ZERO, DVec3::ZERO, &blocking, tuning()),
            None,
        );
    }

    /// Exactly dead ahead has no preferred side, so one is chosen rather
    /// than left to float noise — an agent that dithers between left and
    /// right walks straight into the thing it is dithering about.
    #[test]
    fn a_perfectly_centred_obstacle_still_picks_a_side() {
        let dead_ahead = [Obstacle::new(DVec3::new(1.0, 0.0, 0.0), 0.5)];
        let got = steer_around(DVec3::ZERO, DVec3::X, &dead_ahead, tuning()).unwrap();
        assert!(
            got.y.abs() > 0.05,
            "a centred obstacle left the agent going straight at it: {got:?}",
        );
        // And the same input must give the same answer every time.
        let again = steer_around(DVec3::ZERO, DVec3::X, &dead_ahead, tuning()).unwrap();
        assert_eq!(got, again, "the chosen side is not deterministic");
    }

    /// Clearance is measured surface to surface, so a wide agent keeps a
    /// wide berth without any other number changing.
    #[test]
    fn clearance_is_measured_between_surfaces() {
        let o = [Obstacle::new(DVec3::new(3.0, 0.0, 0.0), 1.0)];
        let gap = nearest_clearance(DVec3::ZERO, &o, 0.5).unwrap();
        assert!(
            (gap - 1.5).abs() < 1e-9,
            "3 m apart, radii 1.0 and 0.5, so 1.5 m of daylight — got {gap}",
        );
    }

    /// Overlapping reports a negative clearance rather than clamping, so a
    /// caller can tell "just touching" from "buried".
    #[test]
    fn overlap_reports_a_negative_clearance() {
        let o = [Obstacle::new(DVec3::new(0.5, 0.0, 0.0), 1.0)];
        let gap = nearest_clearance(DVec3::ZERO, &o, 0.5).unwrap();
        assert!(gap < 0.0, "buried in an obstacle but reported {gap}");
    }

    /// Nothing near means no answer, not an arbitrary large number.
    #[test]
    fn clearance_of_nothing_is_nothing() {
        assert_eq!(nearest_clearance(DVec3::ZERO, &[], 0.5), None);
    }

    /// The floor is not an obstacle.
    ///
    /// A bounding radius is a poor stand-in for a very large flat thing.
    /// In `examples/crates3d` the floor's bounding sphere has a radius of
    /// 17 m, so an agent standing anywhere on it is deep inside that
    /// sphere — and without the inside-it guard the agent is pushed away
    /// from the world origin every tick by a force that means nothing.
    /// The symptom is not an obvious bug; the agent just looks drifty.
    #[test]
    fn a_thing_the_mover_is_standing_inside_is_not_avoided() {
        // The real number, from `examples/crates3d`: a 24 m square floor
        // half a metre thick.
        let floor = [Obstacle::new(DVec3::ZERO, (12.0f64 * 12.0 * 2.0 + 0.25).sqrt())];

        // Walking *toward* the floor's centre is the case that matters.
        // Walking away from it is skipped by the behind-me check anyway,
        // so it would pass with or without the guard and proves nothing.
        for at in [
            DVec3::new(3.0, 0.0, 0.0),
            DVec3::new(-8.0, 5.0, 0.0),
            DVec3::new(0.0, 11.0, 0.0),
        ] {
            let toward_centre = -at.normalize();
            assert_eq!(
                steer_around(at, toward_centre, &floor, tuning()),
                None,
                "standing on the floor at {at:?} and walking across it                  produced a steer away from the thing underfoot",
            );
        }
    }

    /// But a thing merely *close* is still avoided, or the guard would
    /// swallow every obstacle an agent is about to hit.
    #[test]
    fn a_thing_the_mover_is_merely_near_is_still_avoided() {
        let crate_ = [Obstacle::new(DVec3::new(0.9, 0.1, 0.0), 0.5)];
        assert!(
            steer_around(DVec3::ZERO, DVec3::X, &crate_, tuning()).is_some(),
            "a crate just ahead was ignored",
        );
    }
}
