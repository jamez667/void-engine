//! What one walking agent remembers between ticks.

use glam::DVec3;

use crate::avoid::{AvoidTuning, Obstacle};

use super::nav::NavPath;
use super::steer::WalkTuning3D;

/// How many consecutive seconds without progress count as stuck.
///
/// Long enough to survive a legitimate pause — squeezing past a crate,
/// turning a tight corner — and short enough that a wedged agent is
/// noticed before a player would call it broken.
pub const STALL_TIMEOUT: f32 = 1.0;

/// How much closer an agent must get, per tick, to count as progressing.
///
/// Not zero: an agent pressed against a wall still drifts by micrometres
/// as the solver nudges it, and a strict `distance < last_distance` would
/// read that as progress and never report stuck.
pub const PROGRESS_EPSILON: f64 = 1e-4;

/// What an agent is currently doing.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum AgentState {
    /// No goal. Applies no force and leaves the body alone.
    #[default]
    Idle,
    Walking,
    /// Reached the goal and stopped steering, so friction and the sleep
    /// system can take over. This is the desired end state, not a
    /// failure.
    Arrived,
    /// No route to the goal exists. The caller decides what to do.
    Blocked,
}

/// Everything one walking agent carries between ticks.
///
/// The path is `Option` rather than an empty `Vec` because "no route was
/// ever planned" and "the route is finished" want different responses,
/// and an empty vector cannot tell them apart.
#[derive(Clone, Debug, Default)]
pub struct Agent3D {
    pub path: Option<NavPath>,
    pub goal: Option<DVec3>,
    pub tuning: WalkTuning3D,
    pub state: AgentState,
    /// Seconds since the agent last made progress toward its waypoint.
    pub stall_time: f32,
    /// What the agent can see right now, for [`crate::avoid::steer_around`].
    ///
    /// The caller refills this each tick from whatever it uses for
    /// proximity — a broadphase sphere query, a tile lookup, its own list.
    /// Empty means "nothing nearby", which is also the state an agent
    /// starts in, so an existing caller that never fills it gets exactly
    /// the behaviour it had before.
    ///
    /// Held on the agent rather than passed to
    /// [`super::drive_agent`] because it is per-agent state that changes
    /// every tick, and because it keeps `drive_agent`'s signature — which
    /// three call sites already use — unchanged.
    pub obstacles: Vec<Obstacle>,
    /// How hard to steer around what is in [`Self::obstacles`].
    pub avoid: AvoidTuning,
    /// Distance to the current waypoint last tick, for the stall check.
    last_distance: f64,
}

impl Agent3D {
    pub fn new(tuning: WalkTuning3D) -> Self {
        Self {
            path: None,
            goal: None,
            tuning,
            state: AgentState::Idle,
            stall_time: 0.0,
            obstacles: Vec::new(),
            avoid: AvoidTuning::default(),
            last_distance: f64::INFINITY,
        }
    }

    /// Give the agent somewhere to be.
    ///
    /// Clears any existing route: the old one led somewhere else, and
    /// keeping it would walk the agent to the previous goal first. The
    /// caller plans the new route with [`super::replan`], which needs a
    /// tile source this type does not hold.
    pub fn set_goal(&mut self, goal: DVec3) {
        self.goal = Some(goal);
        self.path = None;
        self.state = AgentState::Idle;
        self.reset_progress();
    }

    /// Abandon the goal and the route. The body keeps whatever velocity
    /// it had — stopping dead is the solver's business, not the AI's.
    pub fn stop(&mut self) {
        self.goal = None;
        self.path = None;
        self.state = AgentState::Idle;
        self.reset_progress();
    }

    pub fn is_walking(&self) -> bool {
        self.state == AgentState::Walking
    }

    /// Whether the agent has been getting nowhere for [`STALL_TIMEOUT`].
    ///
    /// Reports; does not act. Recovering means re-planning, which needs
    /// the caller's [`crate::pathfind::TileSource`] — so the decision of
    /// what to do about a stuck agent stays with the caller that owns
    /// the world. See [`super::replan`].
    pub fn stuck(&self) -> bool {
        self.stall_time >= STALL_TIMEOUT
    }

    /// Note how far the agent is from its waypoint, and how long it has
    /// been failing to close that gap.
    pub(super) fn record_progress(&mut self, distance: f64, dt: f32) {
        if distance < self.last_distance - PROGRESS_EPSILON {
            self.stall_time = 0.0;
        } else {
            self.stall_time += dt;
        }
        self.last_distance = distance;
    }

    /// Forget the stall history — after a re-plan, or a new goal, the
    /// old distance refers to a waypoint that no longer exists.
    pub(super) fn reset_progress(&mut self) {
        self.stall_time = 0.0;
        self.last_distance = f64::INFINITY;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A new goal must not inherit the old route, or the agent walks to
    /// where it was last told to go before setting off for the new place.
    #[test]
    fn setting_a_goal_clears_the_previous_route() {
        let mut a = Agent3D::new(WalkTuning3D::default());
        a.path = Some(NavPath {
            plane: super::super::nav::NavPlane::new((4, 4), 1.0, 0.0),
            waypoints: vec![(1, 1), (2, 2)],
            cursor: 0,
        });
        a.set_goal(DVec3::new(5.0, 0.0, 0.0));
        assert!(a.path.is_none(), "the stale route should be gone");
        assert_eq!(a.goal, Some(DVec3::new(5.0, 0.0, 0.0)));
    }

    /// Closing the gap resets the clock.
    #[test]
    fn progress_clears_the_stall_clock() {
        let mut a = Agent3D::new(WalkTuning3D::default());
        a.record_progress(5.0, 0.1);
        a.record_progress(4.0, 0.1);
        assert_eq!(a.stall_time, 0.0);
        assert!(!a.stuck());
    }

    /// Getting nowhere accumulates, and eventually reports stuck.
    #[test]
    fn standing_still_eventually_reports_stuck() {
        let mut a = Agent3D::new(WalkTuning3D::default());
        a.record_progress(5.0, 0.1);
        for _ in 0..(STALL_TIMEOUT / 0.1).ceil() as usize {
            a.record_progress(5.0, 0.1);
        }
        assert!(a.stuck(), "an agent that never gets closer is stuck");
    }

    /// The epsilon is load-bearing: an agent pressed against a wall still
    /// creeps by micrometres as the solver nudges it, and a strict
    /// less-than would read that as progress forever.
    #[test]
    fn micrometre_creep_does_not_count_as_progress() {
        let mut a = Agent3D::new(WalkTuning3D::default());
        let mut d = 5.0;
        a.record_progress(d, 0.1);
        for _ in 0..40 {
            d -= PROGRESS_EPSILON * 0.01;
            a.record_progress(d, 0.1);
        }
        assert!(a.stuck(), "creeping by a fraction of the epsilon is not progress");
    }

    /// Re-planning invalidates the recorded distance, which referred to a
    /// waypoint that may no longer exist.
    #[test]
    fn resetting_progress_forgets_the_old_distance() {
        let mut a = Agent3D::new(WalkTuning3D::default());
        a.record_progress(5.0, 0.1);
        // Past the timeout, not merely up to it.
        a.record_progress(5.0, STALL_TIMEOUT);
        assert!(a.stuck());
        a.reset_progress();
        assert!(!a.stuck(), "a fresh route starts with a clean clock");
    }
}
