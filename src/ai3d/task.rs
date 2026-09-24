//! Stacking crates: the layer that decides *where* a walking agent
//! should be, and what to do when it gets there.
//!
//! [`super::drive_agent`] answers "walk to that point". This answers
//! "which point, and why" — fetch a crate, carry it, put it on the pile,
//! repeat until the pile is the height asked for.
//!
//! # The carry is not simulated, and that is deliberate
//!
//! A carried crate becomes [`crate::physics3d::BodyKind::Kinematic`] and
//! is written to a pose in front of the agent every tick. It therefore
//! exerts **no weight on the agent at all**: a kinematic body has zero
//! inverse mass, contributes nothing to the agent's inertia tensor, and
//! the solver treats it as immovable.
//!
//! That is worth saying plainly, because it is why an agent half a metre
//! tall can carry a crate two metres over its head without toppling. The
//! physical alternative — a real constraint between the two bodies —
//! would need a joint solver this engine does not have, and would put the
//! carried mass on the far side of the tipping limit
//! [`super::WalkTuning3D::for_body`] derives. Carrying is a game
//! mechanic here, not a physics one.
//!
//! What is *still* simulated: the agent walking, every crate it bumps on
//! the way, and the crate it releases from the moment it lets go.
//!
//! # This module does not touch the world
//!
//! [`drive_stacker`] returns a [`StackAction`] describing what it wants
//! done rather than doing it, and takes `&Transform3D` rather than
//! `&mut`. Same reason [`super::drive_agent`] does: an AI that can place
//! bodies teleports them through walls. The one transform that must move
//! — the carried crate's — is handed back as a [`CarryPose`] so applying
//! it is an explicit, greppable act in the caller rather than a hidden
//! side effect here.
//!
//! # What this is not
//!
//! The stack is built and then left. Nothing watches it afterwards, so a
//! tower knocked over after [`StackState::Done`] stays down. Repairing it
//! would mean polling the pile forever, which costs work in the state
//! that is meant to cost nothing — see [`StackState::Done`].
//!
//! # Known gaps in the forklift cycle
//!
//! The mast works: a crate is driven to, lifted, hauled and set down,
//! and the worst single-tick movement of a crate fell from 3.63 m to
//! 0.26 m when the teleporting pickup was replaced by it. Two things are
//! still wrong, both visible in `examples/crates3d`:
//!
//! * **The agent barges its own pile.** Driving up to a tower means the
//!   site tile cannot stay blocked during the final approach, and nothing
//!   yet keeps the chassis off the stack it is placing onto — so the
//!   tower gets nudged. The likely fix is to keep the ring blocked for
//!   the *body* and let only the forks overhang, which means deriving the
//!   approach goal from the chassis footprint rather than from the load.
//!
//! * **It does not retry.** After placing one crate the task reaches
//!   [`StackState::Idle`] and stops rather than fetching the next, so the
//!   cycle runs exactly once.
//!
//! Neither is papered over: `crates3d --verify` asserts the one layer the
//! machine currently manages, so the day the cycle restarts that check
//! fails and says so.

use std::collections::HashMap;

use glam::{DVec3, Quat};

use crate::components::Transform3D;
use crate::pathfind::TileSource;
use crate::physics3d::solver::RESTITUTION_THRESHOLD;

use super::agent::{Agent3D, AgentState};
use super::nav::NavPlane;
use super::steer::WalkTuning3D;

/// A handle to one crate in the caller's world.
///
/// **Not an index.** The task holds a reference to a crate across
/// hundreds of ticks, which is exactly the span over which a caller might
/// despawn something and reindex its storage. `examples/crates3d` gets
/// away with slot index == body index only because it never removes a
/// body, and says so in its own comment; the contract here is that ids
/// are stable and indices are not.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct CrateId(pub u32);

/// One candidate crate, as the caller sees it this tick.
///
/// A snapshot by value, so the task never borrows the caller's world and
/// stays testable without one.
#[derive(Clone, Copy, Debug)]
pub struct CrateInfo {
    pub id: CrateId,
    pub pos: DVec3,
    pub half_extents: [f64; 3],
    /// Something else owns this crate — a player is holding it, another
    /// agent is carrying it. The task must not steal it.
    pub carried_by_other: bool,
    /// Asleep, per [`crate::physics3d::RigidBody::sleeping`].
    pub sleeping: bool,
}

impl CrateInfo {
    /// Full height of this crate, which is what one stack layer adds.
    pub fn layer_height(&self) -> f64 {
        self.half_extents[2].abs() * 2.0
    }
}

/// What the stacker is doing.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum StackState {
    /// Nothing chosen, nothing held.
    #[default]
    Idle,
    /// Walking to a crate.
    Fetching,
    /// Stopped at the crate, lowering the forks and sliding them under.
    ///
    /// This is where the *stop* happens, and the stop is most of what
    /// makes a pickup read as a pickup rather than as a cut.
    Engaging,
    /// Forks under the crate, raising it to travel height.
    Lifting,
    /// Walking to the stack with a crate.
    Hauling,
    /// Stopped at the stack, raising the forks to the layer's height.
    ///
    /// There is deliberately no matching `Lowering`: a crate is released
    /// a few centimetres above the layer below and *falls* the rest,
    /// which gravity already animates. A symmetric descent would be the
    /// mast travelling two metres while the agent stands watching it.
    Raising,
    /// Waiting for the crate just released to come to rest.
    Settling,
    /// The stack is the height asked for. Terminal.
    ///
    /// The agent is stopped, so [`super::drive_agent`] applies no force
    /// and — the part that matters — never calls `wake()`. A finished
    /// stacker falls asleep and costs the solver nothing, which is the
    /// same property `an_arrived_agent_falls_asleep` guards for a plain
    /// walker.
    Done,
}

/// What the task wants the caller to do to the world this tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StackAction {
    /// Nothing.
    None,
    /// Make this crate kinematic and put it at `pose`. Zero its velocity
    /// and wake it; see [`StackTuning`] for why both.
    Pickup { id: CrateId, pose: CarryPose },
    /// Still carried: put it at `pose`, velocity still zero.
    Hold { id: CrateId, pose: CarryPose },
    /// Make it dynamic again at `pose` with this velocity and no spin,
    /// and **wake it**.
    Release { id: CrateId, pose: CarryPose, velocity: DVec3 },
    /// Nothing reachable to pick up. Not an error — a crate may appear.
    NoCrateAvailable,
    /// The stack is finished.
    Finished,
}

/// Where a carried crate goes this tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CarryPose {
    pub pos: DVec3,
    pub rot: Quat,
}

/// The numbers, all derived from the bodies involved rather than chosen.
#[derive(Clone, Copy, Debug)]
pub struct StackTuning {
    /// How many crates high to build.
    pub target_layers: u32,
    /// Planar distance at which a crate can be picked up.
    pub pickup_reach: f64,
    /// Planar distance from the stack column at which a crate is placed.
    pub place_reach: f64,
    /// How far in front of the agent's centre the crate rides.
    pub carry_forward: f64,
    /// How far **above the floor** the fork tops travel while hauling.
    ///
    /// A height in the world, not an offset from the agent. An offset
    /// makes the cargo's height depend on where the solver happens to
    /// have left the body this tick, so a crate two metres overhead
    /// jitters with every contact under the wheels.
    pub carry_up: f64,
    /// How fast the mast travels, in metres per second.
    ///
    /// One crate height per second. Expressed against the crate rather
    /// than as a bare number because what matters is how long the cargo
    /// takes to cross its own size: faster than that and consecutive
    /// frames show it in places that do not overlap, which is what a
    /// teleport looks like even when the motion is continuous.
    ///
    /// Twice a real forklift's ~0.5 m/s, and that is deliberate. At
    /// 0.5 m/s the three-layer build spends 6.2 seconds with the agent
    /// standing still watching its own mast, which is most of the
    /// `--verify` budget and most of a viewer's patience.
    pub lift_rate: f64,
    /// Height of the fork tops when parked, in metres above the floor.
    ///
    /// The tine's own thickness: parking here puts the tine *bottoms* on
    /// the floor, so a crate lifted from rest rises by exactly the depth
    /// of the fork that went under it — which is what a real forklift
    /// does, and is 60 mm rather than the 2.05 m it used to jump.
    pub fork_rest_height: f64,
    /// How long one mast move may take before the crate is given up on.
    ///
    /// Three times the longest legitimate move, so a mast merely slowed
    /// is not mistaken for one that has hung. It exists to catch a NaN
    /// target or a zero rate, not to police pace.
    pub lift_timeout: f32,
    /// Height above the layer below from which a crate is released.
    ///
    /// Bounded above by bouncing, not by taste. A crate released downward
    /// at [`RELEASE_SPEED`] from height `h` arrives at
    /// `sqrt(v0² + 2gh)` — energies add, speeds do not — and past
    /// [`RESTITUTION_THRESHOLD`] the solver applies restitution, so the
    /// crate bounces off the stack instead of settling on it. That caps
    /// the drop at `(thr² - v0²) / 2g` = 46 mm; this takes seven tenths
    /// of it, 33 mm, landing at 0.85 m/s. It is still six times
    /// the solver's own penetration slop, so a crate is never released already
    /// overlapping what it lands on.
    pub drop_clearance: f64,
    /// How far off the column a crate may settle and still count.
    pub settle_xy_tolerance: f64,
    /// How far off nominal height a settled layer may be.
    pub settle_z_tolerance: f64,
    /// How long to wait for a released crate before giving up on it.
    pub settle_timeout: f32,
    /// How far the agent may lean before a carried crate is dropped.
    pub max_carry_tilt_deg: f64,
    /// How long an abandoned crate is ignored for.
    pub blacklist_time: f32,
}

/// How thick the fork tines are, in metres.
///
/// Shared with the caller's mesh: the tines have to be drawn this thick
/// or the crate visibly floats above them. It is also
/// [`StackTuning::fork_rest_height`], because parking the mast at exactly
/// the tine thickness puts the tine bottoms on the floor.
pub const FORK_THICKNESS: f64 = 0.06;

/// The mast: where the forks are, and where they are going.
///
/// Separate from [`StackTuning`] because these change every tick and the
/// tuning does not, and separate from [`StackState`] because the mast
/// keeps travelling across several states rather than restarting at each
/// transition.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Forks {
    /// Height of the tine *top* faces above the floor plane, in metres.
    ///
    /// That is the surface a crate rests on, so a carried crate's centre
    /// sits at `floor_z + height + crate_half_z`.
    pub height: f64,
    /// Where the mast is driving to. Nothing writes [`Forks::height`]
    /// directly; it slews toward this at [`StackTuning::lift_rate`].
    pub target: f64,
}

impl Forks {
    /// Advance the mast one tick. Returns whether it has arrived.
    ///
    /// Rate-limited rather than eased. An ease looks smoother on a graph
    /// and is wrong here: the point of the mast is a *bounded* per-tick
    /// displacement, and an ease's peak rate is higher than its average,
    /// so the number a test can assert against stops being `lift_rate`.
    pub fn step(&mut self, rate: f64, dt: f32) -> bool {
        let delta = self.target - self.height;
        // A NaN target would make `delta.signum()` NaN and poison the
        // mast forever, so it is treated as "already arrived" and the
        // lift timeout catches the stall.
        if !delta.is_finite() {
            return true;
        }
        let step = (rate * dt as f64).abs();
        if delta.abs() <= step {
            self.height = self.target;
            return true;
        }
        self.height += step * delta.signum();
        false
    }

    /// Whether the mast is where it was asked to be.
    pub fn settled(&self) -> bool {
        !(self.target - self.height).is_finite() || (self.target - self.height).abs() <= 1e-9
    }
}

/// The mast height that puts a crate's centre at layer `layer`'s drop
/// height — the tine top, which is the crate's bottom face.
pub fn fork_height_for_layer(
    plane: NavPlane,
    layer: u32,
    crate_half_z: f64,
    tuning: StackTuning,
) -> f64 {
    drop_height_for_layer(plane, layer, crate_half_z, tuning) - plane.floor_z - crate_half_z.abs()
}

/// How fast a crate is pushed down as it is released.
///
/// Not zero: a crate let go at rest drifts, and a drifting crate is one
/// the agent can walk back into before it lands. Small enough that the
/// landing still stays under [`RESTITUTION_THRESHOLD`] — see
/// [`StackTuning::drop_clearance`], which is derived from this.
pub const RELEASE_SPEED: f64 = 0.3;

impl StackTuning {
    /// Derive the numbers from the agent, the crate and the gait.
    ///
    /// Almost nothing here is a free choice. `carry_up` has to clear the
    /// finished stack or the agent mows down its own tower with its
    /// cargo; `drop_clearance` has to stay under the bounce threshold;
    /// the settle tolerances have to admit the overlap the solver
    /// deliberately leaves.
    pub fn for_agent_and_crate(
        agent_half: [f64; 3],
        crate_half: [f64; 3],
        tile_size_m: f64,
        target_layers: u32,
        walk: WalkTuning3D,
    ) -> Self {
        let crate_height = crate_half[2].abs() * 2.0;

        // Far enough forward that the agent and its cargo never touch.
        //
        // The minimum is the two half-extents, which puts the faces
        // exactly flush — and flush is not enough. `obb_contact_manifold`
        // accepts points up to its own contact tolerance *outside* a
        // boundary, so touching faces still generate manifold points and
        // the solver would resolve a contact between the agent and the
        // thing it is holding every tick. A 5 cm margin is two orders
        // above that tolerance.
        const CARRY_GAP: f64 = 0.05;
        let carry_forward = agent_half[0].abs() + crate_half[0].abs() + CARRY_GAP;

        // High enough that the cargo clears the tallest the stack will
        // ever be. Without this the agent walks the crate it is carrying
        // straight through the layers already placed on the way to
        // putting it on top.
        //
        // This is now a height above the **floor**, which is what the
        // mast tracks, rather than an offset from the agent's centre.
        // Losing the `- agent_half` term is not a tweak: an offset from a
        // body the solver is free to shove up and down makes the cargo
        // height depend on contact noise, so a crate two metres overhead
        // twitches every time a wheel rides over something.
        let stack_top = (target_layers.saturating_sub(1)) as f64 * crate_height;
        let carry_up = stack_top + CARRY_GAP;

        // The drop cap, derived rather than picked. A crate released
        // downward at `RELEASE_SPEED` from height `h` arrives at
        // `sqrt(v0² + 2gh)` — energies add, not speeds — so the height
        // that keeps the landing under the bounce threshold is
        // `(thr² - v0²) / 2g`. Seven tenths of it leaves margin.
        let g = crate::physics3d::GRAVITY.z.abs();
        let bounce_cap = (RESTITUTION_THRESHOLD * RESTITUTION_THRESHOLD
            - RELEASE_SPEED * RELEASE_SPEED)
            / (2.0 * g);
        let drop_clearance = bounce_cap * 0.7;

        Self {
            target_layers,
            // Reach is set by the **approach tile**, not by the radii.
            //
            // The agent is sent to a tile *beside* its target rather than
            // onto it, so it deliberately stops a tile away — and because
            // the goal snaps to a tile centre, the true standoff can be
            // half a tile more than that, plus the walker's own
            // `goal_radius` on top. Deriving this from `arrive_radius`
            // instead gave 1.30 m against a measured 2.49 m standoff, so
            // the agent walked to its goal and sat there reporting the
            // crate out of reach forever.
            pickup_reach: tile_size_m * 1.5 + walk.goal_radius + crate_half[0].abs(),
            // Placing stands further off than fetching does, because the
            // ring of tiles around the stack is blocked — see
            // `blocked_tiles`. The nearest tile the agent can legally
            // stand on is two out, and the goal snaps to a tile centre,
            // so the standoff is two tiles plus half a tile of snap plus
            // the walker's own acceptance radius.
            place_reach: tile_size_m * 1.5 + walk.goal_radius + crate_half[0].abs(),
            carry_forward,
            carry_up,
            drop_clearance,
            // These say "is this still a tower", not "is it perfect".
            //
            // A box stays up while its centre of mass is over what holds
            // it, so three quarters of a half-extent still leaves the
            // crate genuinely supported, and a real stack drifts that far
            // as it settles.
            settle_xy_tolerance: crate_half[0].abs() * 0.75,
            // Deriving this from `PENETRATION_SLOP` was wrong, and
            // measurably so. The solver resolves only the *excess* beyond
            // the slop, at `BAUMGARTE` per tick, so a loaded contact
            // reaches an equilibrium overlap well above it — the crate on
            // top presses down continuously. Measured on a settled
            // three-high stack: 10 mm, 60 mm, 90 mm below nominal, so a
            // slop-derived 30 mm rejected two thirds of a stack that was
            // standing perfectly well.
            //
            // A quarter of a layer height is the honest question instead:
            // past that a crate is not resting on the one below, it is
            // somewhere else.
            settle_z_tolerance: crate_height * 0.25,
            // Dominated by `TIME_TO_SLEEP` (0.5 s), plus the fall, plus
            // however long the crate rocks before it settles.
            settle_timeout: crate::physics3d::TIME_TO_SLEEP * 6.0,
            // A healthy walk measures a tenth of a degree. This is 150
            // times that — far past anything walking produces, and well
            // short of horizontal, so a shoved agent drops its cargo
            // while the cargo is still roughly overhead.
            max_carry_tilt_deg: 15.0,
            // Longer than the walker's own stall timeout, so a crate
            // abandoned for stalling is not immediately re-picked.
            blacklist_time: super::STALL_TIMEOUT * 3.0,
            // One crate height per second. See the field docs.
            lift_rate: crate_height,
            // The tine thickness, so the tines rest *on* the floor.
            fork_rest_height: FORK_THICKNESS,
            // Three times the longest move the machine ever makes, which
            // is floor to the top layer.
            lift_timeout: ((stack_top + CARRY_GAP) / crate_height * 3.0) as f32,
        }
    }
}

/// The crate currently being carried.
#[derive(Clone, Copy, Debug)]
pub struct Carry {
    pub id: CrateId,
    pub half_extents: [f64; 3],
}

/// The stacker's memory between ticks.
#[derive(Clone, Debug)]
pub struct StackTask {
    pub state: StackState,
    pub tuning: StackTuning,
    /// The column the stack is built in, fixed on the first placement.
    ///
    /// Recomputing it per layer makes the stack wander and fall over, so
    /// once a layer is down this does not move.
    pub site: Option<(i32, i32)>,
    pub carry: Option<Carry>,
    pub target: Option<CrateId>,
    /// Which crates are already part of the stack, so they are never
    /// picked back up.
    pub placed: Vec<CrateId>,
    /// Crates recently given up on, and how long is left on each.
    blacklist: HashMap<CrateId, f32>,
    /// Seconds spent in [`StackState::Settling`].
    settle_clock: f32,
    /// The direction the cargo is held in, slewed rather than snapped.
    facing: DVec3,
    /// The mast. Public because the caller draws the forks.
    pub forks: Forks,
    /// Seconds spent on the current mast move.
    ///
    /// Its own clock rather than sharing `settle_clock`: a lift and a
    /// settle can never overlap, but sharing one would make a forgotten
    /// reset a silent hang instead of a visible failure.
    lift_clock: f32,
    /// Where the cargo is being slid to while the forks engage.
    ///
    /// A crate does not snap sideways onto the tines any more than it
    /// snaps upward — it slides on, at the same rate the mast lifts.
    ///
    /// Progress is a *fraction*, not a distance to a target. The carry
    /// point moves with the agent every tick, so a slide that chased it
    /// never converged and `Lifting` hung for hundreds of frames.
    slide_from: Option<DVec3>,
    slide_t: f64,
    /// Whether the haul has switched from "walk to the tile beside the
    /// stack" to "line the load up over it".
    ///
    /// One-shot, because the goal it sets clears the agent's path:
    /// setting it every tick re-plans from scratch every tick and the
    /// agent never takes a step.
    final_approach: bool,
}

impl StackTask {
    pub fn new(tuning: StackTuning) -> Self {
        Self {
            state: StackState::Idle,
            tuning,
            site: None,
            carry: None,
            target: None,
            placed: Vec::new(),
            blacklist: HashMap::new(),
            settle_clock: 0.0,
            facing: DVec3::X,
            // Parked, which means the tine bottoms are on the floor.
            forks: Forks {
                height: tuning.fork_rest_height,
                target: tuning.fork_rest_height,
            },
            lift_clock: 0.0,
            slide_from: None,
            slide_t: 1.0,
            final_approach: false,
        }
    }

    /// The horizontal direction the agent is heading, as a unit vector.
    ///
    /// Worth exposing because it is the only honest answer to "which way
    /// is this thing pointing". The body's own rotation is not: nothing
    /// ever yaws it deliberately — it is a box shoved along by a
    /// centre-of-mass force — so its orientation is whatever the contact
    /// solver last left it at, and a mesh oriented from it spins on the
    /// spot. This is slewed rather than snapped, so it turns corners
    /// smoothly instead of flicking between the grid's four directions.
    pub fn facing(&self) -> DVec3 {
        self.facing
    }

    pub fn is_done(&self) -> bool {
        self.state == StackState::Done
    }

    /// How many layers are standing, counted from the world rather than
    /// trusted from a tally.
    ///
    /// A counter drifts the moment a stack topples; the crates are the
    /// truth. Counting rather than remembering is what makes the task
    /// recover from a knocked-over pile on its own.
    pub fn layers_standing(&self, plane: NavPlane, crates: &[CrateInfo]) -> u32 {
        let Some(site) = self.site else { return 0 };
        let Some(h) = self.layer_height(crates) else { return 0 };

        // Each layer is measured against **the one below it**, not
        // against the original column.
        //
        // A settled tower shifts as a unit — the bottom crate creeps as
        // it beds in and everything above rides along with it. Measuring
        // every layer against the column where the stack was started
        // therefore reports a perfectly good three-high tower as zero
        // layers once the base has drifted past the tolerance, which is
        // what it did: 0.49/1.43/2.42 stacked squarely on each other,
        // counted as nothing, because the base had moved 0.57 m.
        //
        // Chaining asks the question that matters: is this crate resting
        // on that one?
        // The base layer is found by height alone, not by position.
        //
        // A settled tower creeps as it beds in — measured at 0.43 m from
        // the tile it was started on, against a 0.375 m tolerance — and
        // anchoring the chain to that tile reports a perfectly good
        // three-high stack as zero layers. Which tile the stack began on
        // stops being interesting the moment the first crate lands; what
        // matters from then on is that each crate is on the one below.
        //
        // Only crates this task actually *placed* can be the base. A
        // loose crate lying on the floor is at exactly the base layer's
        // height, so height alone counts the scrap in the yard as a
        // tower: with three crates on the ground the count came back as
        // two, the mast raised to the third layer, and the very first
        // crate was set down from two and a half metres up.
        let mut column = crates
            .iter()
            .filter(|c| {
                !c.carried_by_other
                    && self.placed.contains(&c.id)
                    && (c.pos.z - (plane.floor_z + h * 0.5)).abs()
                        <= self.tuning.settle_z_tolerance
            })
            .min_by(|a, b| {
                let da = plane.flatten(a.pos - plane.tile_center(site.0, site.1)).length();
                let db = plane.flatten(b.pos - plane.tile_center(site.0, site.1)).length();
                da.total_cmp(&db)
            })
            .map(|c| DVec3::new(c.pos.x, c.pos.y, 0.0))
            .unwrap_or_else(|| plane.tile_center(site.0, site.1));
        let mut n = 0u32;
        while n < self.tuning.target_layers {
            let expect = plane.floor_z + n as f64 * h + h * 0.5;
            let found = crates.iter().find(|c| {
                !c.carried_by_other
                    && self.placed.contains(&c.id)
                    && plane.flatten(c.pos - column).length() <= self.tuning.settle_xy_tolerance
                    && (c.pos.z - expect).abs() <= self.tuning.settle_z_tolerance
            });
            let Some(c) = found else { break };
            // The next layer is measured against where this one actually
            // is, so the tower is allowed to lean without being
            // discounted.
            column = DVec3::new(c.pos.x, c.pos.y, column.z);
            n += 1;
        }
        n
    }

    /// Where the top crate of the stack is, if there is one.
    ///
    /// Used to aim the next drop, so the tower is built on itself rather
    /// than on the spot it was started from.
    pub fn top_of_stack(&self, plane: NavPlane, crates: &[CrateInfo]) -> Option<DVec3> {
        let site = self.site?;
        let h = self.layer_height(crates)?;
        let mut column = plane.tile_center(site.0, site.1);
        let mut found = None;
        for n in 0..self.tuning.target_layers {
            let expect = plane.floor_z + n as f64 * h + h * 0.5;
            let hit = crates.iter().find(|c| {
                !c.carried_by_other
                    && plane.flatten(c.pos - column).length() <= self.tuning.settle_xy_tolerance
                    && (c.pos.z - expect).abs() <= self.tuning.settle_z_tolerance
            })?;
            column = DVec3::new(hit.pos.x, hit.pos.y, column.z);
            found = Some(column);
        }
        found
    }

    /// The layer height, taken from whatever crates exist.
    fn layer_height(&self, crates: &[CrateInfo]) -> Option<f64> {
        crates.first().map(|c| c.layer_height())
    }

    /// Put the task back to the start, keeping its tuning.
    pub fn restart(&mut self) {
        let tuning = self.tuning;
        *self = Self::new(tuning);
    }

    /// Whether the cargo has finished sliding onto the forks.
    fn slide_done(&self) -> bool {
        self.slide_from.is_none() || self.slide_t >= 1.0
    }

    fn blacklisted(&self, id: CrateId) -> bool {
        self.blacklist.contains_key(&id)
    }

    fn blacklist_crate(&mut self, id: CrateId) {
        self.blacklist.insert(id, self.tuning.blacklist_time);
    }

    fn tick_blacklist(&mut self, dt: f32) {
        self.blacklist.retain(|_, t| {
            *t -= dt;
            *t > 0.0
        });
    }

    /// Give up on whatever is being carried or fetched.
    ///
    /// Every failure path routes through here, because the invariant
    /// "at most one crate is kinematic because of this task" has to be
    /// restored from all of them, and a missed path leaks a permanently
    /// kinematic crate — an invisible, immovable obstacle that nothing
    /// in a scene check would attribute to the stacker.
    fn abandon(&mut self, agent: &mut Agent3D) {
        self.target = None;
        self.state = StackState::Idle;
        agent.stop();
    }
}

/// Where the cargo rides, in world space.
///
/// `facing` is the direction the crate is held in. It comes from where
/// the agent is *going*, never from its rotation: nothing ever yaws the
/// agent deliberately — it is a box pushed by a centre-of-mass force — so
/// its orientation is whatever the contact solver last left it at, and
/// deriving "forward" from it points the cargo in an arbitrary direction.
/// The height comes from the **mast and the floor**, never from the
/// agent's own z. An agent bounced twenty millimetres by a contact under
/// its tracks must not bounce a crate two metres overhead with it.
pub fn hold_pose(
    agent_pos: DVec3,
    facing: DVec3,
    forks: Forks,
    crate_half_z: f64,
    floor_z: f64,
    tuning: StackTuning,
) -> CarryPose {
    let f = if facing.length_squared() > 1e-12 {
        DVec3::new(facing.x, facing.y, 0.0).normalize_or_zero()
    } else {
        DVec3::X
    };
    let planar = agent_pos + f * tuning.carry_forward;
    CarryPose {
        pos: DVec3::new(
            planar.x,
            planar.y,
            // The tine top is the crate's bottom face.
            floor_z + forks.height + crate_half_z.abs(),
        ),
        // Square-on for the whole carry, not just at the drop. A crate
        // released even slightly rotated lands on a corner, which falls
        // out of `obb_contact_manifold`'s face-clipping path into its
        // single-point fallback, and a single contact point on a lever
        // arm is what makes a box spin instead of settle.
        rot: Quat::IDENTITY,
    }
}

/// The height a crate's centre must be released from to make layer `n`.
///
/// Derived from the nominal crate height and the layer index, **not**
/// from the measured top of the pile. A crate that settled two
/// millimetres into its neighbour — which the solver leaves on purpose,
/// up to [`crate::physics3d::solver::PENETRATION_SLOP`] — would drag the
/// next drop two millimetres
/// low, and the error would compound up the stack. Measuring is used only
/// to decide whether a layer has settled, never to place the next one.
pub fn drop_height_for_layer(
    plane: NavPlane,
    layer: u32,
    crate_half_z: f64,
    tuning: StackTuning,
) -> f64 {
    let h = crate_half_z.abs() * 2.0;
    plane.floor_z + layer as f64 * h + crate_half_z.abs() + tuning.drop_clearance
}

/// Whether the layer below is genuinely at rest.
///
/// Tests `sleeping`, not "is it moving right now". A crate at the top of
/// a small bounce is instantaneously still and would pass a velocity
/// check; `sleeping` requires half a second of continuous stillness
/// measured *after* the solve, which is exactly the question being asked,
/// and the physics layer already computes it every tick.
pub fn layer_has_settled(
    column: DVec3,
    expected_z: f64,
    crates: &[CrateInfo],
    tuning: StackTuning,
) -> bool {
    crates.iter().any(|c| {
        !c.carried_by_other
            && c.sleeping
            && DVec3::new(c.pos.x - column.x, c.pos.y - column.y, 0.0).length()
                <= tuning.settle_xy_tolerance
            && (c.pos.z - expected_z).abs() <= tuning.settle_z_tolerance
    })
}

/// Advance the stacking behaviour by one tick.
///
/// Runs **before** [`super::drive_agent`], and so before
/// [`crate::physics3d::step`]: this layer decides where the agent wants
/// to be, and the walker then decides how hard to push to get there.
/// Running it afterwards costs a tick of latency on every goal change and
/// picks goals from a pose the agent no longer has.
///
/// The returned [`StackAction`] must be applied by the caller *before*
/// the broadphase is re-hashed, or the narrowphase sees last tick's
/// position for the one body that moved furthest.
pub fn drive_stacker<T: TileSource>(
    task: &mut StackTask,
    agent: &mut Agent3D,
    agent_transform: &Transform3D,
    plane: NavPlane,
    src: &T,
    crates: &[CrateInfo],
    dt: f32,
) -> StackAction {
    task.tick_blacklist(dt);

    if task.state == StackState::Done {
        return StackAction::Finished;
    }

    let pos = agent_transform.pos;

    // Keep the facing filtered rather than snapped. A 4-connected route
    // turns square corners, and snapping the hold direction swings the
    // cargo through an arc of twice the carry offset in a single tick.
    //
    // While the forks are going under a crate, face the **crate** rather
    // than the path. The path is finished by then — the agent has arrived
    // and stopped — so following it leaves the robot pointing whichever
    // way it happened to be walking, which is usually across the crate
    // rather than at it, and the forks reach out sideways past it.
    let engaging_target = matches!(task.state, StackState::Engaging)
        .then(|| task.target)
        .flatten()
        .and_then(|id| crates.iter().find(|c| c.id == id))
        .map(|c| plane.flatten(c.pos - pos));

    // And once the haul is close to the stack, face the **column**. The
    // path leads to a tile beside the tower, not to the tower, so
    // following it leaves the forks pointing past it — and the cargo,
    // which rides out along the forks, lands anywhere but on the pile.
    let approach_target = matches!(task.state, StackState::Hauling | StackState::Raising)
        .then(|| task.site)
        .flatten()
        .map(|s| plane.tile_center(s.0, s.1))
        .filter(|c| plane.flatten(*c - pos).length() <= task.tuning.place_reach)
        .map(|c| plane.flatten(c - pos));

    let want = engaging_target
        .or(approach_target)
        .or_else(|| {
            agent
                .path
                .as_ref()
                .and_then(|p| p.next_world())
                .map(|w| plane.flatten(w - pos))
        })
        .filter(|v| v.length_squared() > 1e-6)
        .unwrap_or(task.facing);
    let slew = (dt as f64 / 0.2).clamp(0.0, 1.0);
    task.facing = (task.facing + (want.normalize_or_zero() - task.facing) * slew)
        .normalize_or_zero();
    if task.facing.length_squared() < 1e-12 {
        task.facing = DVec3::X;
    }

    match task.state {
        StackState::Done => StackAction::Finished,

        StackState::Idle => {
            // Finished? Count the pile rather than trusting a tally.
            if task.site.is_some() && task.layers_standing(plane, crates) >= task.tuning.target_layers
            {
                task.state = StackState::Done;
                agent.stop();
                return StackAction::Finished;
            }

            let Some(next) = choose_crate(task, plane, pos, crates) else {
                return StackAction::NoCrateAvailable;
            };

            // Walk to the tile *next to* the crate, not onto it: a goal
            // inside a solid object is a goal the agent grinds against
            // forever, because the crate is 1 m across and the walker's
            // acceptance radius is under half that.
            let approach = approach_tile(plane, pos, next.pos);
            agent.set_goal(plane.tile_center(approach.0, approach.1));
            // **Fetching ignores the ring.** A loose crate can easily be
            // lying inside it — the agent drops them nearby as it works —
            // and routing around a blocked ring to reach a crate *inside*
            // that ring is impossible, so the agent reports Blocked and
            // gives up on a crate it is standing next to. Measured: the
            // task stalled at one layer with two crates a metre away.
            //
            // The ring exists to stop the agent clipping the tower while
            // *carrying*, which is when the collision actually matters.
            if !super::replan(agent, plane, src, pos, &Default::default()) {
                task.blacklist_crate(next.id);
                return StackAction::NoCrateAvailable;
            }
            task.target = Some(next.id);
            task.state = StackState::Fetching;
            StackAction::None
        }

        StackState::Fetching => {
            let Some(id) = task.target else {
                task.state = StackState::Idle;
                return StackAction::None;
            };
            // Gone: despawned, launched, or fallen off the world.
            let Some(info) = crates.iter().find(|c| c.id == id).copied() else {
                task.abandon(agent);
                return StackAction::None;
            };
            if info.carried_by_other {
                task.blacklist_crate(id);
                task.abandon(agent);
                return StackAction::None;
            }

            // Close enough, and on the floor rather than mid-air.
            let reach = plane.flatten(info.pos - pos).length();
            let grounded = (pos.z - plane.floor_z).abs() <= agent.tuning.ground_tolerance;
            if reach <= task.tuning.pickup_reach && grounded {
                // Stop and put the forks down. The crate stays dynamic
                // and stays where it is: it is not picked up until the
                // tines are actually under it.
                agent.stop();
                task.forks.target = task.tuning.fork_rest_height;
                task.lift_clock = 0.0;
                task.state = StackState::Engaging;
                return StackAction::None;
            }

            // Getting nowhere, or nowhere to get to.
            if agent.stuck() || agent.state == AgentState::Blocked {
                task.blacklist_crate(id);
                task.abandon(agent);
            }
            StackAction::None
        }

        StackState::Engaging => {
            let Some(id) = task.target else {
                task.state = StackState::Idle;
                return StackAction::None;
            };
            let Some(info) = crates.iter().find(|c| c.id == id).copied() else {
                // Nothing is held yet — the crate was dynamic all the way
                // through — so there is nothing to put down.
                task.abandon(agent);
                return StackAction::None;
            };

            task.lift_clock += dt;
            let arrived = task.forks.step(task.tuning.lift_rate, dt);

            // Turned to face it, as well as forks down. The robot arrives
            // walking in whatever direction its last path leg ran, which
            // is usually across the crate rather than at it — and forks
            // that reach out sideways past the thing they are supposed to
            // be going under look exactly as wrong as they are.
            let aimed = crates
                .iter()
                .find(|c| c.id == id)
                .map(|c| plane.flatten(c.pos - pos))
                .filter(|v| v.length_squared() > 1e-9)
                .map(|v| v.normalize_or_zero().dot(task.facing) > 0.985)
                .unwrap_or(true);

            if (!arrived || !aimed) && task.lift_clock < task.tuning.lift_timeout {
                return StackAction::None;
            }
            if task.lift_clock >= task.tuning.lift_timeout {
                task.blacklist_crate(id);
                task.abandon(agent);
                return StackAction::None;
            }

            // The tines are down and under it. *Now* it is cargo: it goes
            // kinematic on this edge and not before, so it sits on the
            // floor under gravity and under contacts right up to the
            // moment the forks take its weight.
            task.carry = Some(Carry { id, half_extents: info.half_extents });
            task.target = None;
            task.slide_from = Some(info.pos);
            task.slide_t = 0.0;
            task.final_approach = false;
            task.lift_clock = 0.0;
            task.forks.target =
                fork_height_for_layer(plane, task.layers_standing(plane, crates), info.half_extents[2], task.tuning)
                    .max(task.tuning.fork_rest_height);
            task.state = StackState::Lifting;

            // Plan the haul *before* committing to the lift, and give the
            // crate straight back if there is nowhere to take it. The
            // return value used to be dropped here, so a failed plan left
            // the agent holding a crate until `stuck()` fired a second
            // later.
            let site = ensure_site(task, plane, pos, crates);
            let approach =
                approach_tile_at(plane, pos, plane.tile_center(site.0, site.1), 1);
            agent.set_goal(plane.tile_center(approach.0, approach.1));
            if !super::replan(agent, plane, src, pos, &task.blocked_tiles(plane)) {
                task.carry = None;
                task.slide_from = None;
                task.blacklist_crate(id);
                task.abandon(agent);
                return StackAction::None;
            }

            StackAction::Pickup {
                id,
                pose: slid_pose(task, pos, info.half_extents[2], plane.floor_z),
            }
        }

        StackState::Lifting => {
            let Some(carry) = task.carry else {
                task.state = StackState::Idle;
                return StackAction::None;
            };
            if !crates.iter().any(|c| c.id == carry.id) {
                // The crate vanished mid-lift. Deliberately *not* a
                // release: there is no body to hand back, and emitting
                // one for a despawned id leaves the caller's lookup
                // returning `None` — so the crate would come back still
                // kinematic if it ever reappeared.
                task.carry = None;
                task.slide_from = None;
                task.blacklist_crate(carry.id);
                task.abandon(agent);
                return StackAction::None;
            }

            let pose = slid_pose(task, pos, carry.half_extents[2], plane.floor_z);

            // Tipped over while lifting: put it down where it is rather
            // than swinging it around.
            let up = (agent_transform.rot * glam::Vec3::Z).as_dvec3();
            let tilt = up.dot(DVec3::Z).clamp(-1.0, 1.0).acos().to_degrees();
            if tilt > task.tuning.max_carry_tilt_deg {
                return release_here(task, agent, pose);
            }

            task.lift_clock += dt;
            if task.lift_clock >= task.tuning.lift_timeout {
                return release_here(task, agent, pose);
            }

            // Advance the slide at the same rate the mast lifts, so the
            // two read as one machine rather than two.
            if let Some(from) = task.slide_from {
                let want = pos + task.facing * task.tuning.carry_forward;
                let span = plane.flatten(want - from).length().max(1e-6);
                task.slide_t =
                    (task.slide_t + task.tuning.lift_rate * dt as f64 / span).min(1.0);
            }

            if task.forks.step(task.tuning.lift_rate, dt) && task.slide_done() {
                task.slide_from = None;
                task.lift_clock = 0.0;
                task.state = StackState::Hauling;
            }
            StackAction::Hold { id: carry.id, pose }
        }

        StackState::Raising => {
            let Some(carry) = task.carry else {
                task.state = StackState::Idle;
                return StackAction::None;
            };
            if !crates.iter().any(|c| c.id == carry.id) {
                task.carry = None;
                task.slide_from = None;
                task.abandon(agent);
                return StackAction::None;
            }
            let Some(site) = task.site else {
                task.state = StackState::Idle;
                return StackAction::None;
            };

            let pose = slid_pose(task, pos, carry.half_extents[2], plane.floor_z);
            let up = (agent_transform.rot * glam::Vec3::Z).as_dvec3();
            let tilt = up.dot(DVec3::Z).clamp(-1.0, 1.0).acos().to_degrees();
            if tilt > task.tuning.max_carry_tilt_deg {
                return release_here(task, agent, pose);
            }

            // Re-read the layer count every tick rather than latching it
            // on entry. A tower knocked from two layers to none while the
            // mast is travelling would otherwise get its next crate
            // placed at 2.5 m and dropped two metres onto the floor.
            let layer = task.layers_standing(plane, crates);
            task.forks.target =
                fork_height_for_layer(plane, layer, carry.half_extents[2], task.tuning);

            task.lift_clock += dt;
            let timed_out = task.lift_clock >= task.tuning.lift_timeout;
            if !task.forks.step(task.tuning.lift_rate, dt) && !timed_out {
                return StackAction::Hold { id: carry.id, pose };
            }
            if timed_out {
                return release_here(task, agent, pose);
            }

            // At height. Let go **from the forks**, not from the column.
            //
            // Releasing at the column is what made the crate jump at this
            // end of the haul: the agent stands off the stack, so the
            // cargo was teleported the remaining distance to the tower on
            // the frame it was let go. A forklift sets its load down where
            // its forks are; if the forks are not over the tower, the
            // agent has not driven close enough, and that is a placement
            // problem rather than something to paper over by flinging.
            let column = plane.tile_center(site.0, site.1);
            let drop = CarryPose {
                pos: DVec3::new(pose.pos.x, pose.pos.y, pose.pos.z),
                rot: Quat::IDENTITY,
            };
            let _ = layer;
            task.carry = None;
            task.slide_from = None;
            task.placed.push(carry.id);
            task.settle_clock = 0.0;
            task.lift_clock = 0.0;
            task.state = StackState::Settling;

            let back = approach_tile_at(plane, pos, column, 1);
            agent.set_goal(plane.tile_center(back.0, back.1));
            super::replan(agent, plane, src, pos, &task.blocked_tiles(plane));

            StackAction::Release {
                id: carry.id,
                pose: drop,
                velocity: DVec3::new(0.0, 0.0, -RELEASE_SPEED),
            }
        }

        StackState::Hauling => {
            let Some(carry) = task.carry else {
                task.state = StackState::Idle;
                return StackAction::None;
            };
            let pose = slid_pose(task, pos, carry.half_extents[2], plane.floor_z);

            // Tipped over: the hold point has swung out over open space
            // and the cargo is sweeping the scene sideways. Put it down
            // where it is rather than carrying on.
            let up = (agent_transform.rot * glam::Vec3::Z).as_dvec3();
            let tilt = up.dot(DVec3::Z).clamp(-1.0, 1.0).acos().to_degrees();
            if tilt > task.tuning.max_carry_tilt_deg {
                return release_here(task, agent, pose);
            }

            let Some(site) = task.site else {
                task.state = StackState::Idle;
                return StackAction::None;
            };
            let column = plane.tile_center(site.0, site.1);
            let layer = task.layers_standing(plane, crates);

            // Place when the *agent* is close enough, not when the cargo
            // happens to be over the column.
            //
            // Gating on the cargo looks more precise and is much worse:
            // the agent is sent to a tile beside the site, so whether its
            // cargo lands over the column depends on it stopping at
            // exactly `carry_forward` while facing the right way. Missing
            // that window means it never places at all — measured, the
            // agent hauled a crate in circles indefinitely with the cargo
            // 0.45 m from a 0.25 m tolerance.
            //
            // The drop pose puts the crate on the column regardless, so
            // the cargo's own position at the moment of release decides
            // nothing.
            let offset = plane.flatten(pos - column).length();
            if offset <= task.tuning.place_reach {
                // Close enough to aim: drive at a standing spot one
                // `carry_forward` short of the tower, so the forks — and
                // the cargo riding on them — end up over the pile.
                //
                // The direction is taken from where the agent *is*, once,
                // rather than from `facing`. Deriving it from `facing`
                // makes the goal move as the robot turns toward it, so
                // the agent chases a point that keeps sliding away and
                // the haul never finishes. Measured: stuck in `Hauling`
                // for the whole run.
                //
                // Set **once**, on the tick the approach begins.
                // `set_goal` clears the agent's path, so calling it every
                // tick re-plans from scratch every tick and the agent
                // never takes a single step — measured as a haul that ran
                // for the entire scene without arriving.
                if !task.final_approach {
                    task.final_approach = true;
                    //
                    // Aim *through* the tower rather than at a standing
                    // spot short of it, and stop on the cargo instead.
                    //
                    // Picking a spot means predicting where the walker
                    // will come to rest, and it does not stop on its goal
                    // — it stops within `goal_radius`, on whichever side
                    // it happened to approach from. Measured: aiming at
                    // `carry_forward` left the agent 1.52 m out with the
                    // load 0.47 m short; aiming at `carry_forward -
                    // goal_radius` put it 0.30 m out with the load 0.75 m
                    // *past*. Splitting the difference is tuning to
                    // noise.
                    //
                    // Driving through and stopping on the load is the
                    // measurement that actually matters, and it has no
                    // constant in it.
                    let in_from = plane.flatten(pos - column).normalize_or_zero();
                    let aim = column - in_from * task.tuning.carry_forward;
                    agent.set_goal(DVec3::new(aim.x, aim.y, plane.floor_z));
                    // Planned with the stack **unblocked**. The standing
                    // spot is inside the site tile, and
                    // `astar_tile_grid` returns no route at all for a
                    // goal that is itself blocked — so with the ring in
                    // place the approach never planned, the agent
                    // reported `Blocked`, and the load only arrived by
                    // the luck of the facing slew dragging it there.
                    //
                    // The ring keeps the agent from clipping the tower
                    // while walking *past* it. Driving up to set a load
                    // down is the one time it has to be let in.
                    super::replan(agent, plane, src, pos, &Default::default());
                }

                // Stop and raise once the **cargo** is over the column.
                // Arriving at its own goal is not the same as having the
                // load in place, and releasing before it is means
                // flinging the crate the rest of the way — the teleport
                // at this end of the haul.
                let cargo_off = plane.flatten(pose.pos - column).length();
                if cargo_off <= task.tuning.settle_xy_tolerance {
                    agent.stop();
                    task.lift_clock = 0.0;
                    task.forks.target =
                        fork_height_for_layer(plane, layer, carry.half_extents[2], task.tuning);
                    task.state = StackState::Raising;
                }
                return StackAction::Hold { id: carry.id, pose };
            }

            // Wedged on the way. Put the crate down rather than carrying
            // it around forever — that also restores the one-crate
            // invariant from this path.
            if agent.stuck() {
                return release_here(task, agent, pose);
            }

            StackAction::Hold { id: carry.id, pose }
        }

        StackState::Settling => {
            task.settle_clock += dt;
            let Some(site) = task.site else {
                task.state = StackState::Idle;
                return StackAction::None;
            };
            let column = plane.tile_center(site.0, site.1);
            let standing = task.layers_standing(plane, crates);

            if standing >= task.tuning.target_layers {
                task.state = StackState::Done;
                agent.stop();
                return StackAction::Finished;
            }

            let h = crates.first().map(|c| c.layer_height()).unwrap_or(1.0);
            let expect = plane.floor_z + standing.saturating_sub(1) as f64 * h + h * 0.5;
            if standing > 0 && layer_has_settled(column, expect, crates, task.tuning) {
                task.state = StackState::Idle;
                return StackAction::None;
            }

            // A crate that has been rocking for several seconds is not
            // going to settle. Re-deriving the pile next tick is the
            // recovery, and it is why `layers_standing` counts rather
            // than remembers.
            if task.settle_clock >= task.tuning.settle_timeout {
                task.state = StackState::Idle;
            }
            StackAction::None
        }
    }
}

/// Where the cargo is *this tick*, sliding onto the forks if it has not
/// finished getting there.
///
/// The crate does not snap sideways onto the tines any more than it snaps
/// upward. `pickup_reach` lets the agent take a crate from up to three
/// metres away — a number that cannot simply be tightened, because the
/// standoff it was derived from was measured — so without this the crate
/// crosses that distance in a single frame, which is the larger half of
/// the teleport.
fn slid_pose(task: &StackTask, agent_pos: DVec3, crate_half_z: f64, floor_z: f64) -> CarryPose {
    let target = hold_pose(
        agent_pos,
        task.facing,
        task.forks,
        crate_half_z,
        floor_z,
        task.tuning,
    );
    let Some(from) = task.slide_from else { return target };

    // Lerp from where the crate was lying to where it rides, by a
    // fraction the caller advances at the mast's own rate. Interpolating
    // toward a *moving* carry point instead never finishes: the point
    // travels with the agent, so the gap never closes.
    let t = task.slide_t.clamp(0.0, 1.0);
    CarryPose {
        pos: DVec3::new(
            from.x + (target.pos.x - from.x) * t,
            from.y + (target.pos.y - from.y) * t,
            // Height is the mast's business and is already animated.
            target.pos.z,
        ),
        rot: target.rot,
    }
}

/// Put the carried crate down where the agent is standing.
fn release_here(task: &mut StackTask, agent: &mut Agent3D, pose: CarryPose) -> StackAction {
    let Some(carry) = task.carry.take() else {
        return StackAction::None;
    };
    task.state = StackState::Idle;
    agent.stop();
    StackAction::Release {
        id: carry.id,
        pose: CarryPose { pos: pose.pos, rot: Quat::IDENTITY },
        velocity: DVec3::new(0.0, 0.0, -RELEASE_SPEED),
    }
}

/// The nearest crate worth fetching.
fn choose_crate(
    task: &StackTask,
    plane: NavPlane,
    from: DVec3,
    crates: &[CrateInfo],
) -> Option<CrateInfo> {
    crates
        .iter()
        .filter(|c| !c.carried_by_other)
        .filter(|c| !task.placed.contains(&c.id))
        .filter(|c| !task.blacklisted(c.id))
        // Not one already sitting in the stack's own column.
        .filter(|c| match task.site {
            Some(s) => {
                plane.flatten(c.pos - plane.tile_center(s.0, s.1)).length()
                    > task.tuning.settle_xy_tolerance
            }
            None => true,
        })
        .min_by(|a, b| {
            let da = plane.flatten(a.pos - from).length();
            let db = plane.flatten(b.pos - from).length();
            da.total_cmp(&db)
        })
        .copied()
}

/// Fix the stack's column if it has not been fixed already.
fn ensure_site(
    task: &mut StackTask,
    plane: NavPlane,
    agent_pos: DVec3,
    crates: &[CrateInfo],
) -> (i32, i32) {
    if let Some(s) = task.site {
        return s;
    }
    // Near the middle of the grid, so the agent can walk all the way
    // round it — an edge tile cannot be approached from every side — and
    // clear of any crate already lying there.
    let (w, h) = plane.dims;
    let centre = ((w / 2) as i32, (h / 2) as i32);
    let mut best = centre;
    'search: for radius in 0..(w.max(h) as i32) {
        for dc in -radius..=radius {
            for dr in -radius..=radius {
                let t = (centre.0 + dc, centre.1 + dr);
                if t.0 < 2 || t.1 < 2 || t.0 >= w as i32 - 2 || t.1 >= h as i32 - 2 {
                    continue;
                }
                let c = plane.tile_center(t.0, t.1);
                let clear = crates
                    .iter()
                    .all(|k| plane.flatten(k.pos - c).length() > k.half_extents[0].abs() * 2.0);
                let away = plane.flatten(agent_pos - c).length() > 1.0;
                if clear && away {
                    best = t;
                    break 'search;
                }
            }
        }
    }
    task.site = Some(best);
    best
}

/// A tile beside `target`, on the side the agent is coming from.
fn approach_tile(plane: NavPlane, from: DVec3, target: DVec3) -> (i32, i32) {
    approach_tile_at(plane, from, target, 1)
}

/// The same, `tiles` tiles out rather than one.
fn approach_tile_at(plane: NavPlane, from: DVec3, target: DVec3, tiles: i32) -> (i32, i32) {
    let d = plane.flatten(from - target);
    let step = if d.length_squared() < 1e-9 {
        DVec3::X
    } else if d.x.abs() >= d.y.abs() {
        DVec3::new(d.x.signum(), 0.0, 0.0)
    } else {
        DVec3::new(0.0, d.y.signum(), 0.0)
    };
    plane.pos_to_tile(target + step * (plane.tile_size_m as f64) * tiles as f64)
}

impl StackTask {
    /// Tiles the agent must route around.
    ///
    /// Once a layer is down, the stack's own tile is one of them — this
    /// is the `extra_blocked` seam `plan_path` leaves open, and without
    /// it the agent walks straight through the pile it is building.
    /// The stack's tile **and its neighbours**.
    ///
    /// One tile is not enough. A tile is only a little wider than the
    /// agent, so standing on the centre of an adjacent tile still leaves
    /// the agent's body overlapping the stack's tile — measured here at
    /// up to a quarter of a metre with a 1 m agent on 1.5 m tiles. The
    /// agent then clips the tower on its way past and knocks it over,
    /// which reads as "the stack collapses on its own" long after the
    /// crate that was placed.
    ///
    /// Blocking the ring costs a detour of one tile and is the difference
    /// between a tower that stands and one that does not.
    fn blocked_tiles(&self, _plane: NavPlane) -> std::collections::HashSet<(i32, i32)> {
        let mut out = std::collections::HashSet::new();
        if !self.placed.is_empty() {
            if let Some(s) = self.site {
                // **The site tile only, not the ring around it.**
                //
                // Blocking the ring kept the agent's body clear of the
                // tower, and it also kept the agent three metres from a
                // tower its forks reach barely one metre over — so the
                // crate had to be flung the remaining distance, which is
                // the teleport at the far end of the haul. A forklift
                // drives up to the stack and sets the load down from
                // where it is standing; it does not stand off and throw.
                //
                // The tower is still protected, by the tile itself and by
                // the cargo riding above it.
                out.insert(s);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane() -> NavPlane {
        NavPlane::new((15, 15), 1.5, 0.0)
    }

    fn tuning() -> StackTuning {
        StackTuning::for_agent_and_crate(
            [0.5, 0.5, 0.25],
            [0.5, 0.5, 0.5],
            1.5,
            3,
            WalkTuning3D::default(),
        )
    }

    /// The mast, parked.
    fn parked(t: StackTuning) -> Forks {
        Forks { height: t.fork_rest_height, target: t.fork_rest_height }
    }

    fn crate_at(id: u32, pos: DVec3, sleeping: bool) -> CrateInfo {
        CrateInfo {
            id: CrateId(id),
            pos,
            half_extents: [0.5, 0.5, 0.5],
            carried_by_other: false,
            sleeping,
        }
    }

    /// The cargo must clear the agent's own body, or the solver spends
    /// every tick resolving a contact between the agent and the thing it
    /// is holding.
    #[test]
    fn the_carry_offset_clears_the_agent() {
        let t = tuning();
        let agent_half_x = 0.5;
        let crate_half_x = 0.5;
        assert!(
            t.carry_forward > agent_half_x + crate_half_x,
            "carry_forward {} does not clear {} + {}",
            t.carry_forward,
            agent_half_x,
            crate_half_x,
        );
    }

    /// And it must clear the finished stack, or the agent walks its cargo
    /// through the tower on the way to the top of it.
    #[test]
    fn the_carry_offset_clears_the_finished_stack() {
        let t = tuning();
        let crate_half_z = 0.5;
        // `carry_up` is now the height of the tine *tops* above the
        // floor, and the tine top is the cargo's bottom face — so it is
        // the cargo bottom directly, with no agent term. The geometry
        // being asserted is unchanged; the frame it is expressed in is
        // not, and the old arithmetic quietly kept passing in the new
        // frame for the wrong reason.
        let cargo_bottom = t.carry_up;
        // Top of a two-high stack, which is what exists while the third
        // layer is being carried in.
        let stack_top = 2.0 * (crate_half_z * 2.0);
        assert!(
            cargo_bottom >= stack_top,
            "cargo rides at {cargo_bottom} which is below the {stack_top} stack top",
        );
    }

    /// The drop must not bounce. This encodes the *derivation* rather
    /// than the number, so raising the clearance to something rounder
    /// fails loudly instead of producing an intermittently bouncing pile.
    #[test]
    fn the_drop_lands_below_the_restitution_threshold() {
        let t = tuning();
        let g = crate::physics3d::GRAVITY.z.abs();
        // Energies add, not speeds: v = sqrt(v0² + 2gh).
        let impact = (RELEASE_SPEED * RELEASE_SPEED + 2.0 * g * t.drop_clearance).sqrt();
        assert!(
            impact < RESTITUTION_THRESHOLD,
            "a crate dropped from {} arrives at {impact:.3} m/s, at or past the \
             {RESTITUTION_THRESHOLD} m/s threshold, and will bounce off the stack",
            t.drop_clearance,
        );
    }

    /// And it must still clear the overlap the solver deliberately keeps,
    /// or a crate is released already intersecting what it lands on.
    #[test]
    fn the_drop_clearance_exceeds_the_penetration_slop() {
        let t = tuning();
        assert!(
            t.drop_clearance > crate::physics3d::solver::PENETRATION_SLOP * 4.0,
            "drop_clearance {} is within the solver's own slop",
            t.drop_clearance,
        );
    }

    /// Layer heights come from the nominal crate size, so settling error
    /// in one layer cannot compound into the next.
    #[test]
    fn each_layer_is_dropped_at_its_nominal_height() {
        let p = plane();
        let t = tuning();
        let z0 = drop_height_for_layer(p, 0, 0.5, t);
        let z1 = drop_height_for_layer(p, 1, 0.5, t);
        let z2 = drop_height_for_layer(p, 2, 0.5, t);
        assert!((z0 - (0.5 + t.drop_clearance)).abs() < 1e-9, "layer 0 at {z0}");
        assert!((z1 - z0 - 1.0).abs() < 1e-9, "layer 1 is {} above layer 0", z1 - z0);
        assert!((z2 - z1 - 1.0).abs() < 1e-9, "layer 2 is {} above layer 1", z2 - z1);
    }

    /// A layer is settled only once the crate is *asleep*. A crate at the
    /// top of a bounce is instantaneously still and would pass a velocity
    /// test.
    #[test]
    fn a_layer_is_not_settled_until_the_crate_is_asleep() {
        let t = tuning();
        let column = DVec3::new(0.0, 0.0, 0.0);
        let awake = [crate_at(1, DVec3::new(0.0, 0.0, 0.5), false)];
        let asleep = [crate_at(1, DVec3::new(0.0, 0.0, 0.5), true)];
        assert!(!layer_has_settled(column, 0.5, &awake, t), "awake must not count");
        assert!(layer_has_settled(column, 0.5, &asleep, t), "asleep should count");
    }

    /// A crate that landed off the column is not a layer, however still
    /// it is — most of it is hanging over the edge.
    #[test]
    fn a_crate_off_the_column_is_not_a_settled_layer() {
        let t = tuning();
        let column = DVec3::ZERO;
        let off = [crate_at(1, DVec3::new(t.settle_xy_tolerance + 0.2, 0.0, 0.5), true)];
        assert!(!layer_has_settled(column, 0.5, &off, t));
    }

    /// The cargo is held square for the whole carry, not just at the
    /// drop: a crate released even slightly rotated lands on a corner,
    /// which is the single-contact-point case that makes a box spin.
    #[test]
    fn the_cargo_is_held_square() {
        let t = tuning();
        let pose = hold_pose(DVec3::ZERO, DVec3::X, parked(t), 0.5, 0.0, t);
        assert_eq!(pose.rot, Quat::IDENTITY);
    }

    /// The hold point follows where the agent is going, in the plane, and
    /// rides on the forks rather than at a fixed offset from the body.
    #[test]
    fn the_cargo_rides_on_the_forks() {
        let t = tuning();
        let pose = hold_pose(DVec3::ZERO, DVec3::X, parked(t), 0.5, 0.0, t);
        assert!(pose.pos.x > 0.0, "should be in front along +X: {:?}", pose.pos);
        assert_eq!(pose.pos.y, 0.0);
        // The tine top is the crate's bottom face — the exact
        // relationship, not merely "above zero", which the old assertion
        // would have passed on for any height at all.
        assert!(
            (pose.pos.z - (t.fork_rest_height + 0.5)).abs() < 1e-9,
            "cargo at {:.3} but the parked tines put it at {:.3}",
            pose.pos.z,
            t.fork_rest_height + 0.5,
        );
    }

    /// The cargo height must come from the floor and the mast, never from
    /// the agent's own z.
    ///
    /// An offset from the body would make a crate two metres overhead
    /// twitch every time a contact under the tracks nudged the agent a
    /// few millimetres.
    #[test]
    fn the_cargo_height_ignores_where_the_agent_is_bounced_to() {
        let t = tuning();
        let low = hold_pose(DVec3::ZERO, DVec3::X, parked(t), 0.5, 0.0, t);
        let bounced = hold_pose(
            DVec3::new(0.0, 0.0, 0.05),
            DVec3::X,
            parked(t),
            0.5,
            0.0,
            t,
        );
        assert_eq!(
            low.pos.z, bounced.pos.z,
            "bouncing the agent 50 mm moved its cargo",
        );
    }

    /// A degenerate facing must not produce a NaN pose — the agent is
    /// stationary on the tick it picks a crate up.
    #[test]
    fn a_stationary_agent_still_gets_a_finite_hold_pose() {
        let t = tuning();
        let pose = hold_pose(DVec3::ZERO, DVec3::ZERO, parked(t), 0.5, 0.0, t);
        assert!(pose.pos.is_finite(), "degenerate facing gave {:?}", pose.pos);
    }

    /// The approach tile is beside the target, never on it: a goal inside
    /// a solid crate is one the agent grinds against forever.
    #[test]
    fn the_approach_tile_is_beside_the_target_not_on_it() {
        let p = plane();
        let target = p.tile_center(7, 7);
        for from in [
            target + DVec3::new(5.0, 0.0, 0.0),
            target + DVec3::new(-5.0, 0.0, 0.0),
            target + DVec3::new(0.0, 5.0, 0.0),
            target + DVec3::new(0.0, -5.0, 0.0),
        ] {
            let t = approach_tile(p, from, target);
            assert_ne!(t, (7, 7), "approach from {from:?} landed on the target tile");
        }
    }

    /// The column is fixed once chosen. A stack whose site is recomputed
    /// per layer wanders and falls over.
    #[test]
    fn the_stack_site_does_not_move_once_it_is_chosen() {
        let p = plane();
        let mut task = StackTask::new(tuning());
        let crates = [crate_at(1, DVec3::new(3.0, 3.0, 0.5), true)];
        let first = ensure_site(&mut task, p, DVec3::ZERO, &crates);
        // Somewhere entirely different, and more crates in the way.
        let more = [
            crate_at(1, DVec3::new(-4.0, 2.0, 0.5), true),
            crate_at(2, DVec3::new(1.0, -3.0, 0.5), true),
        ];
        let second = ensure_site(&mut task, p, DVec3::new(9.0, 9.0, 0.0), &more);
        assert_eq!(first, second, "the site moved after being fixed");
    }

    /// Once a layer is down, the stack's tile is routed around rather
    /// than walked through.
    #[test]
    fn the_stack_tile_is_blocked_once_a_layer_is_placed() {
        let p = plane();
        let mut task = StackTask::new(tuning());
        task.site = Some((7, 7));
        assert!(
            task.blocked_tiles(p).is_empty(),
            "nothing is placed yet, so nothing should be blocked",
        );
        task.placed.push(CrateId(1));
        assert!(
            task.blocked_tiles(p).contains(&(7, 7)),
            "the stack's own tile must be blocked once it holds a crate",
        );
    }

    /// A crate already in the stack is never picked back up, or the agent
    /// dismantles its own tower.
    #[test]
    fn a_placed_crate_is_never_chosen_again() {
        let p = plane();
        let mut task = StackTask::new(tuning());
        task.placed.push(CrateId(1));
        let crates = [
            crate_at(1, DVec3::new(0.5, 0.0, 0.5), true),
            crate_at(2, DVec3::new(4.0, 0.0, 0.5), true),
        ];
        let got = choose_crate(&task, p, DVec3::ZERO, &crates).expect("one is free");
        assert_eq!(got.id, CrateId(2), "should skip the crate already stacked");
    }

    /// A crate someone else is holding is not available.
    #[test]
    fn a_crate_carried_by_someone_else_is_not_chosen() {
        let p = plane();
        let task = StackTask::new(tuning());
        let mut held = crate_at(1, DVec3::new(0.5, 0.0, 0.5), false);
        held.carried_by_other = true;
        let crates = [held, crate_at(2, DVec3::new(6.0, 0.0, 0.5), true)];
        let got = choose_crate(&task, p, DVec3::ZERO, &crates).expect("one is free");
        assert_eq!(got.id, CrateId(2));
    }

    /// A blacklisted crate is skipped, which is what stops the task
    /// re-picking an unreachable crate every tick forever — a livelock
    /// that looks exactly like an idle agent while burning an A* per
    /// tick.
    #[test]
    fn a_blacklisted_crate_is_skipped_until_it_expires() {
        let p = plane();
        let mut task = StackTask::new(tuning());
        let crates = [crate_at(1, DVec3::new(0.5, 0.0, 0.5), true)];
        task.blacklist_crate(CrateId(1));
        assert!(choose_crate(&task, p, DVec3::ZERO, &crates).is_none());

        task.tick_blacklist(task.tuning.blacklist_time + 0.01);
        assert!(
            choose_crate(&task, p, DVec3::ZERO, &crates).is_some(),
            "the blacklist should expire",
        );
    }

    /// Standing layers are counted from the crates, not from a tally, so
    /// a toppled stack is noticed rather than assumed intact.
    #[test]
    fn layers_are_counted_from_the_world() {
        let p = plane();
        let mut task = StackTask::new(tuning());
        task.site = Some((7, 7));
        // Only crates the task placed count toward its tower. A loose
        // crate lying on the floor sits at exactly the base layer's
        // height, so without this the scrap in the yard is counted as a
        // stack — measured in `crates3d` as a count of two before a
        // single crate had been placed, which sent the mast to the third
        // layer and dropped the first crate from two and a half metres.
        task.placed = vec![CrateId(1), CrateId(2)];
        let c = p.tile_center(7, 7);

        let two = [
            crate_at(1, DVec3::new(c.x, c.y, 0.5), true),
            crate_at(2, DVec3::new(c.x, c.y, 1.5), true),
        ];
        assert_eq!(task.layers_standing(p, &two), 2);

        // The top one has rolled away. The count must drop.
        let toppled = [
            crate_at(1, DVec3::new(c.x, c.y, 0.5), true),
            crate_at(2, DVec3::new(c.x + 3.0, c.y, 0.5), true),
        ];
        assert_eq!(task.layers_standing(p, &toppled), 1);
    }

    /// A finished task stays finished and keeps the agent stopped, which
    /// is what lets the whole scene fall asleep.
    #[test]
    fn a_finished_task_stays_finished() {
        let p = plane();
        let mut task = StackTask::new(tuning());
        task.state = StackState::Done;
        let mut agent = Agent3D::new(WalkTuning3D::default());
        struct Open;
        impl TileSource for Open {
            fn dims(&self) -> (u32, u32) {
                (15, 15)
            }
            fn blocks(&self, c: i32, r: i32) -> bool {
                c < 0 || r < 0 || c >= 15 || r >= 15
            }
        }
        let action = drive_stacker(
            &mut task,
            &mut agent,
            &Transform3D::at(DVec3::ZERO),
            p,
            &Open,
            &[],
            1.0 / 60.0,
        );
        assert_eq!(action, StackAction::Finished);
        assert_eq!(task.state, StackState::Done);
    }

}
