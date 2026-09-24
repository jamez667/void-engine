//! Stacking crates: the layer that decides *where* a walking agent
//! should be, and what to do when it gets there.
//!
//! [`super::drive_agent`] answers "walk to that point". This answers
//! "which point, and why" — fetch a crate, carry it, put it on the pile.
//! The task executes one [`super::jobs::Job`] at a time, handed out by a
//! [`super::jobs::JobBoard`], and never finishes: when the board has
//! nothing for it the task idles on [`StackAction::NoJob`] and re-polls
//! next tick, so a crate that appears later or a spot that empties out
//! is picked up without anyone restarting it.
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
//! # Known gaps in the forklift cycle
//!
//! The mast works: a crate is driven to, lifted, hauled and set down,
//! and the worst single-tick movement of a crate fell from 3.63 m to
//! 0.26 m when the teleporting pickup was replaced by it. One thing is
//! still wrong, visible in `examples/crates3d`:
//!
//! * **The agent barges its own pile.** Driving up to a tower means the
//!   site tile cannot stay blocked during the final approach, and nothing
//!   yet keeps the chassis off the stack it is placing onto — so the
//!   tower gets nudged. The likely fix is to keep the ring blocked for
//!   the *body* and let only the forks overhang, which means deriving the
//!   approach goal from the chassis footprint rather than from the load.

use std::collections::HashMap;

use glam::{DVec3, Quat};

use crate::components::Transform3D;
use crate::pathfind::TileSource;
use crate::physics3d::solver::RESTITUTION_THRESHOLD;

use super::agent::{Agent3D, AgentState};
use super::jobs::{Job, JobBoard};
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
    /// Which way the crate is facing.
    ///
    /// Without this the task cannot know which way the box points,
    /// so it squares up to the **world** axes instead — and the crate
    /// then snapped to the mast the instant it became cargo, which is
    /// a box magically straightening itself as it is picked up. A
    /// machine lines up with the load, not with north.
    pub rot: Quat,
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
    /// Stopped at the crate, lowering the forks and turning square to it.
    ///
    /// This is where the *stop* happens, and the stop is most of what
    /// makes a pickup read as a pickup rather than as a cut. The machine
    /// is a standoff back from the crate here, tines clear of it — step 1
    /// of the pickup, "line up but not touching".
    Engaging,
    /// Driving straight in, tines going under the crate.
    ///
    /// Step 2. The crate does not move: the machine does. This is a creep
    /// along the facing rather than a walk, because the walker stops
    /// within `goal_radius` of a goal on whichever side it approached
    /// from, and fork insertion needs a tolerance an order of magnitude
    /// tighter than that.
    Inserting,
    /// Setting the load down on the stack, tines still under it.
    ///
    /// Steps 6 and 7: the mast goes to the layer height plus a clearance,
    /// then lowers onto the stack. The crate is released once the tines
    /// are below it, so it is *placed* rather than dropped.
    Placing,
    /// Stopped short of the stack, creeping square onto the column.
    ///
    /// Step 5. The haul stops a standoff back — far enough that neither
    /// the tines nor the load are over the pile — and this lines the
    /// cargo up with the column before any of it is lifted. Raising while
    /// already on top of the stack is how a machine knocks it over.
    Lining,
    /// Backing straight out until the tines are clear of what was just
    /// set down.
    ///
    /// Step 8, and it is not cosmetic: turning away with the forks still
    /// inside the stack sweeps them through it. The reverse is along the
    /// facing for the same reason the insertion is.
    Withdrawing,
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
    ///
    /// Returns to [`StackState::Idle`] once it has, or on timeout — never
    /// terminal, so the task is ready to poll the board for the next job.
    Settling,
}

/// What the task wants the caller to do to the world this tick.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum StackAction {
    /// Nothing.
    None,
    /// Make this crate kinematic **where it lies** — do not move it. Zero
    /// its velocity, wake it; see [`StackTuning`] for why both.
    ///
    /// No pose: the crate goes kinematic exactly where the solver last
    /// left it, and nothing moves it this tick. The carry offset recorded
    /// in `task.carry` is relative to wherever that is, so it is honest
    /// by construction rather than by writing a pose that happens to
    /// match. The caller also stops the machine dead — the pickup is the
    /// moment the load's actual position becomes load-bearing, and a
    /// machine still coasting on leftover creep velocity would carry the
    /// crate out from under itself before the next tick's carried pose is
    /// even written.
    Pickup { id: CrateId },
    /// Make it dynamic again at `pose` with this velocity and no spin,
    /// and **wake it**.
    Release { id: CrateId, pose: CarryPose, velocity: DVec3 },
    /// Drive the machine **along its own facing** at `speed`, this tick.
    ///
    /// Negative is reverse. The velocity is a world-space vector — these
    /// machines have omni wheels, so translation and heading are
    /// independent and a "speed along the facing" is really just
    /// `facing * speed` computed by the caller before this is built.
    ///
    /// A *velocity* request, not a force: the creep's whole point is a
    /// bounded, predictable approach, and a force would have the mass and
    /// the floor friction between the request and the result. The caller
    /// sets the body's planar velocity and leaves `z` to gravity.
    ///
    /// This is deliberately not routed through [`super::drive_agent`].
    /// That walks tiles and stops within `goal_radius`, which is 0.30 m
    /// and on whichever side it approached from — an order of magnitude
    /// coarser than putting tines into a crate needs.
    Creep { velocity: DVec3 },
    /// Nothing to do this tick: no spot with room has a loose crate to
    /// fill it. Not an error — ask again next tick; a crate may appear or
    /// a stack may fall.
    NoJob,
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
    /// The tallest any spot may be stacked. Sets the haul height
    /// (`carry_up`) and the lift timeout; the board's capacities must not
    /// exceed it.
    pub max_layers: u32,
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
    ///
    /// Tried at 2.0 (twice), both failed: `[1, 2, 1, 0]` against
    /// `[2, 2, 1, 0]` both times, one run stuck in `Engaging` and one in
    /// `Fetching` when the frame budget ran out. Retried again after the
    /// live-block retargeting landed (a layer-1 Raising dropped from
    /// ~62 ticks to ~31, confirming the rate change itself works) and it
    /// still failed the same way both times, so the bottleneck is
    /// elsewhere in the cycle, not the mast. Reverted to the original
    /// rate.
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
    /// How far back from a crate's centre the machine lines up before
    /// driving its tines in.
    ///
    /// Far enough that the tines are clear of the crate at the moment the
    /// machine squares up — otherwise "line up" and "already inside it"
    /// are the same position, and turning to face the crate sweeps the
    /// forks through it.
    pub fork_standoff: f64,
    /// How fast the machine manoeuvres at close quarters, m/s.
    ///
    /// This is the *positioning* speed: shuffling onto the standoff,
    /// and backing out once the load is down. Neither is delicate —
    /// nothing is between the tines and the crate during either — so
    /// it is brisk. Only the insertion itself is slow; see
    /// [`StackTuning::insert_speed`].
    ///
    /// Raised from 2.2 to 3.0 — the walker's own [`WalkTuning3D::speed`],
    /// and the ceiling for that reason: jockeying faster than the
    /// machine can otherwise drive would read as the manoeuvre
    /// cheating past the walk.
    ///
    /// Much slower than [`WalkTuning3D::speed`]. This is the jockeying
    /// speed, and it is slow for the same reason a real one is: the
    /// tolerance is centimetres and the thing being approached falls over
    /// if it is nudged.
    pub creep_speed: f64,
    /// How fast the tines go into a crate, m/s.
    ///
    /// The one genuinely delicate move in the cycle: the tines are
    /// passing under a loose box that will skid if it is nudged, and
    /// the machine stops on a `creep_tolerance` of two centimetres.
    /// Everything else the machine does at close quarters runs at
    /// [`StackTuning::creep_speed`], which is several times this.
    ///
    /// Tried at 0.8: failed twice, both times with a spot short by the
    /// end of the scene's frame budget (`[2, 1, 1, 0]` against
    /// `[2, 2, 1, 0]`) rather than an overlap or an emergency drop — the
    /// faster insert was not the free win it looked like. Reverted to
    /// the original.
    pub insert_speed: f64,
    /// How close, along the fork axis, counts as arrived when creeping.
    ///
    /// An order of magnitude tighter than the walker's `goal_radius`,
    /// which is why creeping exists at all.
    pub creep_tolerance: f64,
    /// How long a creep may take before the crate is given up on.
    pub creep_timeout: f32,
    /// How far above the target layer the mast rises before lowering onto
    /// it — step 6's "plus a little bit".
    ///
    /// The load clears the layer below on the way in and is then set down
    /// on it, rather than being slid across its top face.
    pub place_clearance: f64,
    /// How far the mast drops below the placed crate before the machine
    /// backs out, so the tines come out from under it rather than
    /// dragging it.
    pub tine_drop: f64,
    /// How far back from the stack column a **loaded** machine stops
    /// after the haul, before raising.
    ///
    /// Not `fork_standoff`. That is sized for an empty machine — the
    /// tine tips, a metre out, clear of a crate on the floor. A loaded
    /// machine reaches further than its tines: the load's far face is
    /// at `carry_forward + crate_half`, over half a metre past the tips.
    /// Stopping at the empty standoff put the load's face 5 cm from the
    /// stacked crate, and the walker stops within `goal_radius` of its
    /// goal on either side — so it could arrive a quarter of a metre
    /// *inside* the stack. That is the collision while lining up before
    /// the raise.
    ///
    /// The walker's own slop is folded in here rather than tolerated,
    /// because this is the one approach made by the walker and not by
    /// a creep: the precise move to the pile is `Lining`'s, made at
    /// height once the load is clear of it.
    pub haul_standoff: f64,
}

/// How thick the fork tines are, in metres.
///
/// Shared with the caller's mesh: the tines have to be drawn this thick
/// or the crate visibly floats above them. It is also
/// [`StackTuning::fork_rest_height`], because parking the mast at exactly
/// the tine thickness puts the tine bottoms on the floor.
pub const FORK_THICKNESS: f64 = 0.06;

/// How far the tines reach out in front of the chassis, in metres.
///
/// Shared with the caller's mesh for the same reason [`FORK_THICKNESS`]
/// is: a mesh that disagrees draws tines that stop halfway under the
/// load, or that stick out past it.
///
/// **This must be at least as deep as the cargo.** A fork shorter than
/// the box it is under carries it on the tips, which is wrong to look at
/// and is how a real machine drops its load. The cargo rides back
/// against the mast — see `carry_gap` — so the tines have to span from
/// the chassis face to the crate's far face, which is exactly the
/// crate's own depth.
///
/// One metre suits the 1 m crates this engine's examples use. A game
/// with deeper pallets wants a longer fork, and the assertion in
/// [`StackTuning::for_agent_and_crate`] says so rather than letting the
/// mismatch show up as a visual bug.
pub const FORK_REACH: f64 = 1.0;

/// Daylight left between the tine tips and a crate when the machine
/// squares up to it, in metres.
///
/// Small, but not zero: at exactly zero the tips are touching the crate
/// at the moment the machine turns to face it, and a turn pivots the
/// tips through an arc that clips the box. Ten centimetres is well past
/// the arc a `creep_tolerance` misalignment can produce.
pub const FORK_CLEARANCE: f64 = 0.10;

/// How far behind the fork standoff `Idle`'s fetch goal ends, in metres.
///
/// Continuous lateral-correction gain and speed bounds, shared by
/// `Inserting` and `Lining` — both drive a load toward a target column
/// while only closing one axis explicitly, and both need the same
/// deadband/floor/cap on the other axis or the correction either stalls
/// against friction or limit-cycles across the tolerance forever.
const LATERAL_K: f64 = 2.0; // 1/s
const LATERAL_MIN: f64 = 0.15; // m/s — below what friction cancels, the machine does not move at all
const LATERAL_MAX: f64 = 0.30; // m/s

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
    /// sits at `floor_z + height + carry.offset.z`.
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
///
/// No longer used by the task's placement (kept: re-exported and tested).
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
        max_layers: u32,
        walk: WalkTuning3D,
    ) -> Self {
        let crate_height = crate_half[2].abs() * 2.0;

        // Far enough forward that the agent and its cargo never touch,
        // **and that the tines between them have somewhere to be**.
        //
        // The minimum is the two half-extents, which puts the faces
        // exactly flush — and flush is not enough. `obb_contact_manifold`
        // accepts points up to its own contact tolerance *outside* a
        // boundary, so touching faces still generate manifold points and
        // the solver would resolve a contact between the agent and the
        // thing it is holding every tick.
        //
        // But a gap sized only to dodge that tolerance leaves no room for
        // the forks. The tines are drawn reaching from the chassis to
        // under the load, so the space between the two faces *is* the
        // tine length: at the old 5 cm the mesh had nowhere to go and was
        // drawn straight through the crate, 0.8 m of tine inside a box it
        // was supposed to be carrying.
        //
        // So the load rides hard **against the mast**, with only the
        // solver clearance between them, and the tines run the full
        // depth of the box underneath it. Carrying it out on the tips
        // instead — which a `FORK_REACH` gap did — leaves half the
        // crate hanging off the end of the forks, which is both wrong
        // to look at and how a real machine drops its load.
        //
        // The tine *length* is what has to fit the crate, and that is
        // `FORK_REACH`, checked against the cargo below.
        // Sized for the chassis **corner**, not its face.
        //
        // The machine is a box and its yaw comes from contacts, so it
        // is never exactly square: at 15 degrees off, a 0.5 m
        // half-width chassis reaches 0.61 m along the drive axis
        // rather than 0.50 m. A gap sized to the face lets the corner
        // strike the crate first — measured, the machine pushed the
        // box across the floor instead of getting under it, with the
        // insertion stalled 4 cm short of its own tolerance.
        //
        // `sqrt(2) - 1` is the worst case: the extra reach of a
        // square rotated 45 degrees. Paying it always is cheaper than
        // tracking the yaw, and it costs a few centimetres of how
        // close the load rides.
        let corner_slack = agent_half[0].abs() * (std::f64::consts::SQRT_2 - 1.0);
        let carry_gap = 0.05 + corner_slack;
        let carry_forward = agent_half[0].abs() + crate_half[0].abs() + carry_gap;

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
        let stack_top = (max_layers.saturating_sub(1)) as f64 * crate_height;
        let carry_up = stack_top + carry_gap;

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
            max_layers,
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
            // Far enough back that the **tine tips** are outside the
            // crate, not just the chassis.
            //
            // The tips sit `FORK_REACH` ahead of the machine's centre
            // face, so the standoff has to clear the crate's near face by
            // that much again — otherwise the machine squares up with a
            // metre of fork already inside the box and sweeps it round as
            // it turns. Driving `fork_standoff - carry_forward` forward
            // from here is exactly the insertion.
            fork_standoff: agent_half[0].abs()
                + FORK_REACH
                + crate_half[0].abs()
                + FORK_CLEARANCE,
            // A third of walking pace. Jockeying, not driving — but a
            // machine that takes five seconds to back out of a stack
            // spends most of a scene reversing.
            creep_speed: 3.0,
            insert_speed: 0.45,
            // 2 cm along the fork axis. An order of magnitude tighter
            // than the walker's 0.30 m `goal_radius`, which is the whole
            // reason the creep is not a walk.
            creep_tolerance: 0.02,
            // Generous: a creep is short but slow, and this exists to
            // catch a machine wedged against something, not to police
            // pace. `fork_standoff` at `creep_speed` plus a wide margin.
            creep_timeout: 8.0,
            // Clear the layer below on the way in rather than sliding the
            // load across its top face.
            place_clearance: crate_half[2].abs() * 0.5,
            // Enough that the tines are below the crate's bottom face
            // before the machine reverses, so they come out from under it
            // instead of dragging it back off the stack. Twice the tine
            // thickness, which is comfortably clear of the solver's
            // penetration slop.
            tine_drop: FORK_THICKNESS * 2.0,
            // The load's far face, not the tine tips, is what has to
            // clear the stack — and it is `carry_forward + crate_half`
            // from the machine's centre. Then the stacked crate's own
            // half-width, the same daylight the empty standoff keeps,
            // and the walker's acceptance radius, because the walker
            // stops anywhere inside that on whichever side it arrives.
            haul_standoff: carry_forward
                + crate_half[0].abs()
                + crate_half[0].abs()
                + FORK_CLEARANCE
                + walk.goal_radius,
            // One crate height per second. See the field docs.
            lift_rate: crate_height,
            // The tine thickness, so the tines rest *on* the floor.
            fork_rest_height: FORK_THICKNESS,
            // Three times the longest move the machine ever makes, which
            // is floor to the top layer.
            lift_timeout: ((stack_top + carry_gap) / crate_height * 3.0) as f32,
        }
    }
}

/// The crate currently being carried.
///
/// A rigid child of the machine for the duration of the carry: `offset`
/// and `rot_offset` are recorded once, at pickup, and never touched again
/// until release. Nothing about the carry is recomputed per tick except
/// where the *machine* is — see [`carried_pose`].
#[derive(Clone, Copy, Debug)]
pub struct Carry {
    pub id: CrateId,
    pub half_extents: [f64; 3],
    /// The crate's position relative to the machine, in the facing frame
    /// at the moment of pickup: `x` forward, `y` left, `z` the crate's
    /// centre above the tine top. Fixed for the life of the carry, so the
    /// crate cannot lag or slide relative to the machine — it can only
    /// move exactly as the machine moves.
    pub offset: DVec3,
    /// The crate's rotation relative to the machine's facing at pickup:
    /// `R_z(-psi) * crate_rot`, `psi` the facing yaw. Reapplied on top of
    /// the machine's current facing every tick, so the crate turns with
    /// the machine rather than snapping to it.
    pub rot_offset: Quat,
}

impl Carry {
    /// The residual yaw the placement aim has to cancel for the load to
    /// land square on the box below.
    ///
    /// At pickup the machine squares up to the crate's **nearest** face
    /// (see [`fork_approach_pose`]), so `rot_offset` is typically close to
    /// a multiple of a right angle — the raw offset is usually ~±90° and
    /// carries no useful aiming information. Because the footprint is
    /// square, only the offset's remainder modulo a right angle matters:
    /// reduced into `(-pi/4, pi/4]`, it is the small correction that
    /// squares the crate to whatever is already stacked.
    pub fn square_yaw(&self) -> f64 {
        let d = (self.rot_offset * glam::Vec3::X).as_dvec3();
        let yaw = d.y.atan2(d.x);
        let a = yaw.rem_euclid(std::f64::consts::FRAC_PI_2);
        if a > std::f64::consts::FRAC_PI_4 {
            a - std::f64::consts::FRAC_PI_2
        } else {
            a
        }
    }
}

/// The stacker's memory between ticks.
#[derive(Clone, Debug)]
pub struct StackTask {
    pub state: StackState,
    pub tuning: StackTuning,
    pub carry: Option<Carry>,
    pub target: Option<CrateId>,
    /// The placement being executed. Set on Idle when the board hands one
    /// out, cleared when the crate has settled or the job is abandoned.
    /// The spot is fixed for the job's life; recomputing it per layer is
    /// how a stack wanders.
    pub job: Option<Job>,
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
    /// Whether the haul has switched from "walk to the tile beside the
    /// stack" to "line the load up over it".
    ///
    /// One-shot, because the goal it sets clears the agent's path:
    /// setting it every tick re-plans from scratch every tick and the
    /// agent never takes a step.
    final_approach: bool,
    /// Seconds spent on the current creep, for the stall timeout.
    creep_clock: f32,
    /// Where the machine stood when it began backing out, so the
    /// reverse can be measured as a distance travelled rather than
    /// against the stack it is reversing away from.
    withdraw_from: Option<DVec3>,
    /// Where the machine wants to be pointing this tick, when it is
    /// manoeuvring rather than following a path.
    ///
    /// These machines have omni wheels, so closing a sideways error does
    /// not strictly require turning first — but [`drive_to_pose`] still
    /// turns toward the final heading while it drives, because arriving
    /// square to the crate matters and turning costs nothing extra with
    /// omni wheels. This is that turn request; the drive half is
    /// [`StackAction::Creep`]. Cleared every tick by whoever sets it.
    steer_to: Option<DVec3>,
    /// Whether the placement has reached its standoff pose and moved
    /// on to driving in.
    ///
    /// A latch, and it is load-bearing. `Lining` first drives to a
    /// standing spot and then drives *from* it toward the column, and
    /// both happen inside the one state. Without a record that the
    /// first part is done, it is re-evaluated every tick: the machine
    /// steps forward, is no longer on the spot, is sent back to it,
    /// steps forward again. Measured pinned 4 cm off the standoff for
    /// the whole timeout, heading already perfect, the offset cycling
    /// 0.035 / 0.044 / 0.039 around a 0.040 threshold — one tick
    /// forward, one tick back.
    ///
    /// `Engaging` has the same two halves and does not need this,
    /// because its second half is a different state (`Inserting`) and
    /// the state change is the latch.
    aligned: bool,
}

impl StackTask {
    pub fn new(tuning: StackTuning) -> Self {
        Self {
            state: StackState::Idle,
            tuning,
            carry: None,
            target: None,
            job: None,
            blacklist: HashMap::new(),
            settle_clock: 0.0,
            facing: DVec3::X,
            // Parked, which means the tine bottoms are on the floor.
            forks: Forks {
                height: tuning.fork_rest_height,
                target: tuning.fork_rest_height,
            },
            lift_clock: 0.0,
            final_approach: false,
            creep_clock: 0.0,
            withdraw_from: None,
            steer_to: None,
            aligned: false,
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

    /// Where the carried crate belongs this tick, if one is carried.
    ///
    /// For the caller: call this **after** the physics solve, built from
    /// the machine's final transform for the tick — the only place a
    /// carried crate's pose may be written. Building it from the
    /// pre-step position (as the old `hold_pose` write did, applied
    /// before `drive_stacker` even had a chance to move the agent) is
    /// exactly the one-tick lag that was measured at 5-10 cm/tick while
    /// hauling: the crate trailed the machine by one physics step because
    /// its pose was fixed before that step ran.
    pub fn carried_pose(&self, agent_pos: DVec3, floor_z: f64) -> Option<(CrateId, CarryPose)> {
        let carry = self.carry?;
        Some((
            carry.id,
            carried_pose(agent_pos, self.facing, self.forks, carry, floor_z),
        ))
    }

    /// Whether the machine is manoeuvring itself rather than walking.
    ///
    /// While this is true the caller must **not** run
    /// [`super::drive_agent`]: the task is driving the body with
    /// [`StackAction::Creep`], and the walker would be pushing at the
    /// same time. The two do not agree — the walker steers toward a goal
    /// in whatever direction that lies, while a creep is a specific,
    /// deliberate velocity chosen for centimetre-level positioning — so
    /// with both running the walker fights the creep, which is the
    /// "shifting sideways instead of driving" the creep was meant to fix.
    ///
    /// `agent.stop()` is not enough on its own. It clears the goal, but
    /// the states below are entered and left over many ticks and any one
    /// of them re-planning leaves a path the walker will happily follow.
    /// Gating the call is the version that cannot drift out of step.
    pub fn manoeuvring(&self) -> bool {
        matches!(
            self.state,
            StackState::Engaging
                | StackState::Inserting
                | StackState::Lifting
                | StackState::Raising
                | StackState::Lining
                | StackState::Placing
                | StackState::Withdrawing
        )
    }

    /// Put the task back to the start, keeping its tuning.
    pub fn restart(&mut self) {
        let tuning = self.tuning;
        *self = Self::new(tuning);
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
        self.job = None;
        self.state = StackState::Idle;
        agent.stop();
        // A failed engage (crate vanished, plan failed, blacklisted mid-
        // approach) can leave the mast partway to the crate's height —
        // possibly under the floor, since `Fetching` now targets the
        // crate's own bottom face. Parking it here means the next attempt,
        // on a different crate, starts from a known height instead of
        // wherever this one left off.
        self.forks.target = self.tuning.fork_rest_height;
    }
}

/// Where the cargo rides, in world space, as a rigid child of the machine.
///
/// `facing` is the direction the carry frame is built from. It comes from
/// where the agent is *going*, never from its rotation: nothing ever yaws
/// the agent deliberately — it is a box pushed by a centre-of-mass force —
/// so its orientation is whatever the contact solver last left it at, and
/// deriving "forward" from it points the cargo in an arbitrary direction.
/// The height comes from the **mast and the floor**, never from the
/// agent's own z. An agent bounced twenty millimetres by a contact under
/// its tracks must not bounce a crate two metres overhead with it.
///
/// # Why this squares up before the drop
///
/// A crate set down even slightly rotated lands on a corner, which falls
/// out of `obb_contact_manifold`'s face-clipping path into its
/// single-point fallback, and one contact point on a lever arm is what
/// makes a box spin instead of settle. That is real.
///
/// But there is no snap here to fix it. The crate is a rigid child of the
/// machine for the whole carry — `carry.rot_offset` is fixed at pickup and
/// reapplied on top of `facing` every tick, never recomputed toward some
/// other target — so it lands square only because the *machine* aimed the
/// load square before the release, not because the release itself
/// corrects it. See [`Carry::square_yaw`] and where it is used to twist
/// the approach heading during `Lining`.
pub fn carried_pose(agent_pos: DVec3, facing: DVec3, forks: Forks, carry: Carry, floor_z: f64) -> CarryPose {
    let f = if facing.length_squared() > 1e-12 {
        DVec3::new(facing.x, facing.y, 0.0).normalize_or_zero()
    } else {
        DVec3::X
    };
    let left = DVec3::new(-f.y, f.x, 0.0);
    let pos = agent_pos + f * carry.offset.x + left * carry.offset.y;
    let yaw = f.y.atan2(f.x);
    CarryPose {
        pos: DVec3::new(pos.x, pos.y, floor_z + forks.height + carry.offset.z),
        rot: Quat::from_rotation_z(yaw as f32) * carry.rot_offset,
    }
}

/// The mast height that releases the carried crate `tuning.drop_clearance`
/// above the measured top face at `top_z`.
///
/// The carried crate's bottom sits at `floor_z + forks.height +
/// carry.offset.z − half_z` (see [`carried_pose`]). `carry.offset.z` was
/// recorded at pickup with the crate a few millimetres *sunk* into the
/// floor — the solver's own penetration slop — so aiming the tine top at
/// the top face directly would release the new crate about a centimetre
/// *inside* the one below it. Solving for the mast height that keeps
/// `drop_clearance` above the **measured** top face, rather than a
/// nominal one, cancels that sink out.
pub fn place_fork_height(top_z: f64, carry: Carry, floor_z: f64, tuning: StackTuning) -> f64 {
    (top_z - floor_z) - (carry.offset.z - carry.half_extents[2].abs()) + tuning.drop_clearance
}

/// The height a crate's centre must be released from to make layer `n`.
///
/// Derived from the nominal crate height and the layer index, **not**
/// from the measured top of the pile. A crate that settled two
/// millimetres into its neighbour — which the solver leaves on purpose,
/// up to [`crate::physics3d::solver::PENETRATION_SLOP`] — would drag the
/// next drop two millimetres
/// low, and the error would compound up the stack. Measuring is used to
/// decide whether a layer has settled, and — via [`place_fork_height`] —
/// to place the next one.
///
/// No longer used by the task's placement (kept: re-exported and tested).
pub fn drop_height_for_layer(
    plane: NavPlane,
    layer: u32,
    crate_half_z: f64,
    tuning: StackTuning,
) -> f64 {
    let h = crate_half_z.abs() * 2.0;
    plane.floor_z + layer as f64 * h + crate_half_z.abs() + tuning.drop_clearance
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
/// [`crate::physics3d::step`] — `Creep`, `Release` and the pickup's stop
/// all feed the integrator, same as any other velocity command.
///
/// The carried crate's pose is a different matter: it is written
/// separately, by [`StackTask::carried_pose`], **after** the solve, from
/// the machine's final transform for the tick. Writing it here instead,
/// from the pre-step position, is exactly the one-tick lag this replaced
/// — see that method's docs. A carried crate is excluded from contacts,
/// so its slot in the broadphase lagging a tick behind is harmless; only
/// writing its *pose* early is not.
pub fn drive_stacker<T: TileSource>(
    task: &mut StackTask,
    agent: &mut Agent3D,
    agent_transform: &Transform3D,
    plane: NavPlane,
    src: &T,
    board: &mut JobBoard,
    crates: &[CrateInfo],
    dt: f32,
) -> StackAction {
    task.tick_blacklist(dt);

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
    // Held through the **insertion** too, not just while squaring up.
    //
    // Gating this on `Engaging` alone meant the facing went back to
    // following the path the moment the machine started driving in, so it
    // squared up perfectly and then swung off line over the next metre —
    // measured at 9 and 24 degrees of skew at the moment of pickup, which
    // is the crate visibly snapping straight as it lands on the forks.
    //
    // And it is the crate's own *face normal* that is held, not the
    // bearing to its centre: the bearing moves as the machine closes, so
    // aiming at it turns the machine as it drives and the tines arrive
    // crabbed.
    let engaging_target = matches!(
        task.state,
        StackState::Engaging | StackState::Inserting
    )
    .then(|| task.target)
    .flatten()
    .and_then(|id| crates.iter().find(|c| c.id == id))
    .map(|c| {
        let (_, into) = fork_approach_pose(c.pos, c.rot, pos, task.tuning.fork_standoff);
        into
    });

    // And once the haul is close to the stack, face the **column**. The
    // path leads to a tile beside the tower, not to the tower, so
    // following it leaves the forks pointing past it — and the cargo,
    // which rides out along the forks, lands anywhere but on the pile.
    // Squared to the **stack's own faces**, the same way the pickup is
    // squared to the crate's.
    //
    // Aiming at the bearing to the column has two faults. The bearing
    // moves as the machine closes, so it turns while it drives and
    // arrives crabbed; and it ignores which way the crate already on the
    // pile is lying, so the load is set down skew on top of a box that
    // is square. Taking the face normal of what is already there fixes
    // both, and falls back to the column's own bearing for the first
    // layer, when there is nothing up there to line up with.
    //
    // The heading itself is then twisted by `-aim`, the residual square-up
    // the carried crate still needs — see `Carry::square_yaw`. The
    // *machine's* facing and the *load's* facing are not the same thing
    // once a crate can be picked up at an arbitrary angle: aiming the
    // machine dead at the face normal aims the load `square_yaw` off it,
    // because the load is a rigid child of the machine, not reset to
    // square at pickup. Computed before the closure because `Carry` is
    // `Copy` and the closure would otherwise have to reach back into
    // `task` a second time.
    let aim = task.carry.map(|c| c.square_yaw()).unwrap_or(0.0);
    let approach_target = matches!(
        task.state,
        StackState::Hauling | StackState::Raising | StackState::Lining
    )
    .then(|| task.job.map(|j| j.spot))
    .flatten()
    .map(|spot| board.place_target(spot, plane, crates, task.tuning))
    .filter(|(p, _)| plane.flatten(*p - pos).length() <= task.tuning.place_reach)
    // One path for both arms: `place_target` already falls back to the
    // column (at `Quat::IDENTITY`) when nothing is stacked yet, so bare
    // floor now aims at a cardinal face of the tile rather than the
    // bearing to its centre — tiles are axis-aligned, so this is the
    // honest target either way.
    .map(|(p, rot)| {
        let (_, into) = fork_approach_pose(p, rot, pos, task.tuning.fork_standoff);
        yawed(into, -aim)
    });

    // A manoeuvring heading wins over everything.
    //
    // Set by the states that have to close a sideways error, which a
    // wheeled machine does by turning toward it rather than sliding
    // across. Taken rather than borrowed: it is a request for *this*
    // tick, and leaving it set would keep the machine pointing at a spot
    // it has already reached.
    let want = task
        .steer_to
        .take()
        .or(engaging_target)
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
    // Frozen while the load is being set down.
    //
    // The cargo hangs `carry_forward` out along the facing, so a facing
    // that is still being filtered swings the crate sideways as it is
    // placed: at a metre and a half of reach, one degree of slew is
    // 2.5 cm of lateral travel, and the load is being lowered onto a
    // tower while it happens. That is both the "shifting sideways while
    // placing" and the reason a finished three-high stack fell over.
    //
    // `Placing` is the whole of the set-down, from the mast starting
    // down to the release, so the heading the machine arrived on is the
    // heading it puts the load down on. `Lining` before it is where
    // aiming is allowed to happen.
    if !matches!(task.state, StackState::Placing) {
        let slew = (dt as f64 / 0.2).clamp(0.0, 1.0);
        task.facing = (task.facing + (want.normalize_or_zero() - task.facing) * slew)
            .normalize_or_zero();
        if task.facing.length_squared() < 1e-12 {
            task.facing = DVec3::X;
        }
    }

    // Known gap: the mast only moves in the states that step it, so
    // between jobs the tines stay wherever `Withdrawing` left them —
    // measured at 0.85 m when the next `Engaging` began — and the machine
    // walks to its next crate with the forks in the air. `Withdrawing`
    // sets the rest-height *target*; nothing gets there until `Engaging`.
    //
    // Stepping the mast here for Idle/Fetching/Settling/Hauling was tried
    // and cost a placed crate: `--verify` went from `[2, 2, 1, 0]` to
    // `[2, 1, 1, 0]`, deterministically, with the second crate on spot 1
    // coming off after the next haul. The tines have no collider, so the
    // mechanism is not obvious and has not been measured; until it is,
    // forks riding high between jobs is the lesser fault.

    match task.state {
        StackState::Idle => {
            board.prune(plane, crates, task.tuning);
            let Some(job) = board.next_job(plane, pos, crates, task.tuning, |id| task.blacklisted(id)) else {
                return StackAction::NoJob;
            };
            let Some(next) = crates.iter().find(|c| c.id == job.cargo) else {
                return StackAction::NoJob;
            };

            // Walk to the tile *next to* the crate, not onto it: a goal
            // inside a solid object is a goal the agent grinds against
            // forever, because the crate is 1 m across and the walker's
            // acceptance radius is under half that.
            //
            // Known gap, measured: this puts the machine up to ~1.9 m to one
            // SIDE of the face it must drive in along, which `Engaging`
            // closes with a long sideways crab. Walking instead to a point
            // out on the crate's face-normal line was tried and is worse —
            // the path's final leg then points anywhere (arrivals at 90°
            // off the normal) and a crate near the floor's edge gets a
            // goal off the grid, so the fetch never completes. The real fix
            // is an explicit approach leg after the walk, not a different
            // walk goal.
            let approach = approach_tile_at(plane, pos, next.pos, 1);
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
                task.blacklist_crate(job.cargo);
                return StackAction::NoJob;
            }
            task.target = Some(job.cargo);
            task.job = Some(job);
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
                //
                // Tine *top* to the crate's *bottom* face — "lower or
                // raise tines to box height", the step that used to not
                // exist at all: the forks always went to floor rest
                // height regardless of where the crate was. For a crate
                // resting on the floor this is 0 (tines flush with the
                // floor surface, bottoms 6 cm under it); there is no
                // pallet gap here, and that is accepted — pallet slots
                // are a future item, not this one.
                agent.stop();
                task.forks.target =
                    (info.pos.z - info.half_extents[2].abs() - plane.floor_z).max(0.0);
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
            // Re-targeted every tick — idempotent, and the Fetching-edge
            // set stays — so a crate that moves during the line-up is
            // still met at its own bottom face rather than one it has
            // since left.
            task.forks.target =
                (info.pos.z - info.half_extents[2].abs() - plane.floor_z).max(0.0);
            let arrived = task.forks.step(task.tuning.lift_rate, dt);

            // Step 1: square on to one face, a standoff back, tines clear.
            //
            // The walker put the machine *near* the crate on whatever
            // bearing its last path leg ran — usually across the crate
            // rather than at it. `fork_approach_pose` picks the nearest
            // cardinal face instead, so the drive-in runs along a fork
            // axis and between two faces rather than into a corner.
            let (stand, into) =
                fork_approach_pose(info.pos, info.rot, pos, task.tuning.fork_standoff);
            let lined_up = plane.flatten(pos - stand).length() <= task.tuning.creep_tolerance * 4.0;

            // Turned to face it, as well as forks down. Forks that reach
            // out sideways past the thing they are supposed to be going
            // under look exactly as wrong as they are.
            let aimed = task.facing.normalize_or_zero().dot(into) > 0.985;

            // Get square on to a face of the crate, a standoff back.
            //
            // The same three-leg manoeuvre the placement uses: turn to
            // face the spot, drive to it, turn to point at the crate. A
            // wheeled machine cannot do those at once.
            if !lined_up && task.lift_clock < task.tuning.creep_timeout {
                if let Some(act) =
                    drive_to_pose(task, plane, pos, stand, into, task.tuning.creep_speed)
                {
                    return act;
                }
            }

            if (!arrived || !aimed) && task.lift_clock < task.tuning.lift_timeout {
                return StackAction::None;
            }
            if task.lift_clock >= task.tuning.lift_timeout {
                task.blacklist_crate(id);
                task.abandon(agent);
                return StackAction::None;
            }

            // Lined up, square on, tines down and clear of the crate.
            // Now drive *in* — the machine moves, not the crate.
            task.creep_clock = 0.0;
            task.state = StackState::Inserting;
            StackAction::None
        }

        StackState::Inserting => {
            let Some(id) = task.target else {
                task.state = StackState::Idle;
                return StackAction::None;
            };
            let Some(info) = crates.iter().find(|c| c.id == id).copied() else {
                task.abandon(agent);
                return StackAction::None;
            };

            task.creep_clock += dt;
            if task.creep_clock >= task.tuning.creep_timeout {
                // Wedged on the way in. Give this crate up rather than
                // grinding against whatever is in the way forever.
                task.blacklist_crate(id);
                task.abandon(agent);
                return StackAction::None;
            }

            // The job may have been abandoned from under us — e.g. its
            // spot filled by another route — between ticks. Bail before
            // the crate goes kinematic, or this path leaks a permanently
            // kinematic crate with nowhere to take it.
            let Some(job) = task.job else {
                task.carry = None;
                task.abandon(agent);
                return StackAction::None;
            };

            // Where the machine must stand for the crate to be sitting on
            // its tines: exactly `carry_forward` back along the facing.
            let want = DVec3::new(
                info.pos.x - task.facing.x * task.tuning.carry_forward,
                info.pos.y - task.facing.y * task.tuning.carry_forward,
                pos.z,
            );
            let remaining = creep_remaining(pos, task.facing, want);

            // Hoisted: also used below, both for the lateral correction
            // and for the pickup-edge `Carry::offset`.
            let f = task.facing.normalize_or_zero();
            let f = if f.length_squared() < 0.5 { DVec3::X } else { f };
            let left = DVec3::new(-f.y, f.x, 0.0);

            // The crate's lateral position is re-read and corrected every
            // tick, not just once at `Engaging`'s `lined_up` latch — omni,
            // so a sideways component is a legal command here. `lat > 0`
            // means the crate sits to the machine's left (matches
            // `Carry::offset.y = d.dot(left)` below), and commanding
            // `+left` moves the machine that way, toward the crate's
            // centreline.
            let lat = (info.pos - pos).dot(left);
            // Deadband + floor + cap, or the correction stalls against
            // friction exactly as the forward creep once did, or
            // limit-cycles across the tolerance forever.
            let lat_v = if lat.abs() > task.tuning.creep_tolerance {
                lat.signum() * (lat.abs() * LATERAL_K).clamp(LATERAL_MIN, LATERAL_MAX)
            } else {
                0.0
            };

            if remaining > task.tuning.creep_tolerance || lat.abs() > task.tuning.creep_tolerance {
                // Quick until the tips reach the crate, then slow for the
                // part that actually matters.
                //
                // The whole insertion is only `FORK_REACH` of travel, but
                // the machine starts a standoff back from that — and the
                // approach half has nothing between it and the crate, so
                // creeping it at the delicate speed is time spent moving
                // through empty air.
                let tips_at = FORK_REACH + task.tuning.creep_tolerance;
                let cruise = if remaining > tips_at {
                    task.tuning.creep_speed
                } else {
                    task.tuning.insert_speed
                };
                // Brake into the stop rather than driving at full
                // speed until the tolerance is met.
                //
                // `Creep` sets velocity outright, so a machine still
                // doing `insert_speed` on the tick before it arrives
                // travels another `speed * dt` past the mark — and
                // the thing just past the mark is the crate. Measured
                // without this: the machine drove into the box and
                // pushed it two metres across the floor, with
                // `remaining` stuck at 0.14 m because the target was
                // moving away as fast as the machine closed on it.
                //
                // Tapering over `BRAKE_ZONE` rather than to zero at
                // the tolerance: a speed that reaches zero exactly
                // where the test passes is a machine that creeps the
                // last centimetre for ever.
                const BRAKE_ZONE: f64 = 0.25;
                let fwd = if remaining > task.tuning.creep_tolerance {
                    let taper = (remaining / BRAKE_ZONE).clamp(0.15, 1.0);
                    cruise * taper
                } else {
                    // Forward leg is already done; only the lateral
                    // correction below still has work to do.
                    0.0
                };
                return StackAction::Creep {
                    velocity: f * fwd + left * lat_v,
                };
            }

            // The tines are under it. *Now* it is cargo: it goes
            // kinematic on this edge and not before, so it sits on the
            // floor under gravity and under contacts right up to the
            // moment the forks take its weight.
            //
            // Record its actual pose *relative to the machine*, in the
            // facing frame — this is the whole carry model. The crate
            // goes kinematic where it lies; nothing moves it this tick,
            // and the caller stops the machine dead. There is no slide
            // and no snap because there is nothing left to animate: the
            // offset recorded here **is** where the crate already is, and
            // every later tick reproduces that same rigid relationship
            // from wherever the machine has moved to since.
            //
            // `f`/`left` reused from the hoisted bindings above.
            let d = info.pos - pos;
            let yaw = f.y.atan2(f.x) as f32;
            task.carry = Some(Carry {
                id,
                half_extents: info.half_extents,
                offset: DVec3::new(
                    d.dot(f),
                    d.dot(left),
                    info.pos.z - (plane.floor_z + task.forks.height),
                ),
                rot_offset: Quat::from_rotation_z(-yaw) * info.rot,
            });
            task.target = None;
            task.final_approach = false;
            task.lift_clock = 0.0;
            // Step 3: lift a little, not to full height. Travelling with
            // the load high is what makes a laden machine tip. Relative
            // to where the tines already are, not to the floor rest
            // height — Fetching may have parked them anywhere from 0 up
            // to the crate's own height.
            task.forks.target = task.forks.height + task.tuning.place_clearance;
            task.state = StackState::Lifting;

            // Plan the haul *before* committing to the lift, and give the
            // crate straight back if there is nowhere to take it. The
            // return value used to be dropped here, so a failed plan left
            // the agent holding a crate until `stuck()` fired a second
            // later.
            let approach =
                approach_tile_at(plane, pos, board.column(job.spot, plane), 1);
            agent.set_goal(plane.tile_center(approach.0, approach.1));
            if !super::replan(agent, plane, src, pos, &board.blocked_tiles()) {
                task.carry = None;
                task.blacklist_crate(id);
                task.abandon(agent);
                return StackAction::None;
            }

            StackAction::Pickup { id }
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
                task.blacklist_crate(carry.id);
                task.abandon(agent);
                return StackAction::None;
            }

            let pose = carried_pose(pos, task.facing, task.forks, carry, plane.floor_z);

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

            if task.forks.step(task.tuning.lift_rate, dt) {
                task.lift_clock = 0.0;
                task.state = StackState::Hauling;
            }
            // Stationary — "lift, THEN move". The machine was already
            // stopped dead by the `Pickup` edge; this just holds that
            // stop while the mast finishes travelling.
            StackAction::Creep { velocity: DVec3::ZERO }
        }

        StackState::Raising => {
            let Some(carry) = task.carry else {
                task.state = StackState::Idle;
                return StackAction::None;
            };
            if !crates.iter().any(|c| c.id == carry.id) {
                task.carry = None;
                task.abandon(agent);
                return StackAction::None;
            }

            let pose = carried_pose(pos, task.facing, task.forks, carry, plane.floor_z);
            // Cargo is in hand, so set it down rather than drop to Idle.
            let Some(job) = task.job else {
                return release_here(task, agent, pose);
            };
            let up = (agent_transform.rot * glam::Vec3::Z).as_dvec3();
            let tilt = up.dot(DVec3::Z).clamp(-1.0, 1.0).acos().to_degrees();
            if tilt > task.tuning.max_carry_tilt_deg {
                return release_here(task, agent, pose);
            }

            // Re-read the live top every tick rather than latching it on
            // entry. A tower knocked from two layers to none while the
            // mast is travelling would otherwise get its next crate
            // placed at 2.5 m and dropped two metres onto the floor — the
            // floor fallback in `place_target` lowers the target right
            // along with it.
            let (top, _) = board.place_target(job.spot, plane, crates, task.tuning);
            // Step 6: the measured top face **plus a little bit**, so the
            // load clears the crate below on the way in rather than being
            // dragged across its top face.
            task.forks.target =
                place_fork_height(top.z, carry, plane.floor_z, task.tuning)
                    + task.tuning.place_clearance;

            task.lift_clock += dt;
            let timed_out = task.lift_clock >= task.tuning.lift_timeout;
            if !task.forks.step(task.tuning.lift_rate, dt) && !timed_out {
                return StackAction::Creep { velocity: DVec3::ZERO };
            }
            if timed_out {
                return release_here(task, agent, pose);
            }

            // At height, still standing off the pile. Step 5: go in.
            task.lift_clock = 0.0;
            task.creep_clock = 0.0;
            // A fresh placement starts unaligned; the latch belongs to
            // this one manoeuvre and must not carry over from the last.
            task.aligned = false;
            task.state = StackState::Lining;
            StackAction::Creep { velocity: DVec3::ZERO }
        }

        StackState::Lining => {
            let Some(carry) = task.carry else {
                task.state = StackState::Idle;
                return StackAction::None;
            };
            if !crates.iter().any(|c| c.id == carry.id) {
                task.carry = None;
                task.abandon(agent);
                return StackAction::None;
            }

            let pose = carried_pose(pos, task.facing, task.forks, carry, plane.floor_z);
            // Cargo is in hand, so set it down rather than drop to Idle.
            let Some(job) = task.job else {
                return release_here(task, agent, pose);
            };
            let (column, rot) = board.place_target(job.spot, plane, crates, task.tuning);

            task.creep_clock += dt;
            if task.creep_clock >= task.tuning.creep_timeout {
                // Could not get the load over the column. Put it down
                // where it stands rather than carrying it forever.
                return release_here(task, agent, pose);
            }

            // Creep in until the **cargo** is over the column. The load is
            // already at height, so this passes over the pile rather than
            // through it — which is the whole reason the raise happens
            // before this and not after.
            // Drive **forward** until the cargo is over the column.
            //
            // The load rides at a fixed offset from the machine, in the
            // facing frame — `carry.offset` — so where the cargo ends up
            // is decided by how far the machine drives, not by steering
            // the cargo directly. Creeping along `column - pose.pos`
            // instead asks the machine to move in a direction that
            // does not close that gap, and it sat there until the
            // timeout. Built from the offset rather than the constant
            // `carry_forward`, because a crate can be picked up
            // off-centre from the nominal carry point, and it is the
            // *crate*, not that nominal point, that has to land on the
            // column.
            let f = task.facing.normalize_or_zero();
            let left = DVec3::new(-f.y, f.x, 0.0);
            let want = column - (f * carry.offset.x + left * carry.offset.y);
            let want = DVec3::new(want.x, want.y, pos.z);
            // Get to the pose the load needs, then drive in.
            //
            // `drive_to_pose` is the three-leg manoeuvre — turn, drive,
            // turn — and it is what a wheeled machine can actually do.
            // Correcting position and heading together does not converge;
            // see that function's docs for the measurement.
            //
            // The pose is out at the standoff, not up against the tower:
            // aligning at `carry_forward` puts the load over the stack
            // while the machine is still turning, and the crates touch.
            let (stand, into) =
                fork_approach_pose(column, rot, pos, task.tuning.fork_standoff);
            // Twisted by the load's own residual square-up, so the
            // *cargo* — not just the machine — arrives square to the box
            // below. See `Carry::square_yaw`.
            let into = yawed(into, -carry.square_yaw());

            // Latched: once the standoff pose is reached this is never
            // re-evaluated, or the drive-in below moves the machine off
            // the spot and this sends it straight back. See `aligned`.
            if !task.aligned {
                // The standoff leg: nothing is near the standoff pose on
                // this approach, so it runs at jockeying speed rather than
                // the delicate insert speed — mirrors `Engaging`'s own
                // standoff leg, which uses `creep_speed` for the same
                // reason. `insert_speed` stays reserved for the drive-in
                // below, which is the leg that passes over the pile.
                if let Some(act) =
                    drive_to_pose(task, plane, pos, stand, into, task.tuning.creep_speed)
                {
                    return act;
                }
                task.aligned = true;
            }

            // In over the column, along the facing.
            //
            // The floor is a speed, not a fraction, for the reason
            // `drive_to_pose` documents: the solver still applies
            // friction, and a command below what friction cancels is
            // a machine that does not move at all.
            let remaining = creep_remaining(pos, task.facing, want);

            // The load's lateral position over the column is re-read and
            // corrected every tick, exactly as `Inserting` corrects it
            // against the crate being picked up — `drive_to_pose`'s
            // standoff-pose latch above only gets the machine within
            // `on_spot` (`creep_tolerance * 2`) laterally, and without this
            // the drive-in below closed only the forward axis, so a stack
            // that had crept landed centred on nothing: measured 2.4–4.8 cm
            // off before this. `column - pos` is the vector from the
            // machine to the target column; the crate itself sits
            // `carry.offset.y` further along `left` than the machine does
            // (see `carried_pose`), so that offset is subtracted back out
            // to get the crate's own lateral error.
            let lat = (column - pos).dot(left) - carry.offset.y;
            let lat_v = if lat.abs() > task.tuning.creep_tolerance {
                lat.signum() * (lat.abs() * LATERAL_K).clamp(LATERAL_MIN, LATERAL_MAX)
            } else {
                0.0
            };

            if remaining.abs() > task.tuning.creep_tolerance || lat.abs() > task.tuning.creep_tolerance {
                const BRAKE_ZONE: f64 = 0.25;
                const CRAWL: f64 = 0.35;
                let fwd = if remaining.abs() > task.tuning.creep_tolerance {
                    let taper = (remaining.abs() / BRAKE_ZONE).clamp(0.0, 1.0);
                    (task.tuning.insert_speed * taper).max(CRAWL) * remaining.signum()
                } else {
                    // Forward leg is already done; only the lateral
                    // correction below still has work to do.
                    0.0
                };
                return StackAction::Creep {
                    velocity: task.facing * fwd + left * lat_v,
                };
            }

            // Over the column and at height. Steps 6 and 7 are `Placing`.
            task.lift_clock = 0.0;
            task.state = StackState::Placing;
            StackAction::Creep { velocity: DVec3::ZERO }
        }

        StackState::Placing => {
            let Some(carry) = task.carry else {
                task.state = StackState::Idle;
                return StackAction::None;
            };
            if !crates.iter().any(|c| c.id == carry.id) {
                task.carry = None;
                task.abandon(agent);
                return StackAction::None;
            }

            let pose = carried_pose(pos, task.facing, task.forks, carry, plane.floor_z);
            // Cargo is in hand, so set it down rather than drop to Idle.
            let Some(job) = task.job else {
                return release_here(task, agent, pose);
            };
            let up = (agent_transform.rot * glam::Vec3::Z).as_dvec3();
            let tilt = up.dot(DVec3::Z).clamp(-1.0, 1.0).acos().to_degrees();
            if tilt > task.tuning.max_carry_tilt_deg {
                return release_here(task, agent, pose);
            }

            // Step 7: down onto the measured top face, `drop_clearance`
            // above it, the same target `Raising` closed on.
            let (top, _) = board.place_target(job.spot, plane, crates, task.tuning);
            task.forks.target = place_fork_height(top.z, carry, plane.floor_z, task.tuning);

            task.lift_clock += dt;
            let timed_out = task.lift_clock >= task.tuning.lift_timeout;
            if !task.forks.step(task.tuning.lift_rate, dt) && !timed_out {
                return StackAction::Creep { velocity: DVec3::ZERO };
            }

            // Resting on the stack. Let go **from the forks**, not from
            // the column.
            //
            // Releasing at the column is what made the crate jump at this
            // end of the haul: the agent stands off the stack, so the
            // cargo was teleported the remaining distance to the tower on
            // the frame it was let go. A forklift sets its load down where
            // its forks are; if the forks are not over the tower, the
            // agent has not driven close enough, and that is a placement
            // problem rather than something to paper over by flinging.
            //
            // The rotation is the carried `pose.rot`, not identity. The
            // load was already aimed square to the stack below during
            // `Lining` — see `Carry::square_yaw` and where it twists the
            // approach heading — so it lands square without a snap here.
            // Squaring at the release used to be load-bearing when the
            // carry followed the machine's raw facing outright; now the
            // carry is a rigid child of the machine and the squaring
            // already happened upstream.
            let drop = CarryPose { pos: pose.pos, rot: pose.rot };
            task.carry = None;
            board.record_placed(job.spot, carry.id);
            task.settle_clock = 0.0;
            task.lift_clock = 0.0;
            task.creep_clock = 0.0;
            // Step 8 comes next: drop the tines out from under it and
            // back straight out before turning away.
            task.withdraw_from = Some(pos);
            task.forks.target =
                (task.forks.height - task.tuning.tine_drop).max(task.tuning.fork_rest_height);
            task.state = StackState::Withdrawing;
            let _ = timed_out;

            // The agent is deliberately *not* given a goal here. It backs
            // straight out under `Withdrawing` first; routing away while
            // the tines are still inside the stack is what sweeps them
            // through it.
            agent.stop();

            // The crate is released `drop_clearance` above the *measured*
            // top face and falls that far — not zero. `velocity: ZERO`
            // still stands: that drop is small enough to land under
            // `RESTITUTION_THRESHOLD` (see `place_fork_height` and the
            // tuning derivation), so it does not need the downward nudge
            // `RELEASE_SPEED` gives a crate let go higher up, in mid-air.
            StackAction::Release {
                id: carry.id,
                pose: drop,
                velocity: DVec3::ZERO,
            }
        }

        StackState::Withdrawing => {
            task.creep_clock += dt;
            // The mast drops clear of the crate while the machine reverses
            // — both at once, the way a real one does it.
            task.forks.step(task.tuning.lift_rate, dt);

            let started = task.withdraw_from.unwrap_or(pos);
            let travelled = plane.flatten(pos - started).length();
            // Far enough that the tine tips are clear of the crate's
            // near face — which is the *tine reach*, not the whole
            // carry offset. The load sat `carry_forward` out, but the
            // tips only ever reached `FORK_REACH` past the chassis,
            // so backing the full offset reverses half a metre
            // further than it needs to and costs 290 ticks a cycle.
            let clear = travelled >= FORK_REACH + task.tuning.creep_tolerance;

            if !clear && task.creep_clock < task.tuning.creep_timeout {
                return StackAction::Creep {
                    velocity: -task.facing * task.tuning.creep_speed,
                };
            }

            // Clear of the stack. Step 9: down to transport height, and
            // only now is it safe to turn away and look for the next one.
            task.withdraw_from = None;
            task.creep_clock = 0.0;
            task.forks.target = task.tuning.fork_rest_height;
            task.state = StackState::Settling;
            StackAction::None
        }

        StackState::Hauling => {
            let Some(carry) = task.carry else {
                task.state = StackState::Idle;
                return StackAction::None;
            };
            let pose = carried_pose(pos, task.facing, task.forks, carry, plane.floor_z);

            // Tipped over: the hold point has swung out over open space
            // and the cargo is sweeping the scene sideways. Put it down
            // where it is rather than carrying on.
            let up = (agent_transform.rot * glam::Vec3::Z).as_dvec3();
            let tilt = up.dot(DVec3::Z).clamp(-1.0, 1.0).acos().to_degrees();
            if tilt > task.tuning.max_carry_tilt_deg {
                return release_here(task, agent, pose);
            }

            // Cargo is in hand, so set it down rather than drop to Idle.
            let Some(job) = task.job else {
                return release_here(task, agent, pose);
            };
            let (column, rot) = board.place_target(job.spot, plane, crates, task.tuning);

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
                    //
                    // Stop **short**, by the same standoff a pickup uses.
                    // The haul used to aim through the tower and drive
                    // right up to it, so the load was already over the
                    // pile by the time the mast moved — and raising a
                    // crate through the stack it is about to sit on is
                    // how a machine knocks its own work down. Lining up
                    // and lifting happen out here; going in is `Lining`'s
                    // job, after the mast is at height.
                    //
                    // The standing spot is on a **face normal** of what is
                    // already on the pile, not on the bearing the machine
                    // happened to arrive from. Approaching off-axis puts
                    // the load down rotated against the box below it,
                    // which is the same skew the pickup end had.
                    // `column`/`rot` were already read from `place_target`
                    // above, once for this tick — both arms unified, since
                    // `place_target` itself already falls back to the
                    // column at `Quat::IDENTITY` when nothing is stacked.
                    // Computed once, here, at the latch by design:
                    // `set_goal` clears the path, so recomputing every
                    // tick would re-plan every tick and the agent would
                    // never take a step (see above). `Lining` is what
                    // corrects per tick, once the mast is at height.
                    let (aim, _) =
                        fork_approach_pose(column, rot, pos, task.tuning.haul_standoff);
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
                // Step 4 ends at the standoff, not on top of the pile.
                // Raising happens out here, and only then does the
                // machine go in — which is `Lining`.
                let stood_off = plane.flatten(pos - column).length();
                if stood_off
                    <= task.tuning.haul_standoff + agent.tuning.goal_radius
                    && agent.state == AgentState::Arrived
                {
                    agent.stop();
                    task.lift_clock = 0.0;
                    task.creep_clock = 0.0;
                    task.state = StackState::Raising;
                }
                // `None`, not a `Creep::ZERO`: `Hauling` is the walker's
                // state, not a manoeuvre — `manoeuvring()` does not
                // include it, so `super::drive_agent` is still driving
                // the body this tick. `Creep` would write velocity
                // outright and fight the walker's own command; `None`
                // lets the machine coast the one tick it takes to notice
                // it has arrived, with the load riding over the pile
                // exactly as `carried_pose` places it regardless.
                return StackAction::None;
            }

            // Wedged on the way. Put the crate down rather than carrying
            // it around forever — that also restores the one-crate
            // invariant from this path.
            if agent.stuck() {
                return release_here(task, agent, pose);
            }

            StackAction::None
        }

        StackState::Settling => {
            task.settle_clock += dt;
            // No cargo held here, so there is nothing to set down.
            let Some(job) = task.job else {
                task.job = None;
                task.state = StackState::Idle;
                return StackAction::None;
            };

            // Chain-anchored: is the topmost crate actually resting on
            // the one below it, rather than measured against the fixed
            // tile the stack was started on. See `JobBoard::chain`.
            if board.top_placed(job.spot, plane, crates, task.tuning).is_some_and(|k| k.sleeping) {
                task.job = None;
                task.state = StackState::Idle;
                return StackAction::None;
            }

            // A crate that has been rocking for several seconds is not
            // going to settle. Re-deriving the pile next tick is the
            // recovery, and it is why the chain counts from the world
            // rather than from a tally.
            if task.settle_clock >= task.tuning.settle_timeout {
                task.job = None;
                task.state = StackState::Idle;
            }
            StackAction::None
        }
    }
}

/// Put the carried crate down where the agent is standing.
fn release_here(task: &mut StackTask, agent: &mut Agent3D, pose: CarryPose) -> StackAction {
    let Some(carry) = task.carry.take() else {
        return StackAction::None;
    };
    task.job = None;
    task.state = StackState::Idle;
    agent.stop();
    // Reset the mast rather than leaving it wherever the abandoned carry
    // left it — otherwise the next `Fetching` pickup starts with the
    // tines already at some arbitrary height instead of parked.
    task.forks.target = task.tuning.fork_rest_height;
    // The pose is handed back as-is, including its rotation — an
    // emergency drop (tipped over, timed out, wedged) does not square the
    // crate up the way a normal `Placing` release does, and a crate that
    // lands on a corner here is an accepted caveat of bailing out rather
    // than something this path tries to fix.
    StackAction::Release {
        id: carry.id,
        pose,
        velocity: DVec3::new(0.0, 0.0, -RELEASE_SPEED),
    }
}

/// Where a machine should stand to put its forks into a crate.
///
/// Square on to one face, `standoff` back from the crate's centre along
/// that face's normal, so driving straight forward from here takes the
/// tines in along the fork axis and between the crate's faces rather than
/// across a corner.
///
/// # Why a cardinal face and not the bearing to the crate
///
/// A crate is a box. Approaching on the bearing the machine happens to
/// arrive on means meeting it at whatever angle that is, and a fork
/// entering a box off-axis catches a corner and shoves it. Picking the
/// nearest face makes the approach square by construction, and the four
/// faces are exactly the four directions a 4-connected route can leave
/// from anyway.
pub fn fork_approach_pose(
    crate_pos: DVec3,
    crate_rot: Quat,
    from: DVec3,
    standoff: f64,
) -> (DVec3, DVec3) {
    let d = DVec3::new(from.x - crate_pos.x, from.y - crate_pos.y, 0.0);

    // The crate's **own** four faces, not the world's.
    //
    // Squaring up to world axes lines the machine up with north rather
    // than with the box, so a crate lying at any other angle got picked
    // up skew — and then snapped straight as it became cargo, which
    // reads as the box magically squaring itself on the forks. A machine
    // lines up with its load.
    let fx = (crate_rot * glam::Vec3::X).as_dvec3();
    let face_x = DVec3::new(fx.x, fx.y, 0.0).normalize_or_zero();
    let face_x = if face_x.length_squared() < 0.5 { DVec3::X } else { face_x };
    let face_y = DVec3::new(-face_x.y, face_x.x, 0.0);

    // Whichever of the four is most directly toward the approach.
    let (along_x, along_y) = (d.dot(face_x), d.dot(face_y));
    let normal = if d.length_squared() < 1e-12 {
        face_x
    } else if along_x.abs() >= along_y.abs() {
        face_x * along_x.signum()
    } else {
        face_y * along_y.signum()
    };
    let stand = DVec3::new(
        crate_pos.x + normal.x * standoff,
        crate_pos.y + normal.y * standoff,
        from.z,
    );
    // Facing is *into* the crate, the opposite of the face normal.
    (stand, -normal)
}

/// Rotate a vector about Z by `by` radians, in the XY plane.
///
/// `glam::Quat * DVec3` does not exist — the crate's quaternions only
/// multiply `Vec3`/`DVec3` isn't implemented for `Quat` at all here — so
/// planar aiming corrections go through this instead of building a
/// throwaway quaternion for a rotation that is two dimensional anyway.
fn yawed(v: DVec3, by: f64) -> DVec3 {
    let (s, c) = by.sin_cos();
    DVec3::new(v.x * c - v.y * s, v.x * s + v.y * c, v.z)
}

/// How far a machine still has to creep, along its own facing, to put
/// the tines where they should be.
///
/// Positive means forward, negative means back out. The result is signed
/// distance along `facing` only: sideways error is deliberately ignored,
/// because a fork already lined up on a face has none worth correcting
/// and chasing it would make the machine crab sideways into the crate.
///
/// # Why this is not the walker's job
///
/// [`super::drive_agent`] routes over tiles and stops within
/// `goal_radius` — 0.30 m by default, on whichever side it approached
/// from. Fork insertion needs a tolerance an order of magnitude tighter
/// than that along one axis. So the approach uses the walker to get
/// *near* and this to close the last half metre, which is the same split
/// a real machine makes between driving and jockeying.
pub fn creep_remaining(pos: DVec3, facing: DVec3, want: DVec3) -> f64 {
    let f = DVec3::new(facing.x, facing.y, 0.0).normalize_or_zero();
    if f.length_squared() < 0.5 {
        return 0.0;
    }
    DVec3::new(want.x - pos.x, want.y - pos.y, 0.0).dot(f)
}

/// Move a machine with omni wheels to a pose: position `stand`, heading
/// `face`.
///
/// Returns `None` once it is there, and otherwise the action for this
/// tick.
///
/// # Why this is one move and not three legs
///
/// These machines have omni wheels, so translation and rotation are
/// independent: the body can slide in any direction regardless of which
/// way it points. That makes "get to this pose" a single move — drive
/// straight at the spot while turning to the final heading — rather than
/// the turn/drive/turn a wheeled machine is stuck with.
///
/// It is worth saying why, because the wheeled version was tried and it
/// is genuinely hard. The reachable set from a pose is "along the current
/// heading", so position and heading cannot both be corrected at once:
/// turning does not reduce the distance, and a controller that tests both
/// together has no gradient to follow. Measured going round in circles,
/// the offset grew from 1.7 m to 3.1 m while the cross-track error
/// flipped sign every second. Splitting it into sequential legs fixes the
/// divergence but introduces its own stalls at the hand-off points.
///
/// Omni wheels dissolve the problem rather than solving it. The cost is
/// that the machine strafes, which reads as a hovercraft unless the
/// *model* has omni wheels on it — so the mesh has to show them.
fn drive_to_pose(
    task: &mut StackTask,
    plane: NavPlane,
    pos: DVec3,
    stand: DVec3,
    face: DVec3,
    speed: f64,
) -> Option<StackAction> {
    // Generous next to `creep_tolerance`: this is "close enough to stop",
    // and a tighter one only makes the machine hunt.
    let on_spot = task.tuning.creep_tolerance * 2.0;
    // cos(4 degrees). Tight enough that the tines go in square, loose
    // enough that the facing filter can actually reach it.
    const AIMED: f64 = 0.9976;

    let off = plane.flatten(stand - pos);
    let facing = task.facing.normalize_or_zero();

    // Always turning toward the final heading. With omni wheels this
    // costs nothing — it does not steer the translation.
    task.steer_to = Some(face);

    let there = off.length() <= on_spot;
    let square = facing.dot(face) >= AIMED;
    if there && square {
        return None;
    }

    if there {
        // On the spot, still turning. Hold position rather than drifting
        // off it while the heading catches up.
        return Some(StackAction::Creep { velocity: DVec3::ZERO });
    }

    // Straight at the spot, in world space, easing into the stop so it
    // does not overshoot.
    //
    // The floor on the taper is what stops the last centimetre
    // taking forever. `Creep` sets velocity outright, but the solver
    // still applies friction, and a command below what friction can
    // cancel is a machine that does not move at all — measured
    // stalled 7 mm short of a 40 mm tolerance for a whole timeout,
    // heading already perfect. So the floor is a *speed*, not a
    // fraction: whatever the taper says, the machine is told to move
    // fast enough to actually go.
    const BRAKE_ZONE: f64 = 0.4;
    const CRAWL: f64 = 0.35;
    let taper = (off.length() / BRAKE_ZONE).clamp(0.0, 1.0);
    let speed = (speed * taper).max(CRAWL);
    Some(StackAction::Creep {
        velocity: off.normalize_or_zero() * speed,
    })
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

    /// `offset.z` is recorded a few millimetres *sunk* into the floor at
    /// pickup — here `0.49` against a `0.5` half-height, 1 cm of sink — so
    /// the fork height has to correct for it, not just clear the measured
    /// top face by `drop_clearance`.
    #[test]
    fn place_fork_height_clears_the_measured_top_face() {
        let t = tuning();
        let carry = Carry {
            id: CrateId(1),
            half_extents: [0.5, 0.5, 0.5],
            offset: DVec3::new(0.0, 0.0, 0.49),
            rot_offset: Quat::IDENTITY,
        };
        let top_z = 1.0;
        let floor_z = 0.0;
        let got = place_fork_height(top_z, carry, floor_z, t);
        let want = 1.01 + t.drop_clearance;
        assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
    }

    /// The tines have somewhere to be.
    ///
    /// `carry_forward` is where the load rides, so the gap between the
    /// machine's front face and the load's near face is exactly the space
    /// the forks occupy. Sized only to dodge the solver's contact
    /// tolerance — 5 cm, as it was — the mesh has nowhere to go and gets
    /// drawn straight through the crate. Measured at 0.8 m of tine inside
    /// a box the machine was supposed to be carrying.
    #[test]
    fn the_tines_span_the_whole_crate() {
        let t = tuning();
        // Half the machine and half the crate are both 0.5 here, so the
        // crate is 1.0 deep.
        let crate_depth = 1.0;
        assert!(
            FORK_REACH >= crate_depth - 1e-9,
            "tines reach {FORK_REACH:.2} m under a {crate_depth:.2} m crate \
             — the load is carried on the tips with its far half hanging \
             off the end",
        );

        // And the load sits back against the mast rather than out on the
        // tips: the gap between the chassis face and the crate's near
        // face is the solver clearance, not a fork length.
        let gap = t.carry_forward - 0.5 - 0.5;
        assert!(
            gap > 0.0 && gap < FORK_REACH * 0.5,
            "the load rides {gap:.3} m off the chassis; it should be hard \
             against the mast, not carried at arm's length",
        );
    }

    /// A loaded machine stops far enough back that its **load** clears
    /// the stack, with the walker's slop already paid for.
    ///
    /// `fork_standoff` is sized for an empty machine: tine tips a metre
    /// out, clear of a crate on the floor. A loaded one reaches further
    /// than its tines — the load's far face is at `carry_forward +
    /// crate_half` — and the walker stops within `goal_radius` of its
    /// goal on either side. Stopping at the empty standoff put the load
    /// 5 cm from the stacked crate, so it arrived up to a quarter of a
    /// metre inside it. That was the collision while lining up before
    /// the raise.
    #[test]
    fn the_haul_standoff_clears_the_load_after_walker_slop() {
        let t = tuning();
        let walk = WalkTuning3D::default();
        // The load's far face, out from the machine's centre.
        let load_face = t.carry_forward + 0.5;
        // Worst case: the walker stopped `goal_radius` closer than asked.
        let worst_centre = t.haul_standoff - walk.goal_radius;
        // Where that puts the load's face relative to the column, less
        // the stacked crate's own half-width.
        let gap = (worst_centre - load_face) - 0.5;
        assert!(
            gap >= FORK_CLEARANCE - 1e-9,
            "after the walker's worst-case stop the load is {gap:.3} m from \
             the stacked crate; it must keep at least {FORK_CLEARANCE:.3} m",
        );
    }

    /// And at the standoff the tine *tips* are outside the crate, so
    /// squaring up cannot sweep a metre of fork through it.
    #[test]
    fn the_fork_standoff_keeps_the_tines_clear() {
        let t = tuning();
        // Where the tips sit relative to the crate's near face. The
        // machine's centre is `fork_standoff` out; the tips are
        // `FORK_REACH` ahead of that centre's front face.
        let tips_to_near_face = t.fork_standoff - 0.5 - FORK_REACH - 0.5;
        assert!(
            tips_to_near_face > 0.0,
            "at the standoff the tine tips are {tips_to_near_face:.3} m \
             past the crate's near face — already inside it before the \
             machine has driven anywhere",
        );
    }

    /// The approach is square on to a face, never across a corner.
    ///
    /// A fork entering a box off-axis catches a corner and shoves it,
    /// which is the whole reason this exists rather than just aiming at
    /// the crate's centre from wherever the machine happens to be.
    #[test]
    fn the_fork_approach_squares_up_to_a_face() {
        let c = DVec3::new(4.0, 4.0, 0.5);
        // Coming from several bearings, including diagonals, which are
        // the ones a bearing-based approach gets wrong.
        for from in [
            DVec3::new(0.0, 4.2, 0.35),
            DVec3::new(4.3, 0.0, 0.35),
            DVec3::new(8.0, 3.6, 0.35),
            DVec3::new(1.0, 1.0, 0.35),
            DVec3::new(7.0, 6.5, 0.35),
        ] {
            let (stand, facing) = fork_approach_pose(c, Quat::IDENTITY, from, 1.5);

            // Square: the approach runs along exactly one axis.
            let along = DVec3::new(stand.x - c.x, stand.y - c.y, 0.0);
            let off_axis = along.x.abs().min(along.y.abs());
            assert!(
                off_axis < 1e-9,
                "from {from:?} the standing spot {stand:?} is off both axes \
                 by {off_axis:.4} — that is a corner approach",
            );

            // And the machine faces into the crate from there.
            let to_crate = DVec3::new(c.x - stand.x, c.y - stand.y, 0.0).normalize();
            assert!(
                (facing - to_crate).length() < 1e-9,
                "from {from:?} facing is {facing:?} but the crate is at \
                 {to_crate:?}",
            );
        }
    }

    /// And it stands the asked-for distance back, so the tines start
    /// clear of the crate rather than already inside it.
    #[test]
    fn the_fork_approach_stands_off_by_the_distance_asked_for() {
        let c = DVec3::new(0.0, 0.0, 0.5);
        for standoff in [0.8, 1.5, 2.0] {
            let (stand, _) = fork_approach_pose(c, Quat::IDENTITY, DVec3::new(5.0, 0.3, 0.35), standoff);
            let d = DVec3::new(stand.x - c.x, stand.y - c.y, 0.0).length();
            assert!(
                (d - standoff).abs() < 1e-9,
                "asked for {standoff} m, got {d:.4} m",
            );
        }
    }

    /// The creep measures along the machine's own axis, and is signed.
    #[test]
    fn the_creep_is_signed_distance_along_the_facing() {
        let pos = DVec3::new(0.0, 0.0, 0.35);
        let facing = DVec3::X;

        // A metre ahead.
        let ahead = creep_remaining(pos, facing, DVec3::new(1.0, 0.0, 0.35));
        assert!((ahead - 1.0).abs() < 1e-9, "ahead: {ahead}");

        // Half a metre behind, which is what backing out asks for.
        let behind = creep_remaining(pos, facing, DVec3::new(-0.5, 0.0, 0.35));
        assert!((behind + 0.5).abs() < 1e-9, "behind: {behind}");
    }

    /// Sideways error is ignored rather than corrected.
    ///
    /// A machine already lined up on a face has no sideways error worth
    /// chasing, and steering at one would make it crab into the crate it
    /// is trying to get under.
    #[test]
    fn the_creep_ignores_sideways_error() {
        let pos = DVec3::new(0.0, 0.0, 0.35);
        // A metre ahead but also a metre to the side: the forward part is
        // still exactly one metre.
        let d = creep_remaining(pos, DVec3::X, DVec3::new(1.0, 1.0, 0.35));
        assert!((d - 1.0).abs() < 1e-9, "got {d}, wanted the forward part only");
    }

    /// The carry is a rigid child of the machine: recorded once at
    /// pickup, it reproduces the exact same offset and rotation relative
    /// to the machine no matter where the machine has since moved to.
    ///
    /// Pose A is where the crate would have been recorded (facing +X);
    /// pose B is a later tick, translated and turned 90 degrees. The
    /// crate must have moved by exactly the rigid transform between A and
    /// B — not slid, not snapped, not lagged.
    #[test]
    fn the_carry_is_rigid() {
        let t = tuning();
        let forks = parked(t);
        let agent_a = DVec3::new(1.0, 2.0, 0.25);
        let facing_a = DVec3::X;
        let crate_pos = DVec3::new(2.5, 2.3, 0.56);
        let crate_rot = Quat::from_rotation_z(0.1);

        let left_a = DVec3::new(-facing_a.y, facing_a.x, 0.0);
        let d = crate_pos - agent_a;
        let yaw_a = facing_a.y.atan2(facing_a.x) as f32;
        let carry = Carry {
            id: CrateId(1),
            half_extents: [0.5, 0.5, 0.5],
            offset: DVec3::new(
                d.dot(facing_a),
                d.dot(left_a),
                crate_pos.z - (0.0 + forks.height),
            ),
            rot_offset: Quat::from_rotation_z(-yaw_a) * crate_rot,
        };

        let agent_b = DVec3::new(-3.0, 4.0, 0.25);
        let facing_b = DVec3::Y;
        let pose = carried_pose(agent_b, facing_b, forks, carry, 0.0);

        let r = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let rel = crate_pos - agent_a;
        let rotated = (r * rel.as_vec3()).as_dvec3();
        let want_pos = agent_b + rotated;
        assert!(
            (pose.pos.x - want_pos.x).abs() < 1e-6 && (pose.pos.y - want_pos.y).abs() < 1e-6,
            "got {:?}, wanted {:?}",
            pose.pos,
            want_pos,
        );
        assert!(
            (pose.pos.z - crate_pos.z).abs() < 1e-6,
            "z should track the original height: got {}, wanted {}",
            pose.pos.z,
            crate_pos.z,
        );
        let want_rot = r * crate_rot;
        assert!(
            pose.rot.angle_between(want_rot) < 1e-6,
            "got {:?}, wanted {:?}",
            pose.rot,
            want_rot,
        );

        // A zero facing (degenerate — the agent is stationary on the tick
        // it picks a crate up) must still produce a finite pose.
        let degenerate = carried_pose(agent_b, DVec3::ZERO, forks, carry, 0.0);
        assert!(degenerate.pos.is_finite(), "degenerate facing gave {:?}", degenerate.pos);
    }

}
