//! Where crates go: a row of storage spots and the jobs that fill them.
//!
//! [`StackTask`](super::task::StackTask) executes one job at a time — walk
//! to a crate, carry it, set it down — and knows nothing about what comes
//! after. The board is what decides *which* crate goes on *which* spot
//! next, so the task can run forever rather than stopping once a single
//! pile is built. It never "finishes": [`JobBoard::next_job`] returning
//! `None` means every spot is full or nothing loose is left, and the
//! caller is expected to idle and re-poll rather than treat that as a
//! terminal state — a spot that later loses a crate, or a crate that
//! later appears, makes a job available again on some future poll.

use std::collections::HashSet;

use glam::{DVec3, Quat};

use super::nav::NavPlane;
use super::task::{CrateId, CrateInfo, StackTuning};

/// A storage spot's identity. Mirrors [`CrateId`]: not an index, so a
/// caller can hand it back across ticks without the board's `Vec` order
/// being load-bearing.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SpotId(pub u32);

/// One storage spot: a tile to stack on and how high it may go.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Spot {
    pub tile: (i32, i32),
    pub capacity: u32,
}

/// One unit of work: carry this crate to this spot.
///
/// Not `crate` — reserved word.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Job {
    pub cargo: CrateId,
    pub spot: SpotId,
}

/// Which crates the task has placed on which spot, and the bookkeeping to
/// hand out the next job.
///
/// `placed` is parallel to `spots`: `placed[i]` is the crates this task
/// has set down on `spots[i]`, in no particular order. It is the board's
/// only durable state — everything else ([`JobBoard::standing`],
/// [`JobBoard::chain`]) is recomputed from the world every call, the same
/// "count, don't trust a tally" reasoning the single-site task used for
/// `layers_standing`.
#[derive(Clone, Debug, Default)]
pub struct JobBoard {
    spots: Vec<Spot>,
    placed: Vec<Vec<CrateId>>,
}

impl JobBoard {
    /// A board over these spots, none of them occupied yet.
    pub fn new(spots: Vec<Spot>) -> Self {
        let placed = vec![Vec::new(); spots.len()];
        Self { spots, placed }
    }

    pub fn spots(&self) -> &[Spot] {
        &self.spots
    }

    pub fn spot(&self, id: SpotId) -> Option<Spot> {
        self.spots.get(id.0 as usize).copied()
    }

    /// The crates this task has placed on `id`, empty for an unknown id.
    pub fn placed(&self, id: SpotId) -> &[CrateId] {
        self.placed.get(id.0 as usize).map_or(&[], |v| v.as_slice())
    }

    /// The tallest capacity configured on any spot, 0 if there are none.
    pub fn max_capacity(&self) -> u32 {
        self.spots.iter().map(|s| s.capacity).max().unwrap_or(0)
    }

    /// World-space centre of a spot's tile — the one spot-to-world
    /// conversion, so every caller goes through it rather than
    /// reimplementing `tile_center`.
    pub fn column(&self, id: SpotId, plane: NavPlane) -> DVec3 {
        let tile = self.spot(id).map_or((0, 0), |s| s.tile);
        plane.tile_center(tile.0, tile.1)
    }

    /// The layer height, taken from whatever crates exist.
    fn layer_height(crates: &[CrateInfo]) -> Option<f64> {
        crates.first().map(CrateInfo::layer_height)
    }

    /// The crates standing on `id`, bottom to top, counted from the world
    /// rather than trusted from `placed`.
    ///
    /// A counter drifts the moment a stack topples; the crates are the
    /// truth. Each layer is measured against **the one below it**, not
    /// against the fixed tile.
    ///
    /// A settled tower shifts as a unit — the bottom crate creeps as it
    /// beds in and everything above rides along with it. Measuring every
    /// layer against the tile the stack was started on therefore reports
    /// a perfectly good three-high tower as zero layers once the base has
    /// drifted past the tolerance, which is what it did: 0.49/1.43/2.42
    /// stacked squarely on each other, counted as nothing, because the
    /// base had moved 0.57 m. Chaining asks the question that matters: is
    /// this crate resting on that one?
    ///
    /// The base layer is found by height alone, not by position, and only
    /// among crates this task actually *placed*. A loose crate lying on
    /// the floor sits at exactly the base layer's height, so height alone
    /// would count scrap in the yard as a tower — measured once as three
    /// crates on the ground coming back as two, with the mast raised to
    /// place a third layer on top of nothing.
    pub fn chain<'a>(
        &self,
        id: SpotId,
        plane: NavPlane,
        crates: &'a [CrateInfo],
        tuning: StackTuning,
    ) -> Vec<&'a CrateInfo> {
        let mut out = Vec::new();
        let Some(spot) = self.spot(id) else { return out };
        let Some(h) = Self::layer_height(crates) else { return out };
        let placed = self.placed(id);
        let tile_center = plane.tile_center(spot.tile.0, spot.tile.1);

        let base = crates
            .iter()
            .filter(|c| {
                !c.carried_by_other
                    && placed.contains(&c.id)
                    && (c.pos.z - (plane.floor_z + h * 0.5)).abs() <= tuning.settle_z_tolerance
            })
            .min_by(|a, b| {
                let da = plane.flatten(a.pos - tile_center).length();
                let db = plane.flatten(b.pos - tile_center).length();
                da.total_cmp(&db)
            });
        let Some(base) = base else { return out };
        out.push(base);
        let mut column = DVec3::new(base.pos.x, base.pos.y, 0.0);

        let mut n = 1u32;
        while n < spot.capacity {
            let expect = plane.floor_z + n as f64 * h + h * 0.5;
            let found = crates.iter().find(|c| {
                !c.carried_by_other
                    && placed.contains(&c.id)
                    && plane.flatten(c.pos - column).length() <= tuning.settle_xy_tolerance
                    && (c.pos.z - expect).abs() <= tuning.settle_z_tolerance
            });
            let Some(c) = found else { break };
            // The next layer is measured against where this one actually
            // is, so the tower is allowed to lean without being
            // discounted.
            column = DVec3::new(c.pos.x, c.pos.y, column.z);
            out.push(c);
            n += 1;
        }
        out
    }

    /// How many layers are standing on `id`.
    pub fn standing(&self, id: SpotId, plane: NavPlane, crates: &[CrateInfo], tuning: StackTuning) -> u32 {
        self.chain(id, plane, crates, tuning).len() as u32
    }

    /// The topmost crate standing on `id`, used to aim the next drop so
    /// the tower is built on itself rather than on the spot it started
    /// from.
    pub fn top_placed<'a>(
        &self,
        id: SpotId,
        plane: NavPlane,
        crates: &'a [CrateInfo],
        tuning: StackTuning,
    ) -> Option<&'a CrateInfo> {
        self.chain(id, plane, crates, tuning).last().copied()
    }

    /// Where the next crate on `id` actually goes: the live block, never a
    /// snapshot.
    ///
    /// A settled stack creeps as it beds in (0.57 m of base drift is on
    /// record for this module), so aiming at the tile centre lands the
    /// second layer off the first the moment the base has moved. If a
    /// crate is already standing, the target is *that crate's* xy, its top
    /// face z (`pos.z + half_extents[2]`), and its rotation — the tower is
    /// built on itself. Otherwise the target is the tile centre at floor
    /// height, via [`JobBoard::column`].
    ///
    /// Deliberately not driven by a layer count: a count says how many
    /// crates are stacked, not where the top face actually sits, and it is
    /// the top face — not the count — that the next drop has to clear.
    pub fn place_target(
        &self,
        id: SpotId,
        plane: NavPlane,
        crates: &[CrateInfo],
        tuning: StackTuning,
    ) -> (DVec3, Quat) {
        match self.top_placed(id, plane, crates, tuning) {
            Some(k) => (
                DVec3::new(k.pos.x, k.pos.y, k.pos.z + k.half_extents[2].abs()),
                k.rot,
            ),
            None => (self.column(id, plane), Quat::IDENTITY),
        }
    }

    /// Whether `id` has room for another layer, counted from the world so
    /// a stack that fell mid-job reads as room again before the next
    /// [`JobBoard::prune`] — the task does not have to wait for its own
    /// bookkeeping to notice.
    pub fn has_room(&self, id: SpotId, plane: NavPlane, crates: &[CrateInfo], tuning: StackTuning) -> bool {
        let Some(spot) = self.spot(id) else { return false };
        self.standing(id, plane, crates, tuning) < spot.capacity
    }

    /// Total crates standing across every spot.
    pub fn total_standing(&self, plane: NavPlane, crates: &[CrateInfo], tuning: StackTuning) -> u32 {
        (0..self.spots.len() as u32)
            .map(|i| self.standing(SpotId(i), plane, crates, tuning))
            .sum()
    }

    /// Whether `pos` is over any spot's tile, planar.
    pub fn in_any_column(&self, plane: NavPlane, pos: DVec3, tuning: StackTuning) -> bool {
        (0..self.spots.len() as u32).any(|i| {
            let column = self.column(SpotId(i), plane);
            plane.flatten(pos - column).length() <= tuning.settle_xy_tolerance
        })
    }

    /// The next job to run, or `None` if no spot has room or no crate
    /// qualifies.
    ///
    /// Picks the **first** spot in index order with room, then the
    /// planar-nearest crate to `from` that is not carried by something
    /// else, not already placed on any spot, not `excluded` (the task's
    /// own blacklist, taken as a closure since the field stays private to
    /// it), and not already sitting in any spot's column.
    ///
    /// Known, pre-existing wart: a foreign crate that lands *in* a
    /// column but off the stack is neither counted by [`JobBoard::chain`]
    /// nor fetched by this — it is invisible to the board either way.
    pub fn next_job(
        &self,
        plane: NavPlane,
        from: DVec3,
        crates: &[CrateInfo],
        tuning: StackTuning,
        excluded: impl Fn(CrateId) -> bool,
    ) -> Option<Job> {
        let spot_idx = (0..self.spots.len() as u32)
            .find(|&i| self.has_room(SpotId(i), plane, crates, tuning))?;
        let spot = SpotId(spot_idx);

        let already_placed: HashSet<CrateId> = self.placed.iter().flatten().copied().collect();

        let cargo = crates
            .iter()
            .filter(|c| !c.carried_by_other)
            .filter(|c| !already_placed.contains(&c.id))
            .filter(|c| !excluded(c.id))
            .filter(|c| !self.in_any_column(plane, c.pos, tuning))
            .min_by(|a, b| {
                let da = plane.flatten(a.pos - from).length();
                let db = plane.flatten(b.pos - from).length();
                da.total_cmp(&db)
            })?
            .id;

        Some(Job { cargo, spot })
    }

    /// Tiles a stacked spot occupies, for a caller's pathfinding
    /// `extra_blocked`.
    ///
    /// **The spot tile only, not a ring around it.** Blocking a ring kept
    /// an agent's body clear of the tower, and it also kept the agent
    /// three metres from a tower its forks reach barely one metre over —
    /// so the crate had to be flung the remaining distance, which is the
    /// teleport at the far end of the haul. A forklift drives up to the
    /// stack and sets the load down from where it is standing; it does
    /// not stand off and throw. The tower is still protected, by the tile
    /// itself and by the cargo riding above it.
    pub fn blocked_tiles(&self) -> HashSet<(i32, i32)> {
        self.spots
            .iter()
            .zip(self.placed.iter())
            .filter(|(_, placed)| !placed.is_empty())
            .map(|(spot, _)| spot.tile)
            .collect()
    }

    /// Record that `cargo` was set down on `id`. The only production
    /// write to `placed`.
    pub fn record_placed(&mut self, id: SpotId, cargo: CrateId) {
        if let Some(slot) = self.placed.get_mut(id.0 as usize) {
            slot.push(cargo);
        }
    }

    /// Drop any crate no longer part of a spot's standing chain.
    ///
    /// Run this on **Idle entry only**, not every tick. During `Settling`
    /// the crate just placed rocks before it sleeps and can transiently
    /// leave `chain`'s tolerance; pruning in that window would mark it
    /// loose while it is still sitting on the pile, and the very next
    /// poll would send the task straight back to fetch it. Idle is the
    /// one point where the previous job is genuinely over. Mid-job
    /// collapses are still caught in the meantime — [`JobBoard::has_room`]
    /// and [`JobBoard::standing`] measure positions fresh every tick, so a
    /// stack that falls reads as having room before `prune` ever runs.
    pub fn prune(&mut self, plane: NavPlane, crates: &[CrateInfo], tuning: StackTuning) {
        for i in 0..self.spots.len() {
            let id = SpotId(i as u32);
            let keep: HashSet<CrateId> =
                self.chain(id, plane, crates, tuning).iter().map(|c| c.id).collect();
            self.placed[i].retain(|c| keep.contains(c));
        }
    }
}
