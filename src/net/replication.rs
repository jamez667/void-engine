//! Interest-managed replication: what each client is told, and how little
//! it costs to tell them.
//!
//! Without interest management a server sends every entity to every client
//! and per-client bandwidth grows with the size of the world. That is the
//! wall between a session game and an MMO, and it is a bandwidth wall long
//! before it is a CPU one.
//!
//! # The pipeline
//!
//! Per client, per tick:
//!
//! 1. **Relevancy** — [`SpatialGrid::query_circle_into`] over the client's
//!    viewpoint. Measured at 100k colliders and 1000 clients, this is
//!    ~12 ms of a 33.3 ms tick; see that method's docs for why it is not
//!    the obvious `query_circle`.
//! 2. **Delta** — compare against what this client was last *acknowledged*
//!    to hold and emit only what changed.
//! 3. **Encode** — quantised and bit-packed, components named by their
//!    registry [`NameId`].
//! 4. **Send** — [`send_chunked`], which splits an oversized packet across
//!    datagrams without ever dropping an entity.
//!
//! [`SpatialGrid::query_circle_into`]: crate::collision::SpatialGrid::query_circle_into
//! [`send_chunked`]: crate::net::chunk::send_chunked
//! [`NameId`]: crate::persist::registry::NameId
//!
//! # Why the grid index is not the entity id
//!
//! [`SpatialGrid`] hands back the dense index it assigned at `insert`,
//! which says nothing about which entity that was. The mapping has to be
//! recorded by whoever fills the grid, in insertion order — see
//! [`Relevancy`].
//!
//! That works because the grid is rebuilt every tick, so indices are
//! assigned fresh each time. R4's incremental broadphase would break it:
//! with `update`/`remove`, an index outlives the tick that created it and
//! the parallel vector has to be maintained rather than rebuilt.
//!
//! [`SpatialGrid`]: crate::collision::SpatialGrid
//!
//! # Why generation travels with spawns
//!
//! [`World::despawn`] bumps the slot's generation and frees the index
//! immediately, so a `spawn` later in the *same tick* can hand that index
//! to an unrelated entity. A client tracking entities by index alone would
//! quietly apply one entity's updates to another. Deltas therefore carry
//! only the index — cheap, and correct because a live entity's generation
//! never changes — while spawn and despawn carry the full [`EntityId`], so
//! a reused index is always announced.
//!
//! [`World::despawn`]: crate::World::despawn
//! [`EntityId`]: crate::EntityId
//!
//! # Why there is one baseline per client and not a history
//!
//! A delta is computed against what the client last *acknowledged*, which
//! means the server has to keep that state. The obvious generalisation —
//! retain the last N ticks so a delta can be built against whichever one
//! the client confirms — does not survive arithmetic. At 1000 clients,
//! ~1800 entities in range each and ~32 B of live state per entity, one
//! baseline is ~56 MiB. One *second* of history at 30 Hz is 1.6 GiB, and
//! ten seconds is 16 GiB.
//!
//! So exactly one baseline is kept per client: the last acknowledged
//! state. A client that stops acknowledging does not accumulate history
//! on the server — it is marked stale and gets a full keyframe when it
//! returns. That is the same trade [`rebuild_shed`] already makes on the
//! transport side: accept staleness, never lose state, never grow without
//! bound.
//!
//! [`rebuild_shed`]: crate::net::chunk::Chunkable::rebuild_shed
//!
//! # Who owns what
//!
//! The engine does not drive replication. `run_headless_with` loops over
//! `App::fixed_update` with no per-connection hook, because a server owns
//! its own thread and its own connection table. So the game holds one
//! [`ClientLink`] per connection and calls into it; the engine supplies
//! the type and the rules, not the loop.
//!
//! The tick is likewise the caller's: [`SimCtx`] carries `world`, `input`
//! and `dt`, and the headless loop keeps its tick counter as a local. It
//! is taken as a `u32` here because that is what [`InterpClock`] consumes
//! on the client; a simulation counting ticks in `u64` narrows at this
//! boundary deliberately rather than silently.
//!
//! [`SimCtx`]: crate::SimCtx
//! [`InterpClock`]: crate::net::interp::InterpClock

use std::collections::HashSet;

use crate::ecs::EntityId;

/// Largest acknowledgement message accepted from a client.
///
/// An ack is a tick number and nothing else, so this is generous by an
/// order of magnitude and still refuses anything that could be an attempt
/// to make the server allocate. `framing::read_msg` takes the cap as a
/// mandatory argument precisely so each protocol names its own.
pub const MAX_ACK_BYTES: usize = 64;

/// A client's acknowledgement that it has applied every update up to
/// `tick`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Ack {
    /// The newest tick this client has applied.
    pub tick: u32,
}

impl Ack {
    /// Encode for the reliable stream. Fixed four bytes, little-endian —
    /// an ack is not worth a bit-packer.
    pub fn encode(self) -> [u8; 4] {
        self.tick.to_le_bytes()
    }

    /// Decode one from bytes off the wire.
    ///
    /// Returns `None` for anything that is not exactly four bytes rather
    /// than reading what it can: a wrong-sized ack means the two ends
    /// disagree about the protocol, and guessing at the intent of a
    /// malformed message from a peer is how a parser becomes an exploit.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        let arr: [u8; 4] = bytes.try_into().ok()?;
        Some(Self { tick: u32::from_le_bytes(arr) })
    }
}

/// Why a client's next snapshot cannot be a delta.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum KeyframeReason {
    /// Nothing has been acknowledged yet — a new connection.
    FirstSnapshot,
    /// The client fell further behind than the link will wait.
    Stalled { ticks_behind: u32 },
}

/// What to send a client this tick.
///
/// Three outcomes rather than two, because a budget makes "needs a
/// keyframe" and "gets one now" different questions. Returning
/// `Option<KeyframeReason>` collapsed them, and a caller writing the
/// obvious `if plan.is_some() { keyframe } else { delta }` would then send
/// a *delta* to a client the server knows has a stale baseline — the
/// client applies changes against state it does not have and its view is
/// quietly wrong from then on. Making [`Plan::Deferred`] its own variant
/// means a caller matching exhaustively cannot fall into that branch by
/// accident.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Plan {
    /// Send a delta against the client's baseline.
    Delta,
    /// Send a full keyframe, and report it with [`ClientLink::commit_keyframe`].
    Keyframe(KeyframeReason),
    /// A keyframe is owed but this tick's budget is spent. **Send nothing
    /// to this client.**
    ///
    /// Not a failure: the need is latched and the client is served on a
    /// later tick. Skipping a tick costs that client one frame of
    /// staleness, where a delta would cost it a permanently wrong view and
    /// an unbudgeted keyframe would cost every other client the tick.
    Deferred(KeyframeReason),
}

impl Plan {
    /// True when this tick should produce a keyframe.
    pub fn is_keyframe(self) -> bool { matches!(self, Plan::Keyframe(_)) }

    /// True when nothing may be sent to this client this tick.
    pub fn is_deferred(self) -> bool { matches!(self, Plan::Deferred(_)) }
}

/// How many keyframes a tick may afford, across every client.
///
/// Without one, keyframes are decided per client with no shared limit, and
/// the failure is not gradual. Measured: one keyframe over ~1800 visible
/// entities encodes in 0.388 ms, so 64 clients taking one on the same tick
/// costs 25.9 ms of a 33.3 ms tick and 1000 clients costs **413 ms** —
/// twelve times the entire budget. That is reachable rather than
/// theoretical: a server restart, a shard migration, or one network blip
/// that stalls many clients at once makes every link owe a keyframe on the
/// same tick, and the server stops rather than degrades.
///
/// A budget turns that into a queue. At 13 per tick — about 5 ms — a
/// thousand reconnecting clients are all served within 2.6 seconds at
/// 30 Hz, and no tick is ever blown doing it.
///
/// Caller-owned, like [`ChunkHint`]: reset once per tick, pass to every
/// [`ClientLink::plan`] call in that tick.
///
/// [`ChunkHint`]: crate::net::chunk::ChunkHint
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct KeyframeBudget {
    per_tick: u32,
    spent: u32,
}

impl KeyframeBudget {
    /// A budget of `per_tick` keyframes.
    ///
    /// Sizing it: divide the slice of the tick you will spend on keyframes
    /// by what one costs. At the measured 0.388 ms per keyframe over ~1800
    /// entities, 13 is roughly 5 ms.
    ///
    /// Zero is legal and means "no keyframes this tick", which is a way to
    /// shed load deliberately — every client that needs one is deferred and
    /// keeps its need latched.
    pub fn new(per_tick: u32) -> Self {
        Self { per_tick, spent: 0 }
    }

    /// Start a new tick. Call once per tick, before any `plan`.
    pub fn begin(&mut self) {
        self.spent = 0;
    }

    /// Keyframes still available this tick.
    pub fn remaining(&self) -> u32 { self.per_tick.saturating_sub(self.spent) }

    /// Keyframes granted so far this tick.
    pub fn spent(&self) -> u32 { self.spent }

    /// Take a slot, or report there is none left.
    fn claim(&mut self) -> bool {
        if self.spent >= self.per_tick {
            return false;
        }
        self.spent += 1;
        true
    }
}

impl Default for KeyframeBudget {
    /// 13 per tick: about 5 ms at the measured cost, which drains a
    /// thousand reconnecting clients in 2.6 seconds at 30 Hz.
    fn default() -> Self { Self::new(13) }
}

/// Per-connection replication state: what this client has confirmed, and
/// what it is therefore believed to hold.
///
/// One per connection, owned by the game. Reused across ticks — the
/// baseline set is cleared and refilled rather than reallocated.
pub struct ClientLink {
    /// Newest tick this client has acknowledged. `None` before the first
    /// ack, which is what makes a new connection take a keyframe.
    acked_tick: Option<u32>,
    /// Entities the client is believed to hold, as of `acked_tick`.
    baseline: HashSet<EntityId>,
    /// How far behind the client may fall before the link stops trying to
    /// delta against a baseline it no longer trusts.
    stall_after_ticks: u32,
    /// Set when a keyframe is owed, so the decision survives until it is
    /// actually sent.
    keyframe_owed: Option<KeyframeReason>,
    /// Tick of the last keyframe *sent*, which is not the same as one
    /// acknowledged and must not be conflated with it.
    ///
    /// Staleness is measured from whichever of this and `acked_tick` is
    /// later. Without it, sending a keyframe would not reset the stall
    /// condition — `acked_tick` deliberately does not move on a send — so
    /// the next `plan` would find the client just as far behind and owe
    /// another keyframe, every tick, for as long as the client stayed
    /// quiet.
    keyframed_at: Option<u32>,
}

impl ClientLink {
    /// A link for a freshly connected client.
    ///
    /// `stall_after_ticks` bounds how stale a baseline may get before it
    /// is abandoned. It is not a memory bound — only one baseline is ever
    /// held — but a delta against a very old baseline is both large and
    /// likely wrong, so past some age a keyframe is cheaper and safer.
    /// At 30 Hz, 90 ticks is three seconds.
    pub fn new(stall_after_ticks: u32) -> Self {
        Self {
            acked_tick: None,
            baseline: HashSet::new(),
            stall_after_ticks,
            keyframe_owed: Some(KeyframeReason::FirstSnapshot),
            keyframed_at: None,
        }
    }

    /// The newest acknowledged tick, if any.
    pub fn acked_tick(&self) -> Option<u32> { self.acked_tick }

    /// Entities the client is believed to hold.
    pub fn baseline_len(&self) -> usize { self.baseline.len() }

    /// Record a client's acknowledgement of everything up to `ack.tick`,
    /// as of server tick `now`.
    ///
    /// Out-of-order and replayed acks are ignored rather than rejected:
    /// they are normal on a lossy link, and an ack older than one already
    /// seen carries no information. Returns whether the watermark moved.
    ///
    /// # An ack is peer input, not a fact
    ///
    /// `now` is what makes this safe, and it is a required argument for
    /// that reason. An ack is four attacker-controlled bytes
    /// ([`Ack::decode`] accepts any `u32`), and the watermark it sets is
    /// what [`plan`] measures staleness against. A client that claims to
    /// have applied a tick the server has not yet *sent* is either broken
    /// or lying, and either way the claim cannot be true.
    ///
    /// Accepting it is not a small error. `plan` computes
    /// `tick.saturating_sub(reference)`, so a single forged
    /// `Ack { tick: u32::MAX }` pins that difference at zero forever: the
    /// stall check can never fire again, the client is never re-keyframed
    /// however far it drifts, and the server deltas against a baseline the
    /// client never confirmed — while holding that baseline's memory for
    /// the life of the connection. Measured before this bound existed: 0
    /// keyframes over 10,000 ticks against 109 for an honest silent
    /// client.
    ///
    /// Acks from the future are therefore dropped, not clamped. Clamping
    /// to `now` would still let a peer pin the watermark at the present
    /// tick every tick and never fall behind by construction, which is the
    /// same exploit wearing a politer hat.
    ///
    /// [`plan`]: ClientLink::plan
    pub fn record_ack(&mut self, ack: Ack, now: u32) -> bool {
        if ack.tick > now {
            return false;
        }
        match self.acked_tick {
            Some(t) if ack.tick <= t => false,
            _ => {
                self.acked_tick = Some(ack.tick);
                true
            }
        }
    }

    /// Decide what to send this client for `tick`.
    ///
    /// A keyframe is owed on a new connection, and again whenever the
    /// client has fallen further behind than `stall_after_ticks`. The
    /// decision is latched: once owed, it stays owed until
    /// [`commit_keyframe`] reports one was actually sent, so a keyframe
    /// cannot be lost by the caller asking twice — or by `budget` being
    /// spent when it asked.
    ///
    /// Staleness is measured from the later of the last acknowledgement
    /// and the last keyframe sent. Measuring from the acknowledgement
    /// alone would re-arm the stall on the very next tick — sending does
    /// not advance the ack watermark, by design — and a client that went
    /// quiet would be sent a keyframe every tick until it came back. With
    /// the send counted, a silent client costs one keyframe per
    /// `stall_after_ticks` instead: three seconds apart at 30 Hz.
    ///
    /// `budget` is what stops a thousand clients needing a keyframe on the
    /// same tick from taking 413 ms of a 33.3 ms tick. A client that is
    /// owed one when the budget is spent comes back [`Plan::Deferred`],
    /// which means **send it nothing**: its need stays latched and it is
    /// served on a later tick. Sending a delta instead would apply changes
    /// against a baseline the server knows it does not have.
    ///
    /// [`commit_keyframe`]: ClientLink::commit_keyframe
    pub fn plan(&mut self, tick: u32, budget: &mut KeyframeBudget) -> Plan {
        if self.keyframe_owed.is_none() {
            let reference = match (self.acked_tick, self.keyframed_at) {
                (Some(a), Some(k)) => Some(a.max(k)),
                (Some(a), None)    => Some(a),
                (None, Some(k))    => Some(k),
                (None, None)       => None,
            };
            if let Some(reference) = reference {
                let behind = tick.saturating_sub(reference);
                if behind > self.stall_after_ticks {
                    // Report the gap the client actually has, which is
                    // measured from its own last acknowledgement — the
                    // keyframe send is the server's action, not the
                    // client's progress, so it must not flatter the log.
                    let ticks_behind = self
                        .acked_tick
                        .map_or(tick, |acked| tick.saturating_sub(acked));
                    log::warn!(
                        "event=replication_client_stalled {ticks_behind} ticks behind \
                         (limit {}); a keyframe is owed",
                        self.stall_after_ticks,
                    );
                    self.keyframe_owed = Some(KeyframeReason::Stalled { ticks_behind });
                }
            }
        }

        match self.keyframe_owed {
            None => Plan::Delta,
            // The slot is taken here rather than in `commit_keyframe`, so
            // a caller that asks and then drops the answer still consumes
            // the tick's capacity. Charging on delivery instead would let
            // a loop over a thousand links grant a thousand keyframes and
            // discover the cost only once they were all encoded.
            Some(reason) if budget.claim() => Plan::Keyframe(reason),
            Some(reason) => Plan::Deferred(reason),
        }
    }

    /// Record that a keyframe carrying `visible` was sent at `tick`.
    ///
    /// The baseline becomes exactly what was sent, and `tick` is
    /// remembered so the stall check does not immediately re-arm — see
    /// [`plan`].
    ///
    /// Note the ack watermark is *not* advanced: the client has been sent
    /// this state, not confirmed it, and treating a send as a receipt is
    /// how a lost keyframe turns into a client that never recovers.
    ///
    /// [`plan`]: ClientLink::plan
    pub fn commit_keyframe(&mut self, tick: u32, visible: impl IntoIterator<Item = EntityId>) {
        self.baseline.clear();
        self.baseline.extend(visible);
        self.keyframe_owed = None;
        self.keyframed_at = Some(tick);
    }

    /// Compute which entities entered and left since the baseline.
    ///
    /// `entered` is what the client does not yet hold and must be sent in
    /// full; `left` is what it holds and can no longer see. Both are
    /// written into the caller's buffers so a per-tick, per-client call
    /// allocates nothing.
    pub fn diff(
        &self,
        visible: &[EntityId],
        entered: &mut Vec<EntityId>,
        left: &mut Vec<EntityId>,
    ) {
        entered.clear();
        left.clear();
        for &id in visible {
            if !self.baseline.contains(&id) {
                entered.push(id);
            }
        }
        let seen: HashSet<EntityId> = visible.iter().copied().collect();
        for &id in &self.baseline {
            if !seen.contains(&id) {
                left.push(id);
            }
        }
        // Deterministic order: the baseline is a hash set, so iteration
        // order is seed-dependent and would otherwise vary per process —
        // the same trap `query_pairs` was fixed for.
        left.sort_unstable_by_key(|e| (e.index, e.generation));
        entered.sort_unstable_by_key(|e| (e.index, e.generation));
    }

    /// Record that a delta was sent, moving the baseline to `visible`.
    pub fn commit_delta(&mut self, visible: &[EntityId]) {
        self.baseline.clear();
        self.baseline.extend(visible.iter().copied());
    }
}

/// The set of entities one client can currently see, and the mapping back
/// from grid indices to entities.
///
/// Reused across ticks: the vectors are cleared and refilled rather than
/// reallocated, for the same reason [`AoiScratch`] exists.
///
/// [`AoiScratch`]: crate::collision::AoiScratch
#[derive(Default)]
pub struct Relevancy {
    /// Entity per grid index, in the order they were inserted into the
    /// grid this tick. `entities[i]` is what `SpatialGrid` index `i` means.
    entities: Vec<EntityId>,
}

impl Relevancy {
    pub fn new() -> Self { Self::default() }

    /// Begin a tick. Call before re-filling the grid.
    pub fn begin(&mut self) {
        self.entities.clear();
    }

    /// Record that the next `SpatialGrid::insert` belongs to `id`.
    ///
    /// Must be called once per insert, in the same order, or every index
    /// afterwards maps to the wrong entity. The grid assigns indices
    /// sequentially from zero, so this is a push.
    pub fn push(&mut self, id: EntityId) {
        self.entities.push(id);
    }

    /// The entity a grid index refers to, or `None` if the index was never
    /// recorded — which means the grid and this mapping have diverged.
    pub fn entity(&self, grid_index: u32) -> Option<EntityId> {
        self.entities.get(grid_index as usize).copied()
    }

    /// How many entities were recorded this tick.
    pub fn len(&self) -> usize { self.entities.len() }
    pub fn is_empty(&self) -> bool { self.entities.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_grid_indices_back_to_entities() {
        let mut r = Relevancy::new();
        r.begin();
        r.push(EntityId { index: 7, generation: 2 });
        r.push(EntityId { index: 3, generation: 0 });

        assert_eq!(r.entity(0), Some(EntityId { index: 7, generation: 2 }));
        assert_eq!(r.entity(1), Some(EntityId { index: 3, generation: 0 }));
        assert_eq!(r.entity(2), None, "an unrecorded index must not resolve");
    }

    /// The mapping is rebuilt per tick; last tick's entries must not
    /// survive into this one, or a despawned entity keeps being replicated.
    #[test]
    fn begin_clears_the_previous_tick() {
        let mut r = Relevancy::new();
        r.begin();
        r.push(EntityId { index: 1, generation: 0 });
        assert_eq!(r.len(), 1);

        r.begin();
        assert!(r.is_empty(), "a new tick starts from nothing");
        assert_eq!(r.entity(0), None);
    }

    fn ent(index: u32) -> EntityId { EntityId { index, generation: 0 } }

    /// A budget large enough never to bind, so the assertions around it
    /// keep testing what they tested before budgets existed. Exhaustion
    /// has tests of its own.
    fn budget() -> KeyframeBudget { KeyframeBudget::new(1_000) }

    /// A new connection holds nothing, so its first snapshot cannot be a
    /// delta against anything.
    #[test]
    fn a_new_link_owes_a_keyframe() {
        let mut link = ClientLink::new(90);
        assert_eq!(link.plan(1, &mut budget()), Plan::Keyframe(KeyframeReason::FirstSnapshot));
        assert_eq!(link.acked_tick(), None);
        assert_eq!(link.baseline_len(), 0);
    }

    /// **The subtle one.** Sending a keyframe is not the client receiving
    /// it. If a send advanced the ack watermark, a keyframe lost in
    /// flight would leave the server deltaing against state the client
    /// never got — a client that never recovers.
    #[test]
    fn sending_a_keyframe_does_not_acknowledge_it() {
        let mut link = ClientLink::new(90);
        link.plan(1, &mut budget());
        link.commit_keyframe(1, [ent(1), ent(2)]);

        assert_eq!(link.baseline_len(), 2, "the baseline is what was sent");
        assert_eq!(link.acked_tick(), None, "but nothing has been confirmed");
        assert_eq!(link.plan(2, &mut budget()), Plan::Delta, "the keyframe is no longer owed");
    }

    /// Acks are monotonic. Replays and reorderings are normal on a lossy
    /// link and carry no information.
    #[test]
    fn stale_and_replayed_acks_are_ignored() {
        let mut link = ClientLink::new(90);
        assert!(link.record_ack(Ack { tick: 10 }, 100), "first ack moves the watermark");
        assert_eq!(link.acked_tick(), Some(10));

        assert!(!link.record_ack(Ack { tick: 10 }, 100), "a replay moves nothing");
        assert!(!link.record_ack(Ack { tick: 4 }, 100), "an older ack moves nothing");
        assert_eq!(link.acked_tick(), Some(10), "and the watermark holds");

        assert!(link.record_ack(Ack { tick: 11 }, 100));
        assert_eq!(link.acked_tick(), Some(11));
    }

    /// An ack for a tick the server has not sent is a claim that cannot be
    /// true, and accepting it disables the stall detector permanently:
    /// `plan` measures `tick.saturating_sub(acked)`, so `u32::MAX` pins
    /// that at zero for the life of the connection.
    #[test]
    fn an_ack_from_the_future_is_refused() {
        let mut link = ClientLink::new(10);
        link.plan(1, &mut budget());
        link.commit_keyframe(1, [ent(1)]);

        assert!(!link.record_ack(Ack { tick: u32::MAX }, 5), "a forged ack must not land");
        assert!(!link.record_ack(Ack { tick: 6 }, 5), "one tick ahead is still ahead");
        assert_eq!(link.acked_tick(), None, "nothing forged may reach the watermark");

        // The stall detector must still work afterwards.
        assert!(
            matches!(link.plan(100, &mut budget()), Plan::Keyframe(_)),
            "a client that never legitimately acked must still stall",
        );
    }

    /// The bound is `<= now`, not `< now`: acking the tick just sent is
    /// the normal case, not an attack.
    #[test]
    fn an_ack_for_the_current_tick_is_accepted() {
        let mut link = ClientLink::new(90);
        assert!(link.record_ack(Ack { tick: 7 }, 7), "acking the tick just sent is legitimate");
        assert_eq!(link.acked_tick(), Some(7));
    }

    /// A client that stops acknowledging gets a keyframe rather than the
    /// server retaining history for it — the memory argument in the
    /// module docs, enforced.
    #[test]
    fn a_stalled_client_is_owed_a_keyframe() {
        let mut link = ClientLink::new(90);
        link.plan(1, &mut budget());
        link.commit_keyframe(1, [ent(1)]);
        link.record_ack(Ack { tick: 1 }, 1);

        assert_eq!(link.plan(50, &mut budget()), Plan::Delta, "49 ticks behind is within the limit");
        assert_eq!(link.plan(91, &mut budget()), Plan::Delta, "90 behind is exactly the limit");
        assert_eq!(
            link.plan(92, &mut budget()),
            Plan::Keyframe(KeyframeReason::Stalled { ticks_behind: 91 }),
            "91 behind is past it",
        );
    }

    /// Once owed, a keyframe stays owed. Asking twice must not lose it.
    #[test]
    fn a_keyframe_decision_latches_until_it_is_sent() {
        let mut link = ClientLink::new(10);
        link.plan(1, &mut budget());
        link.commit_keyframe(1, [ent(1)]);
        link.record_ack(Ack { tick: 1 }, 1);

        let Plan::Keyframe(owed) = link.plan(100, &mut budget()) else { panic!("stalled") };
        assert_eq!(link.plan(100, &mut budget()), Plan::Keyframe(owed), "asking again must not clear it");
        assert_eq!(link.plan(101, &mut budget()), Plan::Keyframe(owed));

        link.commit_keyframe(101, [ent(1)]);
        assert_eq!(link.plan(101, &mut budget()), Plan::Delta, "and only sending clears it");
    }

    /// **The bug this field exists for.** A client that stops acking must
    /// cost one keyframe per stall window, not one per tick.
    ///
    /// `commit_keyframe` deliberately does not advance the ack watermark,
    /// so measuring staleness from the ack alone left the client just as
    /// far behind on the very next tick — and owed another keyframe, and
    /// another, for as long as it stayed quiet. An unbounded send loop is
    /// a worse failure than the memory growth the single-baseline design
    /// exists to avoid.
    #[test]
    fn a_silent_client_costs_one_keyframe_per_window_not_one_per_tick() {
        let window = 10;
        let mut link = ClientLink::new(window);
        link.plan(1, &mut budget());
        link.commit_keyframe(1, [ent(1)]);
        link.record_ack(Ack { tick: 1 }, 1);

        // The client goes silent here: no further acks, ever.
        let mut keyframes = 0;
        for tick in 2..=200u32 {
            if link.plan(tick, &mut budget()).is_keyframe() {
                keyframes += 1;
                link.commit_keyframe(tick, [ent(1)]);
            }
        }

        // ~199 ticks over a 10-tick window: a handful, not one per tick.
        assert!(keyframes > 0, "a silent client must still be re-keyframed");
        assert!(
            keyframes <= 200 / window as usize + 1,
            "{keyframes} keyframes over 199 silent ticks is a send loop",
        );
    }

    /// And the stall still reports the client's real lateness, not the
    /// time since the server last talked to itself.
    #[test]
    fn a_stall_reports_lateness_from_the_clients_own_ack() {
        let mut link = ClientLink::new(10);
        link.plan(1, &mut budget());
        link.commit_keyframe(1, [ent(1)]);
        link.record_ack(Ack { tick: 5 }, 5);

        link.plan(20, &mut budget());
        link.commit_keyframe(20, [ent(1)]);

        match link.plan(40, &mut budget()) {
            Plan::Keyframe(KeyframeReason::Stalled { ticks_behind }) => assert_eq!(
                ticks_behind, 35,
                "lateness is measured from the ack at tick 5, not the keyframe at 20",
            ),
            other => panic!("expected a stall, got {other:?}"),
        }
    }

    /// **The thundering herd.** A budget of one means the second client
    /// needing a keyframe on the same tick is deferred, not served — and
    /// deferred is not the same answer as "send a delta".
    #[test]
    fn a_spent_budget_defers_rather_than_sending_a_delta() {
        let mut budget = KeyframeBudget::new(1);
        let mut first = ClientLink::new(90);
        let mut second = ClientLink::new(90);

        assert_eq!(first.plan(1, &mut budget), Plan::Keyframe(KeyframeReason::FirstSnapshot));
        let deferred = second.plan(1, &mut budget);
        assert_eq!(deferred, Plan::Deferred(KeyframeReason::FirstSnapshot));
        assert!(deferred.is_deferred());
        assert!(!deferred.is_keyframe());
        assert_ne!(
            deferred, Plan::Delta,
            "a deferred client must never read as one that can take a delta: \
             its baseline is empty, so a delta would corrupt its view",
        );
    }

    /// A deferred client keeps its claim. The need is latched, so the
    /// next tick with room serves it — nothing is lost by waiting.
    #[test]
    fn a_deferred_client_is_served_on_a_later_tick() {
        let mut spent = KeyframeBudget::new(0);
        let mut link = ClientLink::new(90);

        assert!(link.plan(1, &mut spent).is_deferred(), "no budget at all");
        assert!(link.plan(2, &mut spent).is_deferred(), "still none");

        let mut roomy = KeyframeBudget::new(1);
        assert_eq!(
            link.plan(3, &mut roomy),
            Plan::Keyframe(KeyframeReason::FirstSnapshot),
            "the claim survived two deferrals",
        );
    }

    /// The allowance is per tick, so `begin` restores it. Without the
    /// reset a budget is a whole-run allowance that silently runs out.
    #[test]
    fn begin_restores_the_allowance_each_tick() {
        let mut budget = KeyframeBudget::new(2);
        let mut a = ClientLink::new(90);
        let mut b = ClientLink::new(90);
        let mut c = ClientLink::new(90);

        assert!(a.plan(1, &mut budget).is_keyframe());
        assert!(b.plan(1, &mut budget).is_keyframe());
        assert!(c.plan(1, &mut budget).is_deferred(), "two per tick means two");
        assert_eq!(budget.remaining(), 0);
        assert_eq!(budget.spent(), 2);

        budget.begin();
        assert_eq!(budget.remaining(), 2, "a new tick restores the allowance");
        assert!(c.plan(2, &mut budget).is_keyframe());
    }

    /// A client that needs no keyframe costs nothing, so a busy tick does
    /// not starve the clients that were only ever going to take deltas.
    #[test]
    fn a_delta_does_not_spend_the_budget() {
        let mut budget = KeyframeBudget::new(1);
        let mut link = ClientLink::new(90);

        link.plan(1, &mut budget);
        link.commit_keyframe(1, [ent(1)]);
        link.record_ack(Ack { tick: 1 }, 1);
        budget.begin();

        assert_eq!(link.plan(2, &mut budget), Plan::Delta);
        assert_eq!(budget.remaining(), 1, "a delta must not consume a keyframe slot");
        assert_eq!(budget.spent(), 0);
    }

    /// Sizing, as the docs claim it: 13 per tick is the default, and a
    /// thousand clients drain in the stated number of ticks.
    #[test]
    fn the_default_budget_drains_a_thousand_clients_in_the_stated_time() {
        let mut budget = KeyframeBudget::default();
        assert_eq!(budget.remaining(), 13, "the documented default");

        let mut links: Vec<ClientLink> = (0..1_000).map(|_| ClientLink::new(90)).collect();
        let mut served = 0usize;
        let mut ticks = 0u32;
        while served < links.len() {
            ticks += 1;
            budget.begin();
            for link in links.iter_mut() {
                if let Plan::Keyframe(_) = link.plan(ticks, &mut budget) {
                    link.commit_keyframe(ticks, [ent(1)]);
                    served += 1;
                }
            }
        }
        // 1000 / 13 = 77 ticks, which is 2.6 s at 30 Hz.
        assert_eq!(ticks, 77, "a thousand clients at 13 a tick");
        assert!((ticks as f64 / 30.0) < 3.0, "under three seconds at 30 Hz");
    }

    /// The delta itself: what the client lacks, and what it can no longer
    /// see.
    #[test]
    fn diff_reports_what_entered_and_left() {
        let mut link = ClientLink::new(90);
        link.commit_keyframe(1, [ent(1), ent(2), ent(3)]);

        let (mut entered, mut left) = (Vec::new(), Vec::new());
        link.diff(&[ent(2), ent(3), ent(4)], &mut entered, &mut left);

        assert_eq!(entered, vec![ent(4)], "4 is new to this client");
        assert_eq!(left, vec![ent(1)], "1 is no longer visible");
    }

    /// The baseline is a `HashSet`, whose iteration order is seeded per
    /// process. Without the sort, `left` would differ run to run — the
    /// same trap `query_pairs` was fixed for.
    #[test]
    fn diff_output_is_deterministically_ordered() {
        let mut link = ClientLink::new(90);
        link.commit_keyframe(1, (0..64).map(ent));

        let (mut entered, mut left) = (Vec::new(), Vec::new());
        link.diff(&[], &mut entered, &mut left);

        assert!(entered.is_empty());
        assert_eq!(left.len(), 64);
        assert!(left.windows(2).all(|w| w[0].index < w[1].index), "must be ascending");
    }

    /// Buffers are reused per client per tick, so a diff must not inherit
    /// the previous one's contents.
    #[test]
    fn diff_clears_the_callers_buffers() {
        let mut link = ClientLink::new(90);
        link.commit_keyframe(1, [ent(1)]);

        let mut entered = vec![ent(99)];
        let mut left = vec![ent(98)];
        link.diff(&[ent(1)], &mut entered, &mut left);

        assert!(entered.is_empty(), "stale entries must not survive");
        assert!(left.is_empty());
    }

    /// A reused index is a different entity here too: the client holds
    /// generation 0 and must be told about generation 1 as an arrival.
    #[test]
    fn a_respawned_index_reads_as_entered_and_left() {
        let mut link = ClientLink::new(90);
        link.commit_keyframe(1, [EntityId { index: 4, generation: 0 }]);

        let (mut entered, mut left) = (Vec::new(), Vec::new());
        link.diff(&[EntityId { index: 4, generation: 1 }], &mut entered, &mut left);

        assert_eq!(entered, vec![EntityId { index: 4, generation: 1 }], "the new tenant arrives");
        assert_eq!(left, vec![EntityId { index: 4, generation: 0 }], "the old one departs");
    }

    /// Committing a delta moves the baseline, so the next diff is against
    /// what was just sent.
    #[test]
    fn committing_a_delta_moves_the_baseline() {
        let mut link = ClientLink::new(90);
        link.commit_keyframe(1, [ent(1)]);
        link.commit_delta(&[ent(2), ent(3)]);
        assert_eq!(link.baseline_len(), 2);

        let (mut entered, mut left) = (Vec::new(), Vec::new());
        link.diff(&[ent(2), ent(3)], &mut entered, &mut left);
        assert!(entered.is_empty() && left.is_empty(), "nothing changed since the delta");
    }

    /// An ack round-trips, and a malformed one is refused rather than
    /// guessed at — these bytes come from a peer.
    #[test]
    fn acks_round_trip_and_reject_malformed_input() {
        let a = Ack { tick: 123_456 };
        assert_eq!(Ack::decode(&a.encode()), Some(a));

        assert_eq!(Ack::decode(&[]), None, "empty");
        assert_eq!(Ack::decode(&[1, 2, 3]), None, "too short");
        assert_eq!(Ack::decode(&[1, 2, 3, 4, 5]), None, "too long");
    }

    /// A reused index is a different entity, and the mapping must say so —
    /// this is the case `World::despawn` makes reachable within one tick.
    #[test]
    fn a_reused_index_is_a_distinct_entity() {
        let old = EntityId { index: 4, generation: 1 };
        let new = EntityId { index: 4, generation: 2 };
        assert_ne!(old, new, "generation is what separates them");

        let mut r = Relevancy::new();
        r.begin();
        r.push(new);
        assert_eq!(r.entity(0), Some(new));
        assert_ne!(r.entity(0), Some(old), "the stale generation must not match");
    }
}
