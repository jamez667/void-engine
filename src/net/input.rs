//! Client-to-server input: held state, one-shot commands, and the rule
//! for reconciling frames that arrive out of order.
//!
//! Lifted from `mini-miner-2`'s `net::proto::InputFrame` and
//! `net::host::keep_newest`, which is the shape both games in this
//! workspace converged on from opposite directions — one by design, one
//! by forty-six hand-written latch lines.
//!
//! # Why input has two halves
//!
//! A player's input is never one kind of thing. Walking is a *level*: a
//! key is held, and the newest report of it is the only one that matters
//! — applying an older one walks the character back the way they came.
//! Firing is an *event*: it happened once, and losing it means the player
//! pressed a button and nothing occurred.
//!
//! Encoding both as fields of a per-tick struct is the trap. A 60 Hz
//! client sending into a 30 Hz server emits a zero-valued packet between
//! ticks, which overwrites the one-shot before the simulation reads it.
//! void-claim hit this three separate times — the comments in its
//! `connection.rs` record each wave — and fixed it with a `let prev_x =`
//! line per field, forty-six of them, each latching one value across the
//! gap. mini-miner-2 avoided it by making one-shots a list of events
//! rather than fields to sample.
//!
//! This carries both halves explicitly, so the distinction is in the type
//! rather than in a convention each game rediscovers.
//!
//! # What the engine does not decide
//!
//! The wire format. Both `H` and `C` are the game's own types and the
//! game encodes them — void-claim's vocabulary and mini-miner-2's have
//! nothing in common, and an engine-imposed command enum would fit
//! neither. What is lifted is the *reconciliation rule*, which is
//! identical for both and easy to get subtly wrong.
//!
//! # An authority note worth preserving
//!
//! mini-miner-2's command type is deliberately not its local `Command`:
//! that one carries a `Vec2` the client computed from its own camera, and
//! a server must not take a client's word for a screen-space conversion.
//! Its `Act` variant sends a *menu row index*, so a client can only pick
//! what it was offered rather than naming an action; its `GrabNearest`
//! carries nothing at all, because "what is nearest" is a question about
//! the world that the server answers from its own copy.
//!
//! None of that is enforceable here — `C` is whatever the game says. But
//! a command type is a security boundary, and it is worth designing as
//! one.

use std::collections::HashMap;

/// Which connected player a frame came from.
///
/// `u32` rather than the `u8` mini-miner-2 uses for its two miners: an
/// engine that fixes the ceiling at 255 players is one a game has to stop
/// using at 256, and the conversion at the boundary costs nothing.
pub type PlayerId = u32;

/// Most commands the queue will hold for one player between drains.
///
/// A client sending faster than the server ticks is normal — 60 Hz into
/// 30 Hz is two frames a tick before any jitter — and their commands
/// accumulate until the simulation takes them. A client sending *far*
/// faster is either broken or hostile, and an unbounded merge would let
/// it make the server do arbitrary work in one tick.
///
/// 32 is far above what a player can physically produce (a frame carries
/// the commands from one client tick; a human generating 32 discrete
/// actions inside a single server tick is not playing) and far below what
/// costs anything to process.
pub const MAX_PENDING_COMMANDS: usize = 32;

/// One client's input for one of its frames.
///
/// `H` is the held state — walk direction, throttle, whatever is a level
/// rather than an event. `C` is one discrete ask.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputFrame<H, C> {
    /// Counts up forever on the client. The server keeps the highest it
    /// has seen and echoes it back, so the client knows which of its own
    /// commands have been accounted for and can stop replaying them.
    ///
    /// Monotonic rather than a tick number: a client's tick and a
    /// server's are different clocks, and only the client can say which
    /// of its own frames is newer.
    pub seq: u32,
    /// The level-triggered half. Newest wins outright.
    pub held: H,
    /// The edge-triggered half. Every one must be applied exactly once.
    pub commands: Vec<C>,
}

impl<H: Default, C> Default for InputFrame<H, C> {
    fn default() -> Self {
        Self { seq: 0, held: H::default(), commands: Vec::new() }
    }
}

/// What one player has pending, after reconciling however many frames
/// arrived since the last drain.
#[derive(Clone, Debug)]
pub struct Pending<H, C> {
    /// Highest `seq` seen. What belongs in the next snapshot's ack.
    pub seq: u32,
    /// Held state from the highest-`seq` frame seen.
    pub held: H,
    /// Every command from every frame, oldest first.
    pub commands: Vec<C>,
    /// Commands discarded because [`MAX_PENDING_COMMANDS`] was reached.
    ///
    /// Non-zero means a client is sending faster than the server can
    /// consume, which is worth surfacing rather than swallowing: it is
    /// either a bug in that client or an attempt to flood this one.
    pub dropped: u32,
}

/// Per-player input, reconciled as frames arrive.
///
/// Held by the network layer, drained once per simulation tick.
///
/// # The rule
///
/// Datagrams are neither ordered nor redelivered, so frames from one
/// player can land backwards and several can land between two drains.
/// The two halves reconcile differently, and this is the whole reason
/// the type exists:
///
/// - **Held state** takes the highest `seq`. An older frame describes a
///   keyboard that has already changed.
/// - **Commands accumulate**, from every frame, in arrival order —
///   *including* frames that arrive late. A command is something the
///   player did; arriving out of order makes it old, not imaginary.
///
/// The lifted implementation replaced the whole frame on a newer `seq`,
/// which is right for the first rule and silently drops commands under
/// the second. It is not obviously wrong — it only bites when two frames
/// land between drains, which needs a client faster than the server, and
/// every test of it constructed frames with no commands.
#[derive(Debug)]
pub struct InputQueue<H, C> {
    pending: HashMap<PlayerId, Pending<H, C>>,
    cap: usize,
}

impl<H, C> Default for InputQueue<H, C> {
    fn default() -> Self {
        Self { pending: HashMap::new(), cap: MAX_PENDING_COMMANDS }
    }
}

impl<H: Copy + Default, C> InputQueue<H, C> {
    pub fn new() -> Self {
        Self::default()
    }

    /// A queue with a non-default per-player command cap.
    ///
    /// Zero is legal and means "held state only": every command is
    /// dropped and counted. A game with no discrete actions can say so.
    pub fn with_cap(cap: usize) -> Self {
        Self { pending: HashMap::new(), cap }
    }

    /// Fold one arriving frame into what `who` has pending.
    ///
    /// Safe to call from the network thread for every datagram; the
    /// simulation calls [`drain`](Self::drain) once a tick.
    pub fn accept(&mut self, who: PlayerId, frame: InputFrame<H, C>) {
        let slot = self.pending.entry(who).or_insert_with(|| Pending {
            seq: 0,
            held: H::default(),
            commands: Vec::new(),
            dropped: 0,
        });

        // Held state: newest only. `>=` rather than `>` so a repeated
        // seq — a duplicate datagram, which QUIC permits — is idempotent
        // rather than ignored, and the first frame (seq 0 into a slot
        // initialised at 0) still lands.
        if frame.seq >= slot.seq {
            slot.seq = frame.seq;
            slot.held = frame.held;
        }

        // Commands: all of them, whatever order they arrived in. A late
        // frame's command is late, not fictional.
        for c in frame.commands {
            if self.cap == 0 {
                // A zero cap means "this game has no discrete actions".
                // Counted rather than silently ignored, and returned to
                // before the eviction path — which would otherwise
                // `remove(0)` an empty vector and panic.
                slot.dropped = slot.dropped.saturating_add(1);
                continue;
            }
            if slot.commands.len() >= self.cap {
                // Oldest first. The newest command is the one the player
                // just pressed and is watching for; an older queued
                // action going missing is invisible, a fresh click being
                // eaten is the bug both games already fixed once.
                slot.commands.remove(0);
                slot.dropped = slot.dropped.saturating_add(1);
            }
            slot.commands.push(c);
        }
    }

    /// Take everything pending, leaving the queue empty.
    ///
    /// Called once per simulation tick. Held state does not persist
    /// across a drain: a client that stops sending has stopped holding
    /// keys, and continuing to apply its last direction would walk a
    /// disconnected player into a wall forever.
    pub fn drain(&mut self) -> Vec<(PlayerId, Pending<H, C>)> {
        self.pending.drain().collect()
    }

    /// Highest `seq` seen from `who`, for the ack the next snapshot
    /// carries. Readable without draining.
    pub fn ack_seq(&self, who: PlayerId) -> u32 {
        self.pending.get(&who).map_or(0, |p| p.seq)
    }

    /// Players with anything pending.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Forget a player entirely, on disconnect.
    ///
    /// Without this a queue keeps a slot per player who ever connected,
    /// which on a long-lived server is a slow leak keyed by a number that
    /// never repeats.
    pub fn remove(&mut self, who: PlayerId) {
        self.pending.remove(&who);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
    enum Walk {
        #[default]
        Still,
        N,
        S,
        E,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum Cmd {
        Fire,
        Grab,
        Drop(u8),
    }

    fn frame(seq: u32, held: Walk, commands: Vec<Cmd>) -> InputFrame<Walk, Cmd> {
        InputFrame { seq, held, commands }
    }

    /// Held state is a level: the newest report wins and an older one is
    /// discarded, or a late datagram walks the player backwards for a
    /// tick.
    #[test]
    fn held_state_takes_the_newest_frame() {
        let mut q = InputQueue::new();
        q.accept(1, frame(5, Walk::N, vec![]));
        q.accept(1, frame(3, Walk::S, vec![]));

        let out = q.drain();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].1.held, Walk::N, "the late frame must not win");
        assert_eq!(out[0].1.seq, 5);
    }

    #[test]
    fn a_newer_frame_replaces_held_state() {
        let mut q = InputQueue::new();
        q.accept(1, frame(5, Walk::N, vec![]));
        q.accept(1, frame(6, Walk::E, vec![]));

        let out = q.drain();
        assert_eq!(out[0].1.held, Walk::E);
        assert_eq!(out[0].1.seq, 6);
    }

    /// **The reason this type exists.** Two frames between drains, each
    /// carrying a command. The lifted implementation replaced the whole
    /// frame on a newer seq, so the first command vanished — a button
    /// press that did nothing.
    #[test]
    fn commands_from_every_frame_survive() {
        let mut q = InputQueue::new();
        q.accept(1, frame(5, Walk::N, vec![Cmd::Fire]));
        q.accept(1, frame(6, Walk::E, vec![Cmd::Grab]));

        let out = q.drain();
        assert_eq!(
            out[0].1.commands,
            vec![Cmd::Fire, Cmd::Grab],
            "a newer frame must not discard an earlier frame's commands",
        );
        assert_eq!(out[0].1.held, Walk::E, "while held state still takes the newest");
    }

    /// A command that arrives late is late, not imaginary. Its held state
    /// loses; its command does not.
    #[test]
    fn a_late_frames_commands_still_apply() {
        let mut q = InputQueue::new();
        q.accept(1, frame(9, Walk::N, vec![Cmd::Fire]));
        q.accept(1, frame(4, Walk::S, vec![Cmd::Grab]));

        let out = q.drain();
        assert_eq!(out[0].1.held, Walk::N, "the late frame's held state is stale");
        assert!(
            out[0].1.commands.contains(&Cmd::Grab),
            "but the action the player took is not",
        );
        assert_eq!(out[0].1.commands.len(), 2);
    }

    /// Order is preserved, because a player who drops a thing and then
    /// picks it up did not do the reverse.
    #[test]
    fn commands_keep_their_arrival_order() {
        let mut q = InputQueue::new();
        q.accept(1, frame(1, Walk::Still, vec![Cmd::Drop(1), Cmd::Drop(2)]));
        q.accept(1, frame(2, Walk::Still, vec![Cmd::Drop(3)]));

        let out = q.drain();
        assert_eq!(out[0].1.commands, vec![Cmd::Drop(1), Cmd::Drop(2), Cmd::Drop(3)]);
    }

    /// Two players' frames never displace each other, or one of them
    /// stops moving.
    #[test]
    fn players_are_kept_apart() {
        let mut q = InputQueue::new();
        q.accept(0, frame(9, Walk::E, vec![Cmd::Fire]));
        q.accept(1, frame(2, Walk::N, vec![Cmd::Grab]));

        let mut out = q.drain();
        out.sort_by_key(|(id, _)| *id);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].1.held, Walk::E);
        assert_eq!(out[0].1.commands, vec![Cmd::Fire]);
        assert_eq!(out[1].1.held, Walk::N);
        assert_eq!(out[1].1.commands, vec![Cmd::Grab]);
    }

    /// An unbounded merge lets one client make the server do arbitrary
    /// work in a tick. The cap sheds oldest-first and counts what it shed.
    #[test]
    fn a_flood_is_capped_and_counted() {
        let mut q: InputQueue<Walk, Cmd> = InputQueue::with_cap(4);
        for i in 0..10u8 {
            q.accept(1, frame(i as u32, Walk::Still, vec![Cmd::Drop(i)]));
        }

        let out = q.drain();
        assert_eq!(out[0].1.commands.len(), 4, "held at the cap");
        assert_eq!(out[0].1.dropped, 6, "and said how many it shed");
        assert_eq!(
            out[0].1.commands,
            vec![Cmd::Drop(6), Cmd::Drop(7), Cmd::Drop(8), Cmd::Drop(9)],
            "the newest survive: a fresh press is what the player is watching for",
        );
    }

    /// Draining empties: held state must not persist for a client that
    /// has stopped sending, or a disconnected player walks forever.
    #[test]
    fn draining_leaves_nothing_behind() {
        let mut q = InputQueue::new();
        q.accept(1, frame(1, Walk::N, vec![Cmd::Fire]));
        assert_eq!(q.len(), 1);

        let first = q.drain();
        assert_eq!(first.len(), 1);
        assert!(q.is_empty(), "a drain takes everything");
        assert!(q.drain().is_empty(), "and a second drain finds nothing");
    }

    /// A duplicate datagram is idempotent for held state rather than
    /// ignored — QUIC may redeliver, and `>` would leave the repeat
    /// unapplied while `>=` makes it a no-op.
    #[test]
    fn a_repeated_seq_is_harmless() {
        let mut q = InputQueue::new();
        q.accept(1, frame(7, Walk::N, vec![]));
        q.accept(1, frame(7, Walk::N, vec![]));

        let out = q.drain();
        assert_eq!(out[0].1.seq, 7);
        assert_eq!(out[0].1.held, Walk::N);
    }

    /// The very first frame lands even at seq 0, which is where a client
    /// that has just connected starts counting.
    #[test]
    fn the_first_frame_at_seq_zero_is_accepted() {
        let mut q = InputQueue::new();
        q.accept(1, frame(0, Walk::E, vec![Cmd::Fire]));

        let out = q.drain();
        assert_eq!(out[0].1.held, Walk::E, "seq 0 is a real frame, not an empty slot");
        assert_eq!(out[0].1.commands, vec![Cmd::Fire]);
    }

    /// The ack is readable without draining, because the snapshot that
    /// carries it is built on a different schedule from the tick that
    /// consumes input.
    #[test]
    fn the_ack_is_readable_before_a_drain() {
        let mut q: InputQueue<Walk, Cmd> = InputQueue::new();
        assert_eq!(q.ack_seq(1), 0, "nothing seen from an unknown player");
        q.accept(1, frame(12, Walk::N, vec![]));
        assert_eq!(q.ack_seq(1), 12);
    }

    /// A disconnected player's slot goes, or a long-lived server leaks
    /// one entry per player who ever connected.
    #[test]
    fn removing_a_player_forgets_them() {
        let mut q = InputQueue::new();
        q.accept(1, frame(3, Walk::N, vec![Cmd::Fire]));
        q.remove(1);
        assert!(q.is_empty());
        assert_eq!(q.ack_seq(1), 0);
    }

    /// A cap of zero is a legal way to say "this game has no discrete
    /// actions", and must not panic on the `remove(0)` path.
    #[test]
    fn a_zero_cap_drops_every_command() {
        let mut q: InputQueue<Walk, Cmd> = InputQueue::with_cap(0);
        q.accept(1, frame(1, Walk::N, vec![Cmd::Fire, Cmd::Grab]));

        let out = q.drain();
        assert!(out[0].1.commands.is_empty());
        assert_eq!(out[0].1.dropped, 2);
        assert_eq!(out[0].1.held, Walk::N, "held state is unaffected");
    }
}
