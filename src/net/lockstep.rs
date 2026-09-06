//! Two-peer deterministic lockstep transport.
//!
//! Lockstep is the opposite trade to `net::interp`. Interpolation assumes an
//! authoritative server streaming snapshots of state, and hides latency by
//! rendering the past. Lockstep has no server and sends no state at all: both
//! peers run the *same* simulation from the same seed, exchange only their
//! inputs, and a tick may not run until both peers' inputs for it are in hand.
//! Bandwidth is a few bytes per tick regardless of world size — but every peer
//! stalls at the speed of the slowest link, and any divergence in the sim is
//! permanent and silent. Hence `Checksum`.
//!
//! Because the two peers must stay bit-identical, this module is deliberately
//! paranoid about the wire:
//!
//! * **No floats.** Not in any message, not in the input payload. `f32` sums
//!   reorder across compilers and target features (FMA contraction, x87
//!   80-bit spills), and two peers that disagree in the last mantissa bit
//!   diverge into different worlds within seconds. Quantise anything
//!   continuous to a fixed-point integer before it goes on the wire.
//! * **No `bincode`/`serde_json`.** This crate keeps its dependency list
//!   short on purpose (see Cargo.toml), and the whole protocol is five
//!   messages of fixed shape — a hand-rolled `u16`-length-prefixed encoding
//!   is smaller than the derive machinery it replaces and pins the byte
//!   layout where we can see it. Everything multi-byte is big-endian.
//! * **Inputs are never dropped.** `app.rs`'s `PerfLogger` deliberately
//!   `try_send`s and discards on a full channel rather than stall the game;
//!   that is right for telemetry and fatal here. One missing input frame is
//!   not a dropped log line, it is a peer that can never advance again. The
//!   send side of this module uses an unbounded channel and the socket thread
//!   blocks in `write_all`. Backpressure shows up as a stall, which is
//!   recoverable; a drop is not.
//!
//! # Input delay — why the local player also waits
//!
//! The naive scheme ("send input for tick N, run tick N once both arrive")
//! stalls for a full round-trip on *every* tick. The standard fix is to
//! separate the tick an input is *sampled* on from the tick it is *applied*
//! on: input sampled at tick N is transmitted immediately but both peers
//! apply it at tick `N + delay`. That buys `delay` ticks (~50 ms at
//! [`DEFAULT_INPUT_DELAY`] and 60 Hz) of one-way slack for the packet to
//! land, and the sim runs without stalling as long as latency stays inside
//! that budget. The cost is that your own actions are also delayed by
//! `delay` ticks — which is why the local input is buffered rather than
//! applied at once. Applying local input immediately and remote input late
//! would desync instantly: the two peers would be running different games.
//!
//! # The input edge-flag hazard — read this before wiring a game up
//!
//! `App`'s fixed-update loop clears `InputState`'s pressed/released edge
//! flags after the first step of a frame, because those flags mean "went
//! down *this instant*" and a catch-up frame running five steps would
//! otherwise see the same keypress five times (that bug shipped: one press
//! typed five characters into a login field).
//!
//! Lockstep makes catch-up frames the norm rather than the exception. A
//! stalled peer buffers ticks while it waits, then drains all of them in one
//! frame the moment the remote input lands. So a game **must sample input
//! once per frame into an input frame, and feed that value to every tick it
//! runs** — it must never read `ctx.input` inside a tick. Concretely:
//!
//! ```ignore
//! // ONCE per frame, before any tick runs — in practice at the top of the
//! // first `fixed_update` of the frame, or in `can_advance`.
//! let bits = InputBits::from_input(ctx.input);   // game-defined packing
//! net.submit_local(net.next_local_tick(), bits);
//!
//! // Then, per tick, take the input from the buffer and NOT from ctx.input:
//! let (mine, theirs) = net.inputs_for(tick).unwrap();
//! ```
//!
//! This is not merely an edge-flag nicety: reading `ctx.input` inside a tick
//! is a determinism bug outright. The remote peer has no access to your live
//! `InputState`, so anything sampled there and not sent over the wire is,
//! by definition, state only one of the two peers has.
//!
//! The engine helps as far as it soundly can. [`TickBuffer::inputs_for`] is
//! the *only* way to read an input for a tick, and it hands back both peers'
//! values together, so the natural way to write the game is also the correct
//! one. The engine cannot enforce the rest — it cannot tell whether a game
//! read `ctx.input` — and a heavier mechanism (hiding `ctx.input` behind a
//! per-frame token, say) would complicate every non-networked game to police
//! a rule only networked ones have. So: documented, and shaped, not enforced.
//!
//! # Wiring into the engine loop
//!
//! [`App::can_advance`](crate::app::App::can_advance) is checked before each
//! fixed step; return `!net.stalled()` from it and the engine defers the tick
//! instead of running it with inputs it does not have. Deferred, not dropped
//! — the timestep accumulator keeps the un-run step, so the ticks arrive
//! later rather than time being silently skipped.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::mpsc::{self, Receiver, Sender, TryRecvError};
use std::time::Duration;

/// Bumped whenever the wire format changes shape. Two peers that disagree
/// here cannot safely interpret each other's bytes, so the handshake refuses
/// the connection rather than desyncing later in a way that looks like a
/// simulation bug.
pub const PROTOCOL_VERSION: u32 = 1;

/// Ticks between sampling an input and applying it. Three ticks at 60 Hz is
/// 50 ms of one-way slack — enough for a LAN or a good regional link, and
/// small enough that the added control latency reads as "slightly heavy"
/// rather than broken. Games on worse links should raise it; every extra
/// tick is 16.7 ms of input lag bought for 16.7 ms of latency tolerance.
pub const DEFAULT_INPUT_DELAY: u32 = 3;

/// Cap on a `Chat` payload. Bounded so a hostile or broken peer cannot make
/// us allocate arbitrarily from a length prefix we have not validated.
pub const MAX_CHAT_BYTES: usize = 1024;

/// Largest frame we will accept. Chat is the only variable-length message
/// and it is capped well below this; anything larger is a desynchronised
/// stream or a hostile peer, and we drop the connection rather than trust it.
const MAX_FRAME_BYTES: usize = MAX_CHAT_BYTES + 64;

// Message type tags. Explicit discriminants: these are wire values, so they
// must not shift when someone reorders the enum.
const TAG_HELLO:    u8 = 1;
const TAG_INPUT:    u8 = 2;
const TAG_CHECKSUM: u8 = 3;
const TAG_CHAT:     u8 = 4;
const TAG_BYE:      u8 = 5;

// ── messages ────────────────────────────────────────────────────────────────

/// One protocol message. Deliberately small and closed: every variant is
/// something both peers must agree on to stay in lockstep, and the hot path
/// (`Input`) is a fixed 11 bytes on the wire including its length prefix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Message {
    /// Handshake, sent by both sides immediately on connect. The peers must
    /// agree on protocol version *and* world seed before tick 0 — a seed
    /// mismatch means two different worlds, which presents exactly like a
    /// desync but is unrecoverable, so we catch it up front.
    Hello { protocol_version: u32, world_seed: u64, input_delay: u32 },
    /// The hot path: this peer's input for `tick`. `bits` is opaque to the
    /// engine — the game defines the bitfield — which is what keeps this
    /// module game-agnostic. Quantise; do not pack a float in here.
    Input { tick: u32, bits: u32 },
    /// A hash of the sender's world state at `tick`, for desync detection.
    /// Lockstep failures are otherwise silent: the two worlds drift apart
    /// and nobody notices until the games visibly disagree, long after the
    /// tick that actually caused it. Exchanging a hash every N ticks turns
    /// that into a specific tick number.
    Checksum { tick: u32, hash: u64 },
    /// Out-of-band text. Not part of the simulation and not tick-ordered,
    /// so it can be delivered whenever it lands without affecting determinism.
    Chat { text: String },
    /// Clean shutdown, so the peer can distinguish "quit" from "crashed".
    Bye,
}

/// Wire encoding failed. Only reachable by handing `Chat` more than
/// [`MAX_CHAT_BYTES`]; every other variant is fixed-size and infallible.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EncodeError {
    ChatTooLong(usize),
}

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EncodeError::ChatTooLong(n) => {
                write!(f, "chat message is {n} bytes, limit is {MAX_CHAT_BYTES}")
            }
        }
    }
}

impl std::error::Error for EncodeError {}

/// A frame did not decode. Every variant means the stream is no longer
/// trustworthy — we have lost byte alignment or the peer is not speaking
/// this protocol — so the only correct response is to drop the connection.
/// Resynchronising a lockstep stream mid-flight is not possible: we would
/// not know which inputs we had missed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeError {
    /// Tag byte is not a known message type.
    UnknownTag(u8),
    /// Frame ended mid-field. Expected at least `need` bytes, had `had`.
    Truncated { need: usize, had: usize },
    /// Trailing bytes after a complete message: our idea of the layout and
    /// the sender's disagree, so nothing after this point can be trusted.
    TrailingBytes(usize),
    /// `Chat` payload exceeded [`MAX_CHAT_BYTES`].
    ChatTooLong(usize),
    /// `Chat` payload was not valid UTF-8.
    BadUtf8,
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::UnknownTag(t)     => write!(f, "unknown message tag {t}"),
            DecodeError::Truncated { need, had } => {
                write!(f, "truncated frame: needed {need} bytes, had {had}")
            }
            DecodeError::TrailingBytes(n)  => write!(f, "{n} trailing bytes after message"),
            DecodeError::ChatTooLong(n)    => write!(f, "chat payload {n} bytes exceeds limit"),
            DecodeError::BadUtf8           => write!(f, "chat payload was not valid UTF-8"),
        }
    }
}

impl std::error::Error for DecodeError {}

/// Cursor over a frame body. Every read is bounds-checked, because the bytes
/// come from the network: a malformed length prefix must produce a
/// `DecodeError`, never a panic that takes the game down.
struct Cursor<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a [u8]) -> Self { Self { buf, pos: 0 } }

    fn take(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        let end = self.pos.checked_add(n).ok_or(DecodeError::Truncated {
            need: n,
            had: self.buf.len() - self.pos,
        })?;
        if end > self.buf.len() {
            return Err(DecodeError::Truncated { need: n, had: self.buf.len() - self.pos });
        }
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u8(&mut self) -> Result<u8, DecodeError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, DecodeError> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, DecodeError> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, DecodeError> {
        let b = self.take(8)?;
        Ok(u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]))
    }

    /// Every byte must be consumed. A message that decodes but leaves bytes
    /// over means the sender wrote a layout we do not know — treat it as a
    /// protocol break rather than silently ignoring the tail.
    fn finish(&self) -> Result<(), DecodeError> {
        let left = self.buf.len() - self.pos;
        if left != 0 { return Err(DecodeError::TrailingBytes(left)); }
        Ok(())
    }
}

impl Message {
    /// Encode the message body — the payload *without* the length prefix.
    /// [`encode_frame`](Message::encode_frame) adds that.
    pub fn encode_body(&self) -> Result<Vec<u8>, EncodeError> {
        let mut out = Vec::with_capacity(16);
        match self {
            Message::Hello { protocol_version, world_seed, input_delay } => {
                out.push(TAG_HELLO);
                out.extend_from_slice(&protocol_version.to_be_bytes());
                out.extend_from_slice(&world_seed.to_be_bytes());
                out.extend_from_slice(&input_delay.to_be_bytes());
            }
            Message::Input { tick, bits } => {
                out.push(TAG_INPUT);
                out.extend_from_slice(&tick.to_be_bytes());
                out.extend_from_slice(&bits.to_be_bytes());
            }
            Message::Checksum { tick, hash } => {
                out.push(TAG_CHECKSUM);
                out.extend_from_slice(&tick.to_be_bytes());
                out.extend_from_slice(&hash.to_be_bytes());
            }
            Message::Chat { text } => {
                let bytes = text.as_bytes();
                if bytes.len() > MAX_CHAT_BYTES {
                    return Err(EncodeError::ChatTooLong(bytes.len()));
                }
                out.push(TAG_CHAT);
                // u16 inner length: the frame prefix already bounds this, but
                // carrying it explicitly keeps the body self-describing, so a
                // future variant can follow a chat field without ambiguity.
                out.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
                out.extend_from_slice(bytes);
            }
            Message::Bye => out.push(TAG_BYE),
        }
        Ok(out)
    }

    /// Encode a complete frame: `u16` big-endian body length, then the body.
    /// The length prefix is what lets the reader find message boundaries in
    /// a TCP stream, which delivers bytes and knows nothing about our frames.
    pub fn encode_frame(&self) -> Result<Vec<u8>, EncodeError> {
        let body = self.encode_body()?;
        let mut out = Vec::with_capacity(body.len() + 2);
        out.extend_from_slice(&(body.len() as u16).to_be_bytes());
        out.extend_from_slice(&body);
        Ok(out)
    }

    /// Decode a frame body (no length prefix).
    pub fn decode_body(buf: &[u8]) -> Result<Message, DecodeError> {
        let mut c = Cursor::new(buf);
        let msg = match c.u8()? {
            TAG_HELLO => Message::Hello {
                protocol_version: c.u32()?,
                world_seed:       c.u64()?,
                input_delay:      c.u32()?,
            },
            TAG_INPUT    => Message::Input { tick: c.u32()?, bits: c.u32()? },
            TAG_CHECKSUM => Message::Checksum { tick: c.u32()?, hash: c.u64()? },
            TAG_CHAT => {
                let n = c.u16()? as usize;
                if n > MAX_CHAT_BYTES { return Err(DecodeError::ChatTooLong(n)); }
                let bytes = c.take(n)?;
                let text = std::str::from_utf8(bytes).map_err(|_| DecodeError::BadUtf8)?;
                Message::Chat { text: text.to_string() }
            }
            TAG_BYE => Message::Bye,
            other => return Err(DecodeError::UnknownTag(other)),
        };
        c.finish()?;
        Ok(msg)
    }
}

// ── handshake ───────────────────────────────────────────────────────────────

/// Why a handshake was refused. All three are unrecoverable: there is no
/// negotiation step, because there is nothing to negotiate — a peer running
/// a different protocol or a different world cannot be made compatible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HandshakeError {
    /// Peer speaks a different wire format.
    VersionMismatch { ours: u32, theirs: u32 },
    /// Peer would generate a different world. Presents identically to a
    /// desync but happens before tick 0, so we can name it exactly.
    SeedMismatch { ours: u64, theirs: u64 },
    /// Peers disagree on input delay, so they would apply the same input at
    /// different ticks — a guaranteed desync on the first input.
    DelayMismatch { ours: u32, theirs: u32 },
    /// First message was not a `Hello`.
    NotHello,
}

impl std::fmt::Display for HandshakeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandshakeError::VersionMismatch { ours, theirs } =>
                write!(f, "protocol version mismatch: ours {ours}, theirs {theirs}"),
            HandshakeError::SeedMismatch { ours, theirs } =>
                write!(f, "world seed mismatch: ours {ours}, theirs {theirs}"),
            HandshakeError::DelayMismatch { ours, theirs } =>
                write!(f, "input delay mismatch: ours {ours}, theirs {theirs}"),
            HandshakeError::NotHello =>
                write!(f, "peer's first message was not a Hello"),
        }
    }
}

impl std::error::Error for HandshakeError {}

/// What this peer will announce, and what it demands of the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Handshake {
    pub protocol_version: u32,
    pub world_seed: u64,
    pub input_delay: u32,
}

impl Handshake {
    /// Announce the current protocol version for `world_seed` at the default
    /// input delay.
    pub fn new(world_seed: u64) -> Self {
        Self {
            protocol_version: PROTOCOL_VERSION,
            world_seed,
            input_delay: DEFAULT_INPUT_DELAY,
        }
    }

    pub fn with_input_delay(mut self, delay: u32) -> Self {
        self.input_delay = delay;
        self
    }

    fn to_message(self) -> Message {
        Message::Hello {
            protocol_version: self.protocol_version,
            world_seed: self.world_seed,
            input_delay: self.input_delay,
        }
    }

    /// Check the peer's `Hello` against ours. Pure, so the rejection rules
    /// are testable without a socket — which matters, because these are
    /// exactly the paths that never run during ordinary local testing.
    pub fn check(&self, peer: &Message) -> Result<(), HandshakeError> {
        let Message::Hello { protocol_version, world_seed, input_delay } = peer else {
            return Err(HandshakeError::NotHello);
        };
        // Version first: if the format differs, the seed we just parsed out
        // of their frame may not even be the field they meant to send.
        if *protocol_version != self.protocol_version {
            return Err(HandshakeError::VersionMismatch {
                ours: self.protocol_version,
                theirs: *protocol_version,
            });
        }
        if *world_seed != self.world_seed {
            return Err(HandshakeError::SeedMismatch {
                ours: self.world_seed,
                theirs: *world_seed,
            });
        }
        if *input_delay != self.input_delay {
            return Err(HandshakeError::DelayMismatch {
                ours: self.input_delay,
                theirs: *input_delay,
            });
        }
        Ok(())
    }
}

// ── tick buffer ─────────────────────────────────────────────────────────────

/// Both peers' inputs for one tick, which is the only form in which the
/// simulation is allowed to see them — see the module docs on sampling once
/// per frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TickInputs {
    pub local: u32,
    pub remote: u32,
}

/// Ring of per-tick inputs from both peers, and the stall/ready decision
/// built on it.
///
/// The buffer is a ring of [`TickBuffer::CAPACITY`] slots indexed by
/// `tick % CAPACITY`, each stamped with the tick it holds so a stale slot
/// cannot be mistaken for a live one. A ring rather than a map because the
/// access pattern is a narrow sliding window — we only ever care about ticks
/// between "the one we are about to run" and "that plus input delay" — and a
/// fixed ring never allocates on the hot path.
///
/// A peer that falls more than `CAPACITY` ticks behind would wrap the ring
/// and overwrite inputs it has not consumed. It cannot: it stalls at the
/// first tick it is missing, long before the ring can lap it.
#[derive(Clone, Debug)]
pub struct TickBuffer {
    local:  Vec<Slot>,
    remote: Vec<Slot>,
    /// Next tick the simulation will attempt. Advanced only by `consume`.
    next_tick: u32,
    input_delay: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Slot {
    tick: u32,
    bits: u32,
    filled: bool,
}

impl TickBuffer {
    /// 256 ticks is ~4.3 s at 60 Hz — far more slack than a link that has
    /// not already timed out will ever need, and a power of two so the
    /// index is a mask rather than a division.
    pub const CAPACITY: usize = 256;

    pub fn new(input_delay: u32) -> Self {
        Self {
            local:  vec![Slot::default(); Self::CAPACITY],
            remote: vec![Slot::default(); Self::CAPACITY],
            next_tick: 0,
            input_delay,
        }
    }

    #[inline]
    fn idx(tick: u32) -> usize { (tick as usize) & (Self::CAPACITY - 1) }

    pub fn input_delay(&self) -> u32 { self.input_delay }

    /// The tick the simulation is waiting to run.
    pub fn next_tick(&self) -> u32 { self.next_tick }

    /// The tick that input sampled *right now* will be applied at. Sample
    /// once per frame and submit under this tick; both peers apply it there.
    pub fn scheduled_tick(&self) -> u32 { self.next_tick + self.input_delay }

    /// Record locally-sampled input for `tick`. `tick` is the *application*
    /// tick, normally [`scheduled_tick`](Self::scheduled_tick).
    ///
    /// Submitting for a tick already run is ignored rather than treated as an
    /// error: it is the natural consequence of a frame that ran several
    /// catch-up ticks, and the correct response is to drop it, since that
    /// tick's outcome is already baked into the world on both peers.
    pub fn submit_local(&mut self, tick: u32, bits: u32) {
        if tick < self.next_tick { return; }
        self.local[Self::idx(tick)] = Slot { tick, bits, filled: true };
    }

    /// Record input received from the peer.
    pub fn submit_remote(&mut self, tick: u32, bits: u32) {
        if tick < self.next_tick { return; }
        self.remote[Self::idx(tick)] = Slot { tick, bits, filled: true };
    }

    /// Do we have *both* peers' inputs for `tick`? A tick with only one side
    /// is not runnable: running it with a guessed remote input is rollback
    /// netcode, a different (and much larger) design.
    pub fn ready(&self, tick: u32) -> bool {
        self.inputs_for(tick).is_some()
    }

    /// Both inputs for `tick`, or `None` if either side is missing.
    ///
    /// The `slot.tick == tick` check is what makes the ring safe: an index
    /// collision from a tick `CAPACITY` away leaves a filled slot stamped
    /// with the wrong tick, and this rejects it instead of feeding the sim
    /// an input from 256 ticks ago.
    pub fn inputs_for(&self, tick: u32) -> Option<TickInputs> {
        let l = self.local[Self::idx(tick)];
        let r = self.remote[Self::idx(tick)];
        if l.filled && r.filled && l.tick == tick && r.tick == tick {
            Some(TickInputs { local: l.bits, remote: r.bits })
        } else {
            None
        }
    }

    /// Are we blocked on the peer? True when the next tick's inputs are not
    /// both present. This is the value a game returns (inverted) from
    /// `App::can_advance`.
    pub fn stalled(&self) -> bool {
        !self.ready(self.next_tick)
    }

    /// Take the inputs for the next tick and advance. Returns `None` — and
    /// does not advance — when stalled, so a caller that ignores
    /// [`stalled`](Self::stalled) still cannot run a tick without both
    /// inputs. The only way to move `next_tick` forward is to have actually
    /// consumed a complete pair, which is the invariant the whole scheme
    /// rests on.
    pub fn consume(&mut self) -> Option<TickInputs> {
        let tick = self.next_tick;
        let inputs = self.inputs_for(tick)?;
        // Clear the consumed slots so a wrapped index cannot resurrect them
        // even if the tick stamp were to coincide.
        self.local[Self::idx(tick)].filled = false;
        self.remote[Self::idx(tick)].filled = false;
        self.next_tick = tick.wrapping_add(1);
        Some(inputs)
    }

    /// Ticks of remote input buffered ahead of the current one. Zero means
    /// we are running on inputs that arrived just in time and the next
    /// hiccup stalls us; a healthy connection sits near `input_delay`. Useful
    /// for a connection-quality readout.
    pub fn remote_lead(&self) -> u32 {
        let mut lead = 0;
        for i in 0..Self::CAPACITY as u32 {
            let tick = self.next_tick.wrapping_add(i);
            let r = self.remote[Self::idx(tick)];
            if r.filled && r.tick == tick { lead += 1; } else { break; }
        }
        lead
    }
}

// ── desync detection ────────────────────────────────────────────────────────

/// Peers disagreed about the world at `tick`.
///
/// By the time this fires the two simulations have already diverged, and
/// nothing here can repair that — lockstep has no authoritative copy to
/// resynchronise from. Its value is diagnostic: it names the *tick*, which
/// is the one piece of information that makes a desync tractable to debug.
/// Without it the first symptom is two players describing different worlds
/// some unknowable time after the cause.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Desync {
    pub tick: u32,
    pub local_hash: u64,
    pub remote_hash: u64,
}

impl std::fmt::Display for Desync {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "desync at tick {}: local {:#018x} != remote {:#018x}",
            self.tick, self.local_hash, self.remote_hash,
        )
    }
}

/// Matches up local and remote world hashes by tick and reports the first
/// disagreement.
///
/// The engine owns the protocol and the comparison but deliberately not the
/// hash itself: what constitutes "the world state" is game knowledge, and any
/// hash the engine could compute would either miss game state or include
/// engine state the game does not care about. The game computes a `u64` and
/// hands it over; this decides whether the peers agree.
///
/// Hashes arrive out of step — the peer may be a few ticks ahead or behind —
/// so both sides are held until their counterpart shows up. Unmatched entries
/// are pruned once they fall outside the window, since a hash whose partner
/// never arrived is one the peer never sent, not a divergence.
#[derive(Clone, Debug, Default)]
pub struct DesyncDetector {
    local:  Vec<(u32, u64)>,
    remote: Vec<(u32, u64)>,
    /// First divergence seen. Latched: after peers diverge every later tick
    /// also mismatches, and reporting a flood of them buries the one that
    /// matters — the first.
    reported: Option<Desync>,
}

impl DesyncDetector {
    /// How many unmatched hashes to hold per side before dropping the oldest.
    /// Generous relative to any sane checksum interval, and bounded so a peer
    /// that sends checksums but never matches cannot grow this forever.
    pub const WINDOW: usize = 64;

    pub fn new() -> Self { Self::default() }

    /// Record the local world hash for `tick`. Returns a [`Desync`] if the
    /// peer's hash for that tick is already in hand and differs.
    pub fn record_local(&mut self, tick: u32, hash: u64) -> Option<Desync> {
        Self::push(&mut self.local, tick, hash);
        self.check(tick)
    }

    /// Record a hash received from the peer. Returns a [`Desync`] if our own
    /// hash for that tick is already in hand and differs.
    pub fn record_remote(&mut self, tick: u32, hash: u64) -> Option<Desync> {
        Self::push(&mut self.remote, tick, hash);
        self.check(tick)
    }

    fn push(v: &mut Vec<(u32, u64)>, tick: u32, hash: u64) {
        if let Some(e) = v.iter_mut().find(|(t, _)| *t == tick) {
            e.1 = hash;
            return;
        }
        if v.len() >= Self::WINDOW { v.remove(0); }
        v.push((tick, hash));
    }

    fn check(&mut self, tick: u32) -> Option<Desync> {
        if self.reported.is_some() { return None; }
        let l = self.local.iter().find(|(t, _)| *t == tick)?.1;
        let r = self.remote.iter().find(|(t, _)| *t == tick)?.1;
        // Matched pair either way — drop both so the window stays small.
        self.local.retain(|(t, _)| *t != tick);
        self.remote.retain(|(t, _)| *t != tick);
        if l == r { return None; }
        let d = Desync { tick, local_hash: l, remote_hash: r };
        self.reported = Some(d);
        Some(d)
    }

    /// The first divergence, if the peers have diverged at all.
    pub fn desync(&self) -> Option<Desync> { self.reported }

    /// Have the peers diverged?
    pub fn is_desynced(&self) -> bool { self.reported.is_some() }
}

// ── socket thread ───────────────────────────────────────────────────────────

/// How this peer establishes the connection. After the TCP handshake the two
/// sides are symmetric — there is no server, and neither peer is
/// authoritative — so this only decides who binds and who dials.
#[derive(Debug)]
pub enum Endpoint {
    /// Bind and wait for the other peer. Port 0 asks the OS for a free port,
    /// but then only the socket thread learns which port that was — use
    /// [`Endpoint::Accept`] if the caller needs the address.
    Listen(SocketAddr),
    /// Wait for the peer on a listener the caller has already bound.
    ///
    /// Binding is what publishes the port, so a caller that must know the
    /// address before the peer dials (port 0, or handing the address to
    /// another process) has to bind first and pass the listener in. With
    /// [`Listen`](Endpoint::Listen) the bind happens on the socket thread,
    /// so a peer that dials immediately can arrive before the listener
    /// exists and get connection-refused — a race with no reliable sleep-free
    /// fix on the caller's side.
    Accept(TcpListener),
    /// Dial the other peer.
    Connect(String),
}

/// Connection lifecycle, as observed by the main thread.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NetEvent {
    /// TCP is up and both `Hello`s matched. No tick may run before this.
    Connected,
    /// A tick's worth of input arrived from the peer.
    Input { tick: u32, bits: u32 },
    /// Peer's world hash at a tick, for the game to compare against its own.
    Checksum { tick: u32, hash: u64 },
    /// Out-of-band text from the peer.
    Chat { text: String },
    /// Peer said `Bye` — a clean quit, not a fault.
    PeerLeft,
    /// The connection is over and will not recover. Carries a human-readable
    /// reason; every path into this is unrecoverable by design (see
    /// `DecodeError`), so there is no reconnect story here.
    Disconnected { reason: String },
}

/// Handle to the socket thread. Owns the send channel; the thread owns the
/// socket. The main thread never touches the socket and never blocks on it.
///
/// Follows the crate's established pattern — a named `std::thread::Builder`
/// plus mpsc channels (as in `app.rs`'s `PerfLogger` and `log.rs`'s Loki
/// pusher) — with one deliberate deviation: the outbound channel is
/// **unbounded**, and the thread blocks in `write_all`. `PerfLogger` uses a
/// bounded channel and `try_send` so a stalled consumer cannot stall the
/// game; dropping a log line costs a log line. Dropping an input frame
/// costs the session, permanently and silently, so here backpressure must
/// surface as a stall rather than a loss.
pub struct LockstepPeer {
    tx: Sender<Message>,
    rx: Receiver<NetEvent>,
    connected: bool,
    closed: bool,
    _thread: std::thread::JoinHandle<()>,
}

impl LockstepPeer {
    /// Spawn the socket thread. Returns immediately — the TCP connect and
    /// handshake happen on the thread, and completion arrives as
    /// [`NetEvent::Connected`] from [`poll`](Self::poll). The game must not
    /// run tick 0 until it has seen that event.
    pub fn spawn(endpoint: Endpoint, handshake: Handshake) -> std::io::Result<Self> {
        let (out_tx, out_rx) = mpsc::channel::<Message>();
        let (evt_tx, evt_rx) = mpsc::channel::<NetEvent>();

        let thread = std::thread::Builder::new()
            .name("lockstep-net".into())
            .spawn(move || run_socket(endpoint, handshake, out_rx, evt_tx))?;

        Ok(Self {
            tx: out_tx,
            rx: evt_rx,
            connected: false,
            closed: false,
            _thread: thread,
        })
    }

    /// Drain everything the socket thread has produced since the last call.
    /// Non-blocking; call once per frame. The returned events are in the
    /// order they arrived.
    pub fn poll(&mut self) -> Vec<NetEvent> {
        let mut out = Vec::new();
        loop {
            match self.rx.try_recv() {
                Ok(evt) => {
                    match &evt {
                        NetEvent::Connected => self.connected = true,
                        NetEvent::Disconnected { .. } => {
                            self.connected = false;
                            self.closed = true;
                        }
                        _ => {}
                    }
                    out.push(evt);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // Thread exited without a final event (panicked, or we
                    // already saw its Disconnected). Latch closed either way.
                    self.closed = true;
                    self.connected = false;
                    break;
                }
            }
        }
        out
    }

    /// Handshake completed and the link is live.
    pub fn is_connected(&self) -> bool { self.connected }

    /// The connection has ended and will not come back.
    pub fn is_closed(&self) -> bool { self.closed }

    /// Queue a message. Never blocks and never drops: the channel is
    /// unbounded, so a slow link grows this queue and shows up as a stall on
    /// the *peer*, which is recoverable, rather than a lost input frame,
    /// which is not.
    pub fn send(&self, msg: Message) {
        // Only fails if the socket thread is gone, in which case `poll` has
        // already reported (or is about to report) the disconnect.
        let _ = self.tx.send(msg);
    }

    /// Send this peer's input for an application tick. The hot path.
    pub fn send_input(&self, tick: u32, bits: u32) {
        self.send(Message::Input { tick, bits });
    }

    /// Send a world hash for `tick`. Games typically do this every 60 ticks
    /// or so — often enough to catch a desync within a second, rarely
    /// enough that hashing the world is not itself the frame cost.
    pub fn send_checksum(&self, tick: u32, hash: u64) {
        self.send(Message::Checksum { tick, hash });
    }

    pub fn send_chat(&self, text: impl Into<String>) {
        self.send(Message::Chat { text: text.into() });
    }

    /// Tell the peer we are leaving, so it can report a quit rather than a
    /// timeout.
    pub fn send_bye(&self) {
        self.send(Message::Bye);
    }
}

/// Socket thread body: connect, handshake, then pump both directions until
/// something ends it.
fn run_socket(
    endpoint: Endpoint,
    handshake: Handshake,
    out_rx: Receiver<Message>,
    evt_tx: Sender<NetEvent>,
) {
    let stream = match establish(&endpoint) {
        Ok(s) => s,
        Err(e) => {
            let _ = evt_tx.send(NetEvent::Disconnected { reason: format!("connect failed: {e}") });
            return;
        }
    };

    // The single most important line in this module. Nagle's algorithm holds
    // a small write back until the previous one is acked, hoping to coalesce
    // — which is precisely wrong for lockstep, where every frame is a 13-byte
    // input the peer is *blocked* waiting for. It adds up to ~40 ms per tick
    // and presents as "the netcode is slow" rather than as a socket option.
    if let Err(e) = stream.set_nodelay(true) {
        let _ = evt_tx.send(NetEvent::Disconnected { reason: format!("set_nodelay failed: {e}") });
        return;
    }

    let mut reader = match stream.try_clone() {
        Ok(s) => s,
        Err(e) => {
            let _ = evt_tx.send(NetEvent::Disconnected { reason: format!("clone failed: {e}") });
            return;
        }
    };
    let mut writer = stream;

    // Handshake. Send ours first, then block for theirs: both peers do the
    // same, and because each has already written before it reads, neither
    // can deadlock waiting for a message the other has not sent.
    match handshake.to_message().encode_frame() {
        Ok(bytes) => {
            if let Err(e) = writer.write_all(&bytes) {
                let _ = evt_tx.send(NetEvent::Disconnected { reason: format!("hello write failed: {e}") });
                return;
            }
        }
        Err(e) => {
            let _ = evt_tx.send(NetEvent::Disconnected { reason: format!("hello encode failed: {e}") });
            return;
        }
    }

    match read_frame(&mut reader) {
        Ok(Some(msg)) => {
            if let Err(e) = handshake.check(&msg) {
                let _ = evt_tx.send(NetEvent::Disconnected { reason: e.to_string() });
                return;
            }
        }
        Ok(None) => {
            let _ = evt_tx.send(NetEvent::Disconnected { reason: "peer closed during handshake".into() });
            return;
        }
        Err(e) => {
            let _ = evt_tx.send(NetEvent::Disconnected { reason: format!("handshake read failed: {e}") });
            return;
        }
    }

    if evt_tx.send(NetEvent::Connected).is_err() { return; }

    // Reader thread. Splitting the directions is what lets each side block
    // on its own natural operation — the reader in `read_exact`, the writer
    // in `recv_timeout` — without either starving the other. A single thread
    // would have to poll, and polling a lockstep read adds latency to the
    // one path that must not have any.
    let evt_reader = evt_tx.clone();
    let reader_thread = std::thread::Builder::new()
        .name("lockstep-rx".into())
        .spawn(move || {
            loop {
                match read_frame(&mut reader) {
                    Ok(Some(Message::Input { tick, bits })) => {
                        if evt_reader.send(NetEvent::Input { tick, bits }).is_err() { break; }
                    }
                    Ok(Some(Message::Checksum { tick, hash })) => {
                        if evt_reader.send(NetEvent::Checksum { tick, hash }).is_err() { break; }
                    }
                    Ok(Some(Message::Chat { text })) => {
                        if evt_reader.send(NetEvent::Chat { text }).is_err() { break; }
                    }
                    Ok(Some(Message::Bye)) => {
                        let _ = evt_reader.send(NetEvent::PeerLeft);
                        break;
                    }
                    // A second Hello mid-stream is a protocol break, not a
                    // re-handshake: we are already ticking against the first.
                    Ok(Some(Message::Hello { .. })) => {
                        let _ = evt_reader.send(NetEvent::Disconnected {
                            reason: "unexpected Hello mid-stream".into(),
                        });
                        break;
                    }
                    Ok(None) => {
                        let _ = evt_reader.send(NetEvent::Disconnected {
                            reason: "peer closed the connection".into(),
                        });
                        break;
                    }
                    Err(e) => {
                        let _ = evt_reader.send(NetEvent::Disconnected {
                            reason: format!("read failed: {e}"),
                        });
                        break;
                    }
                }
            }
        });

    if let Err(e) = reader_thread {
        let _ = evt_tx.send(NetEvent::Disconnected { reason: format!("rx thread spawn failed: {e}") });
        return;
    }

    // Writer loop. `recv_timeout` rather than a plain `for msg in out_rx` so
    // the thread wakes periodically and notices the reader having shut down,
    // instead of parking forever on a channel whose sender still exists
    // because the game object is still alive. Same shape as log.rs's pusher.
    loop {
        match out_rx.recv_timeout(Duration::from_millis(250)) {
            Ok(msg) => {
                let is_bye = matches!(msg, Message::Bye);
                let bytes = match msg.encode_frame() {
                    Ok(b) => b,
                    Err(e) => {
                        // Encoding only fails on oversize chat. Drop that one
                        // message and keep the session: unlike an input, a
                        // lost chat line does not desync anything.
                        let _ = evt_tx.send(NetEvent::Chat {
                            text: format!("[not sent: {e}]"),
                        });
                        continue;
                    }
                };
                // Blocking write, by design. See the type docs: backpressure
                // must stall, never drop.
                if let Err(e) = writer.write_all(&bytes) {
                    let _ = evt_tx.send(NetEvent::Disconnected {
                        reason: format!("write failed: {e}"),
                    });
                    break;
                }
                if is_bye {
                    // Half-close so the peer's reader sees EOF promptly
                    // rather than waiting out a timeout.
                    let _ = writer.shutdown(std::net::Shutdown::Write);
                    break;
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                // Nothing to send. Loop so we re-check for shutdown.
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // Game dropped the handle; leave cleanly.
                let _ = writer.shutdown(std::net::Shutdown::Both);
                break;
            }
        }
    }
}

fn establish(endpoint: &Endpoint) -> std::io::Result<TcpStream> {
    match endpoint {
        Endpoint::Listen(addr) => {
            let listener = TcpListener::bind(addr)?;
            let (stream, _peer) = listener.accept()?;
            Ok(stream)
        }
        Endpoint::Accept(listener) => {
            let (stream, _peer) = listener.accept()?;
            Ok(stream)
        }
        Endpoint::Connect(addr) => {
            // `to_socket_addrs` so a hostname works, not just a literal IP.
            let mut last_err = std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "no addresses resolved",
            );
            for sa in addr.to_socket_addrs()? {
                match TcpStream::connect(sa) {
                    Ok(s) => return Ok(s),
                    Err(e) => last_err = e,
                }
            }
            Err(last_err)
        }
    }
}

/// Read one length-prefixed frame. `Ok(None)` means a clean EOF at a frame
/// boundary — the peer closed between messages, which is a normal end.
fn read_frame(r: &mut impl Read) -> std::io::Result<Option<Message>> {
    let mut len_buf = [0u8; 2];
    if !read_exact_or_eof(r, &mut len_buf)? {
        // Clean EOF at a frame boundary: the peer closed between messages.
        return Ok(None);
    }
    let len = u16::from_be_bytes(len_buf) as usize;
    if len == 0 || len > MAX_FRAME_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame length {len} out of range"),
        ));
    }
    let mut body = vec![0u8; len];
    if !read_exact_or_eof(r, &mut body)? {
        // EOF *inside* a frame, unlike the boundary case above: the peer
        // died mid-message, so the stream is genuinely broken.
        return Err(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "peer closed mid-frame",
        ));
    }
    Message::decode_body(&body)
        .map(Some)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
}

/// `read_exact`, but distinguishing "clean EOF before any byte" (`Ok(false)`)
/// from "EOF partway through" (`Err`). `Read::read_exact` collapses both into
/// `UnexpectedEof`, and we need to tell a normal disconnect from a truncated
/// frame.
fn read_exact_or_eof(r: &mut impl Read, buf: &mut [u8]) -> std::io::Result<bool> {
    let mut filled = 0;
    while filled < buf.len() {
        match r.read(&mut buf[filled..]) {
            // EOF. At a frame boundary (nothing read yet) that is a clean
            // disconnect; partway through it is a truncated frame, and the
            // caller must tell those apart.
            Ok(0) => return Ok(false),
            Ok(n) => filled += n,
            Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── framing ────────────────────────────────────────────────────────────

    fn round_trip(msg: Message) {
        let body = msg.encode_body().expect("encode");
        let back = Message::decode_body(&body).expect("decode");
        assert_eq!(msg, back, "round trip changed the message");
    }

    #[test]
    fn round_trips_hello() {
        round_trip(Message::Hello {
            protocol_version: PROTOCOL_VERSION,
            world_seed: 0xDEAD_BEEF_CAFE_F00D,
            input_delay: 3,
        });
    }

    #[test]
    fn round_trips_input() {
        round_trip(Message::Input { tick: 0, bits: 0 });
        round_trip(Message::Input { tick: u32::MAX, bits: u32::MAX });
        round_trip(Message::Input { tick: 12_345, bits: 0b1010_1010 });
    }

    #[test]
    fn round_trips_checksum() {
        round_trip(Message::Checksum { tick: 600, hash: u64::MAX });
        round_trip(Message::Checksum { tick: 0, hash: 0 });
    }

    #[test]
    fn round_trips_chat() {
        round_trip(Message::Chat { text: String::new() });
        round_trip(Message::Chat { text: "hello peer".into() });
        // Multi-byte UTF-8: the length prefix counts bytes, not chars, and
        // getting that backwards truncates mid-codepoint.
        round_trip(Message::Chat { text: "héllo — 世界 🎮".into() });
    }

    #[test]
    fn round_trips_bye() {
        round_trip(Message::Bye);
    }

    #[test]
    fn frame_carries_a_length_prefix() {
        let msg = Message::Input { tick: 7, bits: 3 };
        let frame = msg.encode_frame().unwrap();
        let body = msg.encode_body().unwrap();
        assert_eq!(&frame[..2], &(body.len() as u16).to_be_bytes());
        assert_eq!(&frame[2..], &body[..]);
        // The hot path must stay small: tag + tick + bits + prefix.
        assert_eq!(frame.len(), 11, "input frame grew");
    }

    #[test]
    fn input_frames_are_fixed_size() {
        // Every input frame must be the same length regardless of content,
        // or a length prefix would leak information and, worse, suggest the
        // hot path allocates variably.
        let a = Message::Input { tick: 0, bits: 0 }.encode_frame().unwrap();
        let b = Message::Input { tick: u32::MAX, bits: u32::MAX }.encode_frame().unwrap();
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn rejects_unknown_tag() {
        assert_eq!(Message::decode_body(&[99]), Err(DecodeError::UnknownTag(99)));
    }

    #[test]
    fn rejects_truncated_body() {
        // Input tag with only two of the eight payload bytes.
        let err = Message::decode_body(&[TAG_INPUT, 0, 0]).unwrap_err();
        assert!(matches!(err, DecodeError::Truncated { .. }), "got {err:?}");
    }

    #[test]
    fn rejects_empty_body() {
        let err = Message::decode_body(&[]).unwrap_err();
        assert!(matches!(err, DecodeError::Truncated { .. }), "got {err:?}");
    }

    #[test]
    fn rejects_trailing_bytes() {
        let mut body = Message::Bye.encode_body().unwrap();
        body.push(0xAA);
        assert_eq!(Message::decode_body(&body), Err(DecodeError::TrailingBytes(1)));
    }

    #[test]
    fn rejects_bad_utf8_chat() {
        let mut body = vec![TAG_CHAT];
        body.extend_from_slice(&2u16.to_be_bytes());
        body.extend_from_slice(&[0xFF, 0xFE]); // not valid UTF-8
        assert_eq!(Message::decode_body(&body), Err(DecodeError::BadUtf8));
    }

    #[test]
    fn rejects_oversize_chat_on_encode_and_decode() {
        let big = "x".repeat(MAX_CHAT_BYTES + 1);
        assert_eq!(
            Message::Chat { text: big }.encode_frame(),
            Err(EncodeError::ChatTooLong(MAX_CHAT_BYTES + 1)),
        );
        // And a hostile peer claiming a huge length must be refused before
        // we allocate for it.
        let mut body = vec![TAG_CHAT];
        body.extend_from_slice(&((MAX_CHAT_BYTES + 1) as u16).to_be_bytes());
        assert_eq!(
            Message::decode_body(&body),
            Err(DecodeError::ChatTooLong(MAX_CHAT_BYTES + 1)),
        );
    }

    #[test]
    fn chat_at_the_limit_is_accepted() {
        round_trip(Message::Chat { text: "x".repeat(MAX_CHAT_BYTES) });
    }

    #[test]
    fn wire_layout_is_pinned() {
        // Guards the byte layout itself, so a refactor that silently changes
        // field order or endianness fails here rather than in the field
        // against an old build.
        assert_eq!(
            Message::Input { tick: 1, bits: 2 }.encode_frame().unwrap(),
            vec![0, 9, TAG_INPUT, 0, 0, 0, 1, 0, 0, 0, 2],
        );
        assert_eq!(Message::Bye.encode_frame().unwrap(), vec![0, 1, TAG_BYE]);
    }

    // ── handshake ──────────────────────────────────────────────────────────

    #[test]
    fn handshake_accepts_a_matching_peer() {
        let h = Handshake::new(42);
        assert_eq!(h.check(&h.to_message()), Ok(()));
    }

    #[test]
    fn handshake_rejects_version_mismatch() {
        let h = Handshake::new(42);
        let peer = Message::Hello {
            protocol_version: PROTOCOL_VERSION + 1,
            world_seed: 42,
            input_delay: DEFAULT_INPUT_DELAY,
        };
        assert_eq!(
            h.check(&peer),
            Err(HandshakeError::VersionMismatch {
                ours: PROTOCOL_VERSION,
                theirs: PROTOCOL_VERSION + 1,
            }),
        );
    }

    #[test]
    fn handshake_rejects_seed_mismatch() {
        let h = Handshake::new(42);
        let peer = Message::Hello {
            protocol_version: PROTOCOL_VERSION,
            world_seed: 43,
            input_delay: DEFAULT_INPUT_DELAY,
        };
        assert_eq!(h.check(&peer), Err(HandshakeError::SeedMismatch { ours: 42, theirs: 43 }));
    }

    #[test]
    fn handshake_rejects_delay_mismatch() {
        // Two peers with different delays apply the same input at different
        // ticks — a desync on the very first keypress.
        let h = Handshake::new(42).with_input_delay(3);
        let peer = Message::Hello {
            protocol_version: PROTOCOL_VERSION,
            world_seed: 42,
            input_delay: 5,
        };
        assert_eq!(h.check(&peer), Err(HandshakeError::DelayMismatch { ours: 3, theirs: 5 }));
    }

    #[test]
    fn handshake_rejects_a_non_hello_first_message() {
        let h = Handshake::new(42);
        assert_eq!(h.check(&Message::Input { tick: 0, bits: 0 }), Err(HandshakeError::NotHello));
    }

    #[test]
    fn version_is_checked_before_seed() {
        // If the format differs, the bytes we parsed as a seed may not be a
        // seed at all — reporting "seed mismatch" would send the user
        // chasing the wrong bug.
        let h = Handshake::new(42);
        let peer = Message::Hello {
            protocol_version: PROTOCOL_VERSION + 1,
            world_seed: 999,
            input_delay: DEFAULT_INPUT_DELAY,
        };
        assert!(matches!(h.check(&peer), Err(HandshakeError::VersionMismatch { .. })));
    }

    // ── tick buffer ────────────────────────────────────────────────────────

    #[test]
    fn a_fresh_buffer_is_stalled() {
        let b = TickBuffer::new(DEFAULT_INPUT_DELAY);
        assert!(b.stalled(), "no inputs yet, so tick 0 cannot run");
        assert_eq!(b.next_tick(), 0);
    }

    #[test]
    fn one_sided_input_does_not_make_a_tick_ready() {
        let mut b = TickBuffer::new(0);
        b.submit_local(0, 0b1);
        assert!(!b.ready(0), "local alone must not run a tick");
        assert!(b.stalled());
        assert_eq!(b.inputs_for(0), None);
        assert_eq!(b.consume(), None, "consume must not advance while stalled");
        assert_eq!(b.next_tick(), 0);

        b.submit_remote(0, 0b10);
        assert!(b.ready(0));
        assert!(!b.stalled());
    }

    #[test]
    fn consume_returns_both_inputs_and_advances_once() {
        let mut b = TickBuffer::new(0);
        b.submit_local(0, 7);
        b.submit_remote(0, 9);
        assert_eq!(b.consume(), Some(TickInputs { local: 7, remote: 9 }));
        assert_eq!(b.next_tick(), 1);
        // Tick 0's slots are spent; tick 1 has nothing.
        assert_eq!(b.consume(), None);
        assert_eq!(b.next_tick(), 1);
    }

    #[test]
    fn input_delay_schedules_ahead_of_the_current_tick() {
        let b = TickBuffer::new(3);
        assert_eq!(b.next_tick(), 0);
        assert_eq!(b.scheduled_tick(), 3, "input sampled now applies at 0 + delay");

        let mut b = TickBuffer::new(3);
        for t in 0..5u32 {
            b.submit_local(t, t);
            b.submit_remote(t, t);
        }
        b.consume().unwrap();
        b.consume().unwrap();
        assert_eq!(b.next_tick(), 2);
        assert_eq!(b.scheduled_tick(), 5, "the window slides with the sim");
    }

    #[test]
    fn zero_delay_still_works() {
        // Degenerate but legal: apply input on the tick it was sampled.
        // Stalls on every tick for a full RTT, which is the point of delay,
        // but it must not misbehave arithmetically.
        let b = TickBuffer::new(0);
        assert_eq!(b.scheduled_tick(), b.next_tick());
    }

    #[test]
    fn a_stall_defers_ticks_rather_than_losing_them() {
        // The scenario the engine hook exists for: remote input for tick 1
        // is late, so tick 1 stalls; when it lands, both buffered ticks are
        // still there and drain in order.
        let mut b = TickBuffer::new(0);
        for t in 0..3u32 { b.submit_local(t, 100 + t); }
        b.submit_remote(0, 200);
        b.submit_remote(2, 202); // tick 1 missing — arrives out of order

        assert_eq!(b.consume(), Some(TickInputs { local: 100, remote: 200 }));
        assert!(b.stalled(), "tick 1 has no remote input");
        assert_eq!(b.consume(), None);
        assert_eq!(b.next_tick(), 1, "a stalled tick is deferred, not skipped");

        // The late frame lands.
        b.submit_remote(1, 201);
        assert!(!b.stalled());
        // Both buffered ticks now drain in one frame — the catch-up burst
        // that makes the once-per-frame input rule mandatory.
        assert_eq!(b.consume(), Some(TickInputs { local: 101, remote: 201 }));
        assert_eq!(b.consume(), Some(TickInputs { local: 102, remote: 202 }));
        assert_eq!(b.next_tick(), 3);
    }

    #[test]
    fn out_of_order_arrival_is_buffered_not_dropped() {
        let mut b = TickBuffer::new(0);
        // Remote runs ahead: ticks 0..5 all arrive before we consume any.
        for t in 0..5u32 {
            b.submit_remote(t, 500 + t);
            b.submit_local(t, 100 + t);
        }
        for t in 0..5u32 {
            assert_eq!(
                b.consume(),
                Some(TickInputs { local: 100 + t, remote: 500 + t }),
                "tick {t}",
            );
        }
    }

    #[test]
    fn submitting_for_an_already_run_tick_is_ignored() {
        let mut b = TickBuffer::new(0);
        b.submit_local(0, 1);
        b.submit_remote(0, 2);
        b.consume().unwrap();
        // A late duplicate for tick 0 must not resurrect a consumed tick —
        // the world has already moved past it on both peers.
        b.submit_local(0, 99);
        b.submit_remote(0, 99);
        assert_eq!(b.inputs_for(0), None);
        assert_eq!(b.next_tick(), 1);
    }

    #[test]
    fn a_stale_ring_slot_is_not_mistaken_for_a_live_one() {
        // Index collision: tick 0 and tick CAPACITY share a slot. The tick
        // stamp is what stops the sim eating a 256-tick-old input.
        let mut b = TickBuffer::new(0);
        let far = TickBuffer::CAPACITY as u32;
        b.submit_local(0, 11);
        b.submit_remote(0, 22);
        assert_eq!(TickBuffer::idx(0), TickBuffer::idx(far), "test premise");
        assert!(b.ready(0));
        assert!(!b.ready(far), "same slot, different tick — must not be ready");
        assert_eq!(b.inputs_for(far), None);
    }

    #[test]
    fn remote_lead_reports_buffered_depth() {
        let mut b = TickBuffer::new(3);
        assert_eq!(b.remote_lead(), 0, "nothing buffered yet");
        b.submit_remote(0, 0);
        b.submit_remote(1, 0);
        b.submit_remote(2, 0);
        assert_eq!(b.remote_lead(), 3, "a healthy link sits near input_delay");
        // A gap truncates the run: we are only safe up to the first hole.
        let mut b = TickBuffer::new(3);
        b.submit_remote(0, 0);
        b.submit_remote(2, 0); // hole at 1
        assert_eq!(b.remote_lead(), 1);
    }

    // ── desync detection ───────────────────────────────────────────────────

    #[test]
    fn matching_hashes_are_not_a_desync() {
        let mut d = DesyncDetector::new();
        assert_eq!(d.record_local(60, 0xABC), None, "no peer hash yet");
        assert_eq!(d.record_remote(60, 0xABC), None);
        assert!(!d.is_desynced());
    }

    #[test]
    fn mismatched_hashes_report_the_tick() {
        let mut d = DesyncDetector::new();
        d.record_local(120, 1);
        let hit = d.record_remote(120, 2).expect("must report");
        assert_eq!(hit, Desync { tick: 120, local_hash: 1, remote_hash: 2 });
        assert!(d.is_desynced());
        assert_eq!(d.desync(), Some(hit));
    }

    #[test]
    fn detects_regardless_of_arrival_order() {
        // Remote hash can land before we have computed ours.
        let mut d = DesyncDetector::new();
        assert_eq!(d.record_remote(60, 7), None);
        let hit = d.record_local(60, 8).expect("must report once ours lands");
        assert_eq!(hit.tick, 60);
        assert_eq!(hit.local_hash, 8);
        assert_eq!(hit.remote_hash, 7);
    }

    #[test]
    fn only_the_first_desync_is_reported() {
        // After divergence every later tick mismatches too; reporting them
        // all buries the only one that identifies the cause.
        let mut d = DesyncDetector::new();
        d.record_local(60, 1);
        assert!(d.record_remote(60, 2).is_some());
        d.record_local(120, 3);
        assert_eq!(d.record_remote(120, 4), None, "later mismatches stay quiet");
        assert_eq!(d.desync().unwrap().tick, 60);
    }

    #[test]
    fn unmatched_hashes_do_not_grow_without_bound() {
        // A peer that sends checksums for ticks we never hash must not make
        // this leak.
        let mut d = DesyncDetector::new();
        for t in 0..(DesyncDetector::WINDOW as u32 * 3) {
            assert_eq!(d.record_remote(t, t as u64), None);
        }
        assert!(d.remote.len() <= DesyncDetector::WINDOW);
        assert!(!d.is_desynced(), "unmatched hashes are not divergence");
    }

    #[test]
    fn ticks_are_matched_pairwise_not_positionally() {
        // Peers checksum at different ticks while one runs ahead; only the
        // ticks both sides hashed may be compared.
        let mut d = DesyncDetector::new();
        d.record_local(60, 0xAA);
        d.record_local(120, 0xBB);
        assert_eq!(d.record_remote(120, 0xBB), None, "tick 120 agrees");
        assert!(!d.is_desynced());
    }

    #[test]
    fn full_delayed_handshake_to_tick_flow() {
        // End-to-end of the buffer half: both peers sample once per frame,
        // submit at scheduled_tick, and the sim runs `delay` ticks behind.
        let delay = DEFAULT_INPUT_DELAY;
        let mut b = TickBuffer::new(delay);

        // Frames 0..delay: input is submitted for ticks 3,4,5 — nothing is
        // runnable yet, because ticks 0..2 were never given inputs. A real
        // game primes those with a neutral input at startup.
        for _ in 0..delay {
            let t = b.scheduled_tick();
            b.submit_local(t, 0);
            b.submit_remote(t, 0);
            assert!(b.stalled(), "tick {} not primed", b.next_tick());
        }

        // Prime the opening ticks, as a game does before tick 0.
        for t in 0..delay {
            b.submit_local(t, 0);
            b.submit_remote(t, 0);
        }
        for expected in 0..delay {
            assert_eq!(b.next_tick(), expected);
            assert!(b.consume().is_some(), "primed tick {expected} must run");
        }
        // Now the delayed inputs from the first loop become runnable.
        assert!(!b.stalled());
        assert_eq!(b.next_tick(), delay);
    }

    // ── loopback ───────────────────────────────────────────────────────────

    /// Real sockets over 127.0.0.1: handshake, a few ticks of input in both
    /// directions, chat, and a clean Bye. Kept in the normal test run rather
    /// than `#[ignore]`d because it binds port 0 (never a fixed port, so no
    /// collision with a parallel run or a developer's own server) and every
    /// wait is a bounded `recv_timeout` — it cannot hang the suite, it can
    /// only fail it.
    #[test]
    fn loopback_two_peers_exchange_ticks() {
        const SEED: u64 = 0x5EED;
        // Generous: this is a liveness bound, not a latency assertion. A
        // loaded CI box must not fail here, but a genuinely dead peer must
        // not hang the suite either.
        const WAIT: Duration = Duration::from_secs(5);

        // Bind here, not on the socket thread, and hand the live listener
        // over. Binding first is what makes the test deterministic: the
        // port is accepting connections before the joiner is even spawned,
        // so there is no window in which it can be refused and no sleep
        // pretending to close one.
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local_addr");

        let host = LockstepPeer::spawn(Endpoint::Accept(listener), Handshake::new(SEED))
            .expect("spawn host");
        let join = LockstepPeer::spawn(
            Endpoint::Connect(addr.to_string()),
            Handshake::new(SEED),
        ).expect("spawn joiner");

        let mut host = host;
        let mut join = join;
        // One backlog per peer, carried across every wait_for below.
        let mut host_pending: Vec<NetEvent> = Vec::new();
        let mut join_pending: Vec<NetEvent> = Vec::new();

        // Both peers must report Connected before any tick runs.
        wait_for(&mut host, &mut host_pending, WAIT, |e| matches!(e, NetEvent::Connected))
            .expect("host connected");
        wait_for(&mut join, &mut join_pending, WAIT, |e| matches!(e, NetEvent::Connected))
            .expect("joiner connected");

        // Exchange a few ticks of input in both directions.
        for tick in 0..5u32 {
            host.send_input(tick, 0xA000 + tick);
            join.send_input(tick, 0xB000 + tick);
        }
        for tick in 0..5u32 {
            let got = wait_for(&mut join, &mut join_pending, WAIT, |e| matches!(e, NetEvent::Input { .. }))
                .expect("joiner receives host input");
            assert_eq!(got, NetEvent::Input { tick, bits: 0xA000 + tick });

            let got = wait_for(&mut host, &mut host_pending, WAIT, |e| matches!(e, NetEvent::Input { .. }))
                .expect("host receives joiner input");
            assert_eq!(got, NetEvent::Input { tick, bits: 0xB000 + tick });
        }

        // Checksums and chat share the stream without disturbing tick order.
        host.send_checksum(4, 0xDEAD_BEEF);
        let got = wait_for(&mut join, &mut join_pending, WAIT, |e| matches!(e, NetEvent::Checksum { .. }))
            .expect("checksum arrives");
        assert_eq!(got, NetEvent::Checksum { tick: 4, hash: 0xDEAD_BEEF });

        join.send_chat("gg");
        let got = wait_for(&mut host, &mut host_pending, WAIT, |e| matches!(e, NetEvent::Chat { .. }))
            .expect("chat arrives");
        assert_eq!(got, NetEvent::Chat { text: "gg".into() });

        // A clean Bye must read as a quit, not as a fault.
        join.send_bye();
        let got = wait_for(&mut host, &mut host_pending, WAIT, |e| {
            matches!(e, NetEvent::PeerLeft | NetEvent::Disconnected { .. })
        }).expect("host sees the peer leave");
        assert_eq!(got, NetEvent::PeerLeft, "clean quit must not look like an error");
    }

    /// A peer whose seed differs must be refused at the handshake rather
    /// than allowed to start ticking against a different world.
    #[test]
    fn loopback_rejects_a_seed_mismatch() {
        const WAIT: Duration = Duration::from_secs(5);

        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
        let addr = listener.local_addr().expect("local_addr");

        let mut host = LockstepPeer::spawn(Endpoint::Accept(listener), Handshake::new(1))
            .expect("spawn host");
        let mut join = LockstepPeer::spawn(
            Endpoint::Connect(addr.to_string()),
            Handshake::new(2), // different world
        ).expect("spawn joiner");
        let mut host_pending: Vec<NetEvent> = Vec::new();
        let mut join_pending: Vec<NetEvent> = Vec::new();

        let got = wait_for(&mut host, &mut host_pending, WAIT, |e| matches!(e, NetEvent::Disconnected { .. }))
            .expect("host rejects");
        match got {
            NetEvent::Disconnected { reason } => {
                assert!(reason.contains("seed"), "reason was {reason:?}");
            }
            other => panic!("expected Disconnected, got {other:?}"),
        }
        assert!(!host.is_connected());

        // The joiner must reject it too — neither side may proceed.
        let got = wait_for(&mut join, &mut join_pending, WAIT, |e| matches!(e, NetEvent::Disconnected { .. }))
            .expect("joiner rejects");
        assert!(matches!(got, NetEvent::Disconnected { .. }));
    }

    /// Poll until an event matching `pred` arrives, or the deadline passes.
    /// Polling rather than blocking because `LockstepPeer::poll` is the
    /// non-blocking API the game loop uses; this exercises the same path.
    /// Poll until an event matching `pred` arrives, or the deadline passes.
    ///
    /// `pending` is owned by the caller and carried across calls: `poll`
    /// drains everything available, so an event we are not waiting for yet
    /// (an Input arriving while we wait on a Checksum) must be kept rather
    /// than dropped, or the next `wait_for` would block for something it
    /// has already been handed.
    fn wait_for(
        peer: &mut LockstepPeer,
        pending: &mut Vec<NetEvent>,
        timeout: Duration,
        pred: impl Fn(&NetEvent) -> bool,
    ) -> Option<NetEvent> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            pending.extend(peer.poll());
            if let Some(i) = pending.iter().position(&pred) {
                return Some(pending.remove(i));
            }
            if std::time::Instant::now() >= deadline { return None; }
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}
