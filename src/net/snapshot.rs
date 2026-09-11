//! The snapshot packet: what actually goes on the wire, and how it
//! survives an MTU.
//!
//! This is where the other replication pieces meet. [`Relevancy`] says
//! which entities a client can see, [`ClientLink`] says what it already
//! holds, [`bitpack`] turns a field into bits, and [`send_chunked`] gets
//! the result across a datagram path that may be smaller than the packet.
//!
//! [`Relevancy`]: super::replication::Relevancy
//! [`ClientLink`]: super::replication::ClientLink
//! [`bitpack`]: super::bitpack
//! [`send_chunked`]: super::chunk::send_chunked
//!
//! # The shape, and why it is this shape
//!
//! [`Chunkable`] assumes a packet is "scalar core + bulk header + a
//! splittable item list", because that is the shape a snapshot actually
//! has. This packet is the concrete instance:
//!
//! * **Scalar core** — tick, flags, and which entity is the recipient's
//!   own. Rides every chunk; a few bytes.
//! * **Bulk header** — the name→id table, which is why it is bulk: it is
//!   proportional to how many component types are registered, not to how
//!   many entities are in view. It rides the *first* chunk only.
//! * **Items** — one per entity that entered, left, or changed.
//!
//! Re-cloning a bulk header into every chunk is the production bug
//! `chunk` exists to prevent — 1,325 entities dropped in 44 seconds
//! because two unbounded header vectors made every chunk oversized. The
//! test `the_name_table_rides_only_the_first_chunk` is the guard against
//! reintroducing it here.
//!
//! # Why the name table is sent at all
//!
//! Components go on the wire as a [`NameId`], two bytes rather than a
//! string. But a `NameId` is assigned in registration order and is
//! explicitly *not* stable across runs — the stable thing is the name. So
//! a client cannot be assumed to agree with the server's numbering, and
//! the mapping is sent once per connection, in the header, before any
//! item refers to it.
//!
//! [`NameId`]: crate::persist::registry::NameId
//!
//! # Budget
//!
//! A delta item costs ~16 B and the scalar core ~13 B. Measured by
//! growing a packet one item at a time against the conservative 1200 B
//! floor, 93 items fit a datagram with no bulk header, fewer when the
//! name table rides along.
//!
//! A steady-state delta — a few arrivals and departures plus whatever
//! moved — is one datagram: 40 items encode to 484 B. A full keyframe
//! over a thousand-odd visible entities is not, and chunks; that is
//! precisely what `send_chunked` is for. A 1834-item keyframe lands in
//! 34 datagrams once that connection's [`ChunkHint`] has converged.
//!
//! [`ChunkHint`]: super::chunk::ChunkHint

use glam::DVec2;

use super::bitpack::{BitError, BitReader, BitWriter};
use super::chunk::Chunkable;
use crate::ecs::EntityId;
use crate::persist::registry::{NameId, Registry};

/// Bits per position axis. Over a 1 km sector this is ~1.5 cm, which is
/// finer than a player can perceive at any sane rendering scale.
pub const POS_BITS: u32 = 16;
/// Bits for rotation, over ±π. ~0.09° — well inside what interpolation
/// smooths over anyway.
pub const ROT_BITS: u32 = 12;
/// Bits per velocity axis, over [`MAX_SPEED`].
pub const VEL_BITS: u32 = 16;
/// Velocity range the quantiser covers, in world units per second.
/// Anything faster clamps, which costs a tick of interpolation accuracy
/// on something already moving too fast to see.
pub const MAX_SPEED: f64 = 4_096.0;

/// What happened to one entity since the client's baseline.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ItemKind {
    /// Newly visible, or newly existing. Carries the full [`EntityId`],
    /// because a reused index is a different entity and the client must
    /// be able to tell.
    Entered,
    /// Still visible, state changed. Carries the index only — a live
    /// entity's generation never changes.
    Updated,
    /// No longer visible, or despawned. Carries the full [`EntityId`] for
    /// the same reason [`ItemKind::Entered`] does.
    Left,
}

/// One entity's worth of snapshot.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct EntityItem {
    pub kind: ItemKind,
    pub entity: EntityId,
    /// Sector-local position. See the module docs on why this is not a
    /// world coordinate.
    pub pos: DVec2,
    pub rot: f32,
    pub vel: DVec2,
    /// Which component this item's payload belongs to, interned against
    /// the table in the packet header.
    pub component: NameId,
}

impl EntityItem {
    /// A `Left` item, which carries no state.
    pub fn left(entity: EntityId) -> Self {
        Self {
            kind: ItemKind::Left,
            entity,
            pos: DVec2::ZERO,
            rot: 0.0,
            vel: DVec2::ZERO,
            component: NameId(0),
        }
    }
}

/// One entry of the name→id table.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NameEntry {
    pub name: String,
    pub id: NameId,
}

/// A snapshot for one client at one tick.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotPacket {
    // ── scalar core: rides every chunk ──────────────────────────────
    /// Simulation tick. `u32` because that is what `InterpClock`
    /// consumes; a `u64` simulation narrows here deliberately.
    pub tick: u32,
    /// True on the first chunk of a send. Forwarded onto the wire so the
    /// receiver can tell "blanked for chunking" from "genuinely empty" —
    /// an emptiness heuristic gets a legitimately-empty table wrong.
    pub is_header: bool,
    /// True when this is a full keyframe rather than a delta.
    pub keyframe: bool,
    /// The recipient's own entity, so a client can find itself without a
    /// separate message.
    pub your_entity: EntityId,

    // ── bulk header: rides the first chunk only ─────────────────────
    /// Component name→id mapping. Proportional to registered component
    /// types, not to entities in view, which is what makes it bulk.
    pub names: Vec<NameEntry>,

    // ── splittable ─────────────────────────────────────────────────
    pub items: Vec<EntityItem>,
}

impl SnapshotPacket {
    /// Build the name table from a registry, including only components
    /// that actually reach clients.
    ///
    /// A server-only component has no business appearing in a table sent
    /// to players: it is wasted bytes, and it tells anyone reading the
    /// traffic what the server tracks internally.
    pub fn name_table(registry: &Registry) -> Vec<NameEntry> {
        registry
            .entries()
            .iter()
            .filter(|e| e.replicate.on_the_wire())
            .map(|e| NameEntry { name: e.name.to_string(), id: e.id })
            .collect()
    }

    /// Encode to bits.
    ///
    /// `half_extent` is half the sector size — the range sector-local
    /// positions are quantised over. It is a parameter because
    /// `sector.rs` deliberately has no global sector size: a game picks
    /// one, and precision follows from it.
    pub fn encode(&self, half_extent: f64) -> Vec<u8> {
        let mut w = BitWriter::new();
        self.encode_into(&mut w, half_extent);
        w.finish()
    }

    /// Encode into a reused writer. The per-client, per-tick path.
    pub fn encode_into(&self, w: &mut BitWriter, half_extent: f64) {
        w.clear();
        w.write_bits(self.tick as u64, 32);
        w.write_bit(self.is_header);
        w.write_bit(self.keyframe);
        w.write_varint(self.your_entity.index as u64);
        w.write_varint(self.your_entity.generation as u64);

        // The table is present only when this chunk is the header. A
        // continuation chunk writes a zero count, which is unambiguous
        // because `is_header` distinguishes it from a genuinely empty
        // table on a header chunk.
        w.write_varint(self.names.len() as u64);
        for entry in &self.names {
            w.write_varint(entry.id.0 as u64);
            let bytes = entry.name.as_bytes();
            w.write_varint(bytes.len() as u64);
            for &b in bytes {
                w.write_bits(b as u64, 8);
            }
        }

        w.write_varint(self.items.len() as u64);
        for item in &self.items {
            let kind = match item.kind {
                ItemKind::Entered => 0u64,
                ItemKind::Updated => 1,
                ItemKind::Left    => 2,
            };
            w.write_bits(kind, 2);
            w.write_varint(item.entity.index as u64);
            // Generation rides arrivals and departures only: a live
            // entity's generation cannot change, so an update would be
            // paying for a constant.
            if item.kind != ItemKind::Updated {
                w.write_varint(item.entity.generation as u64);
            }
            if item.kind != ItemKind::Left {
                w.write_varint(item.component.0 as u64);
                w.write_quantised(item.pos.x, half_extent, POS_BITS);
                w.write_quantised(item.pos.y, half_extent, POS_BITS);
                w.write_quantised(item.rot as f64, std::f64::consts::PI, ROT_BITS);
                w.write_quantised(item.vel.x, MAX_SPEED, VEL_BITS);
                w.write_quantised(item.vel.y, MAX_SPEED, VEL_BITS);
            }
        }
    }

    /// Decode a packet off the wire.
    ///
    /// `max_names` and `max_items` bound what a peer can make this
    /// allocate. Without them a corrupt or hostile length prefix is a
    /// one-packet out-of-memory, which is the bug `framing::read_msg`
    /// was fixed for and the same reasoning applies here.
    pub fn decode(
        bytes: &[u8],
        half_extent: f64,
        max_names: usize,
        max_items: usize,
    ) -> Result<Self, BitError> {
        let mut r = BitReader::new(bytes);
        let tick = r.read_bits(32)? as u32;
        let is_header = r.read_bit()?;
        let keyframe = r.read_bit()?;
        let your_entity = EntityId {
            index: r.read_varint()? as u32,
            generation: r.read_varint()? as u32,
        };

        let name_count = r.read_varint()? as usize;
        if name_count > max_names {
            return Err(BitError::OutOfRange);
        }
        let mut names = Vec::with_capacity(name_count);
        for _ in 0..name_count {
            let id = NameId(r.read_varint()? as u16);
            let len = r.read_varint()? as usize;
            // A name cannot be longer than the remaining buffer, so this
            // is checked before allocating rather than after failing.
            if len > r.bits_remaining() / 8 {
                return Err(BitError::Truncated);
            }
            let mut buf = Vec::with_capacity(len);
            for _ in 0..len {
                buf.push(r.read_bits(8)? as u8);
            }
            let name = String::from_utf8(buf).map_err(|_| BitError::OutOfRange)?;
            names.push(NameEntry { name, id });
        }

        let item_count = r.read_varint()? as usize;
        if item_count > max_items {
            return Err(BitError::OutOfRange);
        }
        let mut items = Vec::with_capacity(item_count);
        for _ in 0..item_count {
            let kind = match r.read_bits(2)? {
                0 => ItemKind::Entered,
                1 => ItemKind::Updated,
                2 => ItemKind::Left,
                _ => return Err(BitError::OutOfRange),
            };
            let index = r.read_varint()? as u32;
            let generation = if kind != ItemKind::Updated {
                r.read_varint()? as u32
            } else {
                0
            };
            let entity = EntityId { index, generation };
            if kind == ItemKind::Left {
                items.push(EntityItem::left(entity));
                continue;
            }
            let component = NameId(r.read_varint()? as u16);
            let pos = DVec2::new(
                r.read_quantised(half_extent, POS_BITS)?,
                r.read_quantised(half_extent, POS_BITS)?,
            );
            let rot = r.read_quantised(std::f64::consts::PI, ROT_BITS)? as f32;
            let vel = DVec2::new(
                r.read_quantised(MAX_SPEED, VEL_BITS)?,
                r.read_quantised(MAX_SPEED, VEL_BITS)?,
            );
            items.push(EntityItem { kind, entity, pos, rot, vel, component });
        }

        Ok(Self { tick, is_header, keyframe, your_entity, names, items })
    }
}

impl Chunkable for SnapshotPacket {
    type Item = EntityItem;

    fn items(&self) -> &[EntityItem] { &self.items }

    fn rebuild(&self, items: &[EntityItem], is_header: bool) -> Self {
        Self {
            tick: self.tick,
            is_header,
            keyframe: self.keyframe,
            your_entity: self.your_entity,
            // The load-bearing line: the table rides the header chunk
            // only. Cloning it into every chunk is the production bug.
            names: if is_header { self.names.clone() } else { Vec::new() },
            items: items.to_vec(),
        }
    }

    fn rebuild_shed(&self, items: &[EntityItem]) -> Self {
        Self {
            tick: self.tick,
            is_header: true,
            keyframe: self.keyframe,
            your_entity: self.your_entity,
            names: Vec::new(),
            items: items.to_vec(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::net::chunk::{
        send_chunked, ChunkHint, ChunkResult, DatagramSink, SendOutcome, MIN_DATAGRAM_BUDGET,
    };

    const HALF: f64 = 500.0;

    fn ent(index: u32) -> EntityId { EntityId { index, generation: 0 } }

    fn item(index: u32, kind: ItemKind) -> EntityItem {
        EntityItem {
            kind,
            entity: ent(index),
            pos: DVec2::new(1.5, -2.5),
            rot: 0.25,
            vel: DVec2::new(10.0, -10.0),
            component: NameId(3),
        }
    }

    fn packet(names: usize, items: usize) -> SnapshotPacket {
        SnapshotPacket {
            tick: 7,
            is_header: true,
            keyframe: false,
            your_entity: ent(42),
            names: (0..names)
                .map(|i| NameEntry { name: format!("component{i}"), id: NameId(i as u16) })
                .collect(),
            items: (0..items as u32).map(|i| item(i, ItemKind::Updated)).collect(),
        }
    }

    fn decode(p: &SnapshotPacket) -> SnapshotPacket {
        SnapshotPacket::decode(&p.encode(HALF), HALF, 1024, 4096).expect("must decode")
    }

    /// A sink with a fake MTU, mirroring `chunk`'s — the escalation
    /// policy must be testable without a socket.
    struct FakeSink {
        mtu: usize,
        sent: Vec<Vec<u8>>,
    }

    impl DatagramSink for FakeSink {
        fn send(&mut self, bytes: Vec<u8>) -> SendOutcome {
            if bytes.len() > self.mtu { return SendOutcome::TooLarge; }
            self.sent.push(bytes);
            SendOutcome::Sent
        }
        fn max_datagram_size(&self) -> Option<usize> { Some(self.mtu) }
    }

    #[test]
    fn a_packet_round_trips() {
        let p = packet(3, 5);
        let got = decode(&p);

        assert_eq!(got.tick, 7);
        assert!(got.is_header);
        assert!(!got.keyframe);
        assert_eq!(got.your_entity, ent(42));
        assert_eq!(got.names, p.names, "the name table must survive");
        assert_eq!(got.items.len(), 5);
    }

    /// Quantisation is lossy by design, so the assertion is bounded
    /// error rather than equality.
    #[test]
    fn item_state_survives_within_quantisation_error() {
        let p = packet(1, 1);
        let got = decode(&p);
        let (a, b) = (p.items[0], got.items[0]);

        let pos_step = (2.0 * HALF) / ((1u64 << POS_BITS) - 1) as f64;
        assert!((a.pos.x - b.pos.x).abs() <= pos_step);
        assert!((a.pos.y - b.pos.y).abs() <= pos_step);

        let rot_step = (2.0 * std::f64::consts::PI) / ((1u64 << ROT_BITS) - 1) as f64;
        assert!((a.rot - b.rot).abs() as f64 <= rot_step);

        let vel_step = (2.0 * MAX_SPEED) / ((1u64 << VEL_BITS) - 1) as f64;
        assert!((a.vel.x - b.vel.x).abs() <= vel_step);
        assert_eq!(a.entity, b.entity);
        assert_eq!(a.component, b.component);
    }

    /// Generation rides arrivals and departures, not updates — and the
    /// decoder must agree about which, or every field after drifts.
    #[test]
    fn generation_rides_arrivals_and_departures_only() {
        let p = SnapshotPacket {
            items: vec![
                EntityItem { entity: EntityId { index: 1, generation: 9 },
                             ..item(1, ItemKind::Entered) },
                EntityItem { entity: EntityId { index: 2, generation: 9 },
                             ..item(2, ItemKind::Updated) },
                EntityItem::left(EntityId { index: 3, generation: 9 }),
            ],
            ..packet(1, 0)
        };
        let got = decode(&p);

        assert_eq!(got.items[0].entity, EntityId { index: 1, generation: 9 },
                   "an arrival carries its generation");
        assert_eq!(got.items[2].entity, EntityId { index: 3, generation: 9 },
                   "so does a departure");
        assert_eq!(got.items[1].entity.index, 2, "an update carries only the index");
        assert_eq!(got.items[1].entity.generation, 0,
                   "and its generation is filled in by the client, not the wire");
    }

    /// A `Left` item carries no state, so it must cost meaningfully less
    /// than one that does.
    #[test]
    fn a_departure_is_cheaper_than_an_update() {
        let with_state = SnapshotPacket { items: vec![item(1, ItemKind::Updated)], ..packet(0, 0) };
        let departure  = SnapshotPacket { items: vec![EntityItem::left(ent(1))], ..packet(0, 0) };
        assert!(
            departure.encode(HALF).len() < with_state.encode(HALF).len(),
            "a departure should not pay for position it does not carry",
        );
    }

    /// **The forward-progress invariant**, mirroring `chunk`'s. If the
    /// scalar core plus one item does not fit, no amount of chunking can
    /// deliver the packet and items would be lost.
    #[test]
    fn scalar_core_plus_one_item_fits_the_conservative_budget() {
        let p = packet(0, 1);
        let len = p.rebuild_shed(&p.items).encode(HALF).len();
        assert!(
            len <= MIN_DATAGRAM_BUDGET,
            "shed core + 1 item is {len} B, over the {MIN_DATAGRAM_BUDGET} B floor",
        );
    }

    /// And with a large margin, so a single-item chunk being too large
    /// can only ever be the header's fault.
    #[test]
    fn one_item_is_small() {
        let empty = packet(0, 0).encode(HALF).len();
        let one   = packet(0, 1).encode(HALF).len();
        assert!(one - empty < 32, "one item costs {} B", one - empty);
    }

    #[test]
    fn a_packet_that_fits_is_sent_whole() {
        let mut sink = FakeSink { mtu: MIN_DATAGRAM_BUDGET, sent: Vec::new() };
        let p = packet(4, 10);
        assert_eq!(
            send_chunked(&mut sink, &p, |q| q.encode(HALF), &mut ChunkHint::new(), &"snapshot"),
            ChunkResult::Delivered,
        );
        assert_eq!(sink.sent.len(), 1, "no chunking when it already fits");
    }

    /// **The regression guard.** The name table rides the first chunk
    /// only. Re-cloning a bulk header into every chunk is what made
    /// every chunk oversized in production and cost 1,325 entities.
    #[test]
    fn the_name_table_rides_only_the_first_chunk() {
        let p = packet(24, 120);
        let mut sink = FakeSink { mtu: 400, sent: Vec::new() };
        assert_eq!(
            send_chunked(&mut sink, &p, |q| q.encode(HALF), &mut ChunkHint::new(), &"snapshot"),
            ChunkResult::Delivered,
        );
        assert!(sink.sent.len() > 1, "should have split");

        let first = SnapshotPacket::decode(&sink.sent[0], HALF, 1024, 4096).unwrap();
        assert_eq!(first.names.len(), 24, "the header chunk carries the table");
        assert!(first.is_header);

        for (i, bytes) in sink.sent.iter().enumerate().skip(1) {
            let chunk = SnapshotPacket::decode(bytes, HALF, 1024, 4096).unwrap();
            assert!(chunk.names.is_empty(), "continuation chunk {i} must carry no table");
            assert!(!chunk.is_header, "and must say so on the wire");
        }
    }

    /// Every item is delivered exactly once across the split — the
    /// policy never drops one.
    #[test]
    fn every_item_survives_a_split() {
        let p = packet(8, 100);
        let mut sink = FakeSink { mtu: 300, sent: Vec::new() };
        assert_eq!(
            send_chunked(&mut sink, &p, |q| q.encode(HALF), &mut ChunkHint::new(), &"snapshot"),
            ChunkResult::Delivered,
        );

        let total: usize = sink
            .sent
            .iter()
            .map(|b| SnapshotPacket::decode(b, HALF, 1024, 4096).unwrap().items.len())
            .sum();
        assert_eq!(total, 100, "all items delivered");
    }

    /// A truncated packet is an error, not a panic. These bytes arrive
    /// from a peer.
    #[test]
    fn a_truncated_packet_is_refused() {
        let bytes = packet(2, 4).encode(HALF);
        for cut in [1usize, 4, 9, bytes.len() - 1] {
            assert!(
                SnapshotPacket::decode(&bytes[..cut], HALF, 1024, 4096).is_err(),
                "truncating to {cut} B must not decode",
            );
        }
    }

    /// A hostile length prefix must be refused before it allocates —
    /// the `framing::read_msg` lesson, applied to counts.
    #[test]
    fn absurd_counts_are_refused_before_allocating() {
        let mut w = BitWriter::new();
        w.write_bits(1, 32);
        w.write_bit(true);
        w.write_bit(false);
        w.write_varint(0);
        w.write_varint(0);
        w.write_varint(u32::MAX as u64); // a name table of four billion
        let bytes = w.finish();

        assert_eq!(
            SnapshotPacket::decode(&bytes, HALF, 1024, 4096),
            Err(BitError::OutOfRange),
        );
    }

    /// Only components that actually reach clients appear in the table.
    /// A server-only component is wasted bytes and a hint about what the
    /// server tracks.
    #[test]
    fn the_name_table_lists_only_replicated_components() {
        #[derive(Clone, serde::Serialize, serde::Deserialize)]
        struct Shown(f32);
        #[derive(Clone, serde::Serialize, serde::Deserialize)]
        struct Hidden(f32);
        struct Sparkle;

        let mut reg = Registry::new();
        use crate::persist::registry::{Persist, Replicate};
        reg.register_replicated::<Shown>("shown", Persist::Volatile, Replicate::ToAll).unwrap();
        reg.register::<Hidden>("hidden", Persist::Volatile).unwrap();
        reg.register_replicated_transient::<Sparkle>("sparkle", Replicate::ToOwner).unwrap();

        let table = SnapshotPacket::name_table(&reg);
        let names: Vec<&str> = table.iter().map(|e| e.name.as_str()).collect();

        assert!(names.contains(&"shown"));
        assert!(names.contains(&"sparkle"), "owner-only still reaches the wire");
        assert!(!names.contains(&"hidden"), "a server-only component must not be advertised");
    }
}
