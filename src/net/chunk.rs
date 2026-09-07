//! MTU-aware datagram chunking with a header-shedding escalation policy.
//!
//! The problem this solves is specific to the shape almost every
//! snapshot packet ends up having: one **header** of per-player scalars
//! and bulk lists (your inventory, the market, the leaderboard) plus a
//! **list of items** (the entities in range). The item list is
//! splittable; the header is not. When the whole thing exceeds the path
//! MTU, splitting items alone is not always enough — and the interesting
//! failure is what you do when it isn't.
//!
//! The policy, lifted from `server::net::send_chunked` in void-claim
//! along with the production incident that shaped it:
//!
//! 1. Try to send whole. Almost always fits; costs one syscall.
//! 2. `TooLarge` with n > 1 items: halve the item slice, retry. Ordinary
//!    spillover. The first chunk carries the header, continuation chunks
//!    blank the bulk fields so they are paid for once per tick rather
//!    than once per chunk.
//! 3. `TooLarge` with n == 1: the *header* is oversized, not the item.
//!    Shed the header's bulk payload and retry with the same item.
//!    Never drop the item. An item is world state — a dropped one is an
//!    object that visibly vanishes for the player — whereas a shed
//!    header field is a market list that arrives one tick late.
//! 4. `TooLarge` with a shed header and one item: log an ERROR loudly
//!    and drop. The packet has outgrown datagrams entirely and needs a
//!    reliable stream; silently losing world state is far worse than
//!    saying so.
//!
//! Step 3 is the whole reason this module exists. void-claim's server
//! originally had only steps 1, 2 and "give up on this entity", and
//! logged `single-entity snapshot chunk too large; skipping entity`
//! 1,325 times in 44 seconds. No entity was oversized; two unbounded
//! header vectors were, and they were re-cloned into every chunk, so
//! every chunk was oversized and the server discarded entities one at a
//! time while objects vanished in front of players.
//!
//! The core loop takes a [`DatagramSink`] rather than a `quinn::
//! Connection` so it can be driven by a fake MTU in tests. The original
//! could only be exercised against a live connection, which is why the
//! bug shipped.

/// A packet the chunker can split. Deliberately knows nothing about
/// snapshots, entities, ticks or any other game concept — only that a
/// packet is "some header plus a list of items".
pub trait Chunkable {
    /// One element of the splittable list.
    type Item: Clone;

    /// The splittable payload.
    fn items(&self) -> &[Self::Item];

    /// Rebuild the packet carrying exactly `items`.
    ///
    /// `is_header` is true for the first chunk of a send only. That
    /// chunk carries the bulk header fields; continuation chunks **must
    /// blank them** — re-cloning them into every chunk is precisely the
    /// bug this module exists to prevent. Implementations should also
    /// forward the flag onto the wire, so the receiver can tell "blanked
    /// for chunking" from "genuinely empty"; an emptiness heuristic gets
    /// a legitimately-cleared list wrong.
    fn rebuild(&self, items: &[Self::Item], is_header: bool) -> Self;

    /// Rebuild with the bulk header fields shed entirely — the scalar
    /// core plus `items`, nothing else. Used only by step 3, when even
    /// header + one item will not fit.
    ///
    /// Whatever is shed here must be re-sent on the next tick: this
    /// trades a one-tick-stale header for not losing an item. It is
    /// correct for state that is re-broadcast every tick and wrong for
    /// one-shot events, so keep those in the scalar core.
    fn rebuild_shed(&self, items: &[Self::Item]) -> Self;
}

/// Outcome of one attempted datagram write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendOutcome {
    Sent,
    /// Over the current path MTU — the chunker will split or shed and
    /// retry. Must not be returned for any other failure, or the
    /// chunker will spin trying to make an unsendable packet smaller.
    TooLarge,
    /// The connection is gone. Terminal: the chunker gives up.
    Closed,
}

/// Where chunks go. The quinn dependency lives at the call site's edge
/// (see [`QuinnSink`]) so the escalation policy above is testable with a
/// fake MTU and no sockets.
pub trait DatagramSink {
    fn send(&mut self, bytes: Vec<u8>) -> SendOutcome;

    /// The peer's advertised max datagram size, for log messages only.
    /// `None` if unknown.
    fn max_datagram_size(&self) -> Option<usize> { None }
}

/// Result of a whole [`send_chunked`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkResult {
    /// Every item was delivered (possibly across several datagrams,
    /// possibly with a shed header).
    Delivered,
    /// Step 4: the packet cannot fit a datagram at all and was dropped.
    /// The connection is still fine; the caller should consider a
    /// reliable stream for this packet type.
    Undeliverable,
    /// The connection closed mid-send.
    Closed,
}

/// Conservative floor for a QUIC datagram payload: IPv6's 1280 B minimum
/// link MTU, minus 48 B of IPv6 + UDP headers, minus QUIC short-header
/// and AEAD tag overhead. quinn reports the real (usually larger) value
/// per-connection; this is the smallest any path can hand us, so a
/// packet that fits here fits anywhere.
pub const MIN_DATAGRAM_BUDGET: usize = 1200;

/// Send `packet`, splitting it across datagrams as needed.
///
/// `encode` serialises a rebuilt packet; the chunker is agnostic to the
/// wire format (bitcode, postcard, serde — the engine takes no view).
/// `label` names the packet in the two log lines this can emit; pass
/// something that identifies the tick or packet type, since those logs
/// are the only signal that a packet is outgrowing the transport.
pub fn send_chunked<P, S, E>(
    sink: &mut S,
    packet: &P,
    mut encode: E,
    label: &dyn std::fmt::Display,
) -> ChunkResult
where
    P: Chunkable,
    S: DatagramSink,
    E: FnMut(&P) -> Vec<u8>,
{
    // Step 1: the overwhelmingly common case. One encode, one send.
    match sink.send(encode(packet)) {
        SendOutcome::Sent     => return ChunkResult::Delivered,
        SendOutcome::Closed   => return ChunkResult::Closed,
        SendOutcome::TooLarge => {}
    }

    // A packet with no items has nothing splittable, so step 2 can never
    // help and step 3 is the only move available.
    if packet.items().is_empty() {
        let shed = packet.rebuild_shed(&[]);
        return match sink.send(encode(&shed)) {
            SendOutcome::Sent   => ChunkResult::Delivered,
            SendOutcome::Closed => ChunkResult::Closed,
            SendOutcome::TooLarge => {
                log::error!(
                    "packet undeliverable: item-less shed header exceeds datagram MTU \
                     ({:?} B) for {label}; dropping",
                    sink.max_datagram_size(),
                );
                ChunkResult::Undeliverable
            }
        };
    }

    let mut chunk_size = packet.items().len();
    let mut remaining: &[P::Item] = packet.items();
    // Only the first chunk of a send carries the bulk header.
    let mut first = true;
    // Latches once step 3 fires. Deliberately not cleared between chunks
    // of the same send: if the header didn't fit for chunk 0 it will not
    // fit for chunk 3 either, and re-trying it every chunk would burn a
    // failed datagram each time.
    let mut shed_header = false;

    while !remaining.is_empty() {
        let n = chunk_size.min(remaining.len());
        let (chunk, rest) = remaining.split_at(n);
        let sub = if shed_header {
            packet.rebuild_shed(chunk)
        } else {
            packet.rebuild(chunk, first)
        };

        match sink.send(encode(&sub)) {
            SendOutcome::Sent => {
                first = false;
                remaining = rest;
            }
            SendOutcome::Closed => return ChunkResult::Closed,
            SendOutcome::TooLarge => {
                if n > 1 {
                    // Step 2: ordinary spillover. Halve and retry.
                    chunk_size = n / 2;
                } else if !shed_header {
                    // Step 3: header + a SINGLE item overflows, so the
                    // header is the problem. Shed its bulk payload (it
                    // is re-sent next tick) and retry with the same
                    // item — never drop the item.
                    log::warn!(
                        "packet header exceeds datagram MTU ({:?} B) for {label}; \
                         shedding bulk header fields to keep items",
                        sink.max_datagram_size(),
                    );
                    shed_header = true;
                } else {
                    // Step 4: even the scalar core plus one item will
                    // not fit. Unreachable for a sanely-sized packet
                    // (see the tests pinning scalar header + 1 item
                    // against MIN_DATAGRAM_BUDGET). If it fires, the
                    // packet has outgrown datagrams and needs a
                    // reliable stream — so be loud rather than
                    // silently losing world state.
                    log::error!(
                        "packet undeliverable: scalar header + 1 item exceeds datagram \
                         MTU ({:?} B) for {label}; dropping {} item(s)",
                        sink.max_datagram_size(),
                        remaining.len(),
                    );
                    return ChunkResult::Undeliverable;
                }
            }
        }
    }

    ChunkResult::Delivered
}

// ── quinn adapter (the only part that needs a real connection) ───────────────

/// [`DatagramSink`] over a live `quinn::Connection`.
///
/// Everything above is transport-agnostic; this is the thin edge that
/// translates quinn's error variants into [`SendOutcome`]. Note that
/// only `TooLarge` maps to a retry — `Disabled` (peer refused datagram
/// support) and `ConnectionLost` are terminal, because shrinking the
/// packet cannot help either one.
pub struct QuinnSink<'a>(pub &'a quinn::Connection);

impl DatagramSink for QuinnSink<'_> {
    fn send(&mut self, bytes: Vec<u8>) -> SendOutcome {
        match self.0.send_datagram(bytes.into()) {
            Ok(())                                          => SendOutcome::Sent,
            Err(quinn::SendDatagramError::TooLarge)          => SendOutcome::TooLarge,
            Err(quinn::SendDatagramError::ConnectionLost(_)) => SendOutcome::Closed,
            Err(quinn::SendDatagramError::Disabled)          => SendOutcome::Closed,
            Err(quinn::SendDatagramError::UnsupportedByPeer) => SendOutcome::Closed,
        }
    }

    fn max_datagram_size(&self) -> Option<usize> { self.0.max_datagram_size() }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── a test packet with the shape the policy assumes ──────────────────
    //
    // Scalar core + a bulk header vec + an item list, mirroring
    // void_proto's SnapshotPacket without any of its game meaning.

    #[derive(Clone, Debug, PartialEq)]
    struct TestItem {
        id: u32,
        pos: [f32; 2],
    }

    #[derive(Clone, Debug, PartialEq)]
    struct TestPacket {
        // Scalar core — always present on every chunk.
        tick: u32,
        is_header: bool,
        your_id: u32,
        /// Bulk header field: scales with world state (a market list, a
        /// leaderboard, an A* path). Blanked on continuation chunks,
        /// shed entirely by step 3.
        bulk: Vec<u64>,
        items: Vec<TestItem>,
    }

    impl TestPacket {
        fn new(bulk: usize, items: usize) -> Self {
            Self {
                tick: 7,
                is_header: true,
                your_id: 42,
                bulk: (0..bulk as u64).collect(),
                items: (0..items as u32)
                    .map(|i| TestItem { id: i, pos: [i as f32, -(i as f32)] })
                    .collect(),
            }
        }
    }

    impl Chunkable for TestPacket {
        type Item = TestItem;
        fn items(&self) -> &[TestItem] { &self.items }
        fn rebuild(&self, items: &[TestItem], is_header: bool) -> Self {
            Self {
                tick: self.tick,
                is_header,
                your_id: self.your_id,
                // The load-bearing gate: bulk rides the header chunk only.
                bulk: if is_header { self.bulk.clone() } else { Vec::new() },
                items: items.to_vec(),
            }
        }
        fn rebuild_shed(&self, items: &[TestItem]) -> Self {
            Self {
                tick: self.tick,
                is_header: true,
                your_id: self.your_id,
                bulk: Vec::new(),
                items: items.to_vec(),
            }
        }
    }

    /// Fixed-width stand-in for a real codec. Sizes are what the policy
    /// reacts to, so an exact byte count matters more than realism.
    const SCALAR_CORE: usize = 12; // tick + is_header + your_id
    const PER_BULK: usize = 8;
    const PER_ITEM: usize = 12;

    fn encode(p: &TestPacket) -> Vec<u8> {
        vec![0u8; SCALAR_CORE + p.bulk.len() * PER_BULK + p.items.len() * PER_ITEM]
    }

    /// A sink with a fake MTU that records every datagram it accepted.
    /// This is what makes the escalation policy testable without a
    /// socket — the void-claim original could only be driven by a live
    /// connection, which is how the header bug reached production.
    struct FakeSink {
        mtu: usize,
        sent: Vec<Vec<u8>>,
        /// Rejects everything after this many successful sends, to
        /// exercise the terminal path.
        close_after: Option<usize>,
    }

    impl FakeSink {
        fn new(mtu: usize) -> Self { Self { mtu, sent: Vec::new(), close_after: None } }
    }

    impl DatagramSink for FakeSink {
        fn send(&mut self, bytes: Vec<u8>) -> SendOutcome {
            if let Some(n) = self.close_after {
                if self.sent.len() >= n { return SendOutcome::Closed; }
            }
            if bytes.len() > self.mtu { return SendOutcome::TooLarge; }
            self.sent.push(bytes);
            SendOutcome::Sent
        }
        fn max_datagram_size(&self) -> Option<usize> { Some(self.mtu) }
    }

    /// Decode a recorded datagram back to (bulk_count, item_count).
    /// With the fixed-width codec the lengths are recoverable, which is
    /// enough to assert what each chunk carried.
    fn shape(bytes: &[u8], expect_bulk: usize) -> usize {
        (bytes.len() - SCALAR_CORE - expect_bulk * PER_BULK) / PER_ITEM
    }

    fn run(sink: &mut FakeSink, p: &TestPacket) -> ChunkResult {
        send_chunked(sink, p, encode, &"test-packet")
    }

    // ── (a) the forward-progress invariant ───────────────────────────────

    /// Ported from void_proto's `scalar_header_plus_one_entity_fits_one_
    /// datagram`. The property `send_chunked` depends on: if a scalar
    /// header plus one item does not fit, no amount of chunking can
    /// deliver the packet and items WILL be lost. Pinned against the
    /// conservative IPv6 floor so passing here passes on any path.
    #[test]
    fn scalar_header_plus_one_item_fits_the_conservative_budget() {
        let p = TestPacket::new(0, 1);
        let len = encode(&p.rebuild_shed(&p.items)).len();
        assert!(
            len <= MIN_DATAGRAM_BUDGET,
            "shed header + 1 item is {len} B, over the {MIN_DATAGRAM_BUDGET} B floor; \
             send_chunked cannot split a header, so items would be dropped"
        );
    }

    /// The other half of the same argument: one item is nowhere near the
    /// limit, so a "single-item chunk too large" can only ever be the
    /// header's fault — the misdiagnosis that cost void-claim 1,325 lost
    /// entities.
    #[test]
    fn a_single_item_is_small() {
        let empty = encode(&TestPacket::new(0, 0)).len();
        let one   = encode(&TestPacket::new(0, 1)).len();
        assert!(one - empty < 128, "one item costs {} B", one - empty);
    }

    // ── step 1 ───────────────────────────────────────────────────────────

    #[test]
    fn a_packet_that_fits_is_sent_whole() {
        let mut sink = FakeSink::new(MIN_DATAGRAM_BUDGET);
        assert_eq!(run(&mut sink, &TestPacket::new(4, 10)), ChunkResult::Delivered);
        assert_eq!(sink.sent.len(), 1, "no chunking when it already fits");
    }

    // ── step 2 + (b) bulk fields absent from continuation chunks ─────────

    #[test]
    fn oversized_item_list_is_split_across_chunks() {
        // 64 items × 12 B = 768 B plus header; MTU 200 forces several
        // halvings.
        let p = TestPacket::new(2, 64);
        let mut sink = FakeSink::new(200);
        assert_eq!(run(&mut sink, &p), ChunkResult::Delivered);
        assert!(sink.sent.len() > 1, "should have split");
        // Every item is delivered exactly once, in order — the policy
        // never drops one.
        let total: usize = sink.sent.iter().enumerate()
            .map(|(i, b)| shape(b, if i == 0 { 2 } else { 0 }))
            .sum();
        assert_eq!(total, 64, "all items delivered");
    }

    /// (b) The regression proper: bulk header fields ride the first
    /// chunk only. Re-cloning them into every chunk is what made every
    /// chunk oversized in production.
    #[test]
    fn bulk_header_rides_only_the_first_chunk() {
        let p = TestPacket::new(8, 40);
        let mut sink = FakeSink::new(220);
        assert_eq!(run(&mut sink, &p), ChunkResult::Delivered);
        assert!(sink.sent.len() > 1);

        let first_len = sink.sent[0].len();
        let first_items = shape(&sink.sent[0], 8);
        assert_eq!(first_len, SCALAR_CORE + 8 * PER_BULK + first_items * PER_ITEM,
            "first chunk carries the bulk header");

        for (i, b) in sink.sent.iter().enumerate().skip(1) {
            let items = shape(b, 0);
            assert_eq!(b.len(), SCALAR_CORE + items * PER_ITEM,
                "continuation chunk {i} must carry no bulk header");
        }
    }

    /// `is_header` is what lets the receiver distinguish "blanked for
    /// chunking" from "genuinely empty". An emptiness heuristic gets a
    /// legitimately-cleared list wrong, so the flag must be set on
    /// exactly one chunk per send.
    #[test]
    fn exactly_one_chunk_is_flagged_as_the_header() {
        let p = TestPacket::new(4, 40);
        // Rebuild directly to inspect the flag (the byte codec drops it).
        let header = p.rebuild(&p.items[..1], true);
        let cont   = p.rebuild(&p.items[1..2], false);
        assert!(header.is_header && !header.bulk.is_empty());
        assert!(!cont.is_header && cont.bulk.is_empty(),
            "continuation chunks are flagged and blanked together");
    }

    // ── (c) step 3: shedding rather than item loss ───────────────────────

    /// The incident, reproduced: a bulk header that alone exceeds the
    /// MTU. Splitting items can never help — the fix is to shed the
    /// header, never to drop items.
    #[test]
    fn oversized_bulk_header_sheds_instead_of_losing_items() {
        // 150 bulk entries × 8 B = 1200 B of header alone, against a
        // 200 B MTU: even header + 1 item cannot fit.
        let p = TestPacket::new(150, 6);
        let mut sink = FakeSink::new(200);
        assert_eq!(run(&mut sink, &p), ChunkResult::Delivered);

        // Every chunk is bulk-free (shedding latched), and every item
        // still arrived.
        let total: usize = sink.sent.iter().map(|b| shape(b, 0)).sum();
        assert_eq!(total, 6, "no item may be dropped to fit the header");
        for b in &sink.sent {
            let items = shape(b, 0);
            assert_eq!(b.len(), SCALAR_CORE + items * PER_ITEM,
                "shed chunks carry the scalar core only");
        }
    }

    /// Shedding latches: once the header is known not to fit, later
    /// chunks of the same send must not re-try it and burn a failed
    /// datagram each time.
    #[test]
    fn shedding_latches_for_the_rest_of_the_send() {
        let p = TestPacket::new(150, 12);
        let mut sink = FakeSink::new(60); // 4 items per datagram at most
        assert_eq!(run(&mut sink, &p), ChunkResult::Delivered);
        assert!(sink.sent.len() >= 3);
        for b in &sink.sent {
            assert_eq!((b.len() - SCALAR_CORE) % PER_ITEM, 0, "no bulk anywhere");
        }
    }

    /// An item-less packet has nothing to split, so step 3 is the only
    /// available move — it must still shed rather than declare defeat.
    #[test]
    fn item_less_packet_still_sheds() {
        let p = TestPacket::new(150, 0);
        let mut sink = FakeSink::new(200);
        assert_eq!(run(&mut sink, &p), ChunkResult::Delivered);
        assert_eq!(sink.sent.len(), 1);
        assert_eq!(sink.sent[0].len(), SCALAR_CORE);
    }

    // ── step 4 + terminal conditions ─────────────────────────────────────

    /// When even the scalar core plus one item will not fit, the packet
    /// has outgrown datagrams. Report it rather than silently shredding
    /// world state — this is the signal to move the packet to a
    /// reliable stream.
    #[test]
    fn hopeless_packet_is_reported_undeliverable() {
        let p = TestPacket::new(4, 3);
        // Smaller than SCALAR_CORE + one item: nothing can be sent.
        let mut sink = FakeSink::new(SCALAR_CORE + PER_ITEM - 1);
        assert_eq!(run(&mut sink, &p), ChunkResult::Undeliverable);
        assert!(sink.sent.is_empty());
    }

    #[test]
    fn a_closed_connection_stops_the_send() {
        let p = TestPacket::new(2, 40);
        let mut sink = FakeSink::new(200);
        sink.close_after = Some(2);
        assert_eq!(run(&mut sink, &p), ChunkResult::Closed);
        assert_eq!(sink.sent.len(), 2, "gave up rather than retrying a dead link");
    }

    /// A closed connection on the very first whole-packet attempt is
    /// terminal too — no chunking, no shedding.
    #[test]
    fn closed_on_the_first_attempt_short_circuits() {
        let mut sink = FakeSink::new(MIN_DATAGRAM_BUDGET);
        sink.close_after = Some(0);
        assert_eq!(run(&mut sink, &TestPacket::new(0, 1)), ChunkResult::Closed);
    }

    /// Halving must terminate. A pathological MTU that admits exactly
    /// one item per datagram is the worst case for the search; it must
    /// still deliver every item and not loop.
    #[test]
    fn halving_converges_to_one_item_per_datagram() {
        let p = TestPacket::new(0, 9);
        let mut sink = FakeSink::new(SCALAR_CORE + PER_ITEM);
        assert_eq!(run(&mut sink, &p), ChunkResult::Delivered);
        assert_eq!(sink.sent.len(), 9, "one item per datagram");
        for b in &sink.sent { assert_eq!(shape(b, 0), 1); }
    }
}
