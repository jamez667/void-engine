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

/// What a connection has learned about how many items fit a datagram.
///
/// Halving alone is a poor packer. It starts at the whole item list and
/// divides by two on each rejection, so it lands on the first
/// power-of-two fraction that fits rather than on capacity — measured on
/// a real snapshot, 28 items per chunk where 93 fit, 66 datagrams where
/// 20 would do, and 33% utilisation. Two thirds of every datagram was
/// empty, and at a thousand clients that triples keyframe traffic.
///
/// The fix is not to probe upward within a send: each probe that fails
/// costs a rejected datagram, and for a packet that genuinely admits one
/// item per datagram that turns nine sends into seventeen. Capacity is
/// instead remembered *across* sends, which works because a connection's
/// packet shape barely changes from one tick to the next.
///
/// # What this actually recovers
///
/// Half the gap, not all of it. On that same snapshot: 66 datagrams
/// become 34 and utilisation goes from 33% to 63%, converging after
/// about six ticks and paying no rejections thereafter. A 5000-item
/// keyframe goes from 129 datagrams to 93.
///
/// The remaining third is structural rather than a tuning failure. Only
/// the first chunk of a send carries the bulk header, and only that chunk
/// is measured — a continuation chunk fits more items precisely because
/// it carries less, so recording its size would teach a number the header
/// chunk cannot honour. One learned size therefore serves two chunk
/// shapes, and the continuation chunks run at the header chunk's size
/// with the header's room to spare. Closing that needs a second remembered
/// size, which is a larger change than this one.
///
/// Hold one per connection, alongside whatever else that connection
/// tracks. A fresh hint behaves exactly like the old policy, so a caller
/// with nowhere to keep one loses nothing but the savings.
///
/// Deliberately not linked to `replication::ClientLink` here: this module
/// compiles on the plain `net` axis, where that type does not exist, and
/// an intra-doc link to it is unresolvable there.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ChunkHint {
    /// Largest item count known to have fitted.
    floor: Option<usize>,
    /// Smallest item count known not to have fitted.
    ceiling: Option<usize>,
}

impl ChunkHint {
    /// A hint that has learned nothing. Reproduces the halving-only
    /// policy on its first send.
    pub fn new() -> Self { Self::default() }

    /// Largest item count known to fit, once anything has been sent.
    pub fn known_good(&self) -> Option<usize> { self.floor }

    /// True once capacity is known exactly, at which point sends stop
    /// probing and cost no rejected datagrams at all.
    ///
    /// This is the property that makes remembering worth doing: a design
    /// that kept probing would trade one kind of waste for another.
    pub fn is_converged(&self) -> bool {
        match (self.floor, self.ceiling) {
            (Some(f), Some(c)) => f + 1 >= c,
            _ => false,
        }
    }

    /// Where a send should start, given the list it has to place.
    ///
    /// With nothing learned this is the whole list, which is both the old
    /// behaviour and the right guess: most packets fit whole.
    fn start_for(&self, items: usize) -> usize {
        match self.floor {
            None => items,
            Some(floor) => {
                // With no ceiling known, step up by a quarter: a link whose
                // capacity grew finds it in a few ticks without a large
                // rejection each time.
                //
                // With one known, bisect the gap instead. A fixed step
                // overshoots into the ceiling, gets capped back to the
                // floor, and retries the same size forever — stranded
                // short of capacity while reporting no rejections, which
                // looks like success. Halving the gap always lands
                // strictly between the two and closes it.
                let probe = match self.ceiling {
                    Some(c) if c > floor + 1 => floor + ((c - floor) / 2),
                    Some(_) => floor,
                    None    => floor + (floor / 4).max(1),
                };
                probe.min(items).max(1)
            }
        }
    }

    /// Should the whole-packet attempt be skipped?
    ///
    /// Step 1 sends the entire packet on the assumption it fits. Once a
    /// smaller count is known not to fit, that attempt is a guaranteed
    /// rejection — which is where most of the wasted sends went.
    fn skip_whole_attempt(&self, items: usize) -> bool {
        matches!(self.ceiling, Some(c) if c <= items)
    }

    fn record_fit(&mut self, n: usize) {
        self.floor = Some(self.floor.map_or(n, |f| f.max(n)));
    }

    fn record_reject(&mut self, n: usize) {
        self.ceiling = Some(self.ceiling.map_or(n, |c| c.min(n)));
        // A ceiling at or below the floor means the path shrank — an MTU
        // change or a route flap. Forget the floor rather than carrying a
        // size now known to fail.
        if let (Some(f), Some(c)) = (self.floor, self.ceiling) {
            if f >= c {
                self.floor = if c > 1 { Some(c - 1) } else { None };
            }
        }
    }
}

/// Send `packet`, splitting it across datagrams as needed.
///
/// `encode` serialises a rebuilt packet; the chunker is agnostic to the
/// wire format (bitcode, postcard, serde — the engine takes no view).
/// `label` names the packet in the two log lines this can emit; pass
/// something that identifies the tick or packet type, since those logs
/// are the only signal that a packet is outgrowing the transport.
///
/// `hint` carries what previous sends on this connection learned about
/// datagram capacity; see [`ChunkHint`]. Pass a fresh one to get the
/// original halving-only behaviour.
pub fn send_chunked<P, S, E>(
    sink: &mut S,
    packet: &P,
    mut encode: E,
    hint: &mut ChunkHint,
    label: &dyn std::fmt::Display,
) -> ChunkResult
where
    P: Chunkable,
    S: DatagramSink,
    E: FnMut(&P) -> Vec<u8>,
{
    let total_items = packet.items().len();

    // Step 1: the overwhelmingly common case. One encode, one send.
    // Skipped when the hint already knows a smaller count does not fit,
    // because then this attempt can only be rejected.
    if !hint.skip_whole_attempt(total_items) {
        match sink.send(encode(packet)) {
            SendOutcome::Sent => {
                if total_items > 0 {
                    hint.record_fit(total_items);
                }
                return ChunkResult::Delivered;
            }
            SendOutcome::Closed   => return ChunkResult::Closed,
            SendOutcome::TooLarge => {
                if total_items > 0 {
                    hint.record_reject(total_items);
                }
            }
        }
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

    // Start where this connection last succeeded rather than at the whole
    // list, which is the entire point of the hint: a converged link opens
    // at capacity and never pays a rejection.
    let mut chunk_size = hint.start_for(total_items);
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
                // Only a chunk that carried the bulk header measures the
                // same thing the next send's first chunk will. A shed or
                // continuation chunk fits more items precisely because it
                // is carrying less, so recording it would teach the hint
                // a size the header chunk cannot honour.
                if !shed_header && first {
                    hint.record_fit(n);
                }
                first = false;
                remaining = rest;
            }
            SendOutcome::Closed => return ChunkResult::Closed,
            SendOutcome::TooLarge => {
                if !shed_header && first {
                    hint.record_reject(n);
                }
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
        /// Datagrams refused for being oversized. The cost a probing
        /// policy pays, and what a converged hint must drive to zero.
        rejected: usize,
    }

    impl FakeSink {
        fn new(mtu: usize) -> Self {
            Self { mtu, sent: Vec::new(), close_after: None, rejected: 0 }
        }
    }

    impl DatagramSink for FakeSink {
        fn send(&mut self, bytes: Vec<u8>) -> SendOutcome {
            if let Some(n) = self.close_after {
                if self.sent.len() >= n { return SendOutcome::Closed; }
            }
            if bytes.len() > self.mtu {
                self.rejected += 1;
                return SendOutcome::TooLarge;
            }
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

    /// A fresh hint per call, which reproduces the halving-only policy
    /// exactly — so every assertion below pins the same behaviour it did
    /// before capacity was remembered across sends.
    fn run(sink: &mut FakeSink, p: &TestPacket) -> ChunkResult {
        send_chunked(sink, p, encode, &mut ChunkHint::new(), &"test-packet")
    }

    /// Send with a hint that persists, as a real connection would.
    fn run_hinted(sink: &mut FakeSink, p: &TestPacket, hint: &mut ChunkHint) -> ChunkResult {
        send_chunked(sink, p, encode, hint, &"test-packet")
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

    // ── (d) remembering capacity across sends ────────────────────────────

    /// A fresh hint must behave exactly like the old policy, so a caller
    /// with nowhere to keep one loses nothing but the savings.
    #[test]
    fn a_fresh_hint_reproduces_the_halving_only_policy() {
        let p = TestPacket::new(2, 64);

        let mut a = FakeSink::new(200);
        assert_eq!(run(&mut a, &p), ChunkResult::Delivered);

        let mut b = FakeSink::new(200);
        assert_eq!(run_hinted(&mut b, &p, &mut ChunkHint::new()), ChunkResult::Delivered);

        assert_eq!(a.sent.len(), b.sent.len(), "same datagram count");
        assert_eq!(a.sent, b.sent, "and byte-identical chunks");
    }

    /// **The property that makes remembering worth doing.** A link that
    /// keeps its hint converges on capacity and then stops probing: no
    /// rejected datagrams at all in steady state. A design that kept
    /// probing would trade one kind of waste for another.
    #[test]
    fn a_persistent_hint_converges_and_then_costs_nothing() {
        let p = TestPacket::new(2, 64);
        let mut hint = ChunkHint::new();

        let mut rejects = Vec::new();
        for _ in 0..12 {
            let mut sink = FakeSink::new(200);
            assert_eq!(run_hinted(&mut sink, &p, &mut hint), ChunkResult::Delivered);
            rejects.push(sink.rejected);
            // Every item arrives on every tick, converged or not.
            let total: usize = sink.sent.iter().enumerate()
                .map(|(i, b)| shape(b, if i == 0 { 2 } else { 0 }))
                .sum();
            assert_eq!(total, 64, "all items delivered");
        }

        assert!(hint.is_converged(), "capacity must be pinned down: {hint:?}");
        assert_eq!(
            *rejects.last().unwrap(), 0,
            "a converged link must pay no rejections; sequence was {rejects:?}",
        );
    }

    /// And converging must actually pack better than halving — this is
    /// the 3.3x that motivated the whole thing.
    #[test]
    fn a_converged_hint_uses_fewer_datagrams_than_halving() {
        let p = TestPacket::new(2, 64);

        let mut cold = FakeSink::new(200);
        run(&mut cold, &p);

        let mut hint = ChunkHint::new();
        let mut warm = FakeSink::new(200);
        for _ in 0..12 {
            warm = FakeSink::new(200);
            run_hinted(&mut warm, &p, &mut hint);
        }

        assert!(
            warm.sent.len() < cold.sent.len(),
            "converged {} datagrams vs halving {}",
            warm.sent.len(), cold.sent.len(),
        );
    }

    /// Once a ceiling is known, the whole-packet attempt is a guaranteed
    /// rejection and must be skipped — that is where most of the wasted
    /// sends went.
    #[test]
    fn a_known_ceiling_skips_the_doomed_whole_packet_attempt() {
        let p = TestPacket::new(2, 64);
        let mut hint = ChunkHint::new();

        let mut first = FakeSink::new(200);
        run_hinted(&mut first, &p, &mut hint);
        let cold_rejects = first.rejected;

        let mut second = FakeSink::new(200);
        run_hinted(&mut second, &p, &mut hint);

        assert!(
            second.rejected < cold_rejects,
            "second send still paid {} rejections against {cold_rejects}",
            second.rejected,
        );
    }

    /// A path that shrinks — an MTU change, a route flap — must not leave
    /// the hint recommending a size now known to fail.
    #[test]
    fn a_shrinking_path_retracts_a_stale_floor() {
        let p = TestPacket::new(2, 64);
        let mut hint = ChunkHint::new();

        for _ in 0..8 {
            let mut sink = FakeSink::new(400);
            run_hinted(&mut sink, &p, &mut hint);
        }
        let roomy = hint.known_good().expect("something must have fitted");

        // The path halves underneath us.
        let mut tight = FakeSink::new(120);
        assert_eq!(run_hinted(&mut tight, &p, &mut hint), ChunkResult::Delivered);

        let cramped = hint.known_good().expect("a smaller size must now be known");
        assert!(cramped < roomy, "floor must retract from {roomy} to something smaller");

        // And it still delivers everything on the narrower path.
        let total: usize = tight.sent.iter().enumerate()
            .map(|(i, b)| shape(b, if i == 0 { 2 } else { 0 }))
            .sum();
        assert_eq!(total, 64);
    }

    /// A packet that fits whole teaches the hint that it fits whole, so
    /// the common case stays one encode and one send forever.
    #[test]
    fn a_packet_that_always_fits_never_learns_a_ceiling() {
        let p = TestPacket::new(4, 10);
        let mut hint = ChunkHint::new();

        for _ in 0..5 {
            let mut sink = FakeSink::new(MIN_DATAGRAM_BUDGET);
            assert_eq!(run_hinted(&mut sink, &p, &mut hint), ChunkResult::Delivered);
            assert_eq!(sink.sent.len(), 1, "no chunking when it already fits");
            assert_eq!(sink.rejected, 0, "and no rejections either");
        }
        assert_eq!(hint.known_good(), Some(10));
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
