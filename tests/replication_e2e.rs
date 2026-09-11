//! The replication pipeline, end to end, over several ticks.
//!
//! Every piece has unit tests. None of them proves the pieces *compose*:
//! that a grid index survives the trip to an entity id, that what the
//! client reconstructs from the wire matches what the server believed it
//! could see, and that this keeps holding as entities move in and out of
//! range across ticks. Those are exactly the seams where two modules
//! written hours apart disagree about a convention, and a unit test on
//! either side of the seam sees nothing wrong.
//!
//! An integration test rather than a unit test on purpose: it compiles
//! against the public API exactly as a downstream server crate would, so
//! if the pipeline stops being usable from outside the crate, this fails.
//!
//! The client model here is deliberately dumb — a `HashMap` of what it
//! has been told. That is the point. It knows nothing except what arrived
//! on the wire, so if the server's view and the client's view agree, the
//! wire carried everything it needed to.

#![cfg(feature = "replication")]

use std::collections::HashMap;

use glam::DVec2;
use void_engine::collision::{AoiScratch, SpatialGrid};
use void_engine::components::Transform2D;
use void_engine::ecs::EntityId;
use void_engine::net::chunk::{
    send_chunked, ChunkHint, ChunkResult, DatagramSink, SendOutcome, MIN_DATAGRAM_BUDGET,
};
use void_engine::net::replication::{Ack, ClientLink, KeyframeBudget, Plan, Relevancy};
use void_engine::net::snapshot::{EntityItem, ItemKind, NameEntry, SnapshotPacket};
use void_engine::persist::registry::NameId;
use void_engine::World;

/// Half a sector. Positions are sector-local, so this bounds them.
const HALF: f64 = 500.0;
/// How far a client can see.
const VIEW: f64 = 120.0;
/// Grid cell size.
const CELL: f64 = 60.0;

/// A sink that keeps every datagram, so the test can play the role of the
/// network and hand the bytes to a client.
#[derive(Default)]
struct Wire {
    sent: Vec<Vec<u8>>,
}

impl DatagramSink for Wire {
    fn send(&mut self, bytes: Vec<u8>) -> SendOutcome {
        if bytes.len() > MIN_DATAGRAM_BUDGET {
            return SendOutcome::TooLarge;
        }
        self.sent.push(bytes);
        SendOutcome::Sent
    }
    fn max_datagram_size(&self) -> Option<usize> { Some(MIN_DATAGRAM_BUDGET) }
}

/// What a client believes, rebuilt only from datagrams it received.
#[derive(Default)]
struct ClientView {
    /// Keyed by entity *index*, because that is all an `Updated` carries.
    /// The value keeps the full `EntityId` from the `Entered` that
    /// introduced this tenant, so the tests can still assert identity —
    /// and so a generation arriving out of step is visible rather than
    /// filed under a second key.
    entities: HashMap<u32, (EntityId, DVec2)>,
    /// Name table learned from the header chunk, as a real client would.
    names: HashMap<NameId, String>,
    last_tick: u32,
}

impl ClientView {
    /// Apply every datagram of one tick's send, in order.
    fn apply(&mut self, datagrams: &[Vec<u8>]) {
        for bytes in datagrams {
            let p = SnapshotPacket::decode(bytes, HALF, 1024, 8192)
                .expect("a datagram the server sent must decode");
            self.last_tick = p.tick;
            if p.keyframe && p.is_header {
                // A keyframe replaces the world: anything not in it is
                // gone. Only the header chunk may clear, or continuation
                // chunks would wipe what the header just delivered.
                self.entities.clear();
            }
            for e in &p.names {
                self.names.insert(e.id, e.name.clone());
            }
            for item in &p.items {
                // Keyed by index, not by the full `EntityId`: an `Updated`
                // carries no generation, so keying on the whole id would
                // file it separately from the `Entered` that introduced
                // the entity. `Entered`/`Left` bracket every change of
                // tenant, so the index is unambiguous within a view.
                match item.kind {
                    ItemKind::Entered => {
                        self.entities.insert(item.key(), (item.entity, item.pos));
                    }
                    ItemKind::Updated => {
                        // Keep the identity established by `Entered`; an
                        // update carries no generation to replace it with.
                        match self.entities.get_mut(&item.key()) {
                            Some((_, pos)) => *pos = item.pos,
                            None => { self.entities.insert(item.key(), (item.entity, item.pos)); }
                        }
                    }
                    ItemKind::Left => {
                        // Only if this is still the tenant that left. A
                        // `Left` carries an authoritative generation, so a
                        // departure of the *previous* occupant of a
                        // recycled index must not evict the new one —
                        // which is reachable whenever a driver emits
                        // `Entered` before `Left` within a tick.
                        if self.entities.get(&item.key()).is_some_and(|(id, _)| *id == item.entity) {
                            self.entities.remove(&item.key());
                        }
                    }
                }
            }
        }
    }

    fn ids(&self) -> Vec<EntityId> {
        let mut v: Vec<EntityId> = self.entities.values().map(|(id, _)| *id).collect();
        v.sort_unstable_by_key(|e| (e.index, e.generation));
        v
    }

    /// The full id the client believes occupies `index`, if any.
    fn tenant(&self, index: u32) -> Option<EntityId> {
        self.entities.get(&index).map(|(id, _)| *id)
    }

    /// Position held for an entity, looked up the way a client must —
    /// by index, then checked against the identity it was introduced with.
    fn pos_of(&self, id: EntityId) -> Option<DVec2> {
        match self.entities.get(&id.index) {
            Some((held, pos)) if *held == id => Some(*pos),
            _ => None,
        }
    }
}

/// The server half: one tick of the real pipeline.
///
/// This is the composition under test — relevancy, diff, encode, chunk —
/// written the way a game's `fixed_update` would write it.
struct Server {
    world: World,
    grid: SpatialGrid,
    relevancy: Relevancy,
    scratch: AoiScratch,
    entered: Vec<EntityId>,
    left: Vec<EntityId>,
    names: Vec<NameEntry>,
}

impl Server {
    fn new() -> Self {
        Self {
            world: World::new(),
            grid: SpatialGrid::new(CELL),
            relevancy: Relevancy::new(),
            scratch: AoiScratch::new(),
            entered: Vec::new(),
            left: Vec::new(),
            names: vec![NameEntry { name: "transform2d".to_string(), id: NameId(0) }],
        }
    }

    fn spawn_at(&mut self, x: f64, y: f64) -> EntityId {
        let e = self.world.spawn();
        self.world.insert(e, Transform2D { pos: DVec2::new(x, y), rot: 0.0 });
        e
    }

    fn move_to(&mut self, e: EntityId, x: f64, y: f64) {
        if let Some(t) = self.world.get_mut::<Transform2D>(e) {
            t.pos = DVec2::new(x, y);
        }
    }

    /// Rebuild the broadphase from the world, recording the index→entity
    /// mapping as it goes — what a server tick actually does.
    ///
    /// `clear` rather than a fresh grid: slot numbering restarts from zero
    /// either way, so `Relevancy`'s insertion-order mapping is unaffected,
    /// but the buffers survive the tick.
    fn rebuild_grid(&mut self) {
        self.grid.clear();
        self.relevancy.begin();
        let mut rows: Vec<(EntityId, DVec2)> =
            self.world.iter::<Transform2D>().map(|(id, t)| (id, t.pos)).collect();
        // Deterministic insertion order, so grid indices are reproducible.
        rows.sort_unstable_by_key(|(id, _)| (id.index, id.generation));
        for (id, pos) in rows {
            self.grid.insert(pos, 1.0);
            self.relevancy.push(id);
        }
    }

    /// Who this viewpoint can see, as entity ids.
    fn visible_from(&mut self, eye: DVec2) -> Vec<EntityId> {
        self.grid.query_circle_into(eye, VIEW, 0, &mut self.scratch);
        self.scratch
            .hits
            .iter()
            .map(|&i| {
                self.relevancy
                    .entity(i)
                    .expect("every grid index must map back to an entity")
            })
            .collect()
    }

    /// Build and send one tick's packet for a client.
    fn send_tick(
        &mut self,
        tick: u32,
        eye: DVec2,
        link: &mut ClientLink,
        hint: &mut ChunkHint,
        budget: &mut KeyframeBudget,
        wire: &mut Wire,
    ) -> Vec<EntityId> {
        // One call is one tick, so the keyframe allowance resets here. A
        // budget that is never reset behaves as a whole-run allowance
        // instead: it works until the run is long enough to exhaust it,
        // and then every client silently defers forever.
        budget.begin();

        self.rebuild_grid();
        let visible = self.visible_from(eye);

        // Matched exhaustively rather than tested with `is_some`: a
        // deferred client must not fall into the delta branch, because a
        // delta against a baseline the server knows is stale corrupts the
        // client's view silently.
        let keyframe = match link.plan(tick, budget) {
            Plan::Keyframe(_) => true,
            Plan::Delta => false,
            Plan::Deferred(_) => return Vec::new(),
        };
        let items: Vec<EntityItem> = if keyframe {
            visible.iter().map(|&id| self.item(id, ItemKind::Entered)).collect()
        } else {
            let entered = {
                link.diff(&visible, &mut self.entered, &mut self.left);
                self.entered.clone()
            };
            // Departures first. A recycled index can appear as both a
            // `Left` (old tenant) and an `Entered` (new tenant) in one
            // tick, and a client keyed by index — which is all an
            // `Updated` lets it key by — would have the removal undo the
            // arrival if these came the other way round.
            self.left
                .iter()
                .map(|&id| EntityItem::left(id))
                .chain(entered.iter().map(|&id| self.item(id, ItemKind::Entered)))
                .chain(
                    // Everything still visible and not newly arrived is an
                    // update: positions move every tick.
                    visible
                        .iter()
                        .filter(|id| !entered.contains(id))
                        .map(|&id| self.item(id, ItemKind::Updated)),
                )
                .collect()
        };

        let packet = SnapshotPacket {
            tick,
            is_header: true,
            keyframe,
            your_entity: EntityId { index: 0, generation: 0 },
            names: self.names.clone(),
            items,
        };

        let outcome = send_chunked(wire, &packet, |p| p.encode(HALF), hint, &"e2e");
        assert_eq!(outcome, ChunkResult::Delivered, "tick {tick} must deliver");

        if keyframe {
            link.commit_keyframe(tick, visible.iter().copied());
        } else {
            link.commit_delta(&visible);
        }
        visible
    }

    fn item(&self, id: EntityId, kind: ItemKind) -> EntityItem {
        let pos = self
            .world
            .get::<Transform2D>(id)
            .map(|t| t.pos)
            .unwrap_or(DVec2::ZERO);
        EntityItem { kind, entity: id, pos, rot: 0.0, vel: DVec2::ZERO, component: NameId(0) }
    }
}

/// Quantisation is lossy, so positions compare within one step.
fn close(a: DVec2, b: DVec2) -> bool {
    let step = (2.0 * HALF) / ((1u64 << 16) - 1) as f64;
    (a.x - b.x).abs() <= step && (a.y - b.y).abs() <= step
}

/// **The headline property.** Across several ticks, with entities moving
/// in and out of view, what the client reconstructs from the wire is
/// exactly what the server believed that client could see.
#[test]
fn a_client_view_tracks_the_server_across_ticks() {
    let mut server = Server::new();
    let eye = DVec2::new(0.0, 0.0);

    let near_a = server.spawn_at(10.0, 0.0);
    let near_b = server.spawn_at(-20.0, 30.0);
    let far = server.spawn_at(400.0, 0.0);
    let traveller = server.spawn_at(300.0, 0.0);

    let mut link = ClientLink::new(90);
    let mut hint = ChunkHint::new();
    let mut budget = KeyframeBudget::default();
    let mut client = ClientView::default();

    // Tick 1: first contact, so a keyframe.
    let mut wire = Wire::default();
    let visible = server.send_tick(1, eye, &mut link, &mut hint, &mut budget, &mut wire);
    client.apply(&wire.sent);
    assert!(visible.contains(&near_a) && visible.contains(&near_b));
    assert!(!visible.contains(&far), "far entity must be out of range");
    assert_eq!(client.ids(), sorted(&visible), "keyframe must reproduce the server's view");
    client.acknowledge(&mut link, 1);

    // Tick 2: the traveller arrives, and everything nearby drifts.
    server.move_to(traveller, 60.0, 0.0);
    server.move_to(near_a, 15.0, 5.0);
    let mut wire = Wire::default();
    let visible = server.send_tick(2, eye, &mut link, &mut hint, &mut budget, &mut wire);
    client.apply(&wire.sent);
    assert!(visible.contains(&traveller), "the traveller came into range");
    assert_eq!(client.ids(), sorted(&visible), "a delta must keep the views in step");
    client.acknowledge(&mut link, 2);

    // Tick 3: the traveller leaves again, and must be dropped by the
    // client rather than lingering as a ghost.
    server.move_to(traveller, 480.0, 0.0);
    let mut wire = Wire::default();
    let visible = server.send_tick(3, eye, &mut link, &mut hint, &mut budget, &mut wire);
    client.apply(&wire.sent);
    assert!(!visible.contains(&traveller), "the traveller left");
    assert!(
        client.tenant(traveller.index).is_none(),
        "a departure must remove it, not leave a ghost at its last position",
    );
    assert_eq!(client.ids(), sorted(&visible));
    client.acknowledge(&mut link, 3);

    // Tick 4: nothing moves. The views must still agree.
    let mut wire = Wire::default();
    let visible = server.send_tick(4, eye, &mut link, &mut hint, &mut budget, &mut wire);
    client.apply(&wire.sent);
    assert_eq!(client.ids(), sorted(&visible), "a quiet tick must not desync anything");
}

/// Positions survive the trip, within the quantiser's step.
#[test]
fn positions_arrive_intact() {
    let mut server = Server::new();
    let eye = DVec2::ZERO;
    let e = server.spawn_at(37.5, -42.25);

    let mut link = ClientLink::new(90);
    let mut hint = ChunkHint::new();
    let mut budget = KeyframeBudget::default();
    let mut client = ClientView::default();

    let mut wire = Wire::default();
    server.send_tick(1, eye, &mut link, &mut hint, &mut budget, &mut wire);
    client.apply(&wire.sent);

    let got = client.pos_of(e).expect("the entity must have arrived");
    assert!(close(got, DVec2::new(37.5, -42.25)), "got {got:?}");
}

/// The client learns the component name table from the header chunk, and
/// it survives chunking — which is what lets a `NameId` mean anything on
/// the receiving end.
#[test]
fn the_name_table_reaches_the_client() {
    let mut server = Server::new();
    let mut link = ClientLink::new(90);
    let mut hint = ChunkHint::new();
    let mut budget = KeyframeBudget::default();
    let mut client = ClientView::default();

    server.spawn_at(0.0, 0.0);
    let mut wire = Wire::default();
    server.send_tick(1, DVec2::ZERO, &mut link, &mut hint, &mut budget, &mut wire);
    client.apply(&wire.sent);

    assert_eq!(
        client.names.get(&NameId(0)).map(String::as_str),
        Some("transform2d"),
        "the client must be able to resolve an interned component id",
    );
}

/// A keyframe large enough to chunk must still reconstruct exactly. This
/// is the composition of AoI, encoding and the chunker's split under one
/// assertion.
#[test]
fn a_chunked_keyframe_reconstructs_exactly() {
    let mut server = Server::new();
    let eye = DVec2::ZERO;

    // Pack enough entities inside the view radius to exceed one datagram.
    for i in 0..400 {
        let a = (i as f64) * 0.12;
        let r = 5.0 + (i % 90) as f64;
        server.spawn_at(r * a.cos(), r * a.sin());
    }

    let mut link = ClientLink::new(90);
    let mut hint = ChunkHint::new();
    let mut budget = KeyframeBudget::default();
    let mut client = ClientView::default();

    let mut wire = Wire::default();
    let visible = server.send_tick(1, eye, &mut link, &mut hint, &mut budget, &mut wire);
    assert!(wire.sent.len() > 1, "this keyframe should have chunked");
    client.apply(&wire.sent);

    assert_eq!(
        client.ids(),
        sorted(&visible),
        "{} entities across {} datagrams must reconstruct exactly",
        visible.len(),
        wire.sent.len(),
    );
}

/// A client that goes quiet is re-keyframed *periodically* — not every
/// tick, and not never — and whichever keyframe it eventually receives
/// restores it exactly. Recovery with no history kept on its behalf.
///
/// The first version of this test asserted a keyframe at one hand-picked
/// tick, which failed: keyframes fire every `stall_after_ticks + 1`, each
/// one resetting the window, so the chosen tick fell in a gap. The design
/// was right and the arithmetic was guessed. Asserting the guarantee
/// itself — periodic, spaced, and exact on arrival — is both correct and
/// a stronger claim than a single tick number.
#[test]
fn a_silent_client_is_periodically_restored_by_a_keyframe() {
    const WINDOW: u32 = 5;

    let mut server = Server::new();
    let eye = DVec2::ZERO;
    server.spawn_at(10.0, 10.0);
    let drifter = server.spawn_at(20.0, 0.0);

    let mut link = ClientLink::new(WINDOW);
    let mut hint = ChunkHint::new();
    let mut budget = KeyframeBudget::default();
    let mut client = ClientView::default();

    let mut wire = Wire::default();
    server.send_tick(1, eye, &mut link, &mut hint, &mut budget, &mut wire);
    client.apply(&wire.sent);
    client.acknowledge(&mut link, 1);

    // The client goes quiet here: no further acknowledgements. The world
    // keeps moving and the server keeps sending into the void.
    let mut keyframe_ticks = Vec::new();
    let mut last_keyframe: Option<(u32, Vec<Vec<u8>>, Vec<EntityId>)> = None;

    for tick in 2..=20u32 {
        server.move_to(drifter, 20.0 + (tick % 7) as f64, 0.0);
        let mut sent = Wire::default();
        let visible = server.send_tick(tick, eye, &mut link, &mut hint, &mut budget, &mut sent);

        let head = SnapshotPacket::decode(&sent.sent[0], HALF, 1024, 8192).unwrap();
        if head.keyframe {
            keyframe_ticks.push(tick);
            last_keyframe = Some((tick, sent.sent.clone(), visible));
        }
        // Nothing is applied to the client: every one of these is lost.
    }

    assert!(
        keyframe_ticks.len() >= 2,
        "a silent client must keep being offered keyframes; got {keyframe_ticks:?}",
    );
    for pair in keyframe_ticks.windows(2) {
        let gap = pair[1] - pair[0];
        assert!(
            gap > WINDOW,
            "keyframes {} and {} are {gap} apart, inside the {WINDOW}-tick window — \
             that is the send loop this design exists to avoid",
            pair[0], pair[1],
        );
    }

    // The client comes back and receives the next keyframe it is sent.
    let (tick, datagrams, visible) = last_keyframe.expect("at least one keyframe was sent");
    client.apply(&datagrams);
    assert_eq!(
        client.ids(),
        sorted(&visible),
        "the keyframe at tick {tick} must restore the client exactly, \
         with no history retained for it",
    );
}

/// A despawned index reused by a new entity must not be mistaken for the
/// old one — the case `World::despawn` makes reachable within a tick.
#[test]
fn a_recycled_index_does_not_confuse_the_client() {
    let mut server = Server::new();
    let eye = DVec2::ZERO;
    let first = server.spawn_at(10.0, 0.0);

    let mut link = ClientLink::new(90);
    let mut hint = ChunkHint::new();
    let mut budget = KeyframeBudget::default();
    let mut client = ClientView::default();

    let mut wire = Wire::default();
    server.send_tick(1, eye, &mut link, &mut hint, &mut budget, &mut wire);
    client.apply(&wire.sent);
    client.acknowledge(&mut link, 1);
    assert_eq!(client.tenant(first.index), Some(first));

    // Despawn and immediately respawn: the index comes back with a new
    // generation.
    server.world.despawn(first);
    let second = server.spawn_at(12.0, 0.0);
    assert_eq!(second.index, first.index, "the index must have been recycled");
    assert_ne!(second.generation, first.generation);

    let mut wire = Wire::default();
    let visible = server.send_tick(2, eye, &mut link, &mut hint, &mut budget, &mut wire);
    client.apply(&wire.sent);

    // One index, one tenant: the new occupant, not the old one. Before
    // `EntityItem::key` existed these were two `contains_key` calls on two
    // different keys, which is precisely the fork this test now guards.
    assert_eq!(
        client.tenant(first.index),
        Some(second),
        "the recycled index must hold the new tenant, not the old occupant",
    );
    assert_eq!(client.ids(), sorted(&visible));

    // Tick 3 is where this test used to stop being useful, and where the
    // bug it was written for actually lived.
    //
    // At tick 2 the recycled entity arrives as `Entered`, which carries a
    // generation, so every assertion above passes whatever the client keys
    // by. At tick 3 the same entity is `Updated` — which carries the index
    // only — and a client keying on the full `EntityId` files it under
    // generation 0, a key no `Entered` ever created. It then holds *two*
    // entries for one entity: the real one frozen where its keyframe left
    // it, and a ghost that moves. When the entity finally leaves, `Left`
    // carries the true generation and removes only the real one, so the
    // ghost outlives the connection.
    //
    // Every generation in these tests was 0 until a despawn happened, which
    // is why this survived: the two keys coincided.
    server.move_to(second, 14.0, 0.0);
    let mut wire = Wire::default();
    let visible = server.send_tick(3, eye, &mut link, &mut hint, &mut budget, &mut wire);
    client.apply(&wire.sent);

    assert_eq!(
        client.entities.len(),
        1,
        "an update must not fork the entity into a second entry",
    );
    assert_eq!(
        client.tenant(first.index),
        Some(second),
        "the update must land on the tenant `Entered` introduced, generation intact",
    );
    assert_eq!(
        client.pos_of(second).map(|p| p.x.round()),
        Some(14.0),
        "the update must move the entity the client actually holds",
    );
    assert_eq!(client.ids(), sorted(&visible), "views must still agree after an update");
}

fn sorted(ids: &[EntityId]) -> Vec<EntityId> {
    let mut v = ids.to_vec();
    v.sort_unstable_by_key(|e| (e.index, e.generation));
    v
}

impl ClientView {
    /// The client tells the server what it has applied.
    ///
    /// A client can only have applied a tick the server already sent, so
    /// `tick` doubles as the server's "now" here — the bound
    /// `record_ack` checks against.
    fn acknowledge(&self, link: &mut ClientLink, tick: u32) {
        let bytes = Ack { tick }.encode();
        let ack = Ack::decode(&bytes).expect("an ack must survive its own encoding");
        link.record_ack(ack, tick);
    }
}
