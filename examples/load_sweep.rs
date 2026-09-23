//! Where does a replication tick actually go, and at what client count
//! does it stop fitting?
//!
//! Everything measured in this engine so far targets **500 players on one
//! process**. The question this answers is the one that has to come before
//! any multi-node design: what is the single-node ceiling, and which phase
//! hits it first? That number decides how many nodes a given population
//! needs, which decides how much cross-node handoff traffic exists, which
//! decides whether entity migration or overlapping regions is even viable.
//!
//! Run it:
//!
//! ```text
//! cargo run --release --no-default-features --features replication --example load_sweep
//! ```
//!
//! # What it drives
//!
//! The real pipeline, in the shape `tests/replication_e2e.rs` and
//! `examples/replication_server.rs` use it — not a mock and not a
//! rearrangement:
//!
//! 1. `SpatialGrid::clear` + refill, with `Relevancy` recorded in step.
//! 2. `query_circle_into` per client, through one shared `AoiScratch`.
//! 3. `ClientLink::diff` per client, through one shared `DiffScratch`.
//! 4. Item construction, with `entered` binary-searched exactly as the
//!    drivers do.
//! 5. `send_chunked` into a counting sink, with a per-client `ChunkHint`,
//!    encoding inside the closure exactly as the drivers do.
//!
//! Point 5 is a correction. This harness originally called
//! `encode_into` *and then* `send_chunked` — which encodes again through
//! its closure — and billed them as two separate phases. Both reference
//! drivers call `send_chunked` alone, and its common path does exactly
//! one encode. So the first numbers double-counted serialisation and
//! split it across two columns that were the same work. `encode` and
//! `send` are one phase here because they are one phase in the engine.
//!
//! Deviating from that shape is how a load harness ends up describing a
//! program nobody ships. Three specific traps, all of which this engine
//! has actually fallen into and measured its way out of: `clear()` rather
//! than `new()` (3.75 ms against 3.33 on the same work), scratches reused
//! across clients rather than allocated per call (27.5 ms of pure
//! allocation at 500 clients), and `entered` binary-searched rather than
//! scanned (253 ms against 13.9 on a mass-arrival tick).
//!
//! # What it deliberately does not model
//!
//! No network. `send_chunked` writes into a sink that counts bytes and
//! drops them, so this measures CPU per tick and datagram volume, not
//! bandwidth, latency, loss, or what a real NIC does at 8,000 sockets.
//! Those are the next question, not this one.
//!
//! No game simulation either — entities move on a fixed drift, and there
//! is no physics, no collision resolution, no AI. A real server pays all
//! of that *on top* of what is measured here, so treat every figure as a
//! floor.

use std::hint::black_box;
use std::time::Instant;

use glam::DVec2;
use void_engine::collision::{AoiScratch, SpatialGrid};
use void_engine::components::{Transform2D, Velocity};
use void_engine::ecs::EntityId;
use void_engine::net::chunk::{
    send_chunked, ChunkHint, DatagramSink, SendOutcome, MIN_DATAGRAM_BUDGET,
};
use void_engine::net::replication::{
    ClientLink, DiffScratch, KeyframeBudget, Plan, Relevancy,
};
use void_engine::net::snapshot::{EntityItem, ItemKind, NameEntry, SnapshotPacket};
use void_engine::persist::registry::NameId;
use void_engine::World;

/// Half a sector at the smallest configuration, in world units.
///
/// **The sector grows with the population**, so density stays fixed
/// across a sweep. Holding it constant while scaling entities was the
/// first version of this harness and it measured the wrong thing
/// entirely: entities-per-client-view doubled every row (17 → 31 → 62 →
/// 117 → 226), so the AoI curve was density, not client count, and its
/// apparent 3.7x-per-doubling superlinearity was manufactured by the
/// measurement.
///
/// A real shard adds space as it adds players. The crowding case — many
/// players in *one* place — is a separate sweep below, because it is a
/// different question and deserves its own numbers.
const BASE_HALF: f64 = 5_000.0;
/// Entities the base sector holds. Larger runs grow the sector to match,
/// so this fixes the *density* the whole sweep shares.
const BASE_ENTITIES: f64 = 10_000.0;

/// How far a client can see. Larger than the e2e test's 120 because this
/// is measuring a populated view, not a two-entity one.
const VIEW: f64 = 400.0;

/// Grid cell size for the base density.
///
/// Scaled with density in the crowding sweep rather than held fixed.
/// `SpatialGrid::new` is explicit that mean bucket occupancy in the low
/// single digits is the target and that a badly-sized cell costs up to
/// 64x — so a fixed cell across a 16x density change would measure
/// grid mis-sizing rather than crowding.
const BASE_CELL: f64 = 200.0;

/// Half-extent for a given entity count, holding area-per-entity fixed.
fn half_for(entities: usize) -> f64 {
    BASE_HALF * (entities as f64 / BASE_ENTITIES).sqrt()
}
/// Server tick budget at 30 Hz.
const TICK_BUDGET_MS: f64 = 1000.0 / 30.0;
/// Ticks per measured sample. Enough that the first-tick keyframe is
/// amortised and the steady-state delta path dominates.
const TICKS: u32 = 8;

/// A sink that counts what it is handed and throws it away.
#[derive(Default)]
struct CountingSink {
    datagrams: u64,
    bytes: u64,
}

impl DatagramSink for CountingSink {
    fn send(&mut self, bytes: Vec<u8>) -> SendOutcome {
        if bytes.len() > MIN_DATAGRAM_BUDGET {
            return SendOutcome::TooLarge;
        }
        self.datagrams += 1;
        self.bytes += bytes.len() as u64;
        SendOutcome::Sent
    }
    fn max_datagram_size(&self) -> Option<usize> {
        Some(MIN_DATAGRAM_BUDGET)
    }
}

/// Per-phase milliseconds for one tick.
#[derive(Default, Clone, Copy)]
struct Phases {
    rebuild: f64,
    aoi: f64,
    diff: f64,
    items: f64,
    /// Encode *and* chunk-send, timed together.
    ///
    /// One column rather than two because `send_chunked` encodes through
    /// the closure it is given — the drivers never encode separately, and
    /// timing them apart meant encoding twice and reporting the same work
    /// under two headings.
    send: f64,
    /// Clients that actually received a packet this tick.
    ///
    /// **The honesty check on every per-client number above.**
    /// `KeyframeBudget` deliberately defers clients when the tick's
    /// keyframe allowance is spent, and a deferred client is sent nothing
    /// — so if most clients are deferred, the per-client phases are
    /// measured over a fraction of the population and the whole table
    /// flatters itself. A first run showed 104 datagrams at 16,000
    /// clients, which is what prompted counting this.
    served: u64,
    /// Items across every packet built this tick.
    ///
    /// `seen` reports one sampled client's *visible* set, which is not the
    /// same as what a packet carries: a delta is `left` + `entered` +
    /// updated, and on a moving world the first two are never empty. A
    /// standalone profile that assumed packets were `seen`-sized
    /// accounted for only a quarter of this column's measured cost, and
    /// this counter is what distinguishes "the encoder is slow" from "the
    /// packets are bigger than I assumed".
    items_sent: u64,
}

impl Phases {
    fn total(&self) -> f64 {
        self.rebuild + self.aoi + self.diff + self.items + self.send
    }

    /// Keep the cheapest sample of each phase, for the same reason
    /// `benches/hot_paths.rs` takes a best-of: the expensive samples on a
    /// loaded desktop are scheduler noise, and the floor is what is being
    /// bounded.
    fn keep_best(&mut self, other: Phases) {
        self.rebuild = self.rebuild.min(other.rebuild);
        self.aoi = self.aoi.min(other.aoi);
        self.diff = self.diff.min(other.diff);
        self.items = self.items.min(other.items);
        self.send = self.send.min(other.send);
        // Not a min: these are counts, and the *largest* seen in any
        // sampled tick is what says whether the timings covered the
        // population and how much work they covered.
        self.served = self.served.max(other.served);
        self.items_sent = self.items_sent.max(other.items_sent);
    }

    fn worst() -> Self {
        Self {
            rebuild: f64::MAX,
            aoi: f64::MAX,
            diff: f64::MAX,
            items: f64::MAX,
            send: f64::MAX,
            served: 0,
            items_sent: 0,
        }
    }
}

/// One client's replication state, exactly as a server would hold it.
struct Client {
    eye: DVec2,
    link: ClientLink,
    hint: ChunkHint,
}

/// The server: one world, one grid, one set of scratches, N clients.
struct Sim {
    world: World,
    grid: SpatialGrid,
    relevancy: Relevancy,
    scratch: AoiScratch,
    diff_scratch: DiffScratch,
    entered: Vec<EntityId>,
    left: Vec<EntityId>,
    names: Vec<NameEntry>,
    clients: Vec<Client>,
    movers: Vec<EntityId>,
    /// This run's sector half-extent. Positions are quantised against it,
    /// so it has to travel with the sim rather than be a global.
    half: f64,
}

impl Sim {
    /// `entities` spread over a sector of `half` half-extent, `clients` of
    /// them also being viewpoints — so every client sits where there is
    /// something to see, rather than in empty space where AoI is free.
    fn new(entities: usize, clients: usize, half: f64, cell: f64) -> Self {
        let mut world = World::new();
        let mut movers = Vec::with_capacity(entities);

        // A lattice over the sector. Sector-local, so nothing clamps on
        // encode.
        let side = (entities as f64).sqrt().ceil() as usize;
        let span = half * 2.0 * 0.98;
        let step = span / side as f64;

        for i in 0..entities {
            let (cx, cy) = ((i % side) as f64, (i / side) as f64);
            let pos = DVec2::new(-half * 0.99 + cx * step, -half * 0.99 + cy * step);
            let e = world.spawn();
            world.insert(e, Transform2D { pos, rot: 0.0 });
            // Drift, so deltas are non-empty every tick. A world that does
            // not move measures the cheapest possible diff and flatters
            // everything downstream of it.
            let a = i as f64 * 0.37;
            world.insert(
                e,
                Velocity { linear: DVec2::new(a.cos(), a.sin()) * 12.0, angular: 0.0 },
            );
            movers.push(e);
        }

        let client_states = (0..clients)
            .map(|c| {
                // Spread viewpoints across the populated lattice.
                let i = (c * entities.max(1)) / clients.max(1);
                let (cx, cy) = ((i % side) as f64, (i / side) as f64);
                Client {
                    eye: DVec2::new(-half * 0.99 + cx * step, -half * 0.99 + cy * step),
                    // Three seconds at 30 Hz before a silent client is
                    // re-keyframed, matching the reference driver.
                    link: ClientLink::new(90),
                    hint: ChunkHint::new(),
                }
            })
            .collect();

        Self {
            world,
            grid: SpatialGrid::new(cell),
            relevancy: Relevancy::new(),
            scratch: AoiScratch::new(),
            diff_scratch: DiffScratch::new(),
            entered: Vec::new(),
            left: Vec::new(),
            names: vec![NameEntry { name: "transform2d".to_string(), id: NameId(0) }],
            clients: client_states,
            movers,
            half,
        }
    }

    /// Mean bucket occupancy, so a reader can see the grid is sized
    /// honestly rather than taking the cell size on trust.
    fn mean_occupancy(&self) -> f64 {
        let cells = self.grid.occupied_cells().max(1);
        self.grid.len() as f64 / cells as f64
    }

    /// Advance every entity. Not timed as a replication phase — this is
    /// the game's cost, and the harness is measuring replication.
    fn drift(&mut self, dt: f64) {
        for &e in &self.movers {
            let v = self.world.get::<Velocity>(e).map(|v| v.linear).unwrap_or(DVec2::ZERO);
            if let Some(t) = self.world.get_mut::<Transform2D>(e) {
                t.pos += v * dt;
                // Keep everything sector-local by bouncing at the edge,
                // rather than letting positions drift past `HALF` where
                // the quantiser would clamp and the packet would stop
                // resembling a real one.
                let lim = self.half * 0.99;
                if t.pos.x.abs() > lim {
                    t.pos.x = t.pos.x.clamp(-lim, lim);
                }
                if t.pos.y.abs() > lim {
                    t.pos.y = t.pos.y.clamp(-lim, lim);
                }
            }
        }
    }

    fn item(&self, id: EntityId, kind: ItemKind) -> EntityItem {
        let (pos, vel) = (
            self.world.get::<Transform2D>(id).map(|t| t.pos).unwrap_or(DVec2::ZERO),
            self.world.get::<Velocity>(id).map(|v| v.linear).unwrap_or(DVec2::ZERO),
        );
        // This sweep drives a 2D world, so z is zero and the quaternion is
        // identity. They still cost their wire bits — which is the point
        // of measuring with them present.
        EntityItem {
            kind,
            entity: id,
            pos: pos.extend(0.0),
            rot: glam::Quat::IDENTITY,
            vel: vel.extend(0.0),
            component: NameId(0),
        }
    }

    /// One full server tick for every client, timed by phase.
    fn tick(&mut self, tick: u32, sink: &mut CountingSink) -> Phases {
        let mut p = Phases::default();
        // Budget wide enough to serve everyone, deliberately unlike a live
        // server.
        //
        // `KeyframeBudget::default()` is 13 per tick — about 5 ms of
        // keyframes, sized to protect a real tick from a reconnect storm.
        // That is right for production and wrong for measurement: every
        // client here needs a first keyframe, so at 13/tick a 16,000-client
        // run would take 1,231 ticks before steady state, and sampling at
        // tick 3 measured a nearly-idle server. The first run of this
        // harness reported 104 datagrams at 16,000 clients for exactly
        // that reason.
        //
        // So the budget is lifted to cover the population and the
        // `served` column proves it worked. What this gives up is any
        // claim about keyframe-storm behaviour — that is a separate
        // question and needs the real budget to ask.
        let mut budget = KeyframeBudget::new(self.clients.len() as u32 + 1);
        budget.begin();

        // ── 1. broadphase rebuild, with the relevancy mapping in step ──
        let t0 = Instant::now();
        self.grid.clear();
        self.relevancy.begin();
        let mut rows: Vec<(EntityId, DVec2)> =
            self.world.iter::<Transform2D>().map(|(id, t)| (id, t.pos)).collect();
        rows.sort_unstable_by_key(|(id, _)| (id.index, id.generation));
        for (id, pos) in rows {
            self.grid.insert(pos, 1.0);
            self.relevancy.push(id);
        }
        p.rebuild = t0.elapsed().as_secs_f64() * 1000.0;

        // Per-client work. Indices rather than an iterator because each
        // phase borrows a different part of `self`.
        for ci in 0..self.clients.len() {
            let eye = self.clients[ci].eye;

            // ── 2. area of interest ────────────────────────────────────
            let t = Instant::now();
            self.grid.query_circle_into(eye, VIEW, 0, &mut self.scratch);
            let visible: Vec<EntityId> = self
                .scratch
                .hits
                .iter()
                .filter_map(|&i| self.relevancy.entity(i))
                .collect();
            p.aoi += t.elapsed().as_secs_f64() * 1000.0;

            let keyframe = match self.clients[ci].link.plan(tick, &mut budget) {
                Plan::Keyframe(_) => true,
                Plan::Delta => false,
                // A deferred client is sent nothing, which is the whole
                // point of the budget — and it must be counted as a real
                // outcome, not skipped as if it had no cost.
                Plan::Deferred(_) => continue,
            };
            p.served += 1;

            // ── 3. diff against the client's baseline ──────────────────
            let t = Instant::now();
            if !keyframe {
                self.clients[ci].link.diff(
                    &visible,
                    &mut self.entered,
                    &mut self.left,
                    &mut self.diff_scratch,
                );
            }
            p.diff += t.elapsed().as_secs_f64() * 1000.0;

            // ── 4. build the item list ─────────────────────────────────
            let t = Instant::now();
            let items: Vec<EntityItem> = if keyframe {
                visible.iter().map(|&id| self.item(id, ItemKind::Entered)).collect()
            } else {
                let entered = self.entered.clone();
                self.left
                    .iter()
                    .map(|&id| EntityItem::left(id))
                    .chain(entered.iter().map(|&id| self.item(id, ItemKind::Entered)))
                    .chain(
                        visible
                            .iter()
                            .filter(|id| {
                                entered
                                    .binary_search_by_key(&(id.index, id.generation), |e| {
                                        (e.index, e.generation)
                                    })
                                    .is_err()
                            })
                            .map(|&id| self.item(id, ItemKind::Updated)),
                    )
                    .collect()
            };
            p.items += t.elapsed().as_secs_f64() * 1000.0;
            p.items_sent += items.len() as u64;

            let packet = SnapshotPacket {
                tick,
                is_header: true,
                keyframe,
                your_entity: EntityId { index: ci as u32, generation: 1 },
                names: self.names.clone(),
                items,
            };

            // ── 5. encode and send, as one phase ───────────────────────
            //
            // `send_chunked` encodes through this closure; the drivers do
            // not encode separately, and neither does this.
            let t = Instant::now();
            let half = self.half;
            let hint = &mut self.clients[ci].hint;
            let _ = send_chunked(
                sink,
                &packet,
                |chunk: &SnapshotPacket| black_box(chunk.encode(half)),
                hint,
                &"load",
            );
            p.send += t.elapsed().as_secs_f64() * 1000.0;

            if keyframe {
                self.clients[ci].link.commit_keyframe(tick, visible.iter().copied());
            } else {
                self.clients[ci].link.commit_delta(&visible);
            }
            // Acknowledge immediately: this harness measures the
            // steady-state delta path, and a client that never acks would
            // be re-keyframed every 90 ticks and measure that instead.
            self.clients[ci].link.record_ack(
                void_engine::net::replication::Ack { tick },
                tick,
            );
        }

        p
    }
}

/// Run one configuration and report where the tick went.
fn run(entities: usize, clients: usize, half: f64, cell: f64) -> (Phases, u64, u64, usize, f64) {
    let mut sim = Sim::new(entities, clients, half, cell);
    let mut sink = CountingSink::default();
    let mut best = Phases::worst();
    let mut visible_sample = 0usize;

    for tick in 1..=TICKS {
        sim.drift(1.0 / 30.0);
        let mut per_tick_sink = CountingSink::default();
        let p = sim.tick(tick, &mut per_tick_sink);
        // Skip the first two ticks: tick 1 is all keyframes, tick 2 still
        // carries their commit. Steady state is what a server lives in.
        if tick > 2 {
            best.keep_best(p);
            sink.datagrams = per_tick_sink.datagrams;
            sink.bytes = per_tick_sink.bytes;
        }
    }

    // One more AoI query, untimed, to report how much a client actually
    // sees — the number that explains everything else, and the one that
    // exposed the first version of this harness as measuring density
    // rather than population.
    if let Some(c) = sim.clients.first() {
        sim.grid.query_circle_into(c.eye, VIEW, 0, &mut sim.scratch);
        visible_sample = sim.scratch.hits.len();
    }
    let occ = sim.mean_occupancy();

    (best, sink.datagrams, sink.bytes, visible_sample, occ)
}

/// Print one table of results.
fn table(title: &str, note: &str, rows: &[(usize, usize, f64, f64)]) {
    println!("\n{title}");
    println!("{note}\n");
    println!(
        "{:>7} {:>8} {:>7} {:>6} {:>7} {:>5} | {:>7} {:>7} {:>6} {:>6} {:>9} | {:>9} {:>7}",
        "clients", "entities", "served", "seen", "items/pkt", "occ", "rebuild", "aoi", "diff",
        "items", "encode+send", "total", "of tick",
    );
    println!("{}", "-".repeat(122));

    for &(clients, entities, half, cell) in rows {
        let (p, datagrams, bytes, seen, occ) = run(entities, clients, half, cell);
        let total = p.total();
        let per_pkt = if p.served > 0 { p.items_sent as f64 / p.served as f64 } else { 0.0 };
        println!(
            "{:>7} {:>8} {:>7} {:>6} {:>9.1} {:>5.1} | {:>7.2} {:>7.2} {:>6.2} {:>6.2} {:>9.2} | {:>8.1}ms {:>6.0}%",
            clients, entities, p.served, seen, per_pkt, occ,
            p.rebuild, p.aoi, p.diff, p.items, p.send,
            total, total / TICK_BUDGET_MS * 100.0,
        );
        if (p.served as usize) < clients {
            println!(
                "{:>29} ! only {} of {} clients served — the rest were deferred by the\n\
                 {:>29}   keyframe budget, so the per-client phases above cover a fraction\n\
                 {:>29}   of the population and this row reads low.",
                "", p.served, clients, "", "",
            );
        }
        if total > TICK_BUDGET_MS {
            println!(
                "{:>29} ^ over by {:.1} ms — {} datagrams, {} KiB per tick",
                "",
                total - TICK_BUDGET_MS,
                datagrams,
                bytes / 1024,
            );
        }
    }
}

fn main() {
    println!("Replication tick cost — one process, 30 Hz, budget {TICK_BUDGET_MS:.1} ms");
    println!(
        "Steady-state delta path; first two ticks discarded. No physics, no AI,\n\
         no network — every number is a floor. `seen` is entities in one\n\
         client's view; `occ` is mean grid bucket occupancy."
    );

    // ── population scaling, fixed density ────────────────────────────
    //
    // The sector grows with the entity count, so each client sees about
    // the same amount however many clients there are. This answers "does
    // the pipeline scale with population?" — which is the question a
    // multi-node design actually needs answered.
    let spread: Vec<(usize, usize, f64, f64)> = [500usize, 1_000, 2_000, 4_000, 8_000, 16_000]
        .iter()
        .map(|&c| {
            let e = c * 20;
            (c, e, half_for(e), BASE_CELL)
        })
        .collect();
    table(
        "A. Population scaling — density held fixed, sector grows with the crowd",
        "What a shard looks like as it fills up: more players, more space.",
        &spread,
    );

    // ── crowding, fixed sector ───────────────────────────────────────
    //
    // Everyone into one system. The cell scales with density to keep
    // bucket occupancy in the band `SpatialGrid::new` documents, so this
    // measures crowding rather than a mis-sized grid.
    let base_half = half_for(10_000);
    let crowd: Vec<(usize, usize, f64, f64)> = [500usize, 1_000, 2_000, 4_000, 8_000]
        .iter()
        .map(|&c| {
            let e = c * 20;
            // Density rises as e/10_000, so shrink the cell by its square
            // root to hold occupancy roughly constant.
            let cell = BASE_CELL / (e as f64 / BASE_ENTITIES).sqrt();
            (c, e, base_half, cell)
        })
        .collect();
    table(
        "B. Crowding — one fixed sector, population rising into it",
        "The fleet-fight case: everyone in the same place. Cell size scales with\n\
         density so this measures crowding, not a badly-sized grid.",
        &crowd,
    );

    println!(
        "\nRead the dominant column, not the total. In A it says whether adding\n\
         nodes by region helps at all; in B it says what happens when region\n\
         splitting cannot help, because everyone is in one region."
    );
}
