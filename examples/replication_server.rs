//! Replication against a real QUIC connection, both ends in one process.
//!
//! Everything else in the replication stack is tested against a
//! `FakeSink` with a pretend MTU. That is the right way to test the
//! escalation policy, and it cannot tell you whether the pieces work
//! against a socket — whether a datagram survives the path, whether an
//! acknowledgement finds its way back, whether the threading holds.
//! This runs the whole thing over loopback so that question has an
//! answer.
//!
//! ```text
//! cargo run --release --no-default-features --features replication \
//!     --example replication_server
//! ```
//!
//! # Why two threads
//!
//! [`App::fixed_update`] is synchronous and stays that way: making it
//! async would infect every game's simulation code, which is the whole
//! reason the headless split exists. So the simulation owns a thread and
//! runs [`run_headless_with`] on it, exactly as a dedicated server would.
//!
//! The network side owns a second thread with a current-thread tokio
//! runtime. It has to: accepting a connection and reading a stream are
//! async, and this axis resolves `tokio/rt` but not `rt-multi-thread`,
//! so there is one runtime on one thread by construction.
//!
//! What makes that split cheap is a fact worth stating, because it is
//! not obvious and it decides the design: **`send_datagram` does not
//! need a runtime.** It is synchronous and callable from any thread, so
//! the simulation sends snapshots directly through [`QuinnSink`] on its
//! own thread, with no channel and no hop. Only the acknowledgement path
//! — `accept_uni`, then an async read — is driven on the runtime, and
//! acks cross back over an ordinary `mpsc`.
//!
//! [`App::fixed_update`]: void_engine::App::fixed_update
//! [`run_headless_with`]: void_engine::app_headless::run_headless_with
//! [`QuinnSink`]: void_engine::net::chunk::QuinnSink
//!
//! # What it demonstrates
//!
//! A world of drifting entities, one connected client, and a per-tick
//! pipeline: area-of-interest query, delta against what the client has
//! acknowledged, quantised encode, chunked send. The client applies what
//! arrives, acknowledges the tick, and the two views are compared at the
//! end. They should agree.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{mpsc, Arc};
use std::time::Duration;

use glam::DVec2;
use void_engine::app_headless::{run_headless_with, HeadlessConfig};
use void_engine::collision::{AoiScratch, SpatialGrid};
use void_engine::components::{Transform2D, Velocity};
use void_engine::ecs::EntityId;
use void_engine::net::chunk::{send_chunked, ChunkHint, ChunkResult, QuinnSink};
use void_engine::net::quic::{insecure_skip_verification, server_endpoint, CertSource};
use void_engine::net::replication::{
    Ack, ClientLink, DiffScratch, KeyframeBudget, Plan, Relevancy, MAX_ACK_BYTES,
};
use void_engine::net::snapshot::{EntityItem, ItemKind, NameEntry, SnapshotPacket};
use void_engine::persist::registry::NameId;
use void_engine::{App, SimCtx};

/// Protocol name. Versioned so an incompatible change is a clean
/// handshake rejection rather than a garbled decode.
const ALPN: &[u8] = b"void-engine-replication/1";
/// Half a sector: the range sector-local positions are quantised over.
const HALF_EXTENT: f64 = 500.0;
/// How far a client can see.
const VIEW_RADIUS: f64 = 150.0;
/// Grid cell size. Sized for a low single-digit bucket at this density —
/// see `SpatialGrid::new`, where getting this wrong costs 64x.
const CELL: f64 = 32.0;
/// Ticks to run before reporting.
const TICKS: u64 = 120;
/// Entities in the world.
const ENTITIES: usize = 400;

fn main() {
    // A client that has applied nothing yet. Shared so the simulation can
    // read acknowledgements the network thread receives.
    let acked = Arc::new(AtomicU32::new(0));
    let running = Arc::new(AtomicBool::new(true));
    let (conn_tx, conn_rx) = mpsc::channel::<quinn::Connection>();
    let (report_tx, report_rx) = mpsc::channel::<ClientReport>();

    let net_acked = acked.clone();
    let net_running = running.clone();
    let net = std::thread::spawn(move || {
        net_thread(conn_tx, report_tx, net_acked, net_running);
    });

    // Wait for the server side of the connection before simulating.
    let conn = match conn_rx.recv_timeout(Duration::from_secs(10)) {
        Ok(c) => c,
        Err(_) => {
            eprintln!("no connection within 10s; giving up");
            running.store(false, Ordering::Relaxed);
            let _ = net.join();
            return;
        }
    };
    println!("connected: {} -> {}", conn.remote_address(), ALPN.escape_ascii());

    let sim = Server::new(conn, acked.clone());
    let cfg = HeadlessConfig { hz: 30.0, max_ticks: Some(TICKS), uncapped: false };
    let stats = sim.stats.clone();
    run_headless_with(sim, cfg, || true);

    // Let the last datagrams land before tearing the connection down.
    std::thread::sleep(Duration::from_millis(200));
    running.store(false, Ordering::Relaxed);

    let report = report_rx.recv_timeout(Duration::from_secs(5)).ok();
    let _ = net.join();

    let s = stats.lock().unwrap();
    println!();
    println!("ticks simulated      : {}", s.ticks);
    println!("keyframes sent       : {}", s.keyframes);
    println!("deltas sent          : {}", s.deltas);
    println!("datagrams sent       : {}", s.datagrams);
    println!("bytes sent           : {}", s.bytes);
    println!("undeliverable        : {}", s.undeliverable);
    println!("last acked tick      : {}", acked.load(Ordering::Relaxed));
    match report {
        Some(r) => {
            println!("client entities      : {}", r.entities);
            println!("server visible (last): {}", s.last_visible);
            println!(
                "views agree          : {}",
                if r.entities == s.last_visible { "yes" } else { "NO" }
            );
        }
        None => println!("client report        : none received"),
    }
}

/// What the client rebuilt, reported back at shutdown.
struct ClientReport {
    entities: usize,
}

#[derive(Default)]
struct Stats {
    ticks: u64,
    keyframes: u64,
    deltas: u64,
    datagrams: u64,
    bytes: u64,
    undeliverable: u64,
    last_visible: usize,
}

/// The simulation half: one tick of the real pipeline, written the way a
/// game's `fixed_update` would write it.
struct Server {
    conn: quinn::Connection,
    acked: Arc<AtomicU32>,
    tick: u32,
    grid: SpatialGrid,
    relevancy: Relevancy,
    scratch: AoiScratch,
    link: ClientLink,
    hint: ChunkHint,
    budget: KeyframeBudget,
    entered: Vec<EntityId>,
    left: Vec<EntityId>,
    /// Held across ticks: `diff`'s membership set, reused rather than
    /// rebuilt per call. See `ClientLink::diff`.
    diff_scratch: DiffScratch,
    names: Vec<NameEntry>,
    eye: DVec2,
    stats: Arc<std::sync::Mutex<Stats>>,
}

impl Server {
    fn new(conn: quinn::Connection, acked: Arc<AtomicU32>) -> Self {
        Self {
            conn,
            acked,
            tick: 0,
            grid: SpatialGrid::new(CELL),
            relevancy: Relevancy::new(),
            scratch: AoiScratch::new(),
            // Three seconds at 30 Hz before a silent client is re-keyframed.
            link: ClientLink::new(90),
            hint: ChunkHint::new(),
            budget: KeyframeBudget::default(),
            entered: Vec::new(),
            left: Vec::new(),
            diff_scratch: DiffScratch::new(),
            names: vec![NameEntry { name: "transform2d".to_string(), id: NameId(0) }],
            eye: DVec2::ZERO,
            stats: Arc::new(std::sync::Mutex::new(Stats::default())),
        }
    }

    /// Rebuild the broadphase, recording which entity each grid index
    /// means — what a server tick actually pays.
    ///
    /// `clear` rather than a fresh grid: slot numbering restarts from zero
    /// either way, so `Relevancy`'s insertion-order mapping is unaffected,
    /// but the buffers survive the tick. Measured at 100k colliders on a
    /// 40-unit cell, the rebuild alone goes 10.95 ms to 3.00 ms.
    fn rebuild_grid(&mut self, world: &void_engine::World) {
        self.grid.clear();
        self.relevancy.begin();
        let mut rows: Vec<(EntityId, DVec2)> =
            world.iter::<Transform2D>().map(|(id, t)| (id, t.pos)).collect();
        rows.sort_unstable_by_key(|(id, _)| (id.index, id.generation));
        for (id, pos) in rows {
            self.grid.insert(pos, 1.0);
            self.relevancy.push(id);
        }
    }

    fn item(&self, world: &void_engine::World, id: EntityId, kind: ItemKind) -> EntityItem {
        let (pos, vel) = (
            world.get::<Transform2D>(id).map(|t| t.pos).unwrap_or(DVec2::ZERO),
            world.get::<Velocity>(id).map(|v| v.linear).unwrap_or(DVec2::ZERO),
        );
        EntityItem { kind, entity: id, pos, rot: 0.0, vel, component: NameId(0) }
    }
}

impl App for Server {
    fn init(&mut self, ctx: &mut SimCtx) {
        // A ring of drifting entities, most of them inside the view.
        for i in 0..ENTITIES {
            let a = i as f64 * 0.11;
            let r = 10.0 + (i % 200) as f64;
            let e = ctx.world.spawn();
            ctx.world.insert(e, Transform2D { pos: DVec2::new(r * a.cos(), r * a.sin()), rot: 0.0 });
            ctx.world.insert(
                e,
                Velocity { linear: DVec2::new(-a.sin(), a.cos()) * 8.0, angular: 0.0 },
            );
        }
    }

    fn fixed_update(&mut self, ctx: &mut SimCtx) {
        self.tick = self.tick.wrapping_add(1);
        void_engine::physics::integrate(ctx.world, ctx.dt, 0.0, 0.0);

        // Acknowledgements arrive on the network thread; fold in whatever
        // has landed since the last tick.
        let acked = self.acked.load(Ordering::Relaxed);
        if acked > 0 {
            // `self.tick` is the bound: a client cannot have applied a
            // tick this server has not sent. Without it a forged
            // `u32::MAX` would pin the stall detector open forever.
            self.link.record_ack(Ack { tick: acked }, self.tick);
        }

        self.rebuild_grid(ctx.world);
        self.grid.query_circle_into(self.eye, VIEW_RADIUS, 0, &mut self.scratch);
        let visible: Vec<EntityId> = self
            .scratch
            .hits
            .iter()
            .filter_map(|&i| self.relevancy.entity(i))
            .collect();

        // One call is one tick, so the keyframe allowance resets here.
        self.budget.begin();

        // Matched exhaustively: a deferred client must not fall into the
        // delta branch, because a delta against a baseline the server
        // knows is stale corrupts the client's view silently.
        let keyframe = match self.link.plan(self.tick, &mut self.budget) {
            Plan::Keyframe(_) => true,
            Plan::Delta => false,
            Plan::Deferred(_) => return,
        };

        let items: Vec<EntityItem> = if keyframe {
            visible.iter().map(|&id| self.item(ctx.world, id, ItemKind::Entered)).collect()
        } else {
            self.link.diff(&visible, &mut self.entered, &mut self.left, &mut self.diff_scratch);
            let entered = self.entered.clone();
            let left = self.left.clone();
            // Departures first: a recycled index can be both a `Left` (old
            // tenant) and an `Entered` (new one) in the same tick, and an
            // index-keyed client would have the removal undo the arrival
            // if these came the other way round. See `EntityItem::key`.
            left.iter()
                .map(|&id| EntityItem::left(id))
                .chain(entered.iter().map(|&id| self.item(ctx.world, id, ItemKind::Entered)))
                .chain(
                    // `entered` comes back sorted from `diff`, so this is a
                    // binary search rather than a scan per visible entity.
                    // At 500 clients on a mass-arrival tick the scan
                    // measured 271 ms against a 33.3 ms budget.
                    visible
                        .iter()
                        .filter(|id| {
                            entered.binary_search_by_key(&(id.index, id.generation), |e| {
                                (e.index, e.generation)
                            })
                            .is_err()
                        })
                        .map(|&id| self.item(ctx.world, id, ItemKind::Updated)),
                )
                .collect()
        };

        let packet = SnapshotPacket {
            tick: self.tick,
            is_header: true,
            keyframe,
            your_entity: EntityId { index: 0, generation: 0 },
            names: if keyframe { self.names.clone() } else { Vec::new() },
            items,
        };

        // The send itself. `send_datagram` needs no runtime, so this
        // happens on the simulation thread with no hop.
        let mut sent_bytes = 0u64;
        let mut sent_count = 0u64;
        let outcome = {
            let mut sink = QuinnSink(&self.conn);
            send_chunked(
                &mut sink,
                &packet,
                |p| {
                    let bytes = p.encode(HALF_EXTENT);
                    sent_bytes += bytes.len() as u64;
                    sent_count += 1;
                    bytes
                },
                &mut self.hint,
                &format_args!("tick {}", self.tick),
            )
        };

        if keyframe {
            self.link.commit_keyframe(self.tick, visible.iter().copied());
        } else {
            self.link.commit_delta(&visible);
        }

        let mut s = self.stats.lock().unwrap();
        s.ticks += 1;
        s.last_visible = visible.len();
        if keyframe { s.keyframes += 1 } else { s.deltas += 1 }
        s.bytes += sent_bytes;
        s.datagrams += sent_count;
        if outcome == ChunkResult::Undeliverable {
            s.undeliverable += 1;
        }
    }
}

// ── the network thread ──────────────────────────────────────────────────

fn net_thread(
    conn_tx: mpsc::Sender<quinn::Connection>,
    report_tx: mpsc::Sender<ClientReport>,
    acked: Arc<AtomicU32>,
    running: Arc<AtomicBool>,
) {
    let rt = match tokio::runtime::Builder::new_current_thread().enable_all().build() {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("could not build a runtime: {e}");
            return;
        }
    };

    rt.block_on(async move {
        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let endpoint = match server_endpoint(addr, ALPN, CertSource::self_signed_localhost()) {
            Ok(ep) => ep,
            Err(e) => {
                eprintln!("server endpoint: {e}");
                return;
            }
        };
        let server_addr = endpoint.local_addr().expect("bound endpoint has an address");

        // Accept the client and hand the connection to the simulation.
        let acked_for_reads = acked.clone();
        let accept = tokio::spawn(async move {
            let Some(incoming) = endpoint.accept().await else { return };
            let conn = match incoming.await {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("handshake: {e}");
                    return;
                }
            };
            let _ = conn_tx.send(conn.clone());

            // Read acknowledgements until the connection closes. Each
            // arrives as its own uni stream carrying one framed message.
            // Ends when `accept_uni` errors, which is how a closed
            // connection reports itself.
            while let Ok(mut recv) = conn.accept_uni().await {
                match void_engine::net::framing::read_msg(&mut recv, MAX_ACK_BYTES).await {
                    Ok(bytes) => match Ack::decode(&bytes) {
                        Some(ack) => {
                            // Monotonic: an older ack carries no
                            // information, so never move backwards.
                            let prev = acked_for_reads.load(Ordering::Relaxed);
                            if ack.tick > prev {
                                acked_for_reads.store(ack.tick, Ordering::Relaxed);
                            }
                        }
                        None => eprintln!("malformed ack: {} bytes", bytes.len()),
                    },
                    Err(e) => {
                        if e.kind() != std::io::ErrorKind::UnexpectedEof {
                            eprintln!("ack read: {e}");
                        }
                        break;
                    }
                }
            }
        });

        // The client half, in the same process.
        let client = tokio::spawn(async move { client_task(server_addr, report_tx, running).await });

        let _ = client.await;
        accept.abort();
    });
}

/// A deliberately dumb client: a map of what it has been told, rebuilt
/// only from datagrams. It knows nothing except what arrived on the
/// wire, so if its view matches the server's, the wire carried
/// everything it needed to.
async fn client_task(
    server_addr: SocketAddr,
    report_tx: mpsc::Sender<ClientReport>,
    running: Arc<AtomicBool>,
) {
    let mut endpoint = match quinn::Endpoint::client("127.0.0.1:0".parse().unwrap()) {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("client endpoint: {e}");
            return;
        }
    };
    // Dev only, and loudly named so it cannot be reached by accident: the
    // server above mints a throwaway self-signed certificate, which
    // nothing can verify. A shipped client uses `client_config` against a
    // real certificate.
    let cfg = match insecure_skip_verification(ALPN) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("client config: {e}");
            return;
        }
    };
    endpoint.set_default_client_config(cfg);

    let conn = match endpoint.connect(server_addr, "localhost") {
        Ok(c) => match c.await {
            Ok(c) => c,
            Err(e) => {
                eprintln!("connect: {e}");
                return;
            }
        },
        Err(e) => {
            eprintln!("connect: {e}");
            return;
        }
    };

    // Keyed by entity index: an `Updated` carries no generation, so the
    // full `EntityId` would file it under a different key than the
    // `Entered` that introduced the entity. See `EntityItem::key`.
    let mut entities: HashMap<u32, DVec2> = HashMap::new();
    let mut names: HashMap<NameId, String> = HashMap::new();
    let mut applied = 0u32;

    while running.load(Ordering::Relaxed) {
        // `timeout` rather than `select!`: this axis resolves `tokio/time`
        // but not `tokio/macros`, so the macro does not exist here. It is
        // the better fit anyway — one future, no cancel-safety question.
        let bytes = match tokio::time::timeout(
            Duration::from_millis(100),
            conn.read_datagram(),
        )
        .await
        {
            // A datagram arrived.
            Ok(Ok(bytes)) => bytes,
            // The connection closed: stop reading.
            Ok(Err(_)) => break,
            // Nothing this interval; re-check `running` and wait again.
            Err(_) => continue,
        };

        let packet = match SnapshotPacket::decode(&bytes, HALF_EXTENT, 1024, 8192) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("decode: {e}");
                continue;
            }
        };

        // A keyframe replaces the world. Only the header chunk may clear,
        // or a continuation chunk would wipe what the header delivered.
        if packet.keyframe && packet.is_header {
            entities.clear();
        }
        for entry in &packet.names {
            names.insert(entry.id, entry.name.clone());
        }
        for item in &packet.items {
            // Keyed by index: an `Updated` carries no generation, so the
            // full `EntityId` would file it under a different key than the
            // `Entered` that introduced this entity. See
            // `EntityItem::key`.
            match item.kind {
                ItemKind::Entered | ItemKind::Updated => {
                    entities.insert(item.key(), item.pos);
                }
                ItemKind::Left => {
                    entities.remove(&item.key());
                }
            }
        }

        // Acknowledge the tick once, after applying it. Sending a fresh
        // uni stream per ack keeps this example's framing obvious; a real
        // client would keep one stream open.
        if packet.tick > applied {
            applied = packet.tick;
            if let Ok(mut send) = conn.open_uni().await {
                let _ = void_engine::net::framing::write_msg(&mut send, &Ack { tick: applied }.encode())
                    .await;
                let _ = send.finish();
            }
        }
    }

    println!("client applied up to tick {applied}, holding {} entities", entities.len());
    if let Some(name) = names.get(&NameId(0)) {
        println!("client resolved component id 0 as {name:?}");
    }
    let _ = report_tx.send(ClientReport { entities: entities.len() });
}
