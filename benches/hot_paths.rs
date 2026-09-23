//! Regression guards for the hot paths the MMO-readiness audit measured.
//!
//! These are not micro-benchmarks for tuning; they are *ceilings*. Each one
//! asserts a budget generous enough that ordinary machine-to-machine variance
//! and CI noise pass, but tight enough that an order-of-magnitude regression
//! (an accidental clone in an inner loop, a broadphase rebuilt twice per tick)
//! fails the build instead of being discovered in a profile months later.
//!
//! `cargo bench` prints the numbers and **exits non-zero if any budget is
//! blown**, so CI enforces these by running it as an ordinary step.
//!
//! Baselines recorded on the audit machine, release build:
//!
//! | Path                                    | Measured   | Was       |
//! | --------------------------------------- | ---------- | --------- |
//! | `iter2`, 50k entities x 20 systems      |  1.17 ms   | 11.32 ms  |
//! | same workload over contiguous arrays    |  0.39 ms   |  0.39 ms  |
//! | `iter2`, 250k entities, 1 system        |  0.43 ms   |  2.73 ms  |
//! | collision rebuild+query, 10k colliders  |  3.68 ms   |  5.07 ms  |
//! | AoI, 100k colliders x 1000 clients      |  7.28 ms   |         — |
//!
//! The "Was" column is what the audit measured, when `iter`/`iter2` still
//! collected each query into a heap-allocated `Vec` of raw pointers. Making
//! them lazy closed the gap to contiguous arrays from 28.7x to 2.8x.

use std::hint::black_box;
use std::time::Instant;

use glam::DVec2;
use void_engine::collision::{AoiScratch, SpatialGrid};
use void_engine::components::{Transform2D, Velocity};
use void_engine::World;

#[cfg(feature = "replication")]
use void_engine::ecs::EntityId;
#[cfg(feature = "replication")]
use void_engine::net::bitpack::BitWriter;
#[cfg(feature = "replication")]
use void_engine::net::snapshot::{EntityItem, ItemKind, SnapshotPacket};
#[cfg(feature = "replication")]
use void_engine::persist::registry::NameId;

/// Run `f` `iters` times and return the best per-iteration time in ms.
///
/// Best-of rather than mean: we are bounding the achievable cost, and the
/// worst samples on a shared CI runner are scheduler noise, not the code.
fn best_ms(iters: u32, mut f: impl FnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..iters {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64() * 1000.0);
    }
    best
}

fn world_with(n: usize) -> World {
    let mut w = World::new();
    for i in 0..n {
        let e = w.spawn();
        w.insert(e, Transform2D { pos: DVec2::new(i as f64, 0.0), rot: 0.0 });
        w.insert(e, Velocity { linear: DVec2::ONE, angular: 0.0 });
    }
    w
}

/// The dominant per-tick cost in any real game loop: many systems, each
/// running its own query. `iter`/`iter2` allocate a pointer `Vec` per call,
/// so this scales with (entities x systems), not entities alone.
fn ecs_many_systems() -> f64 {
    let w = world_with(50_000);
    best_ms(5, || {
        let mut acc = 0.0;
        for _ in 0..20 {
            for (_id, t, v) in w.iter2::<Transform2D, Velocity>() {
                acc += black_box(t.pos.x + v.linear.x);
            }
        }
        black_box(acc);
    })
}

/// A single wide query. Guards the per-entity cost independently of the
/// per-query overhead that `ecs_many_systems` dominates.
fn ecs_wide_single_query() -> f64 {
    let w = world_with(250_000);
    best_ms(5, || {
        let mut acc = 0.0;
        for (_id, t, v) in w.iter2::<Transform2D, Velocity>() {
            acc += black_box(t.pos.x + v.linear.x);
        }
        black_box(acc);
    })
}

/// Broadphase rebuild + pair query, which is what a server tick actually
/// pays.
///
/// Times the *fresh-grid* shape deliberately, not `clear`. A driver should
/// use `SpatialGrid::clear` — it retains the buckets and measured 3.75 ms
/// against 3.33 ms here, and 10.95 to 3.00 on a denser cell — but this
/// guard exists to catch algorithmic regressions in insert and
/// `query_pairs`, and allocating fresh is the arm that keeps the two
/// separable. `clear` has its own correctness tests in `collision.rs`.
fn collision_rebuild_and_query(n: usize) -> f64 {
    // Spread colliders over a grid roughly `cell_size` apart so bucket
    // occupancy stays realistic rather than degenerate.
    //
    // Spacing is 20 against a radius of 12, so neighbours actually
    // overlap: 39,402 pairs at 10k. It used to be 30, which put every
    // neighbour beyond the 24-unit reach of two radius-12 squares — the
    // grid returned 55,552 same-cell pairs that could not touch, and once
    // `query_pairs` began rejecting those the workload emitted *nothing*.
    // A guard that only exercises rejection would miss a regression in
    // emission entirely, so the lattice is tightened to produce real
    // pairs and time the whole path.
    let side = (n as f64).sqrt().ceil() as usize;
    best_ms(5, || {
        let mut g = SpatialGrid::new(40.0);
        for i in 0..n {
            let (x, y) = ((i % side) as f64 * 20.0, (i / side) as f64 * 20.0);
            g.insert(DVec2::new(x, y), 12.0);
        }
        black_box(g.query_pairs().len());
    })
}

/// Area-of-interest: one true-radius query per connected client, per tick.
///
/// The budget that matters is the whole tick — 33.3 ms at 30 Hz — and this
/// is only the relevancy half of replication, before anything is encoded or
/// sent. The "Was" column is `query_circle`, whose per-call `HashSet` + `Vec`
/// put this workload over the tick budget on its own; `query_circle_into`
/// reuses a caller-owned scratch and filters to the true disc.
///
/// The grid is built outside the timed closure deliberately: this guards the
/// query path, and `collision_rebuild_and_query` already covers rebuild.
fn aoi_query_per_client() -> f64 {
    let side = (100_000f64).sqrt().ceil() as usize;
    let mut grid = SpatialGrid::new(400.0);
    let mut clients = Vec::with_capacity(1_000);
    for i in 0..100_000usize {
        let (x, y) = ((i % side) as f64 * 31.0, (i / side) as f64 * 31.0);
        grid.insert(DVec2::new(x, y), 2.0);
        // Every hundredth collider doubles as a client viewpoint, so the
        // query centres sit in occupied cells rather than empty space.
        if i % 100 == 0 { clients.push(DVec2::new(x, y)); }
    }

    let mut scratch = AoiScratch::new();
    best_ms(5, || {
        let mut hits = 0usize;
        for c in &clients {
            grid.query_circle_into(*c, 500.0, 0, &mut scratch);
            hits += black_box(scratch.hits.len());
        }
        black_box(hits);
    })
}

/// Snapshot encoding: the other half of a replication tick.
///
/// AoI decides *who* each client sees; this turns that into bytes. It is
/// the half nothing measured until it was written, and the steady-state
/// figure is what says the pipeline fits at all: ~7 ms of relevancy plus
/// this, against 33.3 ms.
///
/// The writer is reused across every packet because that is what the
/// real path does, not because it is much faster: measured, reuse saves
/// about 3% (5.44 ms against 5.61 ms), since one avoided allocation is
/// noise beside ~500 bytes of bit-pushing per packet. Guarding the shape
/// the caller actually uses is the point.
#[cfg(feature = "replication")]
fn snapshot_encode_per_tick(clients: usize, per_client: usize) -> f64 {
    let items: Vec<EntityItem> = (0..per_client as u32)
        .map(|i| EntityItem {
            // A sixteenth are arrivals, the rest position updates, which
            // is roughly what a moving crowd produces.
            kind: if i % 16 == 0 { ItemKind::Entered } else { ItemKind::Updated },
            entity: EntityId { index: i, generation: 1 },
            pos: DVec2::new((i % 900) as f64 - 450.0, (i % 700) as f64 - 350.0).extend(0.0),
            rot: glam::Quat::IDENTITY,
            vel: DVec2::new(9.0, -4.0).extend(0.0),
            component: NameId(3),
        })
        .collect();

    let packets: Vec<SnapshotPacket> = (0..clients)
        .map(|c| SnapshotPacket {
            tick: 100,
            is_header: true,
            keyframe: false,
            your_entity: EntityId { index: c as u32, generation: 1 },
            names: Vec::new(),
            items: items.clone(),
        })
        .collect();

    let mut writer = BitWriter::new();
    best_ms(5, || {
        let mut bytes = 0usize;
        for p in &packets {
            p.encode_into(&mut writer, 500.0);
            bytes += black_box(writer.byte_len());
        }
        black_box(bytes);
    })
}

fn report(label: &str, ms: f64, budget_ms: f64) -> bool {
    let ok = ms <= budget_ms;
    println!(
        "{:<44} {:>8.2} ms   budget {:>7.2} ms   {}",
        label,
        ms,
        budget_ms,
        if ok { "ok" } else { "REGRESSION" }
    );
    ok
}

fn main() {
    // `cargo test --all-targets` builds and runs this binary in debug, where
    // the same code is several times slower — the 50k x 20 case measures
    // ~1.4ms in release and blows the 4ms budget unoptimized. The budgets
    // describe optimized code, so enforcing them in a debug run reports a
    // regression that does not exist. Skip instead: `cargo bench` (release)
    // is where these are enforced, and CI runs it as its own step.
    if cfg!(debug_assertions) {
        println!("hot-path guards skipped: debug build (run `cargo bench` to enforce)");
        return;
    }

    println!("void_engine hot-path guards (release)\n");
    let mut all_ok = true;
    all_ok &= report("ecs iter2: 50k entities x 20 systems", ecs_many_systems(), BUDGET_MANY_SYSTEMS);
    all_ok &= report("ecs iter2: 250k entities, 1 system", ecs_wide_single_query(), BUDGET_WIDE_QUERY);
    all_ok &= report("collision rebuild+query: 10k", collision_rebuild_and_query(10_000), BUDGET_COLLISION_10K);
    all_ok &= report("aoi query: 100k colliders x 1000 clients", aoi_query_per_client(), BUDGET_AOI);
    #[cfg(feature = "replication")]
    {
        all_ok &= report(
            "snapshot encode: 1000 clients x 40 items",
            snapshot_encode_per_tick(1_000, 40),
            BUDGET_ENCODE,
        );
    }
    println!();
    if !all_ok {
        eprintln!("one or more hot paths regressed past budget");
        std::process::exit(1);
    }
}

// Budgets: ~3x the current measurement. Wide enough for a slower CI runner,
// narrow enough to catch a real algorithmic regression.
//
// The ECS budgets were 35.0 / 9.0 when the iterators still collected into a
// heap-allocated Vec per query. Making them lazy took the first from 11.32ms
// to ~1.17ms, so the budgets are retightened here — left at the old values
// they would have happily accepted a full regression back to collecting.
const BUDGET_MANY_SYSTEMS: f64 = 4.0; // measured 1.17 (was 11.32 when collecting)
const BUDGET_WIDE_QUERY: f64 = 1.5; // measured 0.43 (was  2.73 when collecting)
// Re-baselined when `query_pairs` began rejecting non-overlapping pairs
// and the lattice was tightened so it still emits some. The old 15.0 was
// ~3x a measurement taken on a workload that returned 55,552 pairs none
// of which could touch; it would have accepted a full regression back to
// emitting them.
const BUDGET_COLLISION_10K: f64 = 11.0; // measured 3.68 on the overlapping lattice
// Deliberately below the 33.3 ms tick budget as well as ~3x the measurement:
// AoI is one part of a tick that must also encode and send, so a figure that
// merely "fits" is already a regression worth failing on.
//
// This lattice is denser and more uniform than a real world; the same query
// over uniform-random positions across a 10 km square measures ~12.4 ms, and
// `query_circle_into`'s own docs quote that figure. Both are real — bucket
// occupancy is what the cost tracks, so the guard pins the layout it builds.
const BUDGET_AOI: f64 = 22.0; // measured 7.28 on this lattice
// Encoding is the half of a replication tick that AoI does not cover, and
// the two must fit together: ~7 ms of relevancy plus ~6 ms of encoding
// leaves about 20 ms for chunking, sending and the game itself.
//
// Tighter than 3x on purpose. Encode cost is linear in items, so a
// regression here is not a constant factor — 120 items per client instead
// of 40 measures 16.4 ms, and a change that quietly tripled per-item cost
// would still pass a looser budget while eating half the tick.
// Retightened 2026-09-11 when `BitWriter::write_bits` stopped looping
// over `write_bit`: this measured 5.77 ms, then 1.78 ms for identical
// work. Leaving the budget at 15.0 would have silently accepted a full
// regression back to the per-bit loop — the same trap the ECS budgets
// were retightened for, and the reason a budget is ~3x a measurement
// rather than whatever number was true when it was written.
#[cfg(feature = "replication")]
const BUDGET_ENCODE: f64 = 5.0; // measured 1.78 at 1000 x 40 (was 5.77)
