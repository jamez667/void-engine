//! The claim R1 exists to make true: a dedicated server can tick the
//! simulation with no window, no GPU, and no event loop.
//!
//! This is an integration test rather than a unit test on purpose — it
//! compiles against the crate's public API exactly as a downstream server
//! crate would, so if `App`/`SimCtx`/`run_headless` stop being usable from
//! outside the crate, this fails.
//!
//! It runs in every feature configuration, including `--no-default-features`
//! where `wgpu` is not even linked.

use std::cell::Cell;
use std::rc::Rc;

use glam::DVec2;
use void_engine::app_headless::{run_headless_with, Exit, HeadlessConfig};
use void_engine::components::{Transform2D, Velocity};
use void_engine::{App, SimCtx};

/// A minimal server-side game: spawns some entities on init, integrates
/// them every tick. Nothing here can touch a renderer, because `SimCtx`
/// does not have one.
struct MoveSim {
    ticks: u32,
    spawned: usize,
}

impl App for MoveSim {
    fn init(&mut self, ctx: &mut SimCtx) {
        for i in 0..100 {
            let e = ctx.world.spawn();
            ctx.world.insert(e, Transform2D { pos: DVec2::new(i as f64, 0.0), rot: 0.0 });
            ctx.world.insert(e, Velocity { linear: DVec2::new(1.0, 0.0), angular: 0.0 });
        }
        self.spawned = ctx.world.entities().count();
    }

    fn fixed_update(&mut self, ctx: &mut SimCtx) {
        self.ticks += 1;
        void_engine::physics::integrate(ctx.world, ctx.dt, 0.0, 0.0);
    }
}

#[test]
fn a_server_ticks_the_world_with_no_window() {
    let cfg = HeadlessConfig {
        hz: 30.0,
        max_ticks: Some(60),
        uncapped: true,
        ..Default::default()
    };
    let sim = MoveSim { ticks: 0, spawned: 0 };
    let exit = run_headless_with(sim, cfg, || true);
    assert_eq!(exit, Exit::TickLimit, "should stop on the tick limit");
}

/// `init` must run before the first tick, and see a usable world.
#[test]
fn init_runs_and_can_populate_the_world() {
    // The loop owns the `App`, so observe through shared state rather than
    // by reading the value back.
    let spawned = Rc::new(Cell::new(0usize));

    struct Spy(Rc<Cell<usize>>);
    impl App for Spy {
        fn init(&mut self, ctx: &mut SimCtx) {
            for _ in 0..7 {
                ctx.world.spawn();
            }
            self.0.set(ctx.world.entities().count());
        }
        fn fixed_update(&mut self, _ctx: &mut SimCtx) {}
    }

    let cfg = HeadlessConfig {
        hz: 30.0,
        max_ticks: Some(1),
        uncapped: true,
        ..Default::default()
    };
    run_headless_with(Spy(spawned.clone()), cfg, || true);
    assert_eq!(spawned.get(), 7, "init must see a live World");
}

/// The tick count is exact under `uncapped`, which is what makes replays
/// and CI runs reproducible instead of racing a wall clock.
#[test]
fn uncapped_runs_an_exact_number_of_ticks() {
    let count = Rc::new(Cell::new(0u32));

    struct Counter(Rc<Cell<u32>>);
    impl App for Counter {
        fn init(&mut self, _ctx: &mut SimCtx) {}
        fn fixed_update(&mut self, _ctx: &mut SimCtx) {
            self.0.set(self.0.get() + 1);
        }
    }

    let cfg = HeadlessConfig {
        hz: 30.0,
        max_ticks: Some(250),
        uncapped: true,
        ..Default::default()
    };
    let exit = run_headless_with(Counter(count.clone()), cfg, || true);
    assert_eq!(exit, Exit::TickLimit);
    assert_eq!(count.get(), 250, "uncapped mode must run exactly max_ticks");
}

/// `should_run` returning false stops the loop, which is how a server
/// handles SIGTERM.
#[test]
fn should_run_false_stops_the_loop() {
    struct Noop;
    impl App for Noop {
        fn init(&mut self, _ctx: &mut SimCtx) {}
        fn fixed_update(&mut self, _ctx: &mut SimCtx) {}
    }

    let mut calls = 0;
    let exit = run_headless_with(
        Noop,
        // The only site that genuinely wants the struct-update tail: it
        // takes the default rate and runs unbounded, stopping via
        // `should_run` rather than a tick limit.
        HeadlessConfig { uncapped: true, ..Default::default() },
        || {
            calls += 1;
            calls <= 3
        },
    );
    assert_eq!(exit, Exit::Stopped);
}

/// `can_advance` defers a tick instead of running it, and the deferred time
/// is refunded rather than dropped — the lockstep contract, now also
/// honoured by the headless loop.
#[test]
fn can_advance_false_defers_ticks() {
    let ticks = Rc::new(Cell::new(0u32));

    struct Stalled(Rc<Cell<u32>>);
    impl App for Stalled {
        fn init(&mut self, _ctx: &mut SimCtx) {}
        fn fixed_update(&mut self, _ctx: &mut SimCtx) {
            self.0.set(self.0.get() + 1);
        }
        fn can_advance(&self) -> bool {
            false
        }
    }

    let mut iters = 0;
    run_headless_with(
        Stalled(ticks.clone()),
        HeadlessConfig {
            hz: 30.0,
            max_ticks: Some(10),
            uncapped: true,
            ..Default::default()
        },
        || {
            iters += 1;
            iters <= 20
        },
    );
    assert_eq!(ticks.get(), 0, "a stalled sim must not advance");
}

/// The server loop must be able to say whether it is keeping up.
///
/// Before this existed, a headless server that overran its budget simply
/// simulated less world per second with no counter, no log and no error —
/// the windowed loop had `[perf]` and a `PerfSnapshot`, and the one loop
/// with nobody watching it had nothing.
#[test]
fn a_healthy_loop_reports_that_it_is_keeping_up() {
    use std::sync::{Arc, Mutex};
    use void_engine::app_headless::TickHealth;

    struct Idle;
    impl App for Idle {
        fn init(&mut self, _ctx: &mut SimCtx) {}
        fn fixed_update(&mut self, _ctx: &mut SimCtx) {}
    }

    let seen: Arc<Mutex<Vec<TickHealth>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();

    run_headless_with(
        Idle,
        HeadlessConfig {
            hz: 30.0,
            max_ticks: Some(120),
            uncapped: true,
            // Report every window, so an uncapped run still produces one.
            health_every: std::time::Duration::ZERO,
            on_health: Some(Box::new(move |h| sink.lock().unwrap().push(h))),
        },
        || true,
    );

    let reports = seen.lock().unwrap();
    assert!(!reports.is_empty(), "a measured loop must report something");
    let total: u64 = reports.iter().map(|h| h.ticks).sum();
    assert!(total > 0, "reports must account for ticks actually run");
    for h in reports.iter() {
        assert_eq!(h.target_hz, 30.0);
        assert_eq!(h.dropped_s, 0.0, "an idle sim must not lose sim time");
        assert!(h.worst_tick_s < 0.5, "an empty tick should be fast, got {}", h.worst_tick_s);
    }
}

/// Measurement is opt-in: with no sink, the loop does no timing at all.
#[test]
fn health_reporting_is_off_by_default() {
    struct Idle;
    impl App for Idle {
        fn init(&mut self, _ctx: &mut SimCtx) {}
        fn fixed_update(&mut self, _ctx: &mut SimCtx) {}
    }

    let cfg = HeadlessConfig {
        hz: 30.0,
        max_ticks: Some(10),
        uncapped: true,
        ..Default::default()
    };
    assert!(cfg.on_health.is_none(), "a server must opt in to being measured");
    assert_eq!(run_headless_with(Idle, cfg, || true), Exit::TickLimit);
}

/// `SimCtx::dt` reflects the loop's configured rate, so the same
/// `fixed_update` body integrates correctly at 30 and at 60.
#[test]
fn dt_matches_the_configured_tick_rate() {
    let seen = Rc::new(Cell::new(0.0f32));

    struct DtSpy(Rc<Cell<f32>>);
    impl App for DtSpy {
        fn init(&mut self, _ctx: &mut SimCtx) {}
        fn fixed_update(&mut self, ctx: &mut SimCtx) {
            self.0.set(ctx.dt);
        }
    }

    for hz in [30.0f32, 60.0] {
        let cfg = HeadlessConfig {
            hz,
            max_ticks: Some(1),
            uncapped: true,
            ..Default::default()
        };
        run_headless_with(DtSpy(seen.clone()), cfg, || true);
        assert!(
            (seen.get() - 1.0 / hz).abs() < 1e-6,
            "at {hz}Hz dt was {}, expected {}",
            seen.get(),
            1.0 / hz
        );
    }
}
