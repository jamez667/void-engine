//! Phase 2's exit criterion: a server that is killed comes back.
//!
//! The unit tests in `persist::checkpoint` cover the file mechanics. This
//! covers the thing a server operator actually cares about — run, die,
//! restart, and the world is where it was — driven entirely through the
//! public API, the way a game would.
//!
//! A real `kill -9` cannot be issued against ourselves mid-test without
//! taking the harness down too, so the process death is simulated by
//! dropping the world and rebuilding from disk. What that does *not*
//! simulate is a torn write, so that is tested separately by damaging the
//! file directly.

#![cfg(feature = "persist")]

use std::fs;
use std::path::PathBuf;

use glam::DVec2;
use void_engine::components::{Transform2D, Velocity};
use void_engine::persist::{
    capture, load, register_engine_components, restore, save, CheckpointConfig, Registry,
    RngStreams,
};
use void_engine::{App, SimCtx, World};

fn temp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir()
        .join("void_engine_crash_tests")
        .join(format!("{tag}_{}", std::process::id()));
    let _ = fs::remove_dir_all(&d);
    d
}

fn engine_registry() -> Registry {
    let mut r = Registry::new();
    register_engine_components(&mut r).unwrap();
    r
}

/// A server that checkpoints every `every` ticks — the shape a real one
/// has, with the cadence being the only knob that matters.
struct Server {
    cfg: CheckpointConfig,
    registry: Registry,
    every: u64,
    tick: u64,
}

impl App for Server {
    fn init(&mut self, ctx: &mut SimCtx) {
        // Restore-on-boot: if a checkpoint exists, the world is whatever
        // it says, not a fresh one.
        if let Some(snap) = load(&self.cfg).expect("checkpoint set must be readable") {
            self.tick = snap.tick;
            *ctx.world = restore(&snap, &self.registry).expect("checkpoint must restore");
        }
    }

    fn fixed_update(&mut self, ctx: &mut SimCtx) {
        self.tick += 1;
        void_engine::physics::integrate(ctx.world, ctx.dt, 1.0, 1.0);
        if self.tick.is_multiple_of(self.every) {
            let snap = capture(ctx.world, &self.registry, self.tick, &RngStreams::new())
                .expect("capture must succeed");
            save(&self.cfg, &snap).expect("checkpoint write must succeed");
        }
    }
}

/// Spawn a world, tick it, checkpoint, then rebuild from disk as a fresh
/// process would — the core recovery path.
#[test]
fn a_world_survives_a_restart() {
    let cfg = CheckpointConfig::new(temp_dir("restart"));
    let reg = engine_registry();

    // ── first run ───────────────────────────────────────────────────
    let mut world = World::new();
    let mut ids = Vec::new();
    for i in 0..5 {
        let e = world.spawn();
        world.insert(e, Transform2D { pos: DVec2::new(i as f64, 0.0), rot: 0.0 });
        world.insert(e, Velocity { linear: DVec2::new(1.0, 0.0), angular: 0.0 });
        ids.push(e);
    }
    world.despawn(ids[1]);

    for _ in 0..10 {
        void_engine::physics::integrate(&mut world, 1.0 / 30.0, 1.0, 1.0);
    }
    let snap = capture(&world, &reg, 10, &RngStreams::new()).unwrap();
    save(&cfg, &snap).unwrap();

    let positions: Vec<_> = ids
        .iter()
        .map(|&id| world.get::<Transform2D>(id).map(|t| t.pos))
        .collect();

    // ── the process dies here ───────────────────────────────────────
    drop(world);

    // ── second run ──────────────────────────────────────────────────
    let loaded = load(&cfg).unwrap().expect("a checkpoint was written");
    assert_eq!(loaded.tick, 10, "tick must be recovered");
    let recovered = restore(&loaded, &reg).unwrap();

    for (i, &id) in ids.iter().enumerate() {
        assert_eq!(
            recovered.get::<Transform2D>(id).map(|t| t.pos),
            positions[i],
            "entity {i} did not come back where it was",
        );
    }
    assert!(!recovered.alive(ids[1]), "a despawned entity must stay dead");

    let _ = fs::remove_dir_all(&cfg.dir);
}

/// The same thing through the engine's own headless loop, including
/// `init`'s restore-on-boot path.
#[test]
fn a_headless_server_resumes_where_it_stopped() {
    use void_engine::app_headless::{run_headless_with, HeadlessConfig};

    let cfg = CheckpointConfig::new(temp_dir("headless"));

    // First boot: no checkpoint, so it starts fresh and runs 20 ticks,
    // checkpointing every 5.
    let first = Server {
        cfg: cfg.clone(),
        registry: engine_registry(),
        every: 5,
        tick: 0,
    };
    run_headless_with(
        first,
        HeadlessConfig {
            hz: 30.0,
            max_ticks: Some(20),
            uncapped: true,
            ..Default::default()
        },
        || true,
    );

    let after_first = load(&cfg).unwrap().expect("first run must have checkpointed");
    assert_eq!(after_first.tick, 20, "should have checkpointed at tick 20");

    // Second boot: `init` finds the checkpoint and resumes from tick 20,
    // so 10 more ticks land it at 30 rather than back at 10.
    let second = Server {
        cfg: cfg.clone(),
        registry: engine_registry(),
        every: 5,
        tick: 0,
    };
    run_headless_with(
        second,
        HeadlessConfig {
            hz: 30.0,
            max_ticks: Some(10),
            uncapped: true,
            ..Default::default()
        },
        || true,
    );

    let after_second = load(&cfg).unwrap().expect("second run must have checkpointed");
    assert_eq!(
        after_second.tick, 30,
        "the second run must resume from the checkpoint, not restart at zero",
    );

    let _ = fs::remove_dir_all(&cfg.dir);
}

/// A torn write — the crash case the atomic rename exists to survive.
/// The damaged current file must cost one interval, not the world.
#[test]
fn a_torn_write_costs_one_interval_not_the_world() {
    let cfg = CheckpointConfig::new(temp_dir("torn"));
    let reg = engine_registry();

    let mut world = World::new();
    let e = world.spawn();
    world.insert(e, Transform2D { pos: DVec2::new(1.0, 1.0), rot: 0.0 });

    // Checkpoint at tick 10.
    save(&cfg, &capture(&world, &reg, 10, &RngStreams::new()).unwrap()).unwrap();

    // Advance and checkpoint at tick 20.
    world.get_mut::<Transform2D>(e).unwrap().pos = DVec2::new(2.0, 2.0);
    save(&cfg, &capture(&world, &reg, 20, &RngStreams::new()).unwrap()).unwrap();

    // The process died partway through writing the next one. An atomic
    // rename means this cannot actually happen — but if it somehow does,
    // the previous checkpoint must still carry the day.
    fs::write(cfg.current(), b"\xff\xfe truncated").unwrap();

    let recovered = restore(&load(&cfg).unwrap().unwrap(), &reg).unwrap();
    assert_eq!(
        recovered.get::<Transform2D>(e).unwrap().pos,
        DVec2::new(1.0, 1.0),
        "should have fallen back to the tick-10 checkpoint",
    );

    let _ = fs::remove_dir_all(&cfg.dir);
}

/// First boot with no checkpoint is not an error — a server must start.
#[test]
fn a_first_boot_with_no_checkpoint_starts_fresh() {
    let cfg = CheckpointConfig::new(temp_dir("first_boot"));
    assert!(load(&cfg).unwrap().is_none(), "an empty dir must not be an error");
    let _ = fs::remove_dir_all(&cfg.dir);
}
