//! The headless driver: a fixed-tick loop with no window, no GPU and no
//! event loop.
//!
//! This is the half of `app` a dedicated server uses. It is a plain
//! function rather than a trait implementation because there is no OS
//! callback to hang it off — the server owns its thread and simply loops.

use std::time::{Duration, Instant};

use crate::app::{App, SimCtx};
use crate::ecs::World;
use crate::input::InputState;
use crate::time::Timestep;

/// How a headless run ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Exit {
    /// `should_run` returned false.
    Stopped,
    /// Ran the requested number of ticks. Only produced by
    /// [`HeadlessConfig::max_ticks`].
    TickLimit,
}

/// What the loop is managing, sampled on a rollup interval.
///
/// The windowed loop has measured itself since it existed — `[perf]` every
/// second, plus a [`PerfSnapshot`] an overlay can draw. The server loop
/// measured nothing, which is backwards: it is the one with no human
/// watching it. An overloaded server does not stutter or error, it
/// silently simulates less than a second of world per second of wall
/// clock, and the first evidence is players reporting that the game feels
/// wrong.
///
/// [`PerfSnapshot`]: crate::perf::PerfSnapshot
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TickHealth {
    /// Ticks run since the last report.
    pub ticks: u64,
    /// Wall-clock seconds the window covered.
    pub elapsed_s: f64,
    /// Ticks per second actually achieved over the window.
    pub hz: f64,
    /// The rate this loop is configured for.
    pub target_hz: f64,
    /// Sim seconds discarded at the spiral-of-death clamp during this
    /// window. Non-zero means the world fell behind wall clock and the
    /// difference will never be simulated.
    pub dropped_s: f64,
    /// Mean seconds spent inside `fixed_update` per tick this window.
    pub mean_tick_s: f64,
    /// Longest single `fixed_update` this window.
    pub worst_tick_s: f64,
}

impl TickHealth {
    /// Fraction of real time actually simulated, 1.0 when keeping up.
    ///
    /// The number worth alerting on: 0.94 is a server 6% behind and
    /// drifting further every second.
    pub fn realtime_ratio(&self) -> f64 {
        if self.target_hz <= 0.0 { return 1.0 }
        (self.hz / self.target_hz).min(1.0)
    }

    /// Whether the loop kept up over this window.
    pub fn keeping_up(&self) -> bool {
        self.dropped_s <= 0.0 && self.realtime_ratio() >= 0.99
    }
}

/// Knobs for [`run_headless_with`]. `Default` gives a 30 Hz server that
/// runs until stopped.
pub struct HeadlessConfig {
    /// Simulation rate. Defaults to [`crate::time::SERVER_HZ`] (30).
    pub hz: f32,
    /// Stop after this many ticks. `None` runs until `should_run` is false.
    ///
    /// Exists so tests and replays can drive an exact number of ticks
    /// without racing a wall clock.
    pub max_ticks: Option<u64>,
    /// When true, run ticks back-to-back with no sleeping and a synthetic
    /// clock, as fast as the CPU allows.
    ///
    /// This is what makes a server loop testable and a replay reproducible:
    /// wall-clock pacing is exactly the part you do not want when
    /// re-simulating 10,000 ticks in CI. Real servers leave it false.
    pub uncapped: bool,
    /// How often to report [`TickHealth`]. Defaults to one second.
    pub health_every: Duration,
    /// Where to send those reports. `None` measures nothing.
    ///
    /// A callback rather than a returned handle, matching how
    /// `should_run` is already passed: the engine takes no view on how a
    /// server shares this with a health endpoint or a log. Close over an
    /// `Arc<Mutex<_>>` to read it from another thread.
    #[allow(clippy::type_complexity)]
    pub on_health: Option<Box<dyn FnMut(TickHealth)>>,
}

impl Default for HeadlessConfig {
    fn default() -> Self {
        Self {
            hz: crate::time::SERVER_HZ,
            max_ticks: None,
            uncapped: false,
            health_every: Duration::from_secs(1),
            on_health: None,
        }
    }
}

/// Run `app` at [`crate::time::SERVER_HZ`] until it stops itself.
///
/// The loop is the same shape as the windowed one in `app::run` —
/// same accumulator, same `can_advance` deferral and refund, same
/// input-edge consumption rule — minus everything that needs a screen.
/// Simulation written against [`App`] therefore behaves identically on a
/// server and inside a client.
///
/// `should_run` is checked once per iteration, before any ticks. A server
/// typically closes over an `AtomicBool` set by a signal handler.
pub fn run_headless<A: App>(app: A, should_run: impl FnMut() -> bool) -> Exit {
    run_headless_with(app, HeadlessConfig::default(), should_run)
}

/// [`run_headless`] with explicit configuration.
pub fn run_headless_with<A: App>(
    mut app: A,
    cfg: HeadlessConfig,
    mut should_run: impl FnMut() -> bool,
) -> Exit {
    let mut world = World::new();
    // A server has no keyboard. `InputState` is still threaded through so
    // that `SimCtx` has one shape in both loops and game code compiles
    // unchanged; it simply stays empty unless a replay driver fills it.
    let input = InputState::default();
    let mut timestep = Timestep::with_hz(cfg.hz);
    let dt = timestep.dt();

    {
        let mut ctx = SimCtx { world: &mut world, input: &input, dt };
        app.init(&mut ctx);
    }

    let step = Duration::from_secs_f32(dt);
    let mut ticks: u64 = 0;
    let mut last = Instant::now();

    // Rollup state. All of it is dead weight when `on_health` is None:
    // the timing calls are skipped entirely rather than measured and
    // thrown away, so a server that does not want this pays nothing.
    let measuring = cfg.on_health.is_some();
    let mut on_health = cfg.on_health;
    let mut window_start = Instant::now();
    let mut window_ticks: u64 = 0;
    let mut window_tick_s = 0.0f64;
    let mut window_worst_s = 0.0f64;
    let mut window_dropped_base = 0.0f64;

    loop {
        if !should_run() {
            return Exit::Stopped;
        }

        // Uncapped mode feeds the accumulator exactly one step of synthetic
        // time, so the loop is deterministic and does not depend on how
        // fast this machine happens to be.
        let frame_dt = if cfg.uncapped {
            dt
        } else {
            let now = Instant::now();
            let elapsed = now.duration_since(last);
            // Sleep off whatever is left of the step. Unlike the client,
            // there is no spin-wait: a server has nothing to present, so
            // burning a core to hit a sub-millisecond deadline buys
            // nothing. Sleep granularity jitter is absorbed by the
            // accumulator on the next iteration.
            if elapsed < step {
                std::thread::sleep(step - elapsed);
            }
            let now = Instant::now();
            let dt = now.duration_since(last).as_secs_f32();
            last = now;
            dt
        };

        let (steps, _alpha) = timestep.advance(frame_dt);

        for s in 0..steps {
            // Same deferral contract as the windowed loop: a tick that
            // cannot run yet is refunded, not dropped, so a stalled peer
            // does not leave this sim permanently behind.
            if !app.can_advance() {
                timestep.refund(steps - s);
                break;
            }
            let tick_start = measuring.then(Instant::now);
            let mut ctx = SimCtx { world: &mut world, input: &input, dt };
            app.fixed_update(&mut ctx);
            if let Some(t0) = tick_start {
                let spent = t0.elapsed().as_secs_f64();
                window_tick_s += spent;
                window_worst_s = window_worst_s.max(spent);
                window_ticks += 1;
            }

            ticks += 1;
            if let Some(limit) = cfg.max_ticks {
                if ticks >= limit {
                    return Exit::TickLimit;
                }
            }
        }

        if let Some(sink) = on_health.as_mut() {
            let elapsed = window_start.elapsed();
            if elapsed >= cfg.health_every {
                let elapsed_s = elapsed.as_secs_f64();
                let dropped_now = timestep.dropped_seconds();
                sink(TickHealth {
                    ticks: window_ticks,
                    elapsed_s,
                    hz: if elapsed_s > 0.0 { window_ticks as f64 / elapsed_s } else { 0.0 },
                    target_hz: cfg.hz as f64,
                    dropped_s: dropped_now - window_dropped_base,
                    mean_tick_s: if window_ticks > 0 {
                        window_tick_s / window_ticks as f64
                    } else {
                        0.0
                    },
                    worst_tick_s: window_worst_s,
                });
                window_start = Instant::now();
                window_ticks = 0;
                window_tick_s = 0.0;
                window_worst_s = 0.0;
                window_dropped_base = dropped_now;
            }
        }
    }
}
