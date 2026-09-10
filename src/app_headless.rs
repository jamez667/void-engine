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
}

impl Default for HeadlessConfig {
    fn default() -> Self {
        Self { hz: crate::time::SERVER_HZ, max_ticks: None, uncapped: false }
    }
}

/// Run `app` at [`crate::time::SERVER_HZ`] until it stops itself.
///
/// The loop is the same shape as the windowed one in [`crate::app::run`] —
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
            let mut ctx = SimCtx { world: &mut world, input: &input, dt };
            app.fixed_update(&mut ctx);

            ticks += 1;
            if let Some(limit) = cfg.max_ticks {
                if ticks >= limit {
                    return Exit::TickLimit;
                }
            }
        }
    }
}
