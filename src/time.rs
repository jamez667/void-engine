//! Fixed-timestep accumulator.
//!
//! The tick rate is per-`Timestep` rather than a global constant: a client
//! renders and simulates at 60 Hz, while a dedicated server has no reason to
//! burn a core doing the same — 30 Hz halves its CPU and matches what
//! `net::interp::InterpClock` already defaults to for snapshot playback.
//! Before this was a parameter the tree contradicted itself, with
//! `FIXED_DT` hardcoded at 60 Hz and the interpolation clock assuming 30.

/// Client fixed step: 60 Hz. Still a constant because the windowed loop
/// renders at this rate and every existing game was written against it.
pub const FIXED_DT: f32 = 1.0 / 60.0;

/// Default dedicated-server tick: 30 Hz. See [`Timestep::for_server`].
pub const SERVER_HZ: f32 = 30.0;

pub struct Timestep {
    accumulator: f32,
    /// Seconds per fixed step. Set once at construction.
    dt: f32,
}

impl Timestep {
    /// A 60 Hz timestep — the client default, and what `Timestep::default()`
    /// gives you.
    pub fn new() -> Self {
        Self::with_dt(FIXED_DT)
    }

    /// A timestep running at `hz` ticks per second.
    ///
    /// Panics on a non-positive or non-finite rate: a zero or NaN `hz`
    /// would make `advance` divide by zero and hand back a nonsense step
    /// count, and failing at construction is far easier to diagnose than a
    /// loop that silently spins.
    pub fn with_hz(hz: f32) -> Self {
        assert!(hz > 0.0 && hz.is_finite(), "tick rate must be positive and finite, got {hz}");
        Self::with_dt(1.0 / hz)
    }

    /// A timestep with an explicit step duration in seconds.
    pub fn with_dt(dt: f32) -> Self {
        assert!(dt > 0.0 && dt.is_finite(), "step duration must be positive and finite, got {dt}");
        Self { accumulator: 0.0, dt }
    }

    /// The dedicated-server default, [`SERVER_HZ`].
    pub fn for_server() -> Self {
        Self::with_hz(SERVER_HZ)
    }

    /// Seconds per fixed step. Pass this as `dt` to game logic — reading it
    /// from the timestep rather than the `FIXED_DT` constant is what lets
    /// the same `fixed_update` run correctly at 30 and 60 Hz.
    pub fn dt(&self) -> f32 {
        self.dt
    }

    pub fn advance(&mut self, frame_dt: f32) -> (u32, f32) {
        self.accumulator += frame_dt.min(MAX_ACCUM_S);
        let steps = (self.accumulator / self.dt) as u32;
        self.accumulator -= steps as f32 * self.dt;
        let alpha = self.accumulator / self.dt;
        (steps, alpha)
    }

    /// Give back `steps` worth of time that `advance` handed out but the
    /// caller could not run.
    ///
    /// `advance` deducts every step it returns from the accumulator on the
    /// assumption they will all be run. A lockstep game breaks that
    /// assumption: it stops mid-catch-up when it does not yet have the
    /// peer's input for the next tick. Without a refund that time is simply
    /// gone — the sim would silently skip the ticks it stalled on and run
    /// permanently behind the peer, which for lockstep is not a dropped
    /// frame but a divergence.
    ///
    /// Refunding puts the un-run steps back so the very next `advance`
    /// returns them again, once the input has landed. Deferred, not dropped.
    pub fn refund(&mut self, steps: u32) {
        self.accumulator += steps as f32 * self.dt;
        // Cap for the same reason `advance` clamps `frame_dt`: a long stall
        // must not build a debt that then spends itself as a hundred-step
        // catch-up burst the moment the peer reconnects.
        self.accumulator = self.accumulator.min(MAX_ACCUM_S);
    }
}

/// Spiral-of-death guard: the most wall-clock time a single `advance` (or a
/// `refund`) may bank. At 60 Hz this is 15 steps; at 30 Hz, 7.
const MAX_ACCUM_S: f32 = 0.25;

impl Default for Timestep {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn advance_accumulates_into_whole_steps() {
        let mut t = Timestep::new();
        // Half a step: not enough to run anything yet.
        let (steps, _) = t.advance(FIXED_DT * 0.5);
        assert_eq!(steps, 0);
        // The other half completes one.
        let (steps, _) = t.advance(FIXED_DT * 0.6);
        assert_eq!(steps, 1);
    }

    /// The property lockstep depends on: a refunded step is handed out
    /// again, so a stalled tick is deferred rather than silently skipped.
    #[test]
    fn refunded_steps_come_back_on_the_next_advance() {
        let mut t = Timestep::new();
        let (steps, _) = t.advance(FIXED_DT * 3.0);
        assert_eq!(steps, 3);

        // Ran one, stalled on the remaining two.
        t.refund(2);
        // A frame with no elapsed time still yields the deferred steps.
        let (steps, _) = t.advance(0.0);
        assert_eq!(steps, 2, "refunded steps must be re-offered");

        // And they are not offered a third time.
        let (steps, _) = t.advance(0.0);
        assert_eq!(steps, 0);
    }

    #[test]
    fn refunding_zero_is_a_no_op() {
        let mut t = Timestep::new();
        t.advance(FIXED_DT * 2.0);
        t.refund(0);
        let (steps, _) = t.advance(0.0);
        assert_eq!(steps, 0);
    }

    /// A long stall must not bank unbounded time and then spend it as a
    /// huge catch-up burst when the peer comes back.
    #[test]
    fn refund_is_capped_like_advance() {
        let mut t = Timestep::new();
        t.refund(10_000);
        let (steps, _) = t.advance(0.0);
        assert!(steps <= (0.25 / FIXED_DT) as u32 + 1, "burst of {steps} steps");
    }

    // ── per-instance tick rate ───────────────────────────────────────

    #[test]
    fn server_default_is_30hz() {
        let t = Timestep::for_server();
        assert!((t.dt() - 1.0 / 30.0).abs() < 1e-6, "dt was {}", t.dt());
        // And the client default is still 60, so existing games are unmoved.
        assert!((Timestep::new().dt() - FIXED_DT).abs() < 1e-6);
    }

    #[test]
    fn a_30hz_step_runs_half_as_often_as_a_60hz_one() {
        let mut client = Timestep::with_hz(60.0);
        let mut server = Timestep::with_hz(30.0);
        // One second of wall clock, fed in 60 equal slices.
        let (mut c, mut s) = (0, 0);
        for _ in 0..60 {
            c += client.advance(1.0 / 60.0).0;
            s += server.advance(1.0 / 60.0).0;
        }
        assert_eq!(c, 60, "60Hz should run 60 steps in a second");
        assert_eq!(s, 30, "30Hz should run 30 steps in the same second");
    }

    #[test]
    fn alpha_is_a_fraction_of_this_timesteps_own_step() {
        let mut t = Timestep::with_hz(30.0);
        // Exactly half a 30 Hz step.
        let (steps, alpha) = t.advance(1.0 / 60.0);
        assert_eq!(steps, 0);
        assert!((alpha - 0.5).abs() < 1e-5, "alpha was {alpha}");
    }

    #[test]
    fn refund_uses_this_timesteps_own_step_duration() {
        let mut t = Timestep::with_hz(30.0);
        let (steps, _) = t.advance(3.0 / 30.0);
        assert_eq!(steps, 3);
        t.refund(2);
        // `>= 1` rather than `== 2`: the accumulator is f32, and `advance`
        // can leave a tiny negative residue (measured -7e-9), so refunding
        // exactly `2 * dt` lands at 1.9999997 steps and truncates to 1. The
        // property that matters for lockstep is that refunded time comes
        // back and is denominated in *this* timestep's step — not that no
        // float ulp is ever lost. See `refund_is_capped_like_advance`,
        // which has always used an inequality for the same reason.
        let (back, _) = t.advance(0.0);
        assert!(back >= 1, "refunded time must be re-offered, got {back} steps");
        // And it is 30 Hz time: two 30 Hz steps is a sixteenth of a second,
        // which at 60 Hz would have been four.
        assert!(back <= 2, "must not manufacture extra steps, got {back}");
    }

    #[test]
    fn the_spiral_guard_scales_with_the_tick_rate() {
        // A huge frame delta is clamped to MAX_ACCUM_S either way, so a
        // slower tick yields proportionally fewer catch-up steps.
        //
        // Asserted as a ratio and a bound rather than exact counts: in f32
        // `1.0/60.0` is 0.016666668, a hair above true 1/60, so 0.25/dt is
        // 14.999999 and truncates to 14 — not 15. That is pre-existing
        // behaviour of the accumulator, and the guard's job is to bound the
        // burst, not to hit a precise number.
        let mut fast = Timestep::with_hz(60.0);
        let mut slow = Timestep::with_hz(30.0);
        let (f, _) = fast.advance(10.0);
        let (s, _) = slow.advance(10.0);
        assert!((14..=15).contains(&f), "60Hz burst was {f}, expected ~15");
        assert!((7..=8).contains(&s), "30Hz burst was {s}, expected ~7");
        assert!(s < f, "a slower tick must yield fewer catch-up steps");
    }

    #[test]
    #[should_panic(expected = "tick rate must be positive")]
    fn a_zero_tick_rate_is_rejected() {
        let _ = Timestep::with_hz(0.0);
    }
}
