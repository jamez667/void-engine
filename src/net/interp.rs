//! Tick-based render-behind interpolation.
//!
//! Server snapshots arrive at a fixed rate (30 Hz in void_sim), but the
//! client renders at 60-144 Hz. The original scheme reset a wall-clock
//! `snap_alpha` each snapshot and lerped between the last two received
//! frames — which freezes at alpha=1.0 between snapshots and produces a
//! visible pulse when the renderer outpaces the network.
//!
//! Render-behind fixes this by *intentionally* showing remote entities at
//! `latest_tick - 1` plus a sub-tick offset driven by wall-clock time since
//! the most recent snapshot landed. We always interpolate between two
//! already-received snapshots — no extrapolation, no freezing, smooth at any
//! framerate. Trade-off: ~33 ms of perceived latency on every remote entity.
//! Imperceptible at walking + flight speeds, and the unified mechanism
//! replaces two parallel ones (ships + characters).
//!
//! The tick rate is configurable per-clock so games at different rates can
//! reuse the same primitive.

use std::time::Instant;

/// Three-deep snapshot-tick clock for render-behind interpolation.
///
/// We always render the *older* pair (`from2` → `from`) while the newest
/// snapshot (`to`) is held in reserve as the future anchor. Concretely the
/// renderer is "now - 1 tick" behind the latest received frame, so even if
/// a snapshot is a few ms late we have already-buffered data covering the
/// gap — no freeze, no pulse.
///
/// A 2-deep buffer can't do this: it would have to choose between freezing
/// at `t=1.0` (the original pulse) or extrapolating past `to` (rubber-band).
#[derive(Clone, Copy, Debug)]
pub struct InterpClock {
    /// Oldest tracked snapshot tick (render lerps from here, t=0).
    pub from2_tick: u32,
    /// Middle snapshot tick (render lerps toward here, t=1).
    pub from_tick:  u32,
    /// Newest snapshot tick — kept as future anchor, not yet rendered.
    pub to_tick:    u32,
    /// Wall-clock instant the newest snapshot was applied. Drives sub-tick
    /// progress between renders. `None` before any snapshot.
    pub to_recv:    Option<Instant>,
    /// Server tick rate in Hz. Used to translate wall-clock time since the
    /// last snapshot into a sub-tick offset. Defaults to 30.0 (void_sim).
    pub tick_hz:    f32,
}

impl Default for InterpClock {
    fn default() -> Self {
        Self { from2_tick: 0, from_tick: 0, to_tick: 0, to_recv: None, tick_hz: 30.0 }
    }
}

impl InterpClock {
    /// Roll the clock forward: oldest drops, mid becomes oldest, newest
    /// becomes mid, the just-arrived snapshot becomes newest. Call once
    /// per applied snapshot after the per-entity buffers have rolled.
    pub fn record(&mut self, new_tick: u32, now: Instant) {
        self.from2_tick = self.from_tick;
        self.from_tick  = self.to_tick;
        self.to_tick    = new_tick;
        self.to_recv    = Some(now);
    }

    /// Reset on disconnect / forced teleport. Caller clears the per-entity
    /// buffers alongside.
    pub fn clear(&mut self) {
        self.from2_tick = 0;
        self.from_tick  = 0;
        self.to_tick    = 0;
        self.to_recv    = None;
    }

    /// Compute the lerp factor `t ∈ [0, 1]` between the oldest pair of
    /// buffered snapshots (`from2` → `from`). Returns `None` while we're
    /// still warming up (fewer than three snapshots seen) so the caller
    /// can render at the freshest available snapshot directly.
    ///
    /// Math:
    /// ```text
    /// sub_progress = (now - to_recv) * tick_hz, clamped to [0, 1]
    /// target_tick  = to_tick - 2 + sub_progress   // render 2 ticks behind
    /// t            = (target_tick - from2_tick) / (from_tick - from2_tick)
    /// ```
    /// In the happy path (snapshot every tick) `from2 = to-2`, `from = to-1`,
    /// so `target = (to-2) + sub` lies inside `[from2..from]` and `t = sub`
    /// — smoothly advancing from 0 at recv to 1 just before the next
    /// snapshot arrives. When the next snapshot lands, the slots roll
    /// (`from2 ← from`, `from ← to`, `to ← new`), `sub` resets to 0, and
    /// the renderer continues seamlessly from the same world position it
    /// just showed. No freeze, no pulse — at the cost of 1 tick (~33 ms
    /// @ 30 Hz) of perceived latency on remote entities.
    pub fn lerp_t(&self, now: Instant) -> Option<f32> {
        let recv = self.to_recv?;
        if self.from2_tick == 0 || self.from_tick == 0 || self.to_tick == 0 { return None; }
        if self.from_tick <= self.from2_tick { return None; }
        let elapsed = now.saturating_duration_since(recv).as_secs_f32();
        let sub = (elapsed * self.tick_hz).clamp(0.0, 1.0);
        let target = (self.to_tick as f32) - 2.0 + sub;
        let span   = (self.from_tick - self.from2_tick) as f32;
        let t      = (target - self.from2_tick as f32) / span;
        Some(t.clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn warming_up_returns_none() {
        let c = InterpClock::default();
        assert!(c.lerp_t(Instant::now()).is_none());
        // Even after one or two snapshots, still warming up.
        let mut c = InterpClock::default();
        let now = Instant::now();
        c.record(100, now);
        assert!(c.lerp_t(now).is_none());
        c.record(101, now);
        assert!(c.lerp_t(now).is_none(), "need three snapshots for proper render-behind");
    }

    #[test]
    fn third_snapshot_arms_clock_at_zero() {
        let mut c = InterpClock::default();
        let now = Instant::now();
        c.record(100, now);
        c.record(101, now);
        c.record(102, now);
        // target = 102 - 2 + 0 = 100; t = (100 - 100) / (101 - 100) = 0
        // Right after the third snap arrives we render at `from2`'s position
        // — the oldest of the three, two ticks behind real time.
        assert_eq!(c.lerp_t(now), Some(0.0));
    }

    #[test]
    fn lerps_smoothly_across_one_tick() {
        let mut c = InterpClock::default();
        let t0 = Instant::now();
        c.record(50, t0);
        c.record(51, t0);
        c.record(52, t0);
        // At t0 + 0ms: sub=0, target=50, t=0
        let v = c.lerp_t(t0).unwrap();
        assert!((v - 0.0).abs() < 1e-4, "v = {v}");
        // At t0 + 16ms: sub≈0.48, target≈50.48, t≈0.48
        let v = c.lerp_t(t0 + Duration::from_millis(16)).unwrap();
        assert!((v - 0.48).abs() < 0.05, "v = {v}");
        // At t0 + 33ms: sub=0.99→1, target≈51, t=1.0 (about to roll)
        let v = c.lerp_t(t0 + Duration::from_millis(33)).unwrap();
        assert!((v - 1.0).abs() < 0.05, "v = {v}");
    }

    #[test]
    fn rolls_seamlessly_across_snapshot_arrival() {
        // The key invariant: at the moment a new snapshot rolls in, the
        // rendered position is the same as it was the frame before — no
        // discontinuity. After roll, from2 holds what `from` did. So if
        // the previous frame had t≈1.0 on (from2_old, from_old), the
        // next frame must have t≈0.0 on (from2_new=from_old, from_new=to_old).
        let mut c = InterpClock::default();
        let t0 = Instant::now();
        c.record(50, t0);
        c.record(51, t0);
        c.record(52, t0);
        let just_before_roll = t0 + Duration::from_millis(33);
        let v_before = c.lerp_t(just_before_roll).unwrap();
        // New snapshot arrives.
        c.record(53, just_before_roll);
        let v_after = c.lerp_t(just_before_roll).unwrap();
        // Before: rendering at from2=50 + 1.0*(51-50) = 51's position
        // After:  rendering at from2=51 + 0.0*(52-51) = 51's position
        // Both equal `from2_new` = 51 → no jump.
        assert!((v_before - 1.0).abs() < 0.05, "v_before = {v_before}");
        assert!((v_after  - 0.0).abs() < 0.05, "v_after  = {v_after}");
    }

    #[test]
    fn missed_snapshot_still_lerps_in_range() {
        let mut c = InterpClock::default();
        let t0 = Instant::now();
        c.record(50, t0);
        c.record(51, t0);
        c.record(53, t0); // skipped tick 52
        // target = 53 - 2 + 0 = 51; span = 51-50 = 1; t = (51-50)/1 = 1.0
        // We're rendering exactly at `from`'s tick (51), at the upper end
        // of the buffered pair. Acceptable — slight jump on the missed
        // tick is unavoidable without more buffer depth.
        let v = c.lerp_t(t0).unwrap();
        assert!((v - 1.0).abs() < 1e-4, "v = {v}");
    }

    #[test]
    fn saturates_when_no_new_snapshot_arrives() {
        let mut c = InterpClock::default();
        let t0 = Instant::now();
        c.record(10, t0);
        c.record(11, t0);
        c.record(12, t0);
        let stale = t0 + Duration::from_secs(1);
        assert_eq!(c.lerp_t(stale), Some(1.0));
    }

    #[test]
    fn clear_disarms() {
        let mut c = InterpClock::default();
        let now = Instant::now();
        c.record(1, now);
        c.record(2, now);
        c.record(3, now);
        c.clear();
        assert!(c.lerp_t(now).is_none());
    }
}
