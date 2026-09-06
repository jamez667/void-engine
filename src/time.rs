pub const FIXED_DT: f32 = 1.0 / 60.0;

pub struct Timestep {
    accumulator: f32,
}

impl Timestep {
    pub fn new() -> Self {
        Self { accumulator: 0.0 }
    }

    pub fn advance(&mut self, frame_dt: f32) -> (u32, f32) {
        self.accumulator += frame_dt.min(0.25);
        let steps = (self.accumulator / FIXED_DT) as u32;
        self.accumulator -= steps as f32 * FIXED_DT;
        let alpha = self.accumulator / FIXED_DT;
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
        self.accumulator += steps as f32 * FIXED_DT;
        // Cap for the same reason `advance` clamps `frame_dt`: a long stall
        // must not build a debt that then spends itself as a hundred-step
        // catch-up burst the moment the peer reconnects.
        self.accumulator = self.accumulator.min(0.25);
    }
}

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
}
