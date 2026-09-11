//! What the status page reports, gathered into one value.
//!
//! Assembled by the caller rather than read from a global: the engine owns
//! no ledger and no loop handle, and inventing a registry for them would
//! be a worse coupling than passing the numbers in.
//!
//! # Why a snapshot type
//!
//! The HTML and the JSON must agree. If each rendered straight from the
//! live ledger they would sample it at different instants, and an operator
//! comparing the page against the endpoint during an incident would see
//! two different stories and trust neither. So both render from one
//! [`Status`], captured once.

use crate::persist::ledger::{Account, Discrepancy, Ledger};
use crate::persist::store::LedgerStore;

/// A balance-cache drift finding: the cached balance disagrees with the
/// sum of the log.
///
/// The cache exists so a read is O(1) rather than a scan. If it can drift,
/// that is a dupe vector — hence a page that shows it rather than a number
/// only a test ever looks at.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Drift {
    pub account: Account,
    pub asset: String,
    /// What the cache says.
    pub cached: i64,
    /// What the log sums to.
    pub actual: i64,
}

/// The durable writer's state, when there is one.
///
/// Mirrors `ledger_pg::WriterHealth` without depending on it: the `admin`
/// feature implies `ledger`, not `ledger-pg`, and a page that compiled
/// only with Postgres would be useless to the in-memory tier that needs
/// it just as much.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Writer {
    /// No durable backend — the in-memory ledger. Not a fault.
    None,
    Healthy,
    /// Retrying a transient fault. Writes refused; clears itself.
    Degraded(String),
    /// Terminal. Writes refused permanently; needs a human.
    Failed(String),
}

impl Writer {
    /// Whether this state needs someone to look at it.
    pub fn is_alarming(&self) -> bool {
        matches!(self, Writer::Degraded(_) | Writer::Failed(_))
    }

    pub fn label(&self) -> &'static str {
        match self {
            Writer::None => "in-memory",
            Writer::Healthy => "healthy",
            Writer::Degraded(_) => "degraded",
            Writer::Failed(_) => "failed",
        }
    }

    /// The reason, for the states that carry one.
    pub fn reason(&self) -> Option<&str> {
        match self {
            Writer::Degraded(why) | Writer::Failed(why) => Some(why),
            _ => None,
        }
    }
}

/// Everything the page shows, sampled at one instant.
#[derive(Clone, Debug, Default)]
pub struct Status {
    // ── the audits ───────────────────────────────────────────────────
    /// Assets whose books do not sum to zero. **Empty is the only
    /// acceptable value**: a non-zero sum means value entered or left
    /// outside the transfer API.
    pub zero_sum: Vec<Discrepancy>,
    /// Accounts whose cached balance disagrees with the log.
    pub drift: Vec<Drift>,

    // ── the reservation backlog ──────────────────────────────────────
    /// Holds resident in the ledger, live or lapsed. This is what sets
    /// the cost of every spend check.
    pub reservations: usize,
    /// How many of those have lapsed and await a sweep.
    pub lapsed_reservations: usize,

    // ── durability ───────────────────────────────────────────────────
    pub writer: WriterState,
    /// Highest tick durably committed. A checkpoint may never claim a
    /// tick above this.
    pub acked_tick: u64,

    // ── the loop ─────────────────────────────────────────────────────
    /// Most recent tick-health report, if the server is measuring.
    pub tick: Option<TickSummary>,
}

/// Writer state plus its queue depth, which only mean anything together.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WriterState {
    pub state: Writer,
    /// Transfers accepted but not yet committed. A number that only grows
    /// means the writer is losing.
    pub journal_depth: usize,
}

impl Default for WriterState {
    fn default() -> Self {
        Self { state: Writer::None, journal_depth: 0 }
    }
}

/// The loop's own report, flattened from [`TickHealth`].
///
/// [`TickHealth`]: crate::app_headless::TickHealth
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TickSummary {
    pub hz: f64,
    pub target_hz: f64,
    /// Sim seconds discarded in the last window. Non-zero means the world
    /// is running behind wall clock and will never catch up.
    pub dropped_s: f64,
    pub mean_tick_ms: f64,
    pub worst_tick_ms: f64,
}

impl TickSummary {
    pub fn from_health(h: &crate::app_headless::TickHealth) -> Self {
        Self {
            hz: h.hz,
            target_hz: h.target_hz,
            dropped_s: h.dropped_s,
            mean_tick_ms: h.mean_tick_s * 1000.0,
            worst_tick_ms: h.worst_tick_s * 1000.0,
        }
    }

    /// Fraction of real time actually simulated.
    pub fn realtime_ratio(&self) -> f64 {
        if self.target_hz <= 0.0 { return 1.0 }
        (self.hz / self.target_hz).min(1.0)
    }

    pub fn keeping_up(&self) -> bool {
        self.dropped_s <= 0.0 && self.realtime_ratio() >= 0.99
    }
}

impl Status {
    /// Read everything a [`LedgerStore`] can answer.
    ///
    /// For a durable backend. `PgLedger` is the only implementor — the
    /// in-memory [`Ledger`] deliberately is not one, since the trait
    /// exists to describe what a *backend* provides and the core is what
    /// backends are built on. Use [`from_ledger`](Self::from_ledger) for
    /// that tier.
    ///
    /// The Postgres-only signals (writer health, journal depth) are not on
    /// the trait either, so they come from
    /// [`with_writer`](Self::with_writer).
    pub fn from_store(store: &dyn LedgerStore) -> Self {
        Self {
            zero_sum: store.audit_zero_sum(),
            acked_tick: store.acked_tick(),
            ..Default::default()
        }
    }

    /// Read everything from an in-memory [`Ledger`].
    ///
    /// The tier-3-without-Postgres case, and the one this page matters
    /// most to: there is no database to query, so if these audits are not
    /// surfaced here they are not surfaced anywhere.
    ///
    /// Equivalent to [`from_store`](Self::from_store) followed by
    /// [`with_ledger`](Self::with_ledger), for a ledger that is not a
    /// store.
    pub fn from_ledger(ledger: &Ledger, now_tick: u64) -> Self {
        Self {
            zero_sum: ledger.audit_zero_sum(),
            acked_tick: ledger.now_tick(),
            ..Default::default()
        }
        .with_ledger(ledger, now_tick)
    }

    /// Add what only the in-memory core can answer: the balance-cache
    /// audit and the reservation backlog.
    ///
    /// Separate from [`from_store`](Self::from_store) because
    /// `LedgerStore` deliberately does not expose them — the trait is the
    /// contract a backend satisfies, and these are properties of the
    /// in-memory representation that every backend happens to share.
    pub fn with_ledger(mut self, ledger: &Ledger, now_tick: u64) -> Self {
        self.drift = ledger
            .audit_balances_match_entries()
            .into_iter()
            .map(|(account, asset, cached, actual)| Drift { account, asset, cached, actual })
            .collect();
        self.reservations = ledger.reservation_count();
        self.lapsed_reservations = ledger.lapsed_reservations(now_tick);
        self
    }

    /// Add the durable writer's state.
    pub fn with_writer(mut self, state: Writer, journal_depth: usize) -> Self {
        self.writer = WriterState { state, journal_depth };
        self
    }

    /// Add the most recent loop health report.
    pub fn with_tick(mut self, tick: TickSummary) -> Self {
        self.tick = Some(tick);
        self
    }

    /// Whether anything on this page needs attention.
    ///
    /// Deliberately strict: a status page whose headline says "ok" while
    /// something below it is wrong trains an operator to ignore the
    /// headline. Any audit finding, any writer fault, any lost sim time.
    pub fn healthy(&self) -> bool {
        self.zero_sum.is_empty()
            && self.drift.is_empty()
            && !self.writer.state.is_alarming()
            && self.tick.map(|t| t.keeping_up()).unwrap_or(true)
    }

    /// Short reasons the status is not healthy, for the headline.
    pub fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        if !self.zero_sum.is_empty() {
            out.push(format!(
                "{} asset(s) do not sum to zero — value moved outside the API",
                self.zero_sum.len()
            ));
        }
        if !self.drift.is_empty() {
            out.push(format!(
                "{} balance(s) disagree with the log",
                self.drift.len()
            ));
        }
        match &self.writer.state {
            Writer::Degraded(why) => out.push(format!("writer degraded: {why}")),
            Writer::Failed(why) => out.push(format!("writer failed: {why}")),
            _ => {}
        }
        if let Some(t) = self.tick {
            if t.dropped_s > 0.0 {
                out.push(format!(
                    "{:.2}s of simulation dropped — the world is behind wall clock",
                    t.dropped_s
                ));
            } else if !t.keeping_up() {
                out.push(format!(
                    "running at {:.1} Hz against a target of {:.0}",
                    t.hz, t.target_hz
                ));
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clean_status_is_healthy() {
        let s = Status::default();
        assert!(s.healthy());
        assert!(s.problems().is_empty());
    }

    /// The headline must not say "ok" while an audit is failing.
    #[test]
    fn any_audit_finding_makes_it_unhealthy() {
        let s = Status {
            zero_sum: vec![Discrepancy { asset: "credits".into(), imbalance: 5 }],
            ..Default::default()
        };
        assert!(!s.healthy());
        assert_eq!(s.problems().len(), 1);
        assert!(s.problems()[0].contains("outside the API"));
    }

    #[test]
    fn drift_makes_it_unhealthy() {
        let s = Status {
            drift: vec![Drift {
                account: Account::player("alice"),
                asset: "credits".into(),
                cached: 10,
                actual: 7,
            }],
            ..Default::default()
        };
        assert!(!s.healthy());
        assert!(s.problems()[0].contains("disagree with the log"));
    }

    /// An in-memory ledger has no writer, and that is not a fault.
    #[test]
    fn no_durable_writer_is_not_alarming() {
        assert!(!Writer::None.is_alarming());
        assert!(!Writer::Healthy.is_alarming());
        assert!(Writer::Degraded("x".into()).is_alarming());
        assert!(Writer::Failed("x".into()).is_alarming());
        assert!(Status::default().healthy(), "the in-memory tier is healthy by default");
    }

    #[test]
    fn a_failed_writer_reports_its_reason() {
        let s = Status::default().with_writer(Writer::Failed("undefined table".into()), 0);
        assert!(!s.healthy());
        assert!(s.problems()[0].contains("undefined table"));
        assert_eq!(s.writer.state.label(), "failed");
        assert_eq!(s.writer.state.reason(), Some("undefined table"));
    }

    /// Dropped sim time is the N3 signal: the loop does not error, it just
    /// runs less world than wall clock.
    #[test]
    fn dropped_sim_time_is_a_problem_even_at_a_plausible_hz() {
        let s = Status::default().with_tick(TickSummary {
            hz: 29.9,
            target_hz: 30.0,
            dropped_s: 0.4,
            mean_tick_ms: 30.0,
            worst_tick_ms: 120.0,
        });
        assert!(!s.healthy(), "lost simulation must not read as healthy");
        assert!(s.problems()[0].contains("behind wall clock"));
    }

    #[test]
    fn a_loop_below_target_is_reported_even_with_nothing_dropped() {
        let s = Status::default().with_tick(TickSummary {
            hz: 21.0,
            target_hz: 30.0,
            dropped_s: 0.0,
            mean_tick_ms: 40.0,
            worst_tick_ms: 60.0,
        });
        assert!(!s.healthy());
        assert!(s.problems()[0].contains("21.0 Hz"));
        assert!((s.tick.unwrap().realtime_ratio() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn a_healthy_loop_adds_no_problems() {
        let s = Status::default().with_tick(TickSummary {
            hz: 30.0,
            target_hz: 30.0,
            dropped_s: 0.0,
            mean_tick_ms: 2.0,
            worst_tick_ms: 5.0,
        });
        assert!(s.healthy());
        assert!(s.problems().is_empty());
    }

    /// Every failing signal must appear, not just the first — an operator
    /// fixing one and finding another is worse than seeing both.
    #[test]
    fn problems_accumulate() {
        let s = Status {
            zero_sum: vec![Discrepancy { asset: "credits".into(), imbalance: 1 }],
            drift: vec![Drift {
                account: Account::player("bob"),
                asset: "credits".into(),
                cached: 1,
                actual: 2,
            }],
            ..Default::default()
        }
        .with_writer(Writer::Degraded("connection reset".into()), 12)
        .with_tick(TickSummary {
            hz: 10.0,
            target_hz: 30.0,
            dropped_s: 2.0,
            mean_tick_ms: 90.0,
            worst_tick_ms: 200.0,
        });

        let problems = s.problems();
        assert_eq!(problems.len(), 4, "got {problems:?}");
        assert!(!s.healthy());
    }
}
