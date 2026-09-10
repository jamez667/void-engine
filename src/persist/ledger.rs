//! Append-only value tracking: the tier that makes duping detectable.
//!
//! # Why a balance is not a number
//!
//! The obvious design stores `credits: i64` on a component and logs
//! changes beside it. That cannot detect duping, because the log is
//! advisory: any code path that forgets to write an entry silently
//! creates value, and you find out months later from a player.
//!
//! Here a balance is *derived*: [`Ledger::balance`] sums the entries.
//! There is no setter, so there is no path that changes a balance without
//! leaving a record. The record is the value.
//!
//! # Every entry has a counterparty
//!
//! Value moves; it never appears. A mob dropping loot moves it from
//! [`Account::Mint`]; an item consumed on use moves it to
//! [`Account::Burn`]. Both sides of a transfer are written in one call,
//! summing to zero.
//!
//! That is what [`Ledger::audit_zero_sum`] checks. If the whole ledger
//! does not sum to zero for an asset, something created value outside
//! this API — and it says so, with the amount, rather than leaving you to
//! infer it from a player's complaint.
//!
//! # Retries cannot double-spend
//!
//! Every transfer carries an [`IdemKey`] built from the session and the
//! client's sequence number. A retried packet — the classic dupe vector,
//! where a client resends a trade confirmation after a timeout — matches
//! an existing key and returns the *original* receipt instead of moving
//! value twice.
//!
//! # Amounts are integers
//!
//! [`Amount`] is `i64` minor units. Floating point loses exactness above
//! 2^53, so two paths to the same balance can differ in the last bits and
//! reconciliation becomes impossible to do exactly. void-claim's
//! `credits: f64` is the mistake this avoids.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A quantity of an asset, in minor units.
///
/// "Minor unit" is whatever indivisible step the game uses — cents for a
/// currency, whole items for an inventory. The engine does not know or
/// care about the display scale; it only guarantees exact arithmetic.
///
/// `i64` holds a ten-million-account economy at a hundred billion display
/// units each with several orders of magnitude to spare. Sums across the
/// whole ledger are accumulated in `i128` so an audit cannot itself
/// overflow.
pub type Amount = i64;

/// Who holds value.
///
/// The non-player variants are what make double-entry possible: value
/// arriving from a loot table still has a source, so the books balance.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Account {
    /// A player, keyed by a stable game-supplied id — not an `EntityId`,
    /// which is reused as entities despawn and would eventually point a
    /// balance at a different player.
    Player(String),
    /// A shop, bank, guild vault, or any other in-world holder.
    System(String),
    /// Where value enters the economy: quest rewards, loot, admin grants.
    /// Its balance is negative by construction and equals everything ever
    /// created.
    Mint,
    /// Where value leaves: repair costs, consumed items, sinks.
    Burn,
}

impl Account {
    pub fn player(id: impl Into<String>) -> Self {
        Account::Player(id.into())
    }
    pub fn system(id: impl Into<String>) -> Self {
        Account::System(id.into())
    }
    /// True for the two accounts that exist to balance the books rather
    /// than to hold anything.
    pub fn is_economy_boundary(&self) -> bool {
        matches!(self, Account::Mint | Account::Burn)
    }
}

/// What is being moved: `"credits"`, `"item:ore_iron"`, and so on.
///
/// A plain string for the same reason component names are: it is data the
/// game controls, it reads well in an audit, and it does not need a
/// registry of taken numbers.
pub type Asset = String;

/// Deduplication key for a transfer.
///
/// Built from the session and the client's own sequence number, so a
/// resent packet produces the same key. This is the single most important
/// anti-dupe mechanism here: without it, a client that retries a trade on
/// a timeout gets the value twice.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct IdemKey(String);

impl IdemKey {
    /// The normal construction: session identity plus the client's
    /// monotonic sequence number, plus what the action was.
    pub fn new(session: &str, client_seq: u64, action: &str) -> Self {
        // Length-prefixed, not delimiter-joined. A plain
        // `"{session}:{seq}:{action}"` is ambiguous whenever a field can
        // contain the delimiter: ("bob", 5, "5:trade") and
        // ("bob:5", 5, "trade") both render as "bob:5:5:trade", so one
        // player's genuine second purchase is silently deduplicated
        // against the first and the value never moves.
        //
        // That failure is invisible to the zero-sum audit, because
        // nothing is *forged* — the books still balance, a transfer just
        // vanishes. Prefixing each field with its byte length makes the
        // encoding injective, so distinct inputs cannot collide however
        // the caller composes them.
        IdemKey(format!(
            "{}:{session}|{client_seq}|{}:{action}",
            session.len(),
            action.len(),
        ))
    }

    /// For server-originated movements that have no client packet behind
    /// them — a tick-driven payout, an admin grant. The caller is
    /// responsible for uniqueness.
    pub fn server(unique: impl Into<String>) -> Self {
        IdemKey(unique.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One side of one movement. Append-only: entries are never updated or
/// deleted, and a correction is a new compensating entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    /// Position in the log, assigned on commit.
    pub seq: u64,
    pub idem_key: IdemKey,
    /// Simulation tick, for correlating with checkpoints.
    pub tick: u64,
    pub account: Account,
    /// The other side of this movement. Always present — that is the
    /// invariant the zero-sum audit rests on.
    pub counterparty: Account,
    pub asset: Asset,
    /// Signed: negative leaves `account`, positive arrives.
    pub delta: Amount,
    /// Why, from the game's own taxonomy: `"buy_kind"`, `"debt_garnish"`,
    /// `"quest_reward"`. This is what turns a suspicious number into a
    /// diagnosis during an investigation.
    pub reason: String,
    /// Who acted: the player, a GM, or the system. Present from day one
    /// so GM actions are auditable without a schema change.
    pub actor: String,
}

/// A committed transfer: the two entries it wrote.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Receipt {
    pub idem_key: IdemKey,
    pub debit_seq: u64,
    pub credit_seq: u64,
    pub amount: Amount,
    /// True when this returned an existing receipt rather than moving
    /// value again. A caller that cares — a UI wanting to avoid a second
    /// "purchased!" toast — can check it.
    pub deduplicated: bool,
}

/// What a transfer is asking for.
#[derive(Clone, Debug)]
pub struct TransferRequest {
    pub idem_key: IdemKey,
    pub from: Account,
    pub to: Account,
    pub asset: Asset,
    /// Must be positive. Direction is expressed by `from`/`to`, never by
    /// the sign, so a negative amount cannot quietly reverse a transfer.
    pub amount: Amount,
    pub reason: String,
    pub actor: String,
    pub tick: u64,
    /// The hold this transfer consumes, if any.
    ///
    /// Named explicitly rather than matched by amount: two holds on the
    /// same account for the same sum are indistinguishable, and picking
    /// the wrong one is a silent error on money. A mismatched or expired
    /// id is refused outright.
    ///
    /// `None` for an ordinary transfer that reserved nothing.
    pub spends: Option<ReservationId>,
}

/// Why a transfer was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LedgerError {
    /// Amount was zero or negative. Direction belongs in `from`/`to`.
    NonPositiveAmount(Amount),
    /// The account cannot cover it, counting funds already reserved by
    /// in-flight transfers.
    InsufficientFunds { account: Account, asset: Asset, available: Amount, needed: Amount },
    /// `from` and `to` are the same account: a no-op that would still
    /// write two entries and muddy an audit.
    SelfTransfer(Account),
    /// The arithmetic would overflow `i64`.
    Overflow,
    /// A reservation was released or spent that was never taken, or has
    /// already lapsed.
    UnknownReservation,
    /// A transfer named a hold belonging to a different account or asset.
    ///
    /// Refused rather than ignored: spending someone else's hold, or a
    /// hold on a different asset, is either a bug or an attempt.
    ReservationMismatch {
        id: ReservationId,
        expected_account: Account,
        expected_asset: Asset,
    },
    /// The named hold does not cover the amount being spent.
    ReservationTooSmall { id: ReservationId, held: Amount, needed: Amount },
    /// The durable writer has fallen too far behind and the journal has
    /// hit its cap.
    ///
    /// Refusing a purchase is recoverable; growing an unbounded queue
    /// until the process is swapped to death is not. A caller should
    /// treat this as "try again shortly", and an operator should treat a
    /// sustained one as an outage.
    WriterBehind { depth: usize, limit: usize },
    /// The durable writer is retrying a transient fault.
    ///
    /// Distinct from [`LedgerError::WriterFailed`]: this one clears by
    /// itself once the connection returns and reconciliation passes, so a
    /// caller should treat it as "try again shortly" rather than as an
    /// outage needing a human. Writes are refused either way, because
    /// value the durable store cannot accept must not be accepted.
    WriterDegraded(String),
    /// The durable writer died. Every subsequent transfer is refused.
    ///
    /// Continuing to accept value movements with no way to persist them
    /// would be silent data loss wearing the costume of a working ledger.
    WriterFailed(String),
}

impl std::fmt::Display for LedgerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LedgerError::NonPositiveAmount(a) =>
                write!(f, "transfer amount must be positive, got {a}"),
            LedgerError::InsufficientFunds { account, asset, available, needed } =>
                write!(f, "{account:?} has {available} {asset}, needs {needed}"),
            LedgerError::SelfTransfer(a) =>
                write!(f, "transfer from {a:?} to itself"),
            LedgerError::Overflow =>
                write!(f, "transfer would overflow the amount type"),
            LedgerError::UnknownReservation =>
                write!(f, "no such reservation, or it has expired"),
            LedgerError::ReservationMismatch { id, expected_account, expected_asset } =>
                write!(f, "reservation {id:?} is held by {expected_account:?} for \
                          {expected_asset}, not this transfer"),
            LedgerError::ReservationTooSmall { id, held, needed } =>
                write!(f, "reservation {id:?} holds {held}, needs {needed}"),
            LedgerError::WriterBehind { depth, limit } =>
                write!(f, "durable writer is behind: {depth} queued, limit {limit}"),
            LedgerError::WriterDegraded(why) =>
                write!(f, "durable writer is degraded, retrying: {why}"),
            LedgerError::WriterFailed(why) =>
                write!(f, "durable writer has failed: {why}"),
        }
    }
}

impl std::error::Error for LedgerError {}

/// A discrepancy found by reconciliation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Discrepancy {
    pub asset: Asset,
    /// How far from zero the books are. Non-zero means value was created
    /// or destroyed outside this API.
    pub imbalance: i128,
}

/// Handle to a hold on funds.
///
/// A hold is released three ways, and between them they close every path
/// by which funds used to get stuck:
///
/// - **Spent.** A [`TransferRequest`] naming this id in `spends` consumes
///   the hold on success. Only on success: a refused transfer leaves it
///   standing so the caller can retry against it.
/// - **Released.** [`Ledger::release`] gives it back unspent.
/// - **Expired.** The deadline passed. This is the backstop for a caller
///   that panics, disconnects, or simply forgets — the one case the other
///   two cannot cover, and the reason funds can no longer be locked for
///   the life of the process.
///
/// Expiry bites on *read*, not on a sweep: an elapsed hold stops counting
/// against [`Ledger::available`] immediately, so a server that never calls
/// [`Ledger::expire_reservations`] leaks a little memory but never a
/// player's money.
///
/// The id is named explicitly rather than matched by amount, because two
/// holds on one account for the same sum are indistinguishable and
/// picking the wrong one would be a silent error on money.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReservationId(u64);

/// A hold on funds, with a deadline.
///
/// The deadline is what stops an abandoned hold locking money forever.
/// A caller that reserves and then panics, disconnects, or simply forgets
/// used to depress `available` for the life of the process; now the hold
/// lapses and the funds come back on their own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reservation {
    pub account: Account,
    pub asset: Asset,
    pub amount: Amount,
    /// Simulation tick after which this hold no longer counts.
    ///
    /// Ticks rather than wall-clock: the ledger is driven by the sim, a
    /// replay must reproduce exactly, and a server that stalls should not
    /// have its holds expire early because real time kept moving.
    pub expires_after_tick: u64,
}

/// The in-memory ledger.
///
/// Phase 3 keeps everything in memory so the semantics can be proven
/// without a database in the way. Phase 4 puts the same API in front of
/// Postgres; the invariants tested here are what that has to preserve.
#[derive(Default)]
pub struct Ledger {
    entries: Vec<Entry>,
    /// Derived balances, kept incrementally so a read is O(1) rather than
    /// a scan. `audit_balances_match_entries` proves the cache never
    /// drifts from the log — if it can, this is a dupe vector.
    balances: HashMap<(Account, Asset), Amount>,
    /// Committed transfers, so a retry returns the original receipt.
    seen: HashMap<IdemKey, Receipt>,
    /// Funds committed to in-flight transfers but not yet written.
    reservations: HashMap<ReservationId, Reservation>,
    next_seq: u64,
    next_reservation: u64,
    /// Highest tick any transfer has carried.
    ///
    /// Lets `available` judge expiry without every caller threading a
    /// tick through. Monotonic: a late-arriving lower tick does not wind
    /// it back, because that would resurrect holds that had lapsed.
    now_tick: u64,
}

impl Ledger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Move `amount` of `asset` from one account to another.
    ///
    /// Writes two entries — a debit and a credit — or none at all. There
    /// is no partial state: the checks all happen before anything is
    /// appended.
    ///
    /// A repeated [`IdemKey`] returns the original receipt with
    /// `deduplicated: true` and moves nothing.
    pub fn transfer(&mut self, req: TransferRequest) -> Result<Receipt, LedgerError> {
        // Idempotency first: a retry must not even be validated against
        // current balances, because the player may since have spent the
        // money and a second check would wrongly fail a transfer that
        // already succeeded.
        if let Some(prior) = self.seen.get(&req.idem_key) {
            let mut r = prior.clone();
            r.deduplicated = true;
            return Ok(r);
        }

        if req.amount <= 0 {
            return Err(LedgerError::NonPositiveAmount(req.amount));
        }
        if req.from == req.to {
            return Err(LedgerError::SelfTransfer(req.from));
        }

        // Monotonic: a late-arriving lower tick must not wind the clock
        // back, or holds that had already lapsed would come back to life
        // and depress `available` again.
        self.now_tick = self.now_tick.max(req.tick);

        // A named hold must exist, be unexpired, match this account and
        // asset, and cover the amount. Checked before anything is
        // written, so a bad id refuses cleanly.
        if let Some(id) = req.spends {
            let held = self
                .reservations
                .get(&id)
                .filter(|r| r.expires_after_tick >= req.tick)
                .ok_or(LedgerError::UnknownReservation)?;
            if held.account != req.from || held.asset != req.asset {
                return Err(LedgerError::ReservationMismatch {
                    id,
                    expected_account: held.account.clone(),
                    expected_asset: held.asset.clone(),
                });
            }
            if held.amount < req.amount {
                return Err(LedgerError::ReservationTooSmall {
                    id,
                    held: held.amount,
                    needed: req.amount,
                });
            }
        }

        // Mint has no balance to check — it is where value comes from.
        // Everything else must cover the amount.
        //
        // A transfer spending its own hold checks against availability
        // *plus* that hold: the funds were set aside for exactly this, so
        // counting them as unavailable would refuse the spend the
        // reservation existed to guarantee.
        if !matches!(req.from, Account::Mint) {
            let own_hold = req
                .spends
                .and_then(|id| self.reservations.get(&id))
                .map(|r| r.amount)
                .unwrap_or(0);
            let available = self.available_at(&req.from, &req.asset, req.tick) + own_hold;
            if available < req.amount {
                return Err(LedgerError::InsufficientFunds {
                    account: req.from.clone(),
                    asset: req.asset.clone(),
                    available,
                    needed: req.amount,
                });
            }
        }

        // Check both sides for overflow before writing either.
        let from_after = self
            .balance(&req.from, &req.asset)
            .checked_sub(req.amount)
            .ok_or(LedgerError::Overflow)?;
        let to_after = self
            .balance(&req.to, &req.asset)
            .checked_add(req.amount)
            .ok_or(LedgerError::Overflow)?;

        let debit_seq = self.next_seq;
        let credit_seq = self.next_seq + 1;
        self.next_seq += 2;

        self.entries.push(Entry {
            seq: debit_seq,
            idem_key: req.idem_key.clone(),
            tick: req.tick,
            account: req.from.clone(),
            counterparty: req.to.clone(),
            asset: req.asset.clone(),
            delta: -req.amount,
            reason: req.reason.clone(),
            actor: req.actor.clone(),
        });
        self.entries.push(Entry {
            seq: credit_seq,
            idem_key: req.idem_key.clone(),
            tick: req.tick,
            account: req.to.clone(),
            counterparty: req.from.clone(),
            asset: req.asset.clone(),
            delta: req.amount,
            reason: req.reason,
            actor: req.actor,
        });

        self.balances.insert((req.from, req.asset.clone()), from_after);
        self.balances.insert((req.to, req.asset), to_after);

        let receipt = Receipt {
            idem_key: req.idem_key.clone(),
            debit_seq,
            credit_seq,
            amount: req.amount,
            deduplicated: false,
        };
        // Only now: a refused transfer must leave the hold standing so the
        // caller can retry against it. Releasing on entry would turn one
        // failed attempt into lost protection.
        if let Some(id) = req.spends {
            self.reservations.remove(&id);
        }

        self.seen.insert(req.idem_key, receipt.clone());
        Ok(receipt)
    }

    /// An account's balance for an asset, derived from the log.
    pub fn balance(&self, account: &Account, asset: &str) -> Amount {
        self.balances
            .get(&(account.clone(), asset.to_string()))
            .copied()
            .unwrap_or(0)
    }

    /// Balance minus anything reserved by in-flight transfers.
    ///
    /// This is what a spend must check against. Using the raw balance
    /// instead would let a player spend the same funds twice while a
    /// commit is in flight — the exact race the reservation exists for.
    pub fn available(&self, account: &Account, asset: &str) -> Amount {
        self.available_at(account, asset, self.now_tick)
    }

    /// `available` as of a given tick, so an expired hold stops counting
    /// the instant its deadline passes rather than when a sweep happens
    /// to run. Without this, forgetting `expire_reservations` would still
    /// lock funds — the deadline has to bite on read, not on sweep.
    pub fn available_at(&self, account: &Account, asset: &str, now_tick: u64) -> Amount {
        let reserved: Amount = self
            .reservations
            .values()
            .filter(|r| {
                &r.account == account && r.asset == asset && r.expires_after_tick >= now_tick
            })
            .map(|r| r.amount)
            .sum();
        self.balance(account, asset) - reserved
    }

    /// Hold funds against an in-flight transfer.
    ///
    /// Phase 4 needs this because a commit crosses a thread boundary and
    /// takes time; the sim must not let the same funds be spent again
    /// while it is in the air.
    pub fn reserve(
        &mut self,
        account: &Account,
        asset: &str,
        amount: Amount,
        expires_after_tick: u64,
    ) -> Result<ReservationId, LedgerError> {
        if amount <= 0 {
            return Err(LedgerError::NonPositiveAmount(amount));
        }
        let available = self.available(account, asset);
        if available < amount {
            return Err(LedgerError::InsufficientFunds {
                account: account.clone(),
                asset: asset.to_string(),
                available,
                needed: amount,
            });
        }
        let id = ReservationId(self.next_reservation);
        self.next_reservation += 1;
        self.reservations.insert(
            id,
            Reservation {
                account: account.clone(),
                asset: asset.to_string(),
                amount,
                expires_after_tick,
            },
        );
        Ok(id)
    }

    /// Drop every hold whose deadline has passed.
    ///
    /// Call once per tick. Returns how many lapsed, which is worth
    /// logging: a number that climbs means callers are reserving and then
    /// abandoning, and the funds were only recovered because of this
    /// sweep rather than because the code was correct.
    ///
    /// Expired holds already stop counting against `available` the moment
    /// their tick passes — this only reclaims the memory — so a server
    /// that forgets to call it leaks a little space but never locks a
    /// player's money.
    pub fn expire_reservations(&mut self, now_tick: u64) -> usize {
        let before = self.reservations.len();
        self.reservations
            .retain(|_, r| r.expires_after_tick >= now_tick);
        before - self.reservations.len()
    }

    /// Look up a hold, if it exists and has not lapsed.
    pub fn reservation(&self, id: ReservationId, now_tick: u64) -> Option<&Reservation> {
        self.reservations
            .get(&id)
            .filter(|r| r.expires_after_tick >= now_tick)
    }

    /// Release a reservation without spending it.
    pub fn release(&mut self, id: ReservationId) -> Result<(), LedgerError> {
        self.reservations
            .remove(&id)
            .map(|_| ())
            .ok_or(LedgerError::UnknownReservation)
    }

    /// Every entry, oldest first. The audit trail itself.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Entries touching one account, for a support investigation.
    ///
    /// The shared lifetime is explicit because the returned iterator
    /// borrows both the ledger and the account it is filtering on.
    /// Cloning the account instead would work, but this may be walked
    /// over a long history and there is no reason to pay for a `String`
    /// clone per call.
    pub fn history<'a>(&'a self, account: &'a Account) -> impl Iterator<Item = &'a Entry> + 'a {
        self.entries.iter().filter(move |e| &e.account == account)
    }

    /// **The dupe detector.** Every asset must sum to exactly zero across
    /// all accounts, `Mint` and `Burn` included.
    ///
    /// A non-zero result means value entered or left the economy without
    /// going through [`transfer`], which is either a bug or an exploit.
    /// Summed in `i128` so the audit itself cannot overflow.
    ///
    /// [`transfer`]: Ledger::transfer
    pub fn audit_zero_sum(&self) -> Vec<Discrepancy> {
        let mut totals: HashMap<&str, i128> = HashMap::new();
        for e in &self.entries {
            *totals.entry(e.asset.as_str()).or_insert(0) += e.delta as i128;
        }
        let mut out: Vec<Discrepancy> = totals
            .into_iter()
            .filter(|(_, sum)| *sum != 0)
            .map(|(asset, imbalance)| Discrepancy { asset: asset.to_string(), imbalance })
            .collect();
        // Deterministic order so a failing audit reads the same every run.
        out.sort_by(|a, b| a.asset.cmp(&b.asset));
        out
    }

    /// Prove the cached balances match what the log says.
    ///
    /// The cache exists so a balance read is O(1). If it can drift from
    /// the entries, it is itself a dupe vector — a player would spend
    /// money the log says they never had. Returns every account that
    /// disagrees.
    #[allow(clippy::type_complexity)]
    pub fn audit_balances_match_entries(&self) -> Vec<(Account, Asset, Amount, Amount)> {
        let mut derived: HashMap<(Account, Asset), Amount> = HashMap::new();
        for e in &self.entries {
            *derived.entry((e.account.clone(), e.asset.clone())).or_insert(0) += e.delta;
        }

        let mut out = Vec::new();
        // Every key in either map, so a cache entry with no entries
        // behind it is caught as well as the reverse.
        let keys: std::collections::BTreeSet<_> =
            derived.keys().chain(self.balances.keys()).cloned().collect();
        for key in keys {
            let d = derived.get(&key).copied().unwrap_or(0);
            let c = self.balances.get(&key).copied().unwrap_or(0);
            if d != c {
                out.push((key.0, key.1, c, d));
            }
        }
        out
    }

    /// The raw receipt for a key, if this ledger has seen it.
    ///
    /// `pub(crate)` because the trait's `receipt_for` is the public way
    /// to ask; this exists so a durable backend can attach its own
    /// durability rather than inheriting the in-memory answer.
    #[cfg(feature = "ledger-pg")]
    pub(crate) fn receipt_for_inner(&self, key: &IdemKey) -> Option<(Receipt, u64)> {
        let receipt = self.seen.get(key)?.clone();
        // The tick is not on `Receipt`, so recover it from the entry the
        // receipt points at. A durable backend needs it to answer "is this
        // committed yet" against the watermark; without it the only honest
        // answer is "don't know".
        let tick = self
            .entries
            .get(receipt.debit_seq as usize)
            .map(|e| e.tick)
            .unwrap_or(0);
        Some((receipt, tick))
    }

    /// Total ever created, as a positive number.
    ///
    /// Equals `-mint_balance` by construction. Worth surfacing because a
    /// sudden jump is the shape of an exploit even when the books still
    /// balance — a mint is legitimate value creation, so zero-sum will
    /// not flag it.
    pub fn total_minted(&self, asset: &str) -> i128 {
        -(self.balance(&Account::Mint, asset) as i128)
    }
}

impl crate::persist::store::LedgerStore for Ledger {
    fn transfer(
        &mut self,
        req: TransferRequest,
    ) -> Result<crate::persist::store::StoredReceipt, LedgerError> {
        let receipt = Ledger::transfer(self, req)?;
        Ok(crate::persist::store::StoredReceipt {
            receipt,
            // Nothing durable is behind this backend, and saying otherwise
            // would let a caller believe a value movement had survived a
            // crash when it had not.
            durability: crate::persist::store::Durability::Memory,
        })
    }

    fn balance(&self, account: &Account, asset: &str) -> Amount {
        Ledger::balance(self, account, asset)
    }

    fn available(&self, account: &Account, asset: &str) -> Amount {
        Ledger::available(self, account, asset)
    }

    fn reserve(
        &mut self,
        account: &Account,
        asset: &str,
        amount: Amount,
        expires_after_tick: u64,
    ) -> Result<ReservationId, LedgerError> {
        Ledger::reserve(self, account, asset, amount, expires_after_tick)
    }

    fn release(&mut self, id: ReservationId) -> Result<(), LedgerError> {
        Ledger::release(self, id)
    }

    fn history(&self, account: &Account) -> Vec<Entry> {
        Ledger::history(self, account).cloned().collect()
    }

    fn audit_zero_sum(&self) -> Vec<Discrepancy> {
        Ledger::audit_zero_sum(self)
    }

    fn total_minted(&self, asset: &str) -> i128 {
        Ledger::total_minted(self, asset)
    }

    fn acked_tick(&self) -> u64 {
        // No durable store, so nothing can outlive this process and there
        // is nothing to be behind: the highest tick seen is acked by
        // definition.
        self.entries.iter().map(|e| e.tick).max().unwrap_or(0)
    }

    fn receipt_for(&self, key: &IdemKey) -> Option<crate::persist::store::StoredReceipt> {
        self.seen.get(key).map(|r| crate::persist::store::StoredReceipt {
            receipt: r.clone(),
            durability: crate::persist::store::Durability::Memory,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Far enough out that expiry never fires. Tests that care about
    /// expiry set their own deadline.
    const NEVER: u64 = u64::MAX;

    fn req(from: Account, to: Account, amount: Amount, key: &str) -> TransferRequest {
        TransferRequest {
            idem_key: IdemKey::server(key),
            from,
            to,
            asset: "credits".to_string(),
            amount,
            reason: "test".to_string(),
            actor: "system".to_string(),
            tick: 1,
            spends: None,
        }
    }

    fn funded(who: &str, amount: Amount) -> (Ledger, Account) {
        let mut l = Ledger::new();
        let acct = Account::player(who);
        l.transfer(req(Account::Mint, acct.clone(), amount, "seed")).unwrap();
        (l, acct)
    }

    /// Two genuinely different transfers must never share a key.
    ///
    /// A delimiter-joined `"{session}:{seq}:{action}"` is ambiguous
    /// whenever a field can contain the delimiter: ("bob", 5, "5:trade")
    /// and ("bob:5", 5, "trade") both rendered as "bob:5:5:trade", so the
    /// second player's real purchase was silently deduplicated against the
    /// first and the value never moved. Invisible to the zero-sum audit,
    /// because nothing is forged — a transfer simply vanishes.
    #[test]
    fn distinct_inputs_cannot_collide_on_one_idem_key() {
        let a = IdemKey::new("bob", 5, "5:trade");
        let b = IdemKey::new("bob:5", 5, "trade");
        assert_ne!(a, b, "delimiter injection must not forge a collision");

        let c = IdemKey::new("s", 1, "x");
        let d = IdemKey::new("s", 1, "x");
        assert_eq!(c, d, "identical inputs must still produce one key");
    }

    /// The end-to-end consequence: both purchases must land.
    #[test]
    fn a_colliding_shaped_pair_moves_value_twice() {
        let (mut l, alice) = funded("alice", 1000);
        let shop = Account::system("shop");

        let mut first = req(alice.clone(), shop.clone(), 100, "unused");
        first.idem_key = IdemKey::new("bob", 5, "5:trade");
        let mut second = req(alice.clone(), shop.clone(), 700, "unused");
        second.idem_key = IdemKey::new("bob:5", 5, "trade");

        assert!(!l.transfer(first).unwrap().deduplicated);
        assert!(!l.transfer(second).unwrap().deduplicated, "the second must not be swallowed");
        assert_eq!(l.balance(&alice, "credits"), 200);
        assert_eq!(l.balance(&shop, "credits"), 800);
        assert!(l.audit_zero_sum().is_empty());
    }

    #[test]
    fn a_transfer_moves_value_both_ways() {
        let (mut l, alice) = funded("alice", 1000);
        let bob = Account::player("bob");

        l.transfer(req(alice.clone(), bob.clone(), 300, "t1")).unwrap();

        assert_eq!(l.balance(&alice, "credits"), 700);
        assert_eq!(l.balance(&bob, "credits"), 300);
    }

    #[test]
    fn each_transfer_writes_two_entries() {
        let (mut l, alice) = funded("alice", 100);
        let before = l.entries().len();
        l.transfer(req(alice, Account::player("bob"), 10, "t1")).unwrap();
        assert_eq!(l.entries().len(), before + 2, "double-entry: debit and credit");
    }

    /// The books must balance. This is the dupe detector, so it gets the
    /// most direct possible test.
    #[test]
    fn the_books_balance_after_arbitrary_activity() {
        let (mut l, alice) = funded("alice", 5000);
        let bob = Account::player("bob");
        let shop = Account::system("shop");

        l.transfer(req(alice.clone(), bob.clone(), 1200, "a")).unwrap();
        l.transfer(req(bob.clone(), shop.clone(), 700, "b")).unwrap();
        l.transfer(req(shop.clone(), Account::Burn, 200, "c")).unwrap();
        l.transfer(req(Account::Mint, bob.clone(), 50, "d")).unwrap();

        assert!(l.audit_zero_sum().is_empty(), "books did not balance");
        assert!(l.audit_balances_match_entries().is_empty(), "cache drifted from the log");
    }

    /// A retried packet is the classic dupe vector. It must return the
    /// original receipt and move nothing.
    #[test]
    fn a_retried_transfer_does_not_move_value_twice() {
        let (mut l, alice) = funded("alice", 1000);
        let bob = Account::player("bob");
        let key = IdemKey::new("session-7", 42, "trade");

        let mut first = req(alice.clone(), bob.clone(), 250, "unused");
        first.idem_key = key.clone();
        let r1 = l.transfer(first.clone()).unwrap();
        assert!(!r1.deduplicated);

        // The client resends after a timeout.
        let r2 = l.transfer(first).unwrap();
        assert!(r2.deduplicated, "a repeat must be recognised");
        assert_eq!(r1.debit_seq, r2.debit_seq, "must return the original receipt");

        assert_eq!(l.balance(&alice, "credits"), 750, "value moved twice");
        assert_eq!(l.balance(&bob, "credits"), 250);
        assert!(l.audit_zero_sum().is_empty());
    }

    /// A retry must succeed even if the player has since spent the money,
    /// because the original transfer already happened.
    #[test]
    fn a_retry_succeeds_even_when_funds_are_now_short() {
        let (mut l, alice) = funded("alice", 100);
        let bob = Account::player("bob");

        let mut first = req(alice.clone(), bob.clone(), 60, "k1");
        first.idem_key = IdemKey::new("s", 1, "buy");
        l.transfer(first.clone()).unwrap();

        // Alice spends the rest elsewhere.
        l.transfer(req(alice.clone(), Account::Burn, 40, "k2")).unwrap();
        assert_eq!(l.balance(&alice, "credits"), 0);

        // The original packet arrives again. It must not fail.
        let again = l.transfer(first).unwrap();
        assert!(again.deduplicated);
        assert_eq!(l.balance(&alice, "credits"), 0, "a retry must not move value");
    }

    #[test]
    fn spending_more_than_you_have_is_refused() {
        let (mut l, alice) = funded("alice", 100);
        let err = l.transfer(req(alice.clone(), Account::player("bob"), 101, "t")).unwrap_err();
        assert!(matches!(err, LedgerError::InsufficientFunds { available: 100, needed: 101, .. }));
        assert_eq!(l.balance(&alice, "credits"), 100, "a refused transfer must change nothing");
    }

    #[test]
    fn a_refused_transfer_writes_no_entries() {
        let (mut l, alice) = funded("alice", 10);
        let before = l.entries().len();
        let _ = l.transfer(req(alice, Account::player("bob"), 999, "t"));
        assert_eq!(l.entries().len(), before, "a failure must leave no partial state");
    }

    #[test]
    fn a_negative_amount_cannot_reverse_a_transfer() {
        let (mut l, alice) = funded("alice", 100);
        let err = l.transfer(req(alice, Account::player("bob"), -50, "t")).unwrap_err();
        assert_eq!(err, LedgerError::NonPositiveAmount(-50));
    }

    #[test]
    fn a_zero_amount_is_refused() {
        let (mut l, alice) = funded("alice", 100);
        assert_eq!(
            l.transfer(req(alice, Account::player("bob"), 0, "t")).unwrap_err(),
            LedgerError::NonPositiveAmount(0),
        );
    }

    #[test]
    fn a_self_transfer_is_refused() {
        let (mut l, alice) = funded("alice", 100);
        let err = l.transfer(req(alice.clone(), alice, 10, "t")).unwrap_err();
        assert!(matches!(err, LedgerError::SelfTransfer(_)));
    }

    /// Mint has no balance requirement — it is where value enters.
    #[test]
    fn mint_can_pay_out_from_nothing_and_goes_negative() {
        let mut l = Ledger::new();
        let alice = Account::player("alice");
        l.transfer(req(Account::Mint, alice.clone(), 500, "reward")).unwrap();

        assert_eq!(l.balance(&alice, "credits"), 500);
        assert_eq!(l.balance(&Account::Mint, "credits"), -500);
        assert_eq!(l.total_minted("credits"), 500);
        assert!(l.audit_zero_sum().is_empty(), "minting must still balance");
    }

    // ── reservations ────────────────────────────────────────────────

    #[test]
    fn a_reservation_reduces_available_but_not_balance() {
        let (mut l, alice) = funded("alice", 1000);
        let _r = l.reserve(&alice, "credits", 400, NEVER).unwrap();

        assert_eq!(l.balance(&alice, "credits"), 1000, "balance is unchanged");
        assert_eq!(l.available(&alice, "credits"), 600, "available is reduced");
    }

    /// The race the reservation exists for: funds in flight cannot be
    /// spent again.
    #[test]
    fn reserved_funds_cannot_be_spent_twice() {
        let (mut l, alice) = funded("alice", 1000);
        l.reserve(&alice, "credits", 800, NEVER).unwrap();

        let err = l.transfer(req(alice, Account::player("bob"), 300, "t")).unwrap_err();
        assert!(matches!(err, LedgerError::InsufficientFunds { available: 200, .. }));
    }

    #[test]
    fn releasing_a_reservation_restores_availability() {
        let (mut l, alice) = funded("alice", 1000);
        let r = l.reserve(&alice, "credits", 800, NEVER).unwrap();
        l.release(r).unwrap();
        assert_eq!(l.available(&alice, "credits"), 1000);
    }

    #[test]
    fn over_reserving_is_refused() {
        let (mut l, alice) = funded("alice", 100);
        l.reserve(&alice, "credits", 60, NEVER).unwrap();
        assert!(l.reserve(&alice, "credits", 50, NEVER).is_err(), "only 40 remains");
    }

    /// The bug this change exists to fix: a hold used to survive the
    /// transfer it was taken for, depressing `available` forever.
    #[test]
    fn a_spent_reservation_is_released() {
        let (mut l, alice) = funded("alice", 1000);
        let bob = Account::player("bob");
        let hold = l.reserve(&alice, "credits", 300, NEVER).unwrap();
        assert_eq!(l.available(&alice, "credits"), 700, "the hold is counted");

        let mut r = req(alice.clone(), bob.clone(), 300, "spend");
        r.spends = Some(hold);
        l.transfer(r).unwrap();

        assert_eq!(l.balance(&alice, "credits"), 700);
        assert_eq!(
            l.available(&alice, "credits"),
            700,
            "the hold must be gone, not still counted against a balance it already left",
        );
        assert_eq!(l.release(hold), Err(LedgerError::UnknownReservation));
    }

    /// A refused transfer must leave the hold standing, or one failed
    /// attempt would silently drop the protection it was taken for.
    #[test]
    fn a_refused_transfer_keeps_its_reservation() {
        let (mut l, alice) = funded("alice", 1000);
        let hold = l.reserve(&alice, "credits", 300, NEVER).unwrap();

        // Self-transfer is refused after the hold is validated.
        let mut r = req(alice.clone(), alice.clone(), 300, "bad");
        r.spends = Some(hold);
        assert!(l.transfer(r).is_err());

        assert_eq!(l.available(&alice, "credits"), 700, "the hold must survive a refusal");
        l.release(hold).expect("and still be releasable");
    }

    /// Spending a hold belonging to someone else, or on another asset, is
    /// refused rather than quietly ignored.
    #[test]
    fn a_mismatched_reservation_is_refused() {
        let (mut l, alice) = funded("alice", 1000);
        let bob = Account::player("bob");
        l.transfer(req(Account::Mint, bob.clone(), 500, "seed-bob")).unwrap();

        let bobs_hold = l.reserve(&bob, "credits", 200, NEVER).unwrap();
        let mut r = req(alice.clone(), bob.clone(), 100, "steal");
        r.spends = Some(bobs_hold);

        assert!(matches!(
            l.transfer(r),
            Err(LedgerError::ReservationMismatch { .. }),
        ));
        assert_eq!(l.available(&bob, "credits"), 300, "bob's hold is untouched");
    }

    #[test]
    fn a_reservation_smaller_than_the_spend_is_refused() {
        let (mut l, alice) = funded("alice", 1000);
        let hold = l.reserve(&alice, "credits", 100, NEVER).unwrap();
        let mut r = req(alice.clone(), Account::player("bob"), 500, "too-big");
        r.spends = Some(hold);

        assert!(matches!(
            l.transfer(r),
            Err(LedgerError::ReservationTooSmall { held: 100, needed: 500, .. }),
        ));
    }

    #[test]
    fn an_unknown_reservation_id_is_refused() {
        let (mut l, alice) = funded("alice", 1000);
        let mut r = req(alice, Account::player("bob"), 10, "ghost");
        r.spends = Some(ReservationId(9999));
        assert_eq!(l.transfer(r), Err(LedgerError::UnknownReservation));
    }

    /// The deadline has to bite on *read*, not on a sweep — otherwise a
    /// server that forgets to call `expire_reservations` still locks
    /// funds, which is the very failure the deadline exists to prevent.
    #[test]
    fn an_expired_hold_stops_counting_without_a_sweep() {
        let (mut l, alice) = funded("alice", 1000);
        l.reserve(&alice, "credits", 400, 10).unwrap();

        assert_eq!(l.available_at(&alice, "credits", 10), 600, "still held at its deadline");
        assert_eq!(
            l.available_at(&alice, "credits", 11),
            1000,
            "lapsed the tick after, with no sweep having run",
        );
    }

    #[test]
    fn expire_reservations_reclaims_lapsed_holds() {
        let (mut l, alice) = funded("alice", 1000);
        l.reserve(&alice, "credits", 100, 5).unwrap();
        l.reserve(&alice, "credits", 100, 50).unwrap();

        assert_eq!(l.expire_reservations(6), 1, "only the lapsed one goes");
        assert_eq!(l.expire_reservations(6), 0, "and it is gone for good");
        assert_eq!(l.available_at(&alice, "credits", 6), 900, "the live hold still counts");
    }

    /// An expired hold cannot be spent: the funds are no longer set aside,
    /// so honouring it would let a stale request jump the queue.
    #[test]
    fn an_expired_reservation_cannot_be_spent() {
        let (mut l, alice) = funded("alice", 1000);
        let hold = l.reserve(&alice, "credits", 300, 5).unwrap();

        let mut r = req(alice.clone(), Account::player("bob"), 300, "late");
        r.spends = Some(hold);
        r.tick = 6;

        assert_eq!(l.transfer(r), Err(LedgerError::UnknownReservation));
    }

    /// A hold guarantees the spend it was taken for. Other traffic must
    /// not be able to consume the funds underneath it.
    #[test]
    fn a_hold_guarantees_its_own_spend() {
        let (mut l, alice) = funded("alice", 1000);
        let bob = Account::player("bob");
        let hold = l.reserve(&alice, "credits", 900, NEVER).unwrap();

        // Everything else sees only the unreserved remainder.
        assert!(l.transfer(req(alice.clone(), bob.clone(), 200, "other")).is_err());

        // But the reserved spend itself goes through.
        let mut r = req(alice.clone(), bob.clone(), 900, "reserved");
        r.spends = Some(hold);
        l.transfer(r).expect("a hold must guarantee the spend it was taken for");
        assert_eq!(l.balance(&alice, "credits"), 100);
    }

    #[test]
    fn releasing_an_unknown_reservation_errors() {
        let mut l = Ledger::new();
        assert_eq!(l.release(ReservationId(99)), Err(LedgerError::UnknownReservation));
    }

    // ── audits catch what they exist to catch ───────────────────────

    /// Prove the zero-sum audit is not vacuous: forge an entry the way a
    /// bug or an exploit would, and it must be caught.
    #[test]
    fn the_zero_sum_audit_catches_forged_value() {
        let (mut l, alice) = funded("alice", 100);

        // Simulate value appearing with no counterparty — exactly what
        // bypassing the API would do.
        l.entries.push(Entry {
            seq: 999,
            idem_key: IdemKey::server("forged"),
            tick: 1,
            account: alice,
            counterparty: Account::system("nowhere"),
            asset: "credits".to_string(),
            delta: 1_000_000,
            reason: "dupe".to_string(),
            actor: "cheater".to_string(),
        });

        let found = l.audit_zero_sum();
        assert_eq!(found.len(), 1, "the imbalance must be detected");
        assert_eq!(found[0].imbalance, 1_000_000);
        assert_eq!(found[0].asset, "credits");
    }

    /// And prove the cache audit is not vacuous either.
    #[test]
    fn the_balance_audit_catches_a_drifted_cache() {
        let (mut l, alice) = funded("alice", 100);
        l.balances.insert((alice.clone(), "credits".to_string()), 999_999);

        let drift = l.audit_balances_match_entries();
        assert_eq!(drift.len(), 1);
        assert_eq!(drift[0].0, alice);
        assert_eq!((drift[0].2, drift[0].3), (999_999, 100), "(cached, derived)");
    }

    #[test]
    fn a_clean_ledger_audits_clean() {
        let l = Ledger::new();
        assert!(l.audit_zero_sum().is_empty());
        assert!(l.audit_balances_match_entries().is_empty());
    }

    // ── the audit trail is usable ───────────────────────────────────

    #[test]
    fn history_shows_both_sides_of_an_investigation() {
        let (mut l, alice) = funded("alice", 1000);
        let bob = Account::player("bob");
        l.transfer(req(alice.clone(), bob.clone(), 100, "t1")).unwrap();
        l.transfer(req(bob.clone(), alice.clone(), 30, "t2")).unwrap();

        let alice_entries: Vec<_> = l.history(&alice).collect();
        assert_eq!(alice_entries.len(), 3, "seed in, 100 out, 30 in");
        let net: Amount = alice_entries.iter().map(|e| e.delta).sum();
        assert_eq!(net, 930);
    }

    #[test]
    fn entries_record_who_and_why() {
        let mut l = Ledger::new();
        let alice = Account::player("alice");
        l.transfer(TransferRequest {
            idem_key: IdemKey::server("gm-1"),
            from: Account::Mint,
            to: alice.clone(),
            asset: "credits".to_string(),
            amount: 500,
            reason: "gm_grant".to_string(),
            actor: "gm:kate".to_string(),
            tick: 77,
            spends: None,
        })
        .unwrap();

        let e = l.history(&alice).next().unwrap();
        assert_eq!(e.reason, "gm_grant");
        assert_eq!(e.actor, "gm:kate");
        assert_eq!(e.tick, 77, "tick correlates the entry with a checkpoint");
        assert_eq!(e.counterparty, Account::Mint);
    }

    /// Items use the same machinery as currency — a unique item is just
    /// an asset with a total supply of one.
    #[test]
    fn items_are_ledgered_like_currency() {
        let mut l = Ledger::new();
        let alice = Account::player("alice");
        let bob = Account::player("bob");
        let sword = "item:sword_of_dawn";

        let mint = TransferRequest {
            idem_key: IdemKey::server("mint-sword"),
            from: Account::Mint,
            to: alice.clone(),
            asset: sword.to_string(),
            amount: 1,
            reason: "quest_reward".to_string(),
            actor: "system".to_string(),
            tick: 1,
            spends: None,
        };
        l.transfer(mint).unwrap();

        let mut trade = req(alice.clone(), bob.clone(), 1, "trade-sword");
        trade.asset = sword.to_string();
        l.transfer(trade).unwrap();

        assert_eq!(l.balance(&alice, sword), 0, "alice no longer holds it");
        assert_eq!(l.balance(&bob, sword), 1, "bob does");
        assert_eq!(l.total_minted(sword), 1, "exactly one was ever created");
        assert!(l.audit_zero_sum().is_empty());
    }

    #[test]
    fn assets_do_not_bleed_into_each_other() {
        let (mut l, alice) = funded("alice", 100);
        let mut ore = req(alice.clone(), Account::player("bob"), 5, "ore");
        ore.asset = "item:ore".to_string();

        // Alice has no ore, so this must fail on ore, not succeed by
        // borrowing from her credits.
        assert!(l.transfer(ore).is_err());
        assert_eq!(l.balance(&alice, "credits"), 100);
    }
}
