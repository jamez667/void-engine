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
        IdemKey(format!("{session}:{client_seq}:{action}"))
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
    /// A reservation was released that was never taken.
    UnknownReservation,
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
                write!(f, "released a reservation that was never taken"),
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

/// Handle to an in-flight reservation, released when the transfer
/// commits or is abandoned.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReservationId(u64);

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
    reservations: HashMap<ReservationId, (Account, Asset, Amount)>,
    next_seq: u64,
    next_reservation: u64,
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

        // Mint has no balance to check — it is where value comes from.
        // Everything else must cover the amount, reservations included.
        if !matches!(req.from, Account::Mint) {
            let available = self.available(&req.from, &req.asset);
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
        let reserved: Amount = self
            .reservations
            .values()
            .filter(|(a, s, _)| a == account && s == asset)
            .map(|(_, _, amt)| *amt)
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
        self.reservations
            .insert(id, (account.clone(), asset.to_string(), amount));
        Ok(id)
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

#[cfg(test)]
mod tests {
    use super::*;

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
        }
    }

    fn funded(who: &str, amount: Amount) -> (Ledger, Account) {
        let mut l = Ledger::new();
        let acct = Account::player(who);
        l.transfer(req(Account::Mint, acct.clone(), amount, "seed")).unwrap();
        (l, acct)
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
        let _r = l.reserve(&alice, "credits", 400).unwrap();

        assert_eq!(l.balance(&alice, "credits"), 1000, "balance is unchanged");
        assert_eq!(l.available(&alice, "credits"), 600, "available is reduced");
    }

    /// The race the reservation exists for: funds in flight cannot be
    /// spent again.
    #[test]
    fn reserved_funds_cannot_be_spent_twice() {
        let (mut l, alice) = funded("alice", 1000);
        l.reserve(&alice, "credits", 800).unwrap();

        let err = l.transfer(req(alice, Account::player("bob"), 300, "t")).unwrap_err();
        assert!(matches!(err, LedgerError::InsufficientFunds { available: 200, .. }));
    }

    #[test]
    fn releasing_a_reservation_restores_availability() {
        let (mut l, alice) = funded("alice", 1000);
        let r = l.reserve(&alice, "credits", 800).unwrap();
        l.release(r).unwrap();
        assert_eq!(l.available(&alice, "credits"), 1000);
    }

    #[test]
    fn over_reserving_is_refused() {
        let (mut l, alice) = funded("alice", 100);
        l.reserve(&alice, "credits", 60).unwrap();
        assert!(l.reserve(&alice, "credits", 50).is_err(), "only 40 remains");
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
