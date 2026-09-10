//! The contract every ledger backend satisfies.
//!
//! Phase 3 proved the semantics with everything in memory. This is the
//! seam that lets the same semantics run against Postgres without the
//! simulation knowing which it is talking to — and, just as importantly,
//! lets the property tests that proved those semantics be re-run against
//! the database backend unchanged.
//!
//! # Why the trait is sync
//!
//! `App::fixed_update` is sync and stays that way: making it async would
//! infect every game's simulation code and undo the headless split. A
//! Postgres write is not sync, so the durable backend does not implement
//! this trait by blocking on the database — it enqueues, returns a
//! *pending* receipt, and a writer thread commits. See `ledger_pg`.
//!
//! # What a backend must guarantee
//!
//! The invariants are the ones `tests/ledger_properties.rs` asserts, and
//! they are the same whichever backend is underneath:
//!
//! - Every asset sums to zero across all accounts.
//! - A repeated [`IdemKey`] moves value exactly once.
//! - A refused transfer leaves no partial state.
//! - Reserved funds cannot be spent twice.

use super::ledger::{
    Account, Amount, Discrepancy, Entry, IdemKey, LedgerError, Receipt, ReservationId,
    TransferRequest,
};

/// How durable a receipt is.
///
/// The in-memory backend only ever reports [`Durability::Memory`]. The
/// Postgres backend reports [`Durability::Pending`] until the writer
/// thread commits, which is the distinction a trade UI must respect: a
/// pending transfer has *happened* as far as the simulation is
/// concerned, but telling the player "sold!" before it commits is how
/// you end up with a player who saw a confirmation for a transaction
/// that was later lost.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Durability {
    /// No durable store behind this ledger. Lost on process exit.
    Memory,
    /// Accepted and applied to balances, not yet committed.
    Pending,
    /// Committed to durable storage.
    Committed,
}

/// A receipt plus how durable it is.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StoredReceipt {
    pub receipt: Receipt,
    pub durability: Durability,
}

/// The read and write surface a ledger backend provides.
///
/// Deliberately not generic over an async runtime: see the module docs.
pub trait LedgerStore {
    /// Move value, or explain why not.
    ///
    /// Must be atomic: either both entries exist or neither does. A
    /// repeated `idem_key` returns the original receipt without moving
    /// anything, and must do so even if the account could no longer
    /// afford the transfer — the original already happened.
    fn transfer(&mut self, req: TransferRequest) -> Result<StoredReceipt, LedgerError>;

    /// Balance for an account and asset, derived from the log.
    fn balance(&self, account: &Account, asset: &str) -> Amount;

    /// Balance minus funds held by in-flight reservations. This is what
    /// a spend checks against.
    fn available(&self, account: &Account, asset: &str) -> Amount;

    /// Hold funds so an in-flight commit cannot be double-spent.
    fn reserve(
        &mut self,
        account: &Account,
        asset: &str,
        amount: Amount,
    ) -> Result<ReservationId, LedgerError>;

    /// Release a hold without spending it.
    fn release(&mut self, id: ReservationId) -> Result<(), LedgerError>;

    /// Entries touching an account, oldest first — a support
    /// investigation's raw material.
    fn history(&self, account: &Account) -> Vec<Entry>;

    /// **The dupe detector.** Empty means the books balance.
    fn audit_zero_sum(&self) -> Vec<Discrepancy>;

    /// Total ever created for an asset. A jump here is legitimate value
    /// creation, so the zero-sum audit will not flag it — which is
    /// exactly why it is worth watching separately.
    fn total_minted(&self, asset: &str) -> i128;

    /// The highest tick whose value movements are durably committed.
    ///
    /// A checkpoint may never claim a tick above this. Restoring a
    /// checkpoint that shows a purchase the ledger never recorded *is* a
    /// dupe, so the checkpoint writer reads this and records the lower of
    /// the two.
    ///
    /// The in-memory backend has no durable storage, so it reports the
    /// highest tick it has seen: nothing is at risk of outliving it.
    fn acked_tick(&self) -> u64;

    /// Look up a previously-issued receipt by its key.
    fn receipt_for(&self, key: &IdemKey) -> Option<StoredReceipt>;
}
