//! The durable ledger: same semantics, written through to Postgres.
//!
//! # The shape, and why
//!
//! `App::fixed_update` is sync and must stay so — making it async would
//! infect every game's simulation code and undo the headless split. A
//! Postgres commit is not sync. So this backend does not block the tick
//! on the database:
//!
//! ```text
//! tick N   transfer() ──► validate against the in-memory core
//!                     ──► apply to balances, return a PENDING receipt
//!                     ──► push onto the journal
//!                              │
//! writer   ────────────────────┴─► BEGIN; two INSERTs; UPDATE watermark; COMMIT
//! thread                              │
//!                                     └─► acked_tick advances
//! ```
//!
//! The in-memory core is the same [`Ledger`] phase 3 proved, so validation
//! — insufficient funds, self-transfer, idempotency, reservations — is
//! bit-identical to the backend the property tests hammered. Postgres is
//! durability, not a second implementation of the rules. That matters:
//! two implementations of "can this player afford it" would eventually
//! disagree, and the disagreement would be a dupe.
//!
//! # What a crash costs
//!
//! Everything up to `acked_tick` is durable. Transfers accepted after it
//! are lost, exactly as the ticks since the last checkpoint are lost.
//! A checkpoint may never claim a tick above `acked_tick`, because
//! restoring a checkpoint that shows a purchase the ledger never recorded
//! *is* a dupe.
//!
//! # Backpressure
//!
//! An unbounded journal is a memory leak with extra steps. Past
//! [`PgConfig::max_journal`] the ledger refuses new transfers with
//! [`LedgerError::WriterBehind`] rather than growing until the box dies.
//! Refusing a purchase is recoverable; running out of memory is not.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use tokio_postgres::NoTls;

use super::ledger::{
    Account, Amount, Discrepancy, Entry, IdemKey, Ledger, LedgerError, ReservationId,
    TransferRequest,
};
use super::store::{Durability, LedgerStore, StoredReceipt};

mod embedded {
    // Reads `migrations/` at the crate root at compile time, so a server
    // can migrate itself on boot rather than shipping a CLI beside it.
    refinery::embed_migrations!("migrations");
}

/// SQLSTATE for a unique-violation. Checked by code rather than by
/// matching on the message text, which is localised and unstable.
const SQLSTATE_UNIQUE_VIOLATION: &str = "23505";

/// How to reach the database, and how much slack the writer gets.
#[derive(Clone, Debug)]
pub struct PgConfig {
    /// libpq-style connection string.
    pub url: String,
    /// Journal depth past which transfers are refused. The default is
    /// generous enough to absorb a slow fsync or a brief network stall,
    /// and small enough that a genuinely dead writer is noticed in
    /// seconds rather than after the process is swapped to death.
    pub max_journal: usize,
    /// Entries per transaction. Batching amortises round-trips; too large
    /// and a single failure re-does more work.
    pub batch: usize,
}

impl PgConfig {
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into(), max_journal: 10_000, batch: 64 }
    }
}

/// Why the durable ledger could not start.
#[derive(Debug)]
pub enum PgError {
    Connect(tokio_postgres::Error),
    /// The durable log could not be replayed into a consistent state.
    /// Starting anyway would mean serving wrong balances.
    Replay(String),
    Migrate(String),
    Runtime(std::io::Error),
}

impl std::fmt::Display for PgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PgError::Connect(e) => write!(f, "connecting to postgres: {e}"),
            PgError::Replay(e) => write!(f, "replaying the ledger: {e}"),
            PgError::Migrate(e) => write!(f, "running migrations: {e}"),
            PgError::Runtime(e) => write!(f, "starting the writer runtime: {e}"),
        }
    }
}

impl std::error::Error for PgError {}

/// One transfer awaiting durability.
#[derive(Clone, Debug)]
struct Pending {
    idem_key: IdemKey,
    tick: u64,
    debit: Entry,
    credit: Entry,
}

/// Shared between the sim thread and the writer.
struct Shared {
    journal: Mutex<Vec<Pending>>,
    /// Transfers accepted by the sim, ever.
    accepted: AtomicU64,
    /// Transfers actually committed to Postgres, ever.
    ///
    /// Separate from the journal being empty, and that distinction is the
    /// whole point: the writer drains work into a local batch *before*
    /// committing it, so there is a window in which the journal is empty
    /// and nothing is durable. `flush` waiting on the journal would
    /// return success during that window and a restart would find no
    /// rows — which is exactly the bug this pair of counters fixes.
    committed: AtomicU64,
    acked_tick: AtomicU64,
    /// Set when the writer dies. A ledger whose writer is gone must stop
    /// accepting value movements rather than silently becoming in-memory
    /// only — that would be durable-looking data loss.
    writer_failed: Mutex<Option<String>>,
}

/// A [`LedgerStore`] backed by Postgres.
///
/// Validation happens in the in-memory core; this adds durability.
pub struct PgLedger {
    core: Ledger,
    shared: Arc<Shared>,
    max_journal: usize,
    /// Kept so the runtime outlives the writer task.
    _runtime: tokio::runtime::Runtime,
}

impl PgLedger {
    /// Connect, migrate, replay committed state, and start the writer.
    ///
    /// Replay is what makes a restart correct: the in-memory core is
    /// rebuilt from `ledger_entries`, so balances after a restart are
    /// derived from the durable log rather than from a checkpoint that
    /// might be ahead of it.
    pub fn open(cfg: PgConfig) -> Result<Self, PgError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .enable_all()
            .thread_name("ledger-writer")
            .build()
            .map_err(PgError::Runtime)?;

        let url = cfg.url.clone();
        let (mut client, replayed) = runtime.block_on(async move {
            let (mut client, conn) = tokio_postgres::connect(&url, NoTls)
                .await
                .map_err(PgError::Connect)?;
            tokio::spawn(async move {
                if let Err(e) = conn.await {
                    log::error!("[ledger] postgres connection lost: {e}");
                }
            });

            embedded::migrations::runner()
                .run_async(&mut client)
                .await
                .map_err(|e| PgError::Migrate(e.to_string()))?;

            let rows = client
                .query(
                    "SELECT idem_key, side, tick, account_kind, account_id, \
                            counterparty_kind, counterparty_id, asset, delta, reason, actor \
                     FROM ledger_entries ORDER BY seq",
                    &[],
                )
                .await
                .map_err(PgError::Connect)?;

            let acked: i64 = client
                .query_one("SELECT acked_tick FROM ledger_watermark WHERE id", &[])
                .await
                .map_err(PgError::Connect)?
                .get(0);

            Ok::<_, PgError>((client, (rows, acked)))
        })?;

        let (rows, acked) = replayed;
        let mut core = Ledger::new();
        replay_into(&mut core, &rows).map_err(PgError::Replay)?;

        let shared = Arc::new(Shared {
            journal: Mutex::new(Vec::new()),
            accepted: AtomicU64::new(0),
            committed: AtomicU64::new(0),
            acked_tick: AtomicU64::new(acked.max(0) as u64),
            writer_failed: Mutex::new(None),
        });

        let writer_shared = shared.clone();
        let batch = cfg.batch;
        runtime.spawn(async move {
            writer_loop(&mut client, writer_shared, batch).await;
        });

        Ok(Self { core, shared, max_journal: cfg.max_journal, _runtime: runtime })
    }

    /// Block until everything accepted so far is durable.
    ///
    /// For a clean shutdown, and for tests that need to assert on what
    /// actually reached the database.
    pub fn flush(&self, timeout: std::time::Duration) -> Result<(), String> {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if let Some(err) = self.shared.writer_failed.lock().unwrap().clone() {
                return Err(err);
            }
            let accepted = self.shared.accepted.load(Ordering::SeqCst);
            let committed = self.shared.committed.load(Ordering::SeqCst);
            if committed >= accepted {
                return Ok(());
            }
            if std::time::Instant::now() > deadline {
                return Err("timed out waiting for the writer".to_string());
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// How many transfers are waiting to be committed. Worth surfacing:
    /// a number that only grows means the writer is losing.
    pub fn journal_depth(&self) -> usize {
        self.shared.journal.lock().unwrap().len()
    }

    /// The in-memory core, for the audits phase 3 defined.
    pub fn core(&self) -> &Ledger {
        &self.core
    }
}

/// Rebuild an in-memory ledger from committed rows.
fn replay_into(core: &mut Ledger, rows: &[tokio_postgres::Row]) -> Result<(), String> {
    // Replay through the public API so the cache is rebuilt by exactly
    // the code that maintains it normally — a separate rebuild path is a
    // second implementation that can disagree, and a disagreement here is
    // a wrong balance.
    let mut i = 0;
    while i < rows.len() {
        let side: String = rows[i].get("side");
        if side != "d" {
            i += 1;
            continue;
        }
        let idem_key: String = rows[i].get("idem_key");
        let tick: i64 = rows[i].get("tick");
        let asset: String = rows[i].get("asset");
        let delta: i64 = rows[i].get("delta");
        let reason: String = rows[i].get("reason");
        let actor: String = rows[i].get("actor");
        let from = account_from_row(&rows[i], "account_kind", "account_id");
        let to = account_from_row(&rows[i], "counterparty_kind", "counterparty_id");

        // Not `let _ =`. A refused replay means the reconstructed state
        // disagrees with the durable log, and continuing would leave
        // balances silently wrong on the one path that exists to recover
        // from a crash. Better to refuse to start than to start wrong.
        if let Err(e) = core.transfer(TransferRequest {
            idem_key: IdemKey::server(idem_key.clone()),
            from,
            to,
            asset,
            amount: -delta,
            reason,
            actor,
            tick: tick.max(0) as u64,
        }) {
            return Err(format!(
                "replaying entry {idem_key:?} (seq order {i}): {e}"
            ));
        }
        i += 1;
    }
    Ok(())
}

fn account_from_row(row: &tokio_postgres::Row, kind_col: &str, id_col: &str) -> Account {
    let kind: String = row.get(kind_col);
    let id: String = row.get(id_col);
    account_from_parts(&kind, &id)
}

fn account_from_parts(kind: &str, id: &str) -> Account {
    match kind {
        "player" => Account::Player(id.to_string()),
        "system" => Account::System(id.to_string()),
        "mint" => Account::Mint,
        _ => Account::Burn,
    }
}

/// `(kind, id)` for the schema's two columns.
fn account_parts(a: &Account) -> (&'static str, String) {
    match a {
        Account::Player(id) => ("player", id.clone()),
        Account::System(id) => ("system", id.clone()),
        Account::Mint => ("mint", String::new()),
        Account::Burn => ("burn", String::new()),
    }
}

/// Drain the journal into Postgres, forever.
async fn writer_loop(client: &mut tokio_postgres::Client, shared: Arc<Shared>, batch: usize) {
    loop {
        let work: Vec<Pending> = {
            let mut j = shared.journal.lock().unwrap();
            let take = j.len().min(batch);
            j.drain(..take).collect()
        };

        if work.is_empty() {
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            continue;
        }

        let highest = work.iter().map(|p| p.tick).max().unwrap_or(0);
        match commit_batch(client, &work, highest).await {
            Ok(()) => {
                // Only now is it safe to say these ticks are durable.
                shared.acked_tick.fetch_max(highest, Ordering::SeqCst);
                shared.committed.fetch_add(work.len() as u64, Ordering::SeqCst);
            }
            Err(e) => {
                let msg = format!("ledger writer failed: {e}");
                log::error!("[ledger] {msg}");
                *shared.writer_failed.lock().unwrap() = Some(msg);
                // Put the work back so a flush sees it as outstanding
                // rather than silently vanished.
                let mut j = shared.journal.lock().unwrap();
                for (i, p) in work.into_iter().enumerate() {
                    j.insert(i, p);
                }
                return;
            }
        }
    }
}

/// One transaction: every entry in the batch, plus the watermark.
///
/// All or nothing. A partial batch would leave the watermark claiming
/// durability for transfers that were not written.
async fn commit_batch(
    client: &mut tokio_postgres::Client,
    work: &[Pending],
    highest_tick: u64,
) -> Result<(), tokio_postgres::Error> {
    let tx = client.transaction().await?;

    for p in work {
        for (side, e) in [("d", &p.debit), ("c", &p.credit)] {
            let (ak, aid) = account_parts(&e.account);
            let (ck, cid) = account_parts(&e.counterparty);
            let res = tx
                .execute(
                    "INSERT INTO ledger_entries \
                     (idem_key, side, tick, account_kind, account_id, \
                      counterparty_kind, counterparty_id, asset, delta, reason, actor) \
                     VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)",
                    &[
                        &p.idem_key.as_str(),
                        &side,
                        &(p.tick as i64),
                        &ak,
                        &aid,
                        &ck,
                        &cid,
                        &e.asset,
                        &e.delta,
                        &e.reason,
                        &e.actor,
                    ],
                )
                .await;

            if let Err(err) = res {
                // A unique violation means this transfer is already
                // durable — the process restarted mid-batch, say. That is
                // success, not failure: the whole point of the constraint
                // is that a replay cannot double-write.
                let is_dupe = err
                    .as_db_error()
                    .map(|d| d.code().code() == SQLSTATE_UNIQUE_VIOLATION)
                    .unwrap_or(false);
                if !is_dupe {
                    return Err(err);
                }
            }
        }
    }

    tx.execute(
        "UPDATE ledger_watermark SET acked_tick = GREATEST(acked_tick, $1) WHERE id",
        &[&(highest_tick as i64)],
    )
    .await?;

    tx.commit().await
}

impl LedgerStore for PgLedger {
    fn transfer(&mut self, req: TransferRequest) -> Result<StoredReceipt, LedgerError> {
        if let Some(err) = self.shared.writer_failed.lock().unwrap().clone() {
            return Err(LedgerError::WriterFailed(err));
        }
        if self.journal_depth() >= self.max_journal {
            return Err(LedgerError::WriterBehind {
                depth: self.journal_depth(),
                limit: self.max_journal,
            });
        }

        let before = self.core.entries().len();
        let tick = req.tick;
        let receipt = self.core.transfer(req)?;

        // A deduplicated transfer wrote nothing, so there is nothing to
        // journal — and journalling it would try to insert a row the
        // UNIQUE constraint already holds.
        if !receipt.deduplicated {
            let entries = self.core.entries();
            let debit = entries[before].clone();
            let credit = entries[before + 1].clone();
            self.shared.journal.lock().unwrap().push(Pending {
                idem_key: receipt.idem_key.clone(),
                tick: debit.tick,
                debit,
                credit,
            });
            self.shared.accepted.fetch_add(1, Ordering::SeqCst);
        }

        // A dedup hit means the original transfer was already accepted;
        // whether it is *durable* depends on how far the writer has got,
        // which is exactly what the watermark records.
        let durability = if receipt.deduplicated && self.acked_tick() >= tick {
            Durability::Committed
        } else {
            Durability::Pending
        };
        Ok(StoredReceipt { receipt, durability })
    }

    fn balance(&self, account: &Account, asset: &str) -> Amount {
        self.core.balance(account, asset)
    }

    fn available(&self, account: &Account, asset: &str) -> Amount {
        self.core.available(account, asset)
    }

    fn reserve(
        &mut self,
        account: &Account,
        asset: &str,
        amount: Amount,
    ) -> Result<ReservationId, LedgerError> {
        self.core.reserve(account, asset, amount)
    }

    fn release(&mut self, id: ReservationId) -> Result<(), LedgerError> {
        self.core.release(id)
    }

    fn history(&self, account: &Account) -> Vec<Entry> {
        self.core.history(account).cloned().collect()
    }

    fn audit_zero_sum(&self) -> Vec<Discrepancy> {
        self.core.audit_zero_sum()
    }

    fn total_minted(&self, asset: &str) -> i128 {
        self.core.total_minted(asset)
    }

    fn acked_tick(&self) -> u64 {
        self.shared.acked_tick.load(Ordering::SeqCst)
    }

    fn receipt_for(&self, key: &IdemKey) -> Option<StoredReceipt> {
        self.core.receipt_for_inner(key).map(|receipt| {
            let durability = if receipt.deduplicated {
                Durability::Committed
            } else {
                Durability::Pending
            };
            StoredReceipt { receipt, durability }
        })
    }
}
