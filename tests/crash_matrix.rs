//! The crash matrix: kills at each boundary, and a forged dupe that a job
//! must catch rather than a player.
//!
//! Nine boundaries were already covered before this file existed — process
//! death between checkpoints, a torn checkpoint write, first boot, the
//! database vanishing mid-flight, recovery, reads during an outage, a
//! rolled-back transaction, a full journal, and a retry across a restart.
//! This closes the four that were not:
//!
//! 1. A forged row in Postgres, caught by reconciliation. **The exit
//!    criterion**: a dupe must be found by a job, not by a player noticing
//!    their balance is wrong.
//! 2. A checkpoint claiming a tick the ledger never committed.
//! 3. Half of a double-entry pair reaching disk.
//! 4. A genuinely fatal fault refusing to retry forever.
//!
//! Needs `VOID_ENGINE_PG_URL`; skips without it.

#![cfg(feature = "ledger-pg")]

use std::time::Duration;

use void_engine::persist::ledger::{Account, IdemKey, TransferRequest};
use void_engine::persist::ledger_pg::{PgConfig, PgLedger, WriterHealth};
use void_engine::persist::store::LedgerStore;

fn pg_url() -> Option<String> {
    std::env::var("VOID_ENGINE_PG_URL").ok()
}

/// A schema of this test's own, so a forged row cannot leak into another
/// test's books.
fn isolated(tag: &str) -> Option<String> {
    let base = pg_url()?;
    let schema = format!("m_{tag}");
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let (client, conn) = tokio_postgres::connect(&base, tokio_postgres::NoTls)
            .await
            .expect("database must be reachable");
        tokio::spawn(async move { let _ = conn.await; });
        client
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS {schema} CASCADE; CREATE SCHEMA {schema};"
            ))
            .await
            .expect("creating the test schema");
    });
    Some(format!("{base} options=-csearch_path={schema}"))
}

/// Run one statement against a test's schema, for forging damage the
/// public API cannot produce.
fn exec_raw(url: &str, sql: &str) {
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let (client, conn) = tokio_postgres::connect(url, tokio_postgres::NoTls)
            .await
            .expect("database must be reachable");
        tokio::spawn(async move { let _ = conn.await; });
        client.batch_execute(sql).await.expect("raw statement must apply");
    });
}

fn transfer(from: Account, to: Account, amount: i64, key: &str, tick: u64) -> TransferRequest {
    TransferRequest {
        idem_key: IdemKey::server(key),
        from,
        to,
        asset: "credits".to_string(),
        amount,
        reason: "matrix".to_string(),
        actor: "system".to_string(),
        tick,
    }
}

macro_rules! matrix_test {
    ($tag:literal) => {
        match isolated($tag) {
            Some(url) => url,
            None => {
                eprintln!("skipping: VOID_ENGINE_PG_URL not set");
                return;
            }
        }
    };
}

/// **The exit criterion.** Forge value into the durable log the way a bug
/// or an exploit would, and reconciliation must find it.
///
/// This is deliberately done behind the API's back: `transfer` cannot
/// create an unbalanced entry, which is the point of the design. The
/// question this answers is what happens when something *else* does —
/// a migration gone wrong, a direct database edit, a compromised service.
#[test]
fn a_forged_row_is_caught_by_reconciliation() {
    let url = matrix_test!("forged");
    let ledger = PgLedger::open(PgConfig::new(url.clone())).unwrap();

    // A clean ledger reconciles clean.
    ledger.reconcile_now().expect("an empty ledger must reconcile");

    // Now forge credits into existence with no counterparty.
    exec_raw(
        &url,
        "INSERT INTO ledger_entries \
         (idem_key, side, tick, account_kind, account_id, \
          counterparty_kind, counterparty_id, asset, delta, reason, actor) \
         VALUES ('forged-1', 'c', 1, 'player', 'cheater', 'system', 'nowhere', \
                 'credits', 1000000, 'dupe', 'attacker')",
    );

    let err = ledger
        .reconcile_now()
        .expect_err("a forged row must be caught, not ignored");
    assert!(err.contains("credits"), "the offending asset must be named: {err}");
    assert!(err.contains("sum to zero"), "the failure must say what is wrong: {err}");
}

/// Half a transfer reaching disk is the same class of damage and must be
/// caught the same way. A transaction makes this unreachable through the
/// writer, so it is forged directly.
#[test]
fn half_a_double_entry_pair_is_caught() {
    let url = matrix_test!("half_pair");
    let mut ledger = PgLedger::open(PgConfig::new(url.clone())).unwrap();

    ledger
        .transfer(transfer(Account::Mint, Account::player("alice"), 100, "seed", 1))
        .unwrap();
    ledger.flush(Duration::from_secs(10)).unwrap();
    ledger.reconcile_now().expect("a committed transfer must reconcile");

    // Delete one side of the pair, simulating a partial write that the
    // surrounding transaction normally prevents.
    exec_raw(&url, "DELETE FROM ledger_entries WHERE side = 'd'");

    let err = ledger
        .reconcile_now()
        .expect_err("an unpaired entry must be caught");
    assert!(err.contains("credits"), "got {err}");
}

/// A checkpoint may never claim a tick the ledger has not committed.
///
/// Restoring a checkpoint that shows a purchase the ledger never recorded
/// *is* a dupe — the player keeps the goods and the payment never
/// happened. The watermark is what a checkpoint writer must clamp to.
#[test]
fn a_checkpoint_must_not_claim_a_tick_above_the_watermark() {
    let url = matrix_test!("watermark");
    let mut ledger = PgLedger::open(PgConfig::new(url)).unwrap();
    let alice = Account::player("alice");

    ledger.transfer(transfer(Account::Mint, alice.clone(), 500, "seed", 10)).unwrap();
    ledger.flush(Duration::from_secs(10)).unwrap();
    let acked_after_commit = ledger.acked_tick();
    assert!(acked_after_commit >= 10, "the watermark must cover a committed tick");

    // Accept a transfer at a much later tick but do not flush: it is
    // journalled, not durable.
    ledger.transfer(transfer(alice.clone(), Account::Burn, 1, "pending", 9_999)).unwrap();

    // The watermark must not have jumped to the pending tick. A
    // checkpoint taken now must record the acked value, not the sim's
    // current tick, or a restore would resurrect a purchase the ledger
    // never recorded.
    let acked_with_pending = ledger.acked_tick();
    assert!(
        acked_with_pending < 9_999,
        "the watermark claimed tick {acked_with_pending}, but 9999 was never committed",
    );

    // After a flush it may advance.
    ledger.flush(Duration::from_secs(10)).unwrap();
    assert!(ledger.acked_tick() >= 9_999, "a flushed tick must become durable");
}

/// A fatal fault must not retry forever. Dropping the table is
/// unrecoverable by definition: every retry would fail identically.
#[test]
fn a_fatal_fault_stops_rather_than_retrying_forever() {
    let url = matrix_test!("fatal");
    let cfg = PgConfig {
        // Generous, so reaching Failed proves the *classifier* stopped it
        // rather than the retry budget running out.
        max_retries: 10_000,
        retry_base_delay: Duration::from_millis(10),
        retry_max_delay: Duration::from_millis(50),
        ..PgConfig::new(url.clone())
    };
    let mut ledger = PgLedger::open(cfg).unwrap();

    ledger
        .transfer(transfer(Account::Mint, Account::player("alice"), 10, "seed", 1))
        .unwrap();
    ledger.flush(Duration::from_secs(10)).unwrap();

    // Remove the table the writer inserts into. Postgres answers with
    // UNDEFINED_TABLE, which `is_transient` classifies as fatal.
    exec_raw(&url, "DROP TABLE ledger_entries CASCADE");

    // Push work so the writer tries and fails.
    for i in 0..20u64 {
        let _ = ledger.transfer(transfer(
            Account::Mint,
            Account::player("alice"),
            1,
            &format!("f{i}"),
            2 + i,
        ));
        std::thread::sleep(Duration::from_millis(25));
    }

    let mut failed = false;
    for _ in 0..80 {
        if matches!(ledger.health(), WriterHealth::Failed(_)) {
            failed = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        failed,
        "a fatal fault must terminate the writer, not retry; health was {:?}",
        ledger.health(),
    );

    // And it stays failed: no self-healing from an unrecoverable fault.
    std::thread::sleep(Duration::from_millis(200));
    assert!(matches!(ledger.health(), WriterHealth::Failed(_)), "terminal must be terminal");
}
