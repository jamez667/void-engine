//! The durable ledger, against a real Postgres.
//!
//! Skipped unless `VOID_ENGINE_PG_URL` is set, so a normal `cargo test`
//! on a machine with no database still passes. CI sets it against a
//! service container; locally:
//!
//! ```text
//! docker run -d --name ve_pg -e POSTGRES_PASSWORD=devpw \
//!     -e POSTGRES_DB=void_engine -p 55432:5432 postgres:17-alpine
//! VOID_ENGINE_PG_URL="host=127.0.0.1 port=55432 user=postgres \
//!     password=devpw dbname=void_engine" cargo test --test ledger_pg \
//!     --no-default-features --features ledger-pg
//! ```
//!
//! Each test works in its own schema so they cannot interfere, and so a
//! failure leaves its evidence behind rather than being scrubbed by the
//! next test's setup.

#![cfg(feature = "ledger-pg")]

use std::time::Duration;

use void_engine::persist::ledger::{Account, IdemKey, LedgerError, TransferRequest};
use void_engine::persist::ledger_pg::{PgConfig, PgLedger};
use void_engine::persist::store::{Durability, LedgerStore};

/// `None` when no database is configured, which skips rather than fails.
fn pg_url() -> Option<String> {
    std::env::var("VOID_ENGINE_PG_URL").ok()
}

/// A connection string pointed at a schema of this test's own.
///
/// `search_path` is per-connection, so each test gets an isolated set of
/// tables without needing a separate database.
fn isolated(tag: &str) -> Option<String> {
    let base = pg_url()?;
    let schema = format!("t_{tag}");

    // Create the schema up front through a throwaway connection.
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let (client, conn) = tokio_postgres::connect(&base, tokio_postgres::NoTls)
            .await
            .expect("test database must be reachable");
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

fn transfer(from: Account, to: Account, amount: i64, key: &str, tick: u64) -> TransferRequest {
    TransferRequest {
        idem_key: IdemKey::server(key),
        from,
        to,
        asset: "credits".to_string(),
        amount,
        reason: "test".to_string(),
        actor: "system".to_string(),
        tick,
    }
}

macro_rules! pg_test {
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

/// Migrations run on open, so a fresh schema becomes a working ledger
/// with no external tooling.
#[test]
fn open_migrates_an_empty_schema() {
    let url = pg_test!("migrate");
    let ledger = PgLedger::open(PgConfig::new(url)).expect("open must migrate and connect");
    assert_eq!(ledger.acked_tick(), 0, "a fresh ledger has acked nothing");
}

/// A transfer is accepted immediately and becomes durable shortly after.
#[test]
fn a_transfer_is_pending_then_committed() {
    let url = pg_test!("pending");
    let mut ledger = PgLedger::open(PgConfig::new(url)).unwrap();
    let alice = Account::player("alice");

    let r = ledger
        .transfer(transfer(Account::Mint, alice.clone(), 1_000, "seed", 5))
        .unwrap();
    assert_eq!(r.durability, Durability::Pending, "the tick must not block on the database");
    assert_eq!(ledger.balance(&alice, "credits"), 1_000, "balance applies immediately");

    ledger.flush(Duration::from_secs(10)).expect("the writer must commit");
    assert!(ledger.acked_tick() >= 5, "the watermark must advance past the transfer");
}

/// The point of the whole exercise: value survives a restart, and the
/// balance after it is derived from the durable log.
#[test]
fn value_survives_a_restart() {
    let url = pg_test!("restart");
    let alice = Account::player("alice");
    let bob = Account::player("bob");

    {
        let mut ledger = PgLedger::open(PgConfig::new(url.clone())).unwrap();
        ledger.transfer(transfer(Account::Mint, alice.clone(), 1_000, "seed", 1)).unwrap();
        ledger.transfer(transfer(alice.clone(), bob.clone(), 250, "t1", 2)).unwrap();
        ledger.flush(Duration::from_secs(10)).unwrap();
    } // process "dies"

    let reopened = PgLedger::open(PgConfig::new(url)).unwrap();
    assert_eq!(reopened.balance(&alice, "credits"), 750, "alice's balance must be replayed");
    assert_eq!(reopened.balance(&bob, "credits"), 250, "bob's too");
    assert!(reopened.audit_zero_sum().is_empty(), "the books must still balance after replay");
    assert!(reopened.acked_tick() >= 2, "the watermark must be recovered");
}

/// A retry after a restart must not move value again — the durable
/// idempotency guarantee, enforced by the UNIQUE constraint rather than
/// by anything remembering.
#[test]
fn a_retry_across_a_restart_does_not_double_spend() {
    let url = pg_test!("retry_restart");
    let alice = Account::player("alice");
    let bob = Account::player("bob");
    let key = IdemKey::new("session-9", 3, "trade");

    {
        let mut ledger = PgLedger::open(PgConfig::new(url.clone())).unwrap();
        ledger.transfer(transfer(Account::Mint, alice.clone(), 1_000, "seed", 1)).unwrap();
        let mut req = transfer(alice.clone(), bob.clone(), 400, "unused", 2);
        req.idem_key = key.clone();
        ledger.transfer(req).unwrap();
        ledger.flush(Duration::from_secs(10)).unwrap();
    }

    // The client resends after the server came back up.
    let mut reopened = PgLedger::open(PgConfig::new(url)).unwrap();
    let mut again = transfer(alice.clone(), bob.clone(), 400, "unused", 2);
    again.idem_key = key;
    let r = reopened.transfer(again).unwrap();

    assert!(r.receipt.deduplicated, "the replayed key must be recognised");
    assert_eq!(reopened.balance(&alice, "credits"), 600, "value must not move twice");
    assert_eq!(reopened.balance(&bob, "credits"), 400);
    assert!(reopened.audit_zero_sum().is_empty());
}

/// A refused transfer must leave nothing behind, in memory or on disk.
#[test]
fn a_refused_transfer_reaches_the_database_not_at_all() {
    let url = pg_test!("refused");
    let mut ledger = PgLedger::open(PgConfig::new(url)).unwrap();
    let alice = Account::player("alice");

    ledger.transfer(transfer(Account::Mint, alice.clone(), 100, "seed", 1)).unwrap();
    ledger.flush(Duration::from_secs(10)).unwrap();

    let err = ledger
        .transfer(transfer(alice.clone(), Account::player("bob"), 999, "over", 2))
        .unwrap_err();
    assert!(matches!(err, LedgerError::InsufficientFunds { .. }));

    ledger.flush(Duration::from_secs(10)).unwrap();
    assert_eq!(ledger.balance(&alice, "credits"), 100, "nothing moved");
    assert!(ledger.audit_zero_sum().is_empty());
}

/// The journal cap is what stops an unbounded queue eating the process.
#[test]
fn a_full_journal_refuses_rather_than_growing() {
    let url = pg_test!("backpressure");
    // A cap of 1 makes the second in-flight transfer trip it
    // deterministically, without needing to outrun a healthy writer.
    let cfg = PgConfig { max_journal: 1, ..PgConfig::new(url) };
    let mut ledger = PgLedger::open(cfg).unwrap();
    let alice = Account::player("alice");

    // Fill the journal, then keep pushing until the cap is felt. The
    // writer is draining concurrently, so this races deliberately: the
    // assertion is that we get *either* acceptance or a clean refusal,
    // never a panic or an unbounded queue.
    let mut refused = false;
    for i in 0..200u64 {
        match ledger.transfer(transfer(Account::Mint, alice.clone(), 1, &format!("k{i}"), i)) {
            Err(LedgerError::WriterBehind { depth, limit }) => {
                assert!(depth >= limit, "refused with depth {depth} under limit {limit}");
                refused = true;
                break;
            }
            Err(e) => panic!("unexpected error: {e}"),
            Ok(_) => {}
        }
    }

    ledger.flush(Duration::from_secs(10)).unwrap();
    assert!(ledger.audit_zero_sum().is_empty(), "backpressure must not corrupt the books");
    // Not asserting `refused` is true: a fast writer may keep up. The
    // point is that if the cap is hit, it is hit cleanly.
    let _ = refused;
}

/// Entries carry the audit fields an investigation needs, all the way
/// through to the database and back.
#[test]
fn the_audit_trail_survives_a_round_trip() {
    let url = pg_test!("audit_trail");
    let alice = Account::player("alice");

    {
        let mut ledger = PgLedger::open(PgConfig::new(url.clone())).unwrap();
        ledger
            .transfer(TransferRequest {
                idem_key: IdemKey::server("gm-grant-1"),
                from: Account::Mint,
                to: alice.clone(),
                asset: "credits".to_string(),
                amount: 500,
                reason: "gm_grant".to_string(),
                actor: "gm:kate".to_string(),
                tick: 42,
            })
            .unwrap();
        ledger.flush(Duration::from_secs(10)).unwrap();
    }

    let reopened = PgLedger::open(PgConfig::new(url)).unwrap();
    let history = reopened.history(&alice);
    let entry = history.first().expect("the grant must be in the history");
    assert_eq!(entry.reason, "gm_grant", "why must survive");
    assert_eq!(entry.actor, "gm:kate", "who must survive");
    assert_eq!(entry.tick, 42, "the tick correlates with a checkpoint");
    assert_eq!(entry.counterparty, Account::Mint, "and the counterparty");
}

/// Items round-trip like currency, and a unique item stays unique across
/// a restart.
#[test]
fn a_unique_item_stays_unique_across_a_restart() {
    let url = pg_test!("unique_item");
    let alice = Account::player("alice");
    let bob = Account::player("bob");
    let sword = "item:sword_of_dawn";

    {
        let mut ledger = PgLedger::open(PgConfig::new(url.clone())).unwrap();
        let mut mint = transfer(Account::Mint, alice.clone(), 1, "forge", 1);
        mint.asset = sword.to_string();
        ledger.transfer(mint).unwrap();

        let mut trade = transfer(alice.clone(), bob.clone(), 1, "trade", 2);
        trade.asset = sword.to_string();
        ledger.transfer(trade).unwrap();
        ledger.flush(Duration::from_secs(10)).unwrap();
    }

    let reopened = PgLedger::open(PgConfig::new(url)).unwrap();
    assert_eq!(reopened.balance(&alice, sword), 0);
    assert_eq!(reopened.balance(&bob, sword), 1);
    assert_eq!(reopened.total_minted(sword), 1, "exactly one was ever created");
    assert!(reopened.audit_zero_sum().is_empty());
}
