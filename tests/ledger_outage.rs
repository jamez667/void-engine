//! What happens when the database goes away.
//!
//! The durable ledger must never accept a value movement it cannot
//! persist. That is the whole point of the tier, and it is the one
//! property that cannot be checked without actually taking the database
//! away mid-flight — so these tests stop and start a real container.
//!
//! Skipped unless `VOID_ENGINE_PG_URL` **and** `VOID_ENGINE_PG_CONTAINER`
//! are both set, because they need to control the server's lifecycle, not
//! just talk to it. CI sets neither by default: a service container is not
//! stoppable from inside the job, so these run locally and in a dedicated
//! job that owns its own `docker`.
//!
//! ```text
//! VOID_ENGINE_PG_URL="host=127.0.0.1 port=55432 user=postgres \
//!     password=devpw dbname=void_engine" \
//! VOID_ENGINE_PG_CONTAINER=ve_pg \
//!   cargo test --test ledger_outage --no-default-features --features ledger-pg
//! ```

#![cfg(feature = "ledger-pg")]

use std::process::Command;
use std::time::Duration;

use void_engine::persist::ledger::{Account, IdemKey, LedgerError, TransferRequest};
use void_engine::persist::ledger_pg::{PgConfig, PgLedger, WriterHealth};
use void_engine::persist::store::LedgerStore;

fn env_pair() -> Option<(String, String)> {
    let url = std::env::var("VOID_ENGINE_PG_URL").ok()?;
    let container = std::env::var("VOID_ENGINE_PG_CONTAINER").ok()?;
    Some((url, container))
}

/// A schema of this test's own, so a failure leaves its evidence rather
/// than being scrubbed by the next test.
fn isolated(url: &str, tag: &str) -> String {
    let schema = format!("o_{tag}");
    let rt = tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap();
    rt.block_on(async {
        let (client, conn) = tokio_postgres::connect(url, tokio_postgres::NoTls)
            .await
            .expect("database must be reachable at setup");
        tokio::spawn(async move { let _ = conn.await; });
        client
            .batch_execute(&format!(
                "DROP SCHEMA IF EXISTS {schema} CASCADE; CREATE SCHEMA {schema};"
            ))
            .await
            .expect("creating the test schema");
    });
    format!("{url} options=-csearch_path={schema}")
}

fn docker(args: &[&str]) {
    let out = Command::new("docker").args(args).output().expect("docker must be runnable");
    assert!(out.status.success(), "docker {args:?} failed: {}", String::from_utf8_lossy(&out.stderr));
}

fn wait_ready(container: &str) {
    for _ in 0..60 {
        let ok = Command::new("docker")
            .args(["exec", container, "pg_isready", "-U", "postgres"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            return;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
    panic!("postgres did not come back");
}

fn transfer(from: Account, to: Account, amount: i64, key: &str, tick: u64) -> TransferRequest {
    TransferRequest {
        idem_key: IdemKey::server(key),
        from,
        to,
        asset: "credits".to_string(),
        amount,
        reason: "outage-test".to_string(),
        actor: "system".to_string(),
        tick,
        spends: None,
    }
}

macro_rules! outage_test {
    ($tag:literal) => {
        match env_pair() {
            Some((url, container)) => (isolated(&url, $tag), container),
            None => {
                eprintln!("skipping: VOID_ENGINE_PG_URL/_CONTAINER not set");
                return;
            }
        }
    };
}

/// The headline property: while the database is gone, the ledger refuses
/// value movements rather than accepting ones it cannot persist.
#[test]
fn writes_are_refused_while_the_database_is_down() {
    let (url, container) = outage_test!("refuse");
    let cfg = PgConfig {
        // Long enough that the outage does not exhaust retries and turn
        // this into the terminal case, which is a different test.
        max_retries: 60,
        retry_base_delay: Duration::from_millis(50),
        retry_max_delay: Duration::from_millis(200),
        ..PgConfig::new(url)
    };
    let mut ledger = PgLedger::open(cfg).expect("opens while the database is up");
    let alice = Account::player("alice");

    ledger.transfer(transfer(Account::Mint, alice.clone(), 1_000, "seed", 1)).unwrap();
    ledger.flush(Duration::from_secs(10)).expect("seed must commit");
    assert_eq!(ledger.health(), WriterHealth::Healthy);

    docker(&["stop", &container]);

    // Push work at it until the writer notices the connection is gone and
    // flips to degraded. The first few may be accepted into the journal:
    // the sim does not learn of the fault until the writer tries to
    // commit, which is inherent to not blocking the tick on the database.
    let mut refused: Option<LedgerError> = None;
    for i in 0..400u64 {
        match ledger.transfer(transfer(alice.clone(), Account::Burn, 1, &format!("d{i}"), 2 + i)) {
            Err(e) => {
                refused = Some(e);
                break;
            }
            Ok(_) => std::thread::sleep(Duration::from_millis(25)),
        }
    }

    let err = refused.expect("the ledger must eventually refuse while the database is down");
    assert!(
        matches!(err, LedgerError::WriterDegraded(_) | LedgerError::WriterBehind { .. }),
        "expected a degraded or backpressure refusal, got {err}",
    );

    docker(&["start", &container]);
    wait_ready(&container);
}

/// And the other half: once the database comes back, the ledger heals
/// itself without a restart, and the work queued during the outage lands.
#[test]
fn the_ledger_recovers_when_the_database_returns() {
    let (url, container) = outage_test!("recover");
    let cfg = PgConfig {
        max_retries: 120,
        retry_base_delay: Duration::from_millis(50),
        retry_max_delay: Duration::from_millis(200),
        ..PgConfig::new(url)
    };
    let mut ledger = PgLedger::open(cfg).expect("opens while the database is up");
    let alice = Account::player("alice");

    ledger.transfer(transfer(Account::Mint, alice.clone(), 500, "seed", 1)).unwrap();
    ledger.flush(Duration::from_secs(10)).unwrap();

    docker(&["stop", &container]);

    // Queue something during the outage, ignoring refusals — the point is
    // that whatever *is* accepted must survive.
    for i in 0..40u64 {
        let _ = ledger.transfer(transfer(alice.clone(), Account::Burn, 1, &format!("r{i}"), 2 + i));
        std::thread::sleep(Duration::from_millis(25));
    }

    docker(&["start", &container]);
    wait_ready(&container);

    // The writer should reconnect, drain, reconcile, and clear the flag
    // on its own. No restart, no operator.
    let mut healed = false;
    for _ in 0..120 {
        if ledger.health() == WriterHealth::Healthy {
            healed = true;
            break;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    assert!(healed, "the ledger must heal itself; health was {:?}", ledger.health());

    // And it accepts work again.
    ledger
        .transfer(transfer(alice.clone(), Account::Burn, 1, "after", 999))
        .expect("writes must be accepted after recovery");
    ledger.flush(Duration::from_secs(20)).expect("post-recovery work must commit");

    // The books must still balance, in memory and therefore in the log.
    assert!(ledger.audit_zero_sum().is_empty(), "an outage must not unbalance the books");
}

/// Reads keep working throughout. Refusing them would break exactly the
/// tooling an operator needs during an outage.
#[test]
fn reads_keep_working_while_degraded() {
    let (url, container) = outage_test!("reads");
    let cfg = PgConfig {
        max_retries: 60,
        retry_base_delay: Duration::from_millis(50),
        retry_max_delay: Duration::from_millis(200),
        ..PgConfig::new(url)
    };
    let mut ledger = PgLedger::open(cfg).expect("opens while the database is up");
    let alice = Account::player("alice");

    ledger.transfer(transfer(Account::Mint, alice.clone(), 750, "seed", 1)).unwrap();
    ledger.flush(Duration::from_secs(10)).unwrap();

    docker(&["stop", &container]);

    // Drive it into a degraded state.
    for i in 0..400u64 {
        if ledger.transfer(transfer(alice.clone(), Account::Burn, 1, &format!("x{i}"), 2 + i)).is_err() {
            break;
        }
        std::thread::sleep(Duration::from_millis(25));
    }

    // Reads answer from the in-memory core, which is fed by the durable
    // log, so they remain correct and available.
    assert!(ledger.balance(&alice, "credits") > 0, "balance must still be readable");
    assert!(!ledger.history(&alice).is_empty(), "history must still be readable");
    assert!(ledger.audit_zero_sum().is_empty(), "the audit must still run");

    docker(&["start", &container]);
    wait_ready(&container);
}
