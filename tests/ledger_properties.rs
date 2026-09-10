//! Phase 3's exit criterion: no sequence of transfers, retries and
//! competing spends can create or destroy value.
//!
//! The unit tests in `persist::ledger` cover named cases — a retry, an
//! overdraft, a self-transfer. This covers the cases nobody thought of,
//! by generating long random sequences and asserting the invariants hold
//! after every one.
//!
//! No proptest dependency: a seeded LCG plus a fixed set of seeds gives
//! reproducible sequences without adding a crate. A failure prints its
//! seed, so it can be replayed exactly.

#![cfg(feature = "ledger")]

use void_engine::persist::ledger::{Account, Amount, IdemKey, Ledger, TransferRequest};
use void_engine::rng::Lcg;

const ASSETS: [&str; 3] = ["credits", "item:ore", "item:sword"];

fn accounts() -> Vec<Account> {
    vec![
        Account::player("alice"),
        Account::player("bob"),
        Account::player("carol"),
        Account::system("shop"),
        Account::system("bank"),
        Account::Burn,
    ]
}

/// Assert the two invariants that must hold no matter what happened.
fn assert_invariants(l: &Ledger, seed: u64, step: usize) {
    let imbalances = l.audit_zero_sum();
    assert!(
        imbalances.is_empty(),
        "seed {seed} step {step}: value was created or destroyed: {imbalances:?}",
    );

    let drift = l.audit_balances_match_entries();
    assert!(
        drift.is_empty(),
        "seed {seed} step {step}: cached balance drifted from the log: {drift:?}",
    );
}

/// Drive a long random sequence of operations and check the invariants
/// after each one.
fn run_sequence(seed: u64, steps: usize) {
    let mut rng = Lcg::new(seed);
    let mut l = Ledger::new();
    let accts = accounts();

    // Seed the economy from Mint so there is something to move around.
    for (i, a) in accts.iter().enumerate() {
        if a.is_economy_boundary() {
            continue;
        }
        for asset in ASSETS {
            l.transfer(TransferRequest {
                idem_key: IdemKey::server(format!("seed-{i}-{asset}")),
                from: Account::Mint,
                to: a.clone(),
                asset: asset.to_string(),
                amount: 1_000,
                reason: "seed".to_string(),
                actor: "system".to_string(),
                tick: 0,
                spends: None,
            })
            .expect("seeding from Mint must succeed");
        }
    }

    // Keep a pool of previously-used keys so retries actually collide.
    let mut used_keys: Vec<IdemKey> = Vec::new();

    for step in 0..steps {
        let pick = |rng: &mut Lcg, n: usize| (rng.next() % n as u64) as usize;

        let from = accts[pick(&mut rng, accts.len())].clone();
        let to = accts[pick(&mut rng, accts.len())].clone();
        let asset = ASSETS[pick(&mut rng, ASSETS.len())];
        // Deliberately spans zero, negatives and amounts larger than any
        // balance, so refusals are exercised as much as successes.
        let amount: Amount = (rng.next() % 3_000) as i64 - 500;

        // One in four operations replays an earlier key — the retry path.
        let idem_key = if !used_keys.is_empty() && rng.next().is_multiple_of(4) {
            used_keys[pick(&mut rng, used_keys.len())].clone()
        } else {
            let k = IdemKey::new("session", step as u64, "op");
            used_keys.push(k.clone());
            k
        };

        let _ = l.transfer(TransferRequest {
            idem_key,
            from,
            to,
            asset: asset.to_string(),
            amount,
            reason: "fuzz".to_string(),
            actor: "test".to_string(),
            tick: step as u64,
            spends: None,
        });

        // Occasionally hold and release funds, so reservations interleave
        // with transfers rather than being tested in isolation.
        if rng.next().is_multiple_of(8) {
            let who = accts[pick(&mut rng, accts.len())].clone();
            let amt = (rng.next() % 400) as i64 + 1;
            // Deadlines spread across the run so some holds lapse naturally
            // and some are released explicitly — both paths must keep the
            // books balanced.
            let expiry = step as u64 + (rng.next() % 50);
            if let Ok(r) = l.reserve(&who, asset, amt, expiry) {
                if rng.next().is_multiple_of(2) {
                    l.release(r).expect("a held reservation must release");
                }
                // Otherwise it stays held, which must reduce availability
                // without ever affecting the books.
            }
        }

        assert_invariants(&l, seed, step);
    }
}

/// The headline property, over several independent sequences.
#[test]
fn no_sequence_of_operations_can_create_or_destroy_value() {
    for seed in [1, 7, 42, 1337, 99_991] {
        run_sequence(seed, 400);
    }
}

/// A long single run, to catch anything that only emerges with depth.
#[test]
fn a_long_sequence_holds_the_invariants() {
    run_sequence(0xDEADBEEF, 3_000);
}

/// Retries specifically: hammer the same few keys and assert the ledger
/// never moves value more than once per key.
#[test]
fn heavy_retry_pressure_moves_each_transfer_exactly_once() {
    let mut l = Ledger::new();
    let alice = Account::player("alice");
    let bob = Account::player("bob");

    l.transfer(TransferRequest {
        idem_key: IdemKey::server("seed"),
        from: Account::Mint,
        to: alice.clone(),
        asset: "credits".to_string(),
        amount: 10_000,
        reason: "seed".to_string(),
        actor: "system".to_string(),
        tick: 0,
        spends: None,
    })
    .unwrap();

    // Ten distinct transfers, each submitted twenty times.
    for i in 0..10u64 {
        for _ in 0..20 {
            let _ = l.transfer(TransferRequest {
                idem_key: IdemKey::new("s", i, "buy"),
                from: alice.clone(),
                to: bob.clone(),
                asset: "credits".to_string(),
                amount: 100,
                reason: "buy".to_string(),
                actor: "alice".to_string(),
                tick: i,
                spends: None,
            });
        }
    }

    assert_eq!(l.balance(&bob, "credits"), 1_000, "each transfer must land exactly once");
    assert_eq!(l.balance(&alice, "credits"), 9_000);
    assert!(l.audit_zero_sum().is_empty());
    assert!(l.audit_balances_match_entries().is_empty());
}

/// Total supply of a unique item must stay at one however it is traded.
#[test]
fn a_unique_item_is_never_duplicated() {
    let mut rng = Lcg::new(24);
    let mut l = Ledger::new();
    let accts = accounts();
    let sword = "item:sword";

    l.transfer(TransferRequest {
        idem_key: IdemKey::server("forge"),
        from: Account::Mint,
        to: accts[0].clone(),
        asset: sword.to_string(),
        amount: 1,
        reason: "quest_reward".to_string(),
        actor: "system".to_string(),
        tick: 0,
        spends: None,
    })
    .unwrap();

    for step in 0..500u64 {
        let from = accts[(rng.next() % accts.len() as u64) as usize].clone();
        let to = accts[(rng.next() % accts.len() as u64) as usize].clone();
        let _ = l.transfer(TransferRequest {
            idem_key: IdemKey::new("s", step, "trade"),
            from,
            to,
            asset: sword.to_string(),
            amount: 1,
            reason: "trade".to_string(),
            actor: "test".to_string(),
            tick: step,
            spends: None,
        });

        // Exactly one sword exists, and it is in exactly one place.
        let held: Amount = accts.iter().map(|a| l.balance(a, sword)).sum();
        assert_eq!(held, 1, "step {step}: the sword was duplicated or lost");
        assert_eq!(l.total_minted(sword), 1, "step {step}: a second sword was minted");
    }

    assert!(l.audit_zero_sum().is_empty());
}
