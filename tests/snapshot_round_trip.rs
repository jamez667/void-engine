//! Phase 1's exit criterion: a restored world is indistinguishable from
//! the original.
//!
//! An integration test rather than a unit test on purpose — it compiles
//! against the public API exactly as a game would, so if `Registry`,
//! `capture` or `restore` stop being usable from outside the crate, this
//! fails.
//!
//! The invariant under test is stronger than "the same components come
//! back". Entity ids, generations and the free list must be reproduced
//! *verbatim*: a saved id that renumbers on load is a dangling reference
//! in every record that mentioned it.

#![cfg(feature = "persist")]

use glam::DVec2;
use void_engine::components::{Collider, Destructible2D, Particle, Transform2D, Velocity};
use void_engine::persist::{
    capture, from_bytes, register_engine_components, restore, restore_rng, to_bytes, Persist,
    Registry, RngStreams, Snapshot, SnapshotError,
};
use void_engine::rng::Pcg32;
use void_engine::World;

fn engine_registry() -> Registry {
    let mut r = Registry::new();
    register_engine_components(&mut r).expect("engine components must register cleanly");
    r
}

/// A world with holes in it: entities spawned, some despawned, so the
/// free list is non-empty and generations have advanced.
fn populated_world() -> (World, Vec<void_engine::EntityId>) {
    let mut w = World::new();
    let mut ids = Vec::new();
    for i in 0..10 {
        let e = w.spawn();
        w.insert(e, Transform2D { pos: DVec2::new(i as f64, i as f64 * 2.0), rot: i as f32 });
        w.insert(e, Velocity { linear: DVec2::new(1.0, -1.0), angular: 0.5 });
        if i % 3 == 0 {
            w.insert(e, Collider::circle(4.0 + i as f32));
        }
        ids.push(e);
    }
    // Punch holes so the free list and generation counters are exercised.
    w.despawn(ids[2]);
    w.despawn(ids[7]);
    (w, ids)
}

#[test]
fn a_restored_world_matches_the_original() {
    let reg = engine_registry();
    let (world, ids) = populated_world();

    let snap = capture(&world, &reg, 42, &RngStreams::new()).unwrap();
    let restored = restore(&snap, &reg).unwrap();

    // Every live entity keeps its exact id, generation and component data.
    for (i, &id) in ids.iter().enumerate() {
        assert_eq!(world.alive(id), restored.alive(id), "liveness differs for entity {i}");
        if !world.alive(id) {
            continue;
        }
        let a = world.get::<Transform2D>(id).unwrap();
        let b = restored.get::<Transform2D>(id).expect("live entity lost its Transform2D");
        assert_eq!((a.pos, a.rot), (b.pos, b.rot), "transform differs for entity {i}");

        assert_eq!(
            world.get::<Collider>(id).map(|c| (c.radius, c.size)),
            restored.get::<Collider>(id).map(|c| (c.radius, c.size)),
            "collider differs for entity {i}",
        );
    }

    let before: Vec<_> = world.entities().collect();
    let after: Vec<_> = restored.entities().collect();
    assert_eq!(before, after, "the live entity set must be identical");
}

/// The property that makes saved cross-references safe. A despawned slot
/// is reused with a bumped generation; if a restore renumbered, the stale
/// id would wrongly resolve.
#[test]
fn generations_and_the_free_list_survive_exactly() {
    let reg = engine_registry();
    let (world, ids) = populated_world();

    let snap = capture(&world, &reg, 0, &RngStreams::new()).unwrap();
    let mut restored = restore(&snap, &reg).unwrap();

    let stale = ids[2]; // despawned before the snapshot
    assert!(!restored.alive(stale), "a despawned id must not come back alive");

    // The next spawn must reuse the freed slot with a fresh generation,
    // exactly as the original world would have.
    let mut original = world;
    let a = original.spawn();
    let b = restored.spawn();
    assert_eq!(a.index, b.index, "restore changed which slot is reused next");
    assert_eq!(a.generation, b.generation, "restore changed the generation counter");
    assert_ne!(b.generation, stale.generation, "the reused slot must reject the stale id");
}

#[test]
fn a_snapshot_round_trips_through_bytes() {
    let reg = engine_registry();
    let (world, ids) = populated_world();

    let snap = capture(&world, &reg, 7, &RngStreams::new()).unwrap();
    let bytes = to_bytes(&snap).unwrap();
    let decoded: Snapshot = from_bytes(&bytes).unwrap();
    assert_eq!(decoded.tick, 7);

    let restored = restore(&decoded, &reg).unwrap();
    let live = ids.iter().find(|&&i| world.alive(i)).copied().unwrap();
    assert_eq!(
        world.get::<Transform2D>(live).unwrap().pos,
        restored.get::<Transform2D>(live).unwrap().pos,
    );
}

/// Transient components are deliberately not saved — restoring mid-flight
/// sparks is worse than letting them lapse.
#[test]
fn transient_components_are_not_restored() {
    let reg = engine_registry();
    let mut w = World::new();
    let e = w.spawn();
    w.insert(e, Transform2D { pos: DVec2::ZERO, rot: 0.0 });
    w.insert(e, Particle {
        lifetime: 1.0, max_lifetime: 1.0,
        color_start: [1.0; 4], color_end: [0.0; 4],
        size_start: 2.0, size_end: 0.0, tag: 0,
    });

    let snap = capture(&w, &reg, 0, &RngStreams::new()).unwrap();
    let restored = restore(&snap, &reg).unwrap();

    assert!(restored.get::<Transform2D>(e).is_some(), "volatile state must survive");
    assert!(restored.get::<Particle>(e).is_none(), "transient state must not");
}

/// RNG streams resume at their captured position, not from their seed.
#[test]
fn rng_streams_resume_mid_stream() {
    let reg = engine_registry();
    let (world, _) = populated_world();

    let mut worldgen = Pcg32::seed(0xABCDEF, 1);
    let mut loot = Pcg32::seed(0xABCDEF, 2);
    for _ in 0..500 {
        worldgen.next_u32();
        loot.next_u32();
    }

    let streams = RngStreams::new().add("worldgen", &worldgen).add("loot", &loot);
    let snap = capture(&world, &reg, 0, &streams).unwrap();

    let mut back = restore_rng(&snap);
    let mut r_worldgen = back.remove("worldgen").expect("worldgen stream missing");
    let mut r_loot = back.remove("loot").expect("loot stream missing");

    assert_eq!(worldgen.next_u32(), r_worldgen.next_u32(), "worldgen diverged");
    assert_eq!(loot.next_u32(), r_loot.next_u32(), "loot diverged");
    // And the two streams stay independent after restore.
    assert_ne!(r_worldgen.next_u32(), r_loot.next_u32());
}

/// A component saved under an old name still loads once an alias exists.
/// This is what makes a badly-chosen name recoverable.
#[test]
fn a_renamed_component_still_loads() {
    let mut old_reg = Registry::new();
    old_reg.register::<Transform2D>("xform", Persist::Volatile).unwrap();

    let mut w = World::new();
    let e = w.spawn();
    w.insert(e, Transform2D { pos: DVec2::new(3.0, 4.0), rot: 1.5 });
    let snap = capture(&w, &old_reg, 0, &RngStreams::new()).unwrap();

    // The type has since been renamed in the registry, with an alias.
    let mut new_reg = Registry::new();
    new_reg.register::<Transform2D>("transform2d", Persist::Volatile).unwrap();
    new_reg.rename("xform", "transform2d").unwrap();

    let restored = restore(&snap, &new_reg).unwrap();
    assert_eq!(restored.get::<Transform2D>(e).unwrap().pos, DVec2::new(3.0, 4.0));
}

/// A snapshot naming a component this build does not know must fail
/// loudly, not silently drop the data.
#[test]
fn an_unknown_component_is_rejected() {
    let mut full = Registry::new();
    full.register::<Transform2D>("transform2d", Persist::Volatile).unwrap();

    let mut w = World::new();
    let e = w.spawn();
    w.insert(e, Transform2D { pos: DVec2::ZERO, rot: 0.0 });
    let snap = capture(&w, &full, 0, &RngStreams::new()).unwrap();

    let empty = Registry::new();
    let err = restore(&snap, &empty).err().expect("an unregistered component must fail");
    assert_eq!(err, SnapshotError::UnknownComponent("transform2d".into()));
}

/// A schema bump with no migration must be an error rather than a
/// misread — the fields moved, so the old bytes mean something else now.
#[test]
fn a_schema_mismatch_is_rejected() {
    let mut v1 = Registry::new();
    v1.register_versioned::<Destructible2D>("destructible2d", Persist::Volatile, 1).unwrap();

    let mut w = World::new();
    let e = w.spawn();
    w.insert(e, Destructible2D::new(5.0, 100.0));
    let snap = capture(&w, &v1, 0, &RngStreams::new()).unwrap();

    let mut v2 = Registry::new();
    v2.register_versioned::<Destructible2D>("destructible2d", Persist::Volatile, 2).unwrap();

    let err = restore(&snap, &v2).err().expect("a schema bump must fail");
    // Matched rather than compared whole: `detail` carries the underlying
    // `MigrateError`'s message, and pinning that string would make this
    // test fail on any rewording of an error nobody is asserting about.
    match err {
        SnapshotError::SchemaMismatch { name, found, expected, detail } => {
            assert_eq!(name, "destructible2d");
            assert_eq!((found, expected), (1, 2));
            assert!(
                detail.contains("no migration registered"),
                "the error should say which step is missing, got {detail:?}",
            );
        }
        other => panic!("expected a schema mismatch, got {other:?}"),
    }
}

/// The point of the migration chain: a component whose shape changed can
/// still load a save written before the change.
///
/// Before this existed, a version bump was unconditionally fatal to every
/// existing save — `registry.rs` promised "the load path runs migrations
/// forward" in three places and no such machinery was there.
#[test]
fn a_registered_migration_brings_an_old_save_forward() {
    let mut v1 = Registry::new();
    v1.register_versioned::<Destructible2D>("destructible2d", Persist::Volatile, 1).unwrap();

    let mut w = World::new();
    let e = w.spawn();
    w.insert(e, Destructible2D::new(5.0, 100.0));
    let snap = capture(&w, &v1, 0, &RngStreams::new()).unwrap();

    let mut v2 = Registry::new();
    v2.register_versioned::<Destructible2D>("destructible2d", Persist::Volatile, 2).unwrap();
    // A migration that genuinely rewrites the bytes, so this test can
    // tell "the chain ran and its output was decoded" from "the chain was
    // skipped and the original bytes were decoded". A no-op step proves
    // only the first half, which is the trap: it passes whether or not
    // the migration is wired into the load path at all.
    //
    // This is what a real migration looks like: decode a mirror of the
    // old shape, change it, re-encode. Here the mirror is the same type,
    // and the change doubles every radius.
    v2.register_migration("destructible2d", 1, |bytes| {
        let mut col: Vec<Option<Destructible2D>> =
            bincode::deserialize(bytes).map_err(|e| e.to_string())?;
        for slot in col.iter_mut().flatten() {
            slot.radius *= 2.0;
        }
        bincode::serialize(&col).map_err(|e| e.to_string())
    })
    .unwrap();

    let restored = restore(&snap, &v2).expect("a migrated save must load");
    let got = restored
        .get::<Destructible2D>(e)
        .expect("the component must survive the migration");
    assert_eq!(
        got.radius, 10.0,
        "the migration doubles the radius, so 5.0 must arrive as 10.0 — \
         seeing 5.0 means the chain never ran and the original bytes were \
         decoded straight through",
    );
    assert_eq!(got.mass, 100.0, "fields the migration did not touch must survive");
}

/// Steps chain: a save two versions behind runs both migrations, in
/// order. Declaring a 1→3 jump is deliberately impossible, because a
/// chain of small steps is what lets a save written at *any* intermediate
/// version load.
#[test]
fn migrations_chain_across_several_versions_in_order() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static ORDER: AtomicUsize = AtomicUsize::new(0);

    let mut v1 = Registry::new();
    v1.register_versioned::<Destructible2D>("destructible2d", Persist::Volatile, 1).unwrap();

    let mut w = World::new();
    let e = w.spawn();
    w.insert(e, Destructible2D::new(3.0, 50.0));
    let snap = capture(&w, &v1, 0, &RngStreams::new()).unwrap();

    ORDER.store(0, Ordering::SeqCst);
    let mut v3 = Registry::new();
    v3.register_versioned::<Destructible2D>("destructible2d", Persist::Volatile, 3).unwrap();
    v3.register_migration("destructible2d", 1, |b| {
        assert_eq!(ORDER.swap(1, Ordering::SeqCst), 0, "1->2 must run first");
        Ok(b.to_vec())
    })
    .unwrap();
    v3.register_migration("destructible2d", 2, |b| {
        assert_eq!(ORDER.swap(2, Ordering::SeqCst), 1, "2->3 must run second");
        Ok(b.to_vec())
    })
    .unwrap();

    restore(&snap, &v3).expect("a two-step migration must load");
    assert_eq!(ORDER.load(Ordering::SeqCst), 2, "both steps must have run");
}

/// A gap in the chain is still fatal. The difference migrations make is
/// that a game *can* close the gap — not that a missing step is ignored,
/// which would read bytes in the wrong shape and corrupt state silently.
#[test]
fn a_gap_in_the_migration_chain_is_still_an_error() {
    let mut v1 = Registry::new();
    v1.register_versioned::<Destructible2D>("destructible2d", Persist::Volatile, 1).unwrap();

    let mut w = World::new();
    let e = w.spawn();
    w.insert(e, Destructible2D::new(1.0, 2.0));
    let snap = capture(&w, &v1, 0, &RngStreams::new()).unwrap();

    let mut v3 = Registry::new();
    v3.register_versioned::<Destructible2D>("destructible2d", Persist::Volatile, 3).unwrap();
    // Only the second half of the chain: 1->2 is missing.
    v3.register_migration("destructible2d", 2, |b| Ok(b.to_vec())).unwrap();

    let err = restore(&snap, &v3).err().expect("a gap must fail");
    match err {
        SnapshotError::SchemaMismatch { detail, .. } => assert!(
            detail.contains("from version 1 to 2"),
            "the error should name the missing step, got {detail:?}",
        ),
        other => panic!("expected a schema mismatch, got {other:?}"),
    }
}

/// A save from a *newer* build cannot be migrated: steps only run
/// forward. Refusing is the honest answer — the alternative is reading
/// fields this build does not know about as though they were fields it
/// does.
#[test]
fn a_component_from_a_future_build_is_refused() {
    let mut newer = Registry::new();
    newer.register_versioned::<Destructible2D>("destructible2d", Persist::Volatile, 5).unwrap();

    let mut w = World::new();
    let e = w.spawn();
    w.insert(e, Destructible2D::new(1.0, 2.0));
    let snap = capture(&w, &newer, 0, &RngStreams::new()).unwrap();

    let mut older = Registry::new();
    older.register_versioned::<Destructible2D>("destructible2d", Persist::Volatile, 2).unwrap();

    let err = restore(&snap, &older).err().expect("a future component must fail");
    match err {
        SnapshotError::SchemaMismatch { detail, .. } => assert!(
            detail.contains("newer than this build"),
            "the error should say the save is from the future, got {detail:?}",
        ),
        other => panic!("expected a schema mismatch, got {other:?}"),
    }
}

/// A failing migration surfaces its own message rather than a bare
/// version comparison — otherwise a game debugging a bad upgrade learns
/// only that two numbers differ.
#[test]
fn a_failing_migration_reports_why() {
    let mut v1 = Registry::new();
    v1.register_versioned::<Destructible2D>("destructible2d", Persist::Volatile, 1).unwrap();

    let mut w = World::new();
    let e = w.spawn();
    w.insert(e, Destructible2D::new(1.0, 2.0));
    let snap = capture(&w, &v1, 0, &RngStreams::new()).unwrap();

    let mut v2 = Registry::new();
    v2.register_versioned::<Destructible2D>("destructible2d", Persist::Volatile, 2).unwrap();
    v2.register_migration("destructible2d", 1, |_| Err("field 'shape' was never set".into()))
        .unwrap();

    let err = restore(&snap, &v2).err().expect("a failing migration must fail");
    match err {
        SnapshotError::SchemaMismatch { detail, .. } => assert!(
            detail.contains("field 'shape' was never set"),
            "the migration's own message should reach the caller, got {detail:?}",
        ),
        other => panic!("expected a schema mismatch, got {other:?}"),
    }
}

/// A migration for a step at or above the current version could never
/// run, so registering one is rejected rather than silently ignored — a
/// typo there leaves a registration that looks complete.
#[test]
fn a_migration_that_could_never_run_is_rejected() {
    let mut r = Registry::new();
    r.register_versioned::<Destructible2D>("destructible2d", Persist::Volatile, 2).unwrap();
    assert!(r.register_migration("destructible2d", 2, |b| Ok(b.to_vec())).is_err());
    assert!(r.register_migration("destructible2d", 7, |b| Ok(b.to_vec())).is_err());
    assert!(r.register_migration("destructible2d", 1, |b| Ok(b.to_vec())).is_ok());
}

/// A snapshot from a future engine is refused rather than partially read:
/// loading it would silently discard fields this build cannot see.
#[test]
fn a_future_format_is_refused() {
    let reg = engine_registry();
    let (world, _) = populated_world();
    let mut snap = capture(&world, &reg, 0, &RngStreams::new()).unwrap();
    snap.format_version = 999;

    let err = restore(&snap, &reg).err().expect("a future format must be refused");
    assert!(matches!(err, SnapshotError::FutureFormat { found: 999, .. }));
}

/// An empty world round-trips without special-casing.
#[test]
fn an_empty_world_round_trips() {
    let reg = engine_registry();
    let snap = capture(&World::new(), &reg, 0, &RngStreams::new()).unwrap();
    let restored = restore(&snap, &reg).unwrap();
    assert_eq!(restored.entities().count(), 0);
}
