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
    assert_eq!(err, SnapshotError::SchemaMismatch {
        name: "destructible2d".into(), found: 1, expected: 2,
    });
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
