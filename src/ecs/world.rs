use std::any::{Any, TypeId};
use std::collections::HashMap;
use super::entity::EntityId;

/// Type-erased view over a `ComponentStorage<T>` so `despawn` can wipe any
/// component slot without knowing T at the call site. Without this, a slot
/// reused by a new entity inherits the old occupant's components.
// Require `Sync` on component storages so `&World` implements `Sync`
// automatically — needed by server-side rayon workloads (per-session
// snapshot build fans out to a worker pool with a shared `&World`
// reference). All game components are plain `Clone` data structs
// already; nothing in the tree stores `Rc<T>` or bare `Cell<T>`, so
// the added bound is a compile-time formality.
trait AnyStorage: Any + Send + Sync {
    fn clear_slot(&mut self, index: usize);
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

struct ComponentStorage<T> {
    data: Vec<Option<T>>,
}

impl<T: Send + Sync + 'static> ComponentStorage<T> {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn insert(&mut self, index: usize, val: T) {
        if index >= self.data.len() {
            self.data.resize_with(index + 1, || None);
        }
        self.data[index] = Some(val);
    }

    fn get(&self, index: usize) -> Option<&T> {
        self.data.get(index)?.as_ref()
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.data.get_mut(index)?.as_mut()
    }

    fn remove(&mut self, index: usize) {
        if index < self.data.len() {
            self.data[index] = None;
        }
    }
}

impl<T: Send + Sync + 'static> AnyStorage for ComponentStorage<T> {
    fn clear_slot(&mut self, index: usize) {
        if index < self.data.len() {
            self.data[index] = None;
        }
    }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}

pub struct World {
    generations: Vec<u32>,
    free_list: Vec<u32>,
    alive: Vec<bool>,
    components: HashMap<TypeId, Box<dyn AnyStorage>>,
}

impl World {
    pub fn new() -> Self {
        Self {
            generations: Vec::new(),
            free_list: Vec::new(),
            alive: Vec::new(),
            components: HashMap::new(),
        }
    }

    /// True when `spawn` will allocate fresh indices rather than reuse
    /// freed ones — i.e. every id it returns from now on is larger than
    /// every id it has returned so far.
    ///
    /// Lets a bulk spawner record a watermark and identify "everything I
    /// just created" by `index >= mark`, instead of snapshotting the
    /// whole world before and after. That difference is O(1) vs O(world)
    /// per batch, which is the difference between linear and quadratic
    /// when the batch runs thousands of times at startup.
    pub fn spawns_are_monotonic(&self) -> bool { self.free_list.is_empty() }

    /// The next index `spawn` would allocate when `spawns_are_monotonic`.
    pub fn next_index(&self) -> u32 { self.generations.len() as u32 }

    pub fn spawn(&mut self) -> EntityId {
        if let Some(index) = self.free_list.pop() {
            let gen = self.generations[index as usize];
            self.alive[index as usize] = true;
            EntityId { index, generation: gen }
        } else {
            let index = self.generations.len() as u32;
            self.generations.push(0);
            self.alive.push(true);
            EntityId { index, generation: 0 }
        }
    }

    pub fn despawn(&mut self, id: EntityId) {
        if !self.alive(id) {
            return;
        }
        // Wipe every component slot for this index — otherwise a new entity
        // popping this slot off the free list would inherit stale components.
        let idx = id.index as usize;
        for storage in self.components.values_mut() {
            storage.clear_slot(idx);
        }
        self.generations[idx] += 1;
        self.alive[idx] = false;
        self.free_list.push(id.index);
    }

    pub fn alive(&self, id: EntityId) -> bool {
        self.alive
            .get(id.index as usize)
            .copied()
            .unwrap_or(false)
            && self.generations.get(id.index as usize).copied() == Some(id.generation)
    }

    fn storage<T: Send + Sync + 'static>(&self) -> Option<&ComponentStorage<T>> {
        self.components.get(&TypeId::of::<T>())?.as_any().downcast_ref::<ComponentStorage<T>>()
    }

    fn storage_mut<T: Send + Sync + 'static>(&mut self) -> &mut ComponentStorage<T> {
        self.components
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(ComponentStorage::<T>::new()) as Box<dyn AnyStorage>)
            .as_any_mut()
            .downcast_mut::<ComponentStorage<T>>()
            .unwrap()
    }

    pub fn insert<T: Send + Sync + 'static>(&mut self, id: EntityId, val: T) {
        if !self.alive(id) {
            return;
        }
        self.storage_mut::<T>().insert(id.index as usize, val);
    }

    pub fn get<T: Send + Sync + 'static>(&self, id: EntityId) -> Option<&T> {
        if !self.alive(id) {
            return None;
        }
        self.storage::<T>()?.get(id.index as usize)
    }

    pub fn get_mut<T: Send + Sync + 'static>(&mut self, id: EntityId) -> Option<&mut T> {
        if !self.alive(id) {
            return None;
        }
        self.storage_mut::<T>().get_mut(id.index as usize)
    }

    pub fn has<T: Send + Sync + 'static>(&self, id: EntityId) -> bool {
        self.get::<T>(id).is_some()
    }

    pub fn remove<T: Send + Sync + 'static>(&mut self, id: EntityId) {
        self.storage_mut::<T>().remove(id.index as usize);
    }

    pub fn entities(&self) -> impl Iterator<Item = EntityId> + '_ {
        self.alive
            .iter()
            .enumerate()
            .filter(|(_, &a)| a)
            .map(move |(i, _)| EntityId {
                index: i as u32,
                generation: self.generations[i],
            })
    }

    /// How many distinct component types have storage registered.
    ///
    /// Exposed for tests that pin the "reading does not allocate storage"
    /// property; games have no reason to care.
    #[doc(hidden)]
    pub fn component_kinds(&self) -> usize {
        self.components.len()
    }

    /// Iterate all entities with component T.
    ///
    /// Lazy: no allocation, no intermediate buffer. This used to `collect`
    /// into a `Vec<(EntityId, *const T)>` and then re-yield it — the raw
    /// pointers existed only to escape the borrow checker, since an
    /// iterator closing over `&self` cannot also yield references derived
    /// from it. Splitting the borrows (`alive` and `generations` separately
    /// from the storage) removes the conflict, so the pointers and the
    /// allocation both go away.
    ///
    /// That allocation was not incidental. Query cost scales with
    /// (entities × systems), not entities, because every system pays a
    /// fresh heap allocation and a full scan; measured at 50k entities over
    /// 20 systems, ~83% of the time was the collect and only ~7% the actual
    /// iteration. See `benches/hot_paths.rs`.
    pub fn iter<T: Send + Sync + 'static>(&self) -> impl Iterator<Item = (EntityId, &T)> {
        let alive = &self.alive;
        let gens = &self.generations;
        self.storage::<T>()
            .into_iter()
            .flat_map(move |storage| storage.data.iter().enumerate())
            .filter_map(move |(i, slot)| {
                let val = slot.as_ref()?;
                if *alive.get(i).unwrap_or(&false) {
                    Some((EntityId { index: i as u32, generation: gens[i] }, val))
                } else {
                    None
                }
            })
    }

    /// Iterate all entities with component T, mutably.
    ///
    /// Lazy, for the reasons on [`World::iter`]. `alive`/`generations` are
    /// borrowed immutably and the storage mutably, from disjoint fields of
    /// `self` — the borrow checker accepts that split, which is what lets
    /// the raw-pointer `Vec` go away here too.
    ///
    /// Unlike the old version this no longer inserts an empty storage as a
    /// side effect of iterating an unknown component; iterating nothing
    /// simply yields nothing.
    pub fn iter_mut<T: Send + Sync + 'static>(&mut self) -> impl Iterator<Item = (EntityId, &mut T)> {
        let alive = &self.alive;
        let gens = &self.generations;
        self.components
            .get_mut(&TypeId::of::<T>())
            .and_then(|s| s.as_any_mut().downcast_mut::<ComponentStorage<T>>())
            .into_iter()
            .flat_map(move |storage| {
                storage.data.iter_mut().enumerate().filter_map(move |(i, slot)| {
                    let val = slot.as_mut()?;
                    if *alive.get(i).unwrap_or(&false) {
                        Some((EntityId { index: i as u32, generation: gens[i] }, val))
                    } else {
                        None
                    }
                })
            })
    }

    /// Iterate entities that have both A and B. Walks A's storage, O(1)
    /// lookup into B — one pass, two components.
    ///
    /// Lazy, for the reasons on [`World::iter`]. Both storages are resolved
    /// once up front rather than per element: `storage::<B>()` is a
    /// `TypeId` hash lookup plus a downcast, so hoisting it out of the loop
    /// matters as much as dropping the allocation did.
    pub fn iter2<A: Send + Sync + 'static, B: Send + Sync + 'static>(
        &self,
    ) -> impl Iterator<Item = (EntityId, &A, &B)> {
        let alive = &self.alive;
        let gens = &self.generations;
        // `zip` on the Options gives "both present or nothing", which is the
        // empty-iterator case the old `_ => vec![]` arm handled.
        self.storage::<A>()
            .zip(self.storage::<B>())
            .into_iter()
            .flat_map(move |(sa, sb)| {
                sa.data.iter().enumerate().filter_map(move |(i, slot_a)| {
                    let a = slot_a.as_ref()?;
                    if !alive.get(i).copied().unwrap_or(false) {
                        return None;
                    }
                    let b = sb.get(i)?;
                    Some((EntityId { index: i as u32, generation: gens[i] }, a, b))
                })
            })
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    struct Pos(i32, i32);
    #[derive(Clone, Debug, PartialEq)]
    struct Vel(i32);
    #[derive(Clone, Debug, PartialEq)]
    struct Tag;

    // -- identity and liveness ---------------------------------------

    #[test]
    fn spawn_yields_distinct_live_ids() {
        let mut w = World::new();
        let a = w.spawn();
        let b = w.spawn();
        assert_ne!(a, b);
        assert!(w.alive(a) && w.alive(b));
    }

    #[test]
    fn despawn_kills_only_its_own_entity() {
        let mut w = World::new();
        let a = w.spawn();
        let b = w.spawn();
        w.despawn(a);
        assert!(!w.alive(a));
        assert!(w.alive(b), "despawn must not affect unrelated entities");
    }

    #[test]
    fn despawning_twice_is_a_no_op() {
        let mut w = World::new();
        let a = w.spawn();
        w.despawn(a);
        w.despawn(a);
        // The second despawn must not push the index onto the free list a
        // second time -- that would hand the same slot to two live entities.
        let x = w.spawn();
        let y = w.spawn();
        assert_ne!(x.index, y.index, "a slot was handed out twice");
    }

    /// The core of the generational scheme: a reused slot must not answer
    /// to the old id.
    #[test]
    fn a_reused_slot_rejects_the_stale_id() {
        let mut w = World::new();
        let old = w.spawn();
        w.despawn(old);
        let new = w.spawn();
        assert_eq!(new.index, old.index, "expected the slot to be reused");
        assert_ne!(new.generation, old.generation, "generation must advance");
        assert!(!w.alive(old));
        assert!(w.alive(new));
    }

    #[test]
    fn an_out_of_range_id_is_not_alive() {
        let mut w = World::new();
        let real = w.spawn();
        let far = EntityId { index: 999, generation: 0 };
        assert!(!w.alive(far), "out-of-range index must not report alive");
        assert!(w.get::<Pos>(far).is_none());
        assert!(w.alive(real));
    }

    // -- component storage -------------------------------------------

    #[test]
    fn insert_get_and_remove_round_trip() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(w.get::<Pos>(e).is_none());
        w.insert(e, Pos(3, 4));
        assert_eq!(w.get::<Pos>(e), Some(&Pos(3, 4)));
        assert!(w.has::<Pos>(e));
        w.remove::<Pos>(e);
        assert!(w.get::<Pos>(e).is_none());
        assert!(!w.has::<Pos>(e));
    }

    #[test]
    fn insert_overwrites_the_previous_value() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Pos(1, 1));
        w.insert(e, Pos(2, 2));
        assert_eq!(w.get::<Pos>(e), Some(&Pos(2, 2)));
    }

    #[test]
    fn get_mut_mutates_in_place() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Vel(1));
        w.get_mut::<Vel>(e).unwrap().0 = 42;
        assert_eq!(w.get::<Vel>(e), Some(&Vel(42)));
    }

    #[test]
    fn components_are_keyed_by_type_not_slot() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Pos(1, 2));
        w.insert(e, Vel(9));
        assert_eq!(w.get::<Pos>(e), Some(&Pos(1, 2)));
        assert_eq!(w.get::<Vel>(e), Some(&Vel(9)));
    }

    #[test]
    fn insert_on_a_dead_entity_is_ignored() {
        let mut w = World::new();
        let e = w.spawn();
        w.despawn(e);
        w.insert(e, Pos(1, 1));
        assert!(w.get::<Pos>(e).is_none());
    }

    /// The invariant the `despawn` comment calls out by name: a recycled
    /// slot must not inherit the previous occupant components.
    #[test]
    fn a_recycled_slot_inherits_no_components() {
        let mut w = World::new();
        let old = w.spawn();
        w.insert(old, Pos(7, 7));
        w.insert(old, Vel(7));
        w.insert(old, Tag);
        w.despawn(old);

        let new = w.spawn();
        assert_eq!(new.index, old.index, "expected slot reuse");
        assert!(w.get::<Pos>(new).is_none(), "stale Pos leaked into a new entity");
        assert!(w.get::<Vel>(new).is_none(), "stale Vel leaked into a new entity");
        assert!(w.get::<Tag>(new).is_none(), "stale Tag leaked into a new entity");
    }

    // -- iteration ---------------------------------------------------

    #[test]
    fn iter_visits_only_live_entities_with_the_component() {
        let mut w = World::new();
        let a = w.spawn();
        let b = w.spawn();
        let c = w.spawn();
        w.insert(a, Pos(1, 0));
        w.insert(b, Pos(2, 0));
        // c deliberately has no Pos.
        let _ = c;
        w.despawn(b);

        let seen: Vec<_> = w.iter::<Pos>().map(|(id, p)| (id, p.clone())).collect();
        assert_eq!(seen, vec![(a, Pos(1, 0))],
            "iter must skip dead and component-less entities");
    }

    #[test]
    fn iter_yields_ids_whose_generation_matches() {
        let mut w = World::new();
        let e = w.spawn();
        w.despawn(e);
        let reused = w.spawn();
        w.insert(reused, Pos(5, 5));

        let ids: Vec<_> = w.iter::<Pos>().map(|(id, _)| id).collect();
        assert_eq!(ids, vec![reused]);
        assert!(w.alive(ids[0]), "an id handed out by iter must be alive");
    }

    #[test]
    fn iter_mut_writes_are_visible_afterwards() {
        let mut w = World::new();
        let a = w.spawn();
        let b = w.spawn();
        w.insert(a, Vel(1));
        w.insert(b, Vel(2));
        for (_, v) in w.iter_mut::<Vel>() {
            v.0 *= 10;
        }
        assert_eq!(w.get::<Vel>(a), Some(&Vel(10)));
        assert_eq!(w.get::<Vel>(b), Some(&Vel(20)));
    }

    #[test]
    fn iter2_yields_only_entities_having_both() {
        let mut w = World::new();
        let both = w.spawn();
        let only_a = w.spawn();
        let only_b = w.spawn();
        w.insert(both, Pos(1, 1));
        w.insert(both, Vel(1));
        w.insert(only_a, Pos(2, 2));
        w.insert(only_b, Vel(2));

        let seen: Vec<_> = w.iter2::<Pos, Vel>().map(|(id, _, _)| id).collect();
        assert_eq!(seen, vec![both]);
    }

    #[test]
    fn iter2_skips_despawned_entities() {
        let mut w = World::new();
        let a = w.spawn();
        let b = w.spawn();
        for e in [a, b] {
            w.insert(e, Pos(0, 0));
            w.insert(e, Vel(0));
        }
        w.despawn(a);
        let seen: Vec<_> = w.iter2::<Pos, Vel>().map(|(id, _, _)| id).collect();
        assert_eq!(seen, vec![b]);
    }

    #[test]
    fn iterating_an_unknown_component_is_empty_not_a_panic() {
        let w = World::new();
        assert_eq!(w.iter::<Pos>().count(), 0);
        assert_eq!(w.iter2::<Pos, Vel>().count(), 0);
    }

    #[test]
    fn entities_lists_exactly_the_live_set() {
        let mut w = World::new();
        let a = w.spawn();
        let b = w.spawn();
        let c = w.spawn();
        w.despawn(b);
        let mut live: Vec<_> = w.entities().collect();
        live.sort_by_key(|e| e.index);
        assert_eq!(live, vec![a, c]);
    }

    // -- the monotonic-spawn watermark contract ----------------------

    #[test]
    fn spawns_are_monotonic_until_a_slot_is_freed() {
        let mut w = World::new();
        assert!(w.spawns_are_monotonic());
        let mark = w.next_index();
        let a = w.spawn();
        let b = w.spawn();
        assert!(w.spawns_are_monotonic(), "no despawn yet, so still monotonic");
        assert!(a.index >= mark && b.index >= mark,
            "ids issued after the mark must be at or above it");

        w.despawn(a);
        assert!(!w.spawns_are_monotonic(),
            "a freed slot breaks the watermark contract");
    }

    /// The iterators are lazy: taking a few items must not walk the whole
    /// storage. Guards against a future refactor quietly reintroducing the
    /// `collect()` these used to do, which made every query cost a full
    /// scan plus a heap allocation regardless of how much was consumed.
    #[test]
    fn iterators_are_lazy_not_collected() {
        let mut w = World::new();
        for i in 0..10_000 {
            let e = w.spawn();
            w.insert(e, Pos(i, i));
        }
        // `next()` on a collecting iterator would have already scanned all
        // 10k slots and allocated; on a lazy one it stops at the first hit.
        let mut it = w.iter::<Pos>();
        assert_eq!(it.next().map(|(_, p)| p.clone()), Some(Pos(0, 0)));
        assert_eq!(it.next().map(|(_, p)| p.clone()), Some(Pos(1, 1)));
        drop(it);

        // Same for the two-component query.
        for (i, _) in w.iter::<Pos>().take(3).enumerate() {
            let _ = i;
        }
        assert_eq!(w.iter::<Pos>().take(5).count(), 5);
    }

    /// Reading through `iter_mut` must not create storage as a side effect.
    /// The old implementation used `entry().or_insert_with(..)`, so merely
    /// iterating an unknown component registered an empty storage for it —
    /// which then cost every `despawn` an extra slot wipe forever after.
    #[test]
    fn iter_mut_over_an_unknown_component_registers_nothing() {
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, Pos(1, 1));

        assert_eq!(w.component_kinds(), 1, "only Pos should be registered");
        assert_eq!(w.iter_mut::<Vel>().count(), 0, "no Vel exists yet");
        assert_eq!(
            w.component_kinds(),
            1,
            "iterating an absent component must not register storage for it"
        );

        // And it still works normally once something is actually inserted.
        w.insert(e, Vel(5));
        assert_eq!(w.component_kinds(), 2);
        assert_eq!(w.iter_mut::<Vel>().count(), 1);
    }

    #[test]
    fn next_index_predicts_the_next_fresh_slot() {
        let mut w = World::new();
        let predicted = w.next_index();
        let e = w.spawn();
        assert_eq!(e.index, predicted);
    }
}
