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

    /// Iterate all entities with component T. Single allocation (raw-pointer vec → safe refs).
    pub fn iter<T: Send + Sync + 'static>(&self) -> impl Iterator<Item = (EntityId, &T)> {
        let pairs: Vec<(EntityId, *const T)> = match self.storage::<T>() {
            None => vec![],
            Some(storage) => storage
                .data
                .iter()
                .enumerate()
                .filter_map(|(i, slot)| {
                    let val = slot.as_ref()?;
                    if *self.alive.get(i).unwrap_or(&false) {
                        Some((EntityId { index: i as u32, generation: self.generations[i] }, val as *const T))
                    } else {
                        None
                    }
                })
                .collect(),
        };
        pairs.into_iter().map(|(id, ptr)| (id, unsafe { &*ptr }))
    }

    pub fn iter_mut<T: Send + Sync + 'static>(&mut self) -> impl Iterator<Item = (EntityId, &mut T)> {
        let alive = &self.alive;
        let gens = &self.generations;
        let storage = self
            .components
            .entry(TypeId::of::<T>())
            .or_insert_with(|| Box::new(ComponentStorage::<T>::new()) as Box<dyn AnyStorage>)
            .as_any_mut()
            .downcast_mut::<ComponentStorage<T>>()
            .unwrap();

        let pairs: Vec<(EntityId, *mut T)> = storage
            .data
            .iter_mut()
            .enumerate()
            .filter_map(|(i, slot)| {
                let val = slot.as_mut()?;
                if *alive.get(i).unwrap_or(&false) {
                    Some((EntityId { index: i as u32, generation: gens[i] }, val as *mut T))
                } else {
                    None
                }
            })
            .collect();

        pairs.into_iter().map(|(id, ptr)| (id, unsafe { &mut *ptr }))
    }

    /// Iterate entities that have both A and B. Walks A's storage, O(1) lookup into B.
    /// Eliminates the collect-then-double-get pattern: one pass, two components.
    pub fn iter2<A: Send + Sync + 'static, B: Send + Sync + 'static>(&self) -> impl Iterator<Item = (EntityId, &A, &B)> {
        let pairs: Vec<(EntityId, *const A, *const B)> = match (self.storage::<A>(), self.storage::<B>()) {
            (Some(sa), Some(sb)) => sa
                .data
                .iter()
                .enumerate()
                .filter_map(|(i, slot_a)| {
                    let a = slot_a.as_ref()?;
                    if !self.alive.get(i).copied().unwrap_or(false) {
                        return None;
                    }
                    let b = sb.get(i)?;
                    Some((
                        EntityId { index: i as u32, generation: self.generations[i] },
                        a as *const A,
                        b as *const B,
                    ))
                })
                .collect(),
            _ => vec![],
        };
        pairs.into_iter().map(|(id, pa, pb)| (id, unsafe { &*pa }, unsafe { &*pb }))
    }
}

impl Default for World {
    fn default() -> Self {
        Self::new()
    }
}
