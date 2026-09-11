//! Interest-managed replication: what each client is told, and how little
//! it costs to tell them.
//!
//! Without interest management a server sends every entity to every client
//! and per-client bandwidth grows with the size of the world. That is the
//! wall between a session game and an MMO, and it is a bandwidth wall long
//! before it is a CPU one.
//!
//! # The pipeline
//!
//! Per client, per tick:
//!
//! 1. **Relevancy** — [`SpatialGrid::query_circle_into`] over the client's
//!    viewpoint. Measured at 100k colliders and 1000 clients, this is
//!    ~12 ms of a 33.3 ms tick; see that method's docs for why it is not
//!    the obvious `query_circle`.
//! 2. **Delta** — compare against what this client was last *acknowledged*
//!    to hold and emit only what changed.
//! 3. **Encode** — quantised and bit-packed, components named by their
//!    registry [`NameId`].
//! 4. **Send** — [`send_chunked`], which splits an oversized packet across
//!    datagrams without ever dropping an entity.
//!
//! [`SpatialGrid::query_circle_into`]: crate::collision::SpatialGrid::query_circle_into
//! [`send_chunked`]: crate::net::chunk::send_chunked
//! [`NameId`]: crate::persist::registry::NameId
//!
//! # Why the grid index is not the entity id
//!
//! [`SpatialGrid`] hands back the dense index it assigned at `insert`,
//! which says nothing about which entity that was. The mapping has to be
//! recorded by whoever fills the grid, in insertion order — see
//! [`Relevancy`].
//!
//! That works because the grid is rebuilt every tick, so indices are
//! assigned fresh each time. R4's incremental broadphase would break it:
//! with `update`/`remove`, an index outlives the tick that created it and
//! the parallel vector has to be maintained rather than rebuilt.
//!
//! [`SpatialGrid`]: crate::collision::SpatialGrid
//!
//! # Why generation travels with spawns
//!
//! [`World::despawn`] bumps the slot's generation and frees the index
//! immediately, so a `spawn` later in the *same tick* can hand that index
//! to an unrelated entity. A client tracking entities by index alone would
//! quietly apply one entity's updates to another. Deltas therefore carry
//! only the index — cheap, and correct because a live entity's generation
//! never changes — while spawn and despawn carry the full [`EntityId`], so
//! a reused index is always announced.
//!
//! [`World::despawn`]: crate::World::despawn
//! [`EntityId`]: crate::EntityId

use crate::ecs::EntityId;

/// The set of entities one client can currently see, and the mapping back
/// from grid indices to entities.
///
/// Reused across ticks: the vectors are cleared and refilled rather than
/// reallocated, for the same reason [`AoiScratch`] exists.
///
/// [`AoiScratch`]: crate::collision::AoiScratch
#[derive(Default)]
pub struct Relevancy {
    /// Entity per grid index, in the order they were inserted into the
    /// grid this tick. `entities[i]` is what `SpatialGrid` index `i` means.
    entities: Vec<EntityId>,
}

impl Relevancy {
    pub fn new() -> Self { Self::default() }

    /// Begin a tick. Call before re-filling the grid.
    pub fn begin(&mut self) {
        self.entities.clear();
    }

    /// Record that the next `SpatialGrid::insert` belongs to `id`.
    ///
    /// Must be called once per insert, in the same order, or every index
    /// afterwards maps to the wrong entity. The grid assigns indices
    /// sequentially from zero, so this is a push.
    pub fn push(&mut self, id: EntityId) {
        self.entities.push(id);
    }

    /// The entity a grid index refers to, or `None` if the index was never
    /// recorded — which means the grid and this mapping have diverged.
    pub fn entity(&self, grid_index: u32) -> Option<EntityId> {
        self.entities.get(grid_index as usize).copied()
    }

    /// How many entities were recorded this tick.
    pub fn len(&self) -> usize { self.entities.len() }
    pub fn is_empty(&self) -> bool { self.entities.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_grid_indices_back_to_entities() {
        let mut r = Relevancy::new();
        r.begin();
        r.push(EntityId { index: 7, generation: 2 });
        r.push(EntityId { index: 3, generation: 0 });

        assert_eq!(r.entity(0), Some(EntityId { index: 7, generation: 2 }));
        assert_eq!(r.entity(1), Some(EntityId { index: 3, generation: 0 }));
        assert_eq!(r.entity(2), None, "an unrecorded index must not resolve");
    }

    /// The mapping is rebuilt per tick; last tick's entries must not
    /// survive into this one, or a despawned entity keeps being replicated.
    #[test]
    fn begin_clears_the_previous_tick() {
        let mut r = Relevancy::new();
        r.begin();
        r.push(EntityId { index: 1, generation: 0 });
        assert_eq!(r.len(), 1);

        r.begin();
        assert!(r.is_empty(), "a new tick starts from nothing");
        assert_eq!(r.entity(0), None);
    }

    /// A reused index is a different entity, and the mapping must say so —
    /// this is the case `World::despawn` makes reachable within one tick.
    #[test]
    fn a_reused_index_is_a_distinct_entity() {
        let old = EntityId { index: 4, generation: 1 };
        let new = EntityId { index: 4, generation: 2 };
        assert_ne!(old, new, "generation is what separates them");

        let mut r = Relevancy::new();
        r.begin();
        r.push(new);
        assert_eq!(r.entity(0), Some(new));
        assert_ne!(r.entity(0), Some(old), "the stale generation must not match");
    }
}
