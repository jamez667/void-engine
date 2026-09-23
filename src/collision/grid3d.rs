//! Uniform-grid broadphase in three dimensions.
//!
//! The 3D counterpart to [`crate::collision::grid`], and the piece of Phase 5 that is
//! genuinely *algorithmically* different rather than one axis wider. The
//! 2D grid keys cells on `(partition, cx, cy)` and walks a doubly-nested
//! loop; this keys on `(partition, cx, cy, cz)` and walks a triply-nested
//! one. Everything downstream of that — the slot/generation bookkeeping,
//! the free list, [`crate::collision::grid::AoiScratch`] — is dimension-free and is
//! reused rather than duplicated.
//!
//! # Sizing it
//!
//! The 2D advice holds, but the constant does not. Aim for a mean bucket
//! occupancy in the low single digits; with a third axis, a cell that held
//! `n` colliders in 2D holds roughly `n^(3/2)` for the same cell size and
//! density, so a grid ported straight across will be coarser than it looks.
//! A cell a little *larger* than the typical collider diameter is the
//! usual starting point.
//!
//! # What this deliberately does not do
//!
//! No sweep, no persistent pair cache, no contact islands. Like the 2D
//! half it hands back a candidate pair list and leaves filtering and
//! resolution to the caller — see [`crate::collision::narrow3d`] for the geometry.

use std::collections::HashMap;

use glam::DVec3;

use super::grid::AoiScratch;

/// Stable handle to a collider slot, carrying a generation so a stale id
/// fails closed rather than addressing the slot's next occupant.
///
/// The same rule — and the same failure it guards — as
/// [`super::grid::ColliderId`] and [`crate::EntityId`].
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct ColliderId3D {
    pub index: u32,
    pub generation: u32,
}

/// Uniform spatial hash over 3D space.
pub struct SpatialGrid3D {
    cell_size: f64,
    /// `(partition, cx, cy, cz) -> collider slots`. The extra axis versus
    /// the 2D key is the whole structural difference.
    cells: HashMap<(u32, i32, i32, i32), Vec<u32>>,
    /// Per-slot (centre, bounding-sphere radius).
    bounds: Vec<(DVec3, f64)>,
    parts: Vec<u32>,
    alive: Vec<bool>,
    gens: Vec<u32>,
    free: Vec<u32>,
    live: usize,
}

impl SpatialGrid3D {
    /// Build an empty grid with the given cell size, in world units.
    pub fn new(cell_size: f64) -> Self {
        Self {
            // A non-positive cell size would make `cell` divide by zero
            // and hash every collider into one bucket, silently turning
            // the broadphase into an O(n^2) scan.
            cell_size: if cell_size > 0.0 { cell_size } else { 1.0 },
            cells: HashMap::new(),
            bounds: Vec::new(),
            parts: Vec::new(),
            alive: Vec::new(),
            gens: Vec::new(),
            free: Vec::new(),
            live: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    pub fn slot_count(&self) -> usize {
        self.bounds.len()
    }

    pub fn occupied_cells(&self) -> usize {
        self.cells.len()
    }

    /// Drop every collider, keeping the allocated buckets.
    ///
    /// Buckets are emptied in place rather than removed, matching the 2D
    /// grid: a world that refills to a similar shape reuses the same
    /// allocations.
    pub fn clear(&mut self) {
        for bucket in self.cells.values_mut() {
            bucket.clear();
        }
        self.bounds.clear();
        self.parts.clear();
        self.alive.clear();
        self.gens.clear();
        self.free.clear();
        self.live = 0;
    }

    /// Insert a collider by bounding sphere. Returns its slot index.
    pub fn insert(&mut self, pos: DVec3, rad: f64) -> u32 {
        self.insert_partitioned(pos, rad, 0)
    }

    /// Insert into a named partition. Partitions never see each other in
    /// queries, which is how a game keeps (say) separate floors or sectors
    /// from colliding across the gap.
    pub fn insert_partitioned(&mut self, pos: DVec3, rad: f64, partition: u32) -> u32 {
        let idx = if let Some(i) = self.free.pop() {
            self.bounds[i as usize] = (pos, rad);
            self.parts[i as usize] = partition;
            self.alive[i as usize] = true;
            i
        } else {
            self.bounds.push((pos, rad));
            self.parts.push(partition);
            self.alive.push(true);
            self.gens.push(0);
            (self.bounds.len() - 1) as u32
        };
        self.live += 1;
        self.hook(idx);
        idx
    }

    /// Insert and return a generation-checked handle.
    pub fn insert_tracked(&mut self, pos: DVec3, rad: f64, partition: u32) -> ColliderId3D {
        let index = self.insert_partitioned(pos, rad, partition);
        ColliderId3D { index, generation: self.gens[index as usize] }
    }

    /// Whether a handle still names a live collider.
    pub fn contains(&self, id: ColliderId3D) -> bool {
        self.alive.get(id.index as usize).copied().unwrap_or(false)
            && self.gens.get(id.index as usize) == Some(&id.generation)
    }

    /// Remove a collider by slot index.
    ///
    /// As in 2D, finish consuming any query output that named this slot
    /// before removing it: a bare index will happily address the slot's
    /// next occupant.
    pub fn remove(&mut self, idx: u32) {
        let i = idx as usize;
        if i >= self.alive.len() || !self.alive[i] {
            return;
        }
        self.unhook(idx);
        self.alive[i] = false;
        // Bump on removal so any handle taken before this stops resolving.
        self.gens[i] = self.gens[i].wrapping_add(1);
        self.free.push(idx);
        self.live -= 1;
    }

    /// Move a collider, rehashing it into its new cells.
    pub fn update(&mut self, idx: u32, pos: DVec3, rad: f64) {
        let i = idx as usize;
        if i >= self.alive.len() || !self.alive[i] {
            return;
        }
        self.unhook(idx);
        self.bounds[i] = (pos, rad);
        self.hook(idx);
    }

    /// The cell a point falls in.
    #[inline]
    fn cell(&self, pos: DVec3) -> (i32, i32, i32) {
        (
            (pos.x / self.cell_size).floor() as i32,
            (pos.y / self.cell_size).floor() as i32,
            (pos.z / self.cell_size).floor() as i32,
        )
    }

    /// The inclusive cell range a bounding sphere spans.
    #[inline]
    fn span(&self, pos: DVec3, rad: f64) -> ((i32, i32, i32), (i32, i32, i32)) {
        (self.cell(pos - DVec3::splat(rad)), self.cell(pos + DVec3::splat(rad)))
    }

    fn hook(&mut self, idx: u32) {
        let (pos, rad) = self.bounds[idx as usize];
        let part = self.parts[idx as usize];
        let ((x0, y0, z0), (x1, y1, z1)) = self.span(pos, rad);
        for cz in z0..=z1 {
            for cy in y0..=y1 {
                for cx in x0..=x1 {
                    self.cells.entry((part, cx, cy, cz)).or_default().push(idx);
                }
            }
        }
    }

    fn unhook(&mut self, idx: u32) {
        let (pos, rad) = self.bounds[idx as usize];
        let part = self.parts[idx as usize];
        let ((x0, y0, z0), (x1, y1, z1)) = self.span(pos, rad);
        for cz in z0..=z1 {
            for cy in y0..=y1 {
                for cx in x0..=x1 {
                    if let Some(bucket) = self.cells.get_mut(&(part, cx, cy, cz)) {
                        if let Some(p) = bucket.iter().position(|&v| v == idx) {
                            bucket.swap_remove(p);
                        }
                    }
                }
            }
        }
    }

    /// Candidate overlapping pairs, as slot indices.
    ///
    /// Sorted before returning, for the reason the 2D version documents:
    /// `HashMap` iteration order depends on `RandomState`'s per-process
    /// seed, so an unsorted list differs between a server and a client
    /// replaying the same tick.
    pub fn query_pairs(&self) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for ((part, cx, cy, cz), bucket) in &self.cells {
            for (i, &a) in bucket.iter().enumerate() {
                for &b in &bucket[i + 1..] {
                    // A pair spanning several shared cells would be
                    // reported once per cell. Report it only from the
                    // lowest cell both occupy, which is the one
                    // containing the minimum corner of their overlap.
                    let (pa, ra) = self.bounds[a as usize];
                    let (pb, rb) = self.bounds[b as usize];
                    let reach = ra + rb;
                    let d = pa - pb;
                    if d.x.abs() > reach || d.y.abs() > reach || d.z.abs() > reach {
                        continue;
                    }
                    if d.length_squared() > reach * reach {
                        continue;
                    }
                    let ((ax0, ay0, az0), _) = self.span(pa, ra);
                    let ((bx0, by0, bz0), _) = self.span(pb, rb);
                    let owner = (
                        *part,
                        ax0.max(bx0),
                        ay0.max(by0),
                        az0.max(bz0),
                    );
                    if owner != (*part, *cx, *cy, *cz) {
                        continue;
                    }
                    out.push(if a < b { (a, b) } else { (b, a) });
                }
            }
        }
        out.sort_unstable();
        out
    }

    /// Slots whose bounding sphere centre lies within `radius` of
    /// `center`, written into `scratch`.
    ///
    /// The hot path for per-client area-of-interest, and the 3D analogue
    /// of `SpatialGrid::query_circle_into` — including reusing its
    /// [`AoiScratch`], which is dimension-free.
    pub fn query_sphere_into(
        &self,
        center: DVec3,
        radius: f64,
        partition: u32,
        scratch: &mut AoiScratch,
    ) {
        scratch.begin(self.slot_count());
        let r2 = radius * radius;
        let ((x0, y0, z0), (x1, y1, z1)) = self.span(center, radius);
        for cz in z0..=z1 {
            for cy in y0..=y1 {
                for cx in x0..=x1 {
                    let Some(bucket) = self.cells.get(&(partition, cx, cy, cz)) else {
                        continue;
                    };
                    for &idx in bucket {
                        // A collider spanning several cells appears in
                        // each, so the visit is deduplicated by stamp
                        // rather than by hashing.
                        if !scratch.first_visit(idx) {
                            continue;
                        }
                        if self.bounds[idx as usize].0.distance_squared(center) <= r2 {
                            scratch.hits.push(idx);
                        }
                    }
                }
            }
        }
        scratch.hits.sort_unstable();
    }

    /// Allocating convenience wrapper over [`Self::query_sphere_into`].
    pub fn query_sphere(&self, center: DVec3, radius: f64) -> Vec<u32> {
        let mut scratch = AoiScratch::default();
        self.query_sphere_into(center, radius, 0, &mut scratch);
        std::mem::take(&mut scratch.hits)
    }

    /// The bounding sphere of a live slot.
    pub fn bounds_of(&self, idx: u32) -> Option<(DVec3, f64)> {
        if self.alive.get(idx as usize).copied().unwrap_or(false) {
            Some(self.bounds[idx as usize])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_colliders_are_reported_once() {
        let mut g = SpatialGrid3D::new(10.0);
        g.insert(DVec3::ZERO, 1.0);
        g.insert(DVec3::new(1.0, 0.0, 0.0), 1.0);
        assert_eq!(g.query_pairs(), vec![(0, 1)]);
    }

    /// A pair whose spheres span several shared cells must still be
    /// reported once. Without the owner-cell rule it comes back once per
    /// shared cell, and the caller resolves the same contact repeatedly —
    /// which reads as objects being flung apart.
    #[test]
    fn a_pair_spanning_many_cells_is_still_reported_once() {
        // Cell size 1, radii 3: each sphere spans a 7x7x7 block.
        let mut g = SpatialGrid3D::new(1.0);
        g.insert(DVec3::ZERO, 3.0);
        g.insert(DVec3::new(1.0, 1.0, 1.0), 3.0);
        assert_eq!(g.query_pairs(), vec![(0, 1)]);
    }

    #[test]
    fn distant_colliders_are_not_paired() {
        let mut g = SpatialGrid3D::new(10.0);
        g.insert(DVec3::ZERO, 1.0);
        g.insert(DVec3::new(50.0, 0.0, 0.0), 1.0);
        assert!(g.query_pairs().is_empty());
    }

    /// The whole point of the third axis: colliders separated only in z
    /// must land in *different cells*, not merely be rejected afterwards
    /// by the distance check.
    ///
    /// Asserting on `query_pairs` alone does not test this — the distance
    /// check rejects a z-separated pair whether or not the cell key
    /// includes z, so an earlier revision passed with `cz` hardcoded to
    /// zero. What the key actually controls is how many candidates the
    /// broadphase considers, so that is what is measured: with a correct
    /// 3D key these colliders share no bucket, and `occupied_cells` is
    /// the sum of two disjoint spans rather than one shared one.
    #[test]
    fn separation_in_z_puts_colliders_in_different_cells() {
        let mut flat = SpatialGrid3D::new(10.0);
        flat.insert(DVec3::ZERO, 1.0);
        let one_collider_cells = flat.occupied_cells();

        let mut g = SpatialGrid3D::new(10.0);
        g.insert(DVec3::ZERO, 1.0);
        g.insert(DVec3::new(0.0, 0.0, 500.0), 1.0);

        assert_eq!(
            g.occupied_cells(),
            one_collider_cells * 2,
            "two colliders 500 apart in z occupy {} cells, not the {} two \
             disjoint colliders should — they are sharing buckets, so the \
             cell key is ignoring the third axis",
            g.occupied_cells(),
            one_collider_cells * 2,
        );
        assert!(g.query_pairs().is_empty());
    }

    #[test]
    fn partitions_do_not_see_each_other() {
        let mut g = SpatialGrid3D::new(10.0);
        g.insert_partitioned(DVec3::ZERO, 1.0, 0);
        g.insert_partitioned(DVec3::new(0.5, 0.0, 0.0), 1.0, 1);
        assert!(g.query_pairs().is_empty());
    }

    #[test]
    fn a_sphere_query_finds_what_is_inside_it() {
        let mut g = SpatialGrid3D::new(10.0);
        let a = g.insert(DVec3::ZERO, 1.0);
        let b = g.insert(DVec3::new(5.0, 0.0, 0.0), 1.0);
        let _far = g.insert(DVec3::new(0.0, 0.0, 100.0), 1.0);

        let mut hits = g.query_sphere(DVec3::ZERO, 10.0);
        hits.sort_unstable();
        assert_eq!(hits, vec![a, b]);
    }

    /// A collider spanning many cells appears in each of them, so a query
    /// touching several must still return it once.
    #[test]
    fn a_query_does_not_return_the_same_collider_twice() {
        let mut g = SpatialGrid3D::new(1.0);
        g.insert(DVec3::ZERO, 5.0);
        assert_eq!(g.query_sphere(DVec3::ZERO, 5.0).len(), 1);
    }

    #[test]
    fn moving_a_collider_rehashes_it() {
        let mut g = SpatialGrid3D::new(10.0);
        let a = g.insert(DVec3::ZERO, 1.0);
        assert_eq!(g.query_sphere(DVec3::new(100.0, 0.0, 0.0), 5.0), Vec::<u32>::new());
        g.update(a, DVec3::new(100.0, 0.0, 0.0), 1.0);
        assert_eq!(g.query_sphere(DVec3::new(100.0, 0.0, 0.0), 5.0), vec![a]);
        assert!(g.query_sphere(DVec3::ZERO, 5.0).is_empty());
    }

    #[test]
    fn removing_a_collider_unhooks_it_from_every_cell() {
        let mut g = SpatialGrid3D::new(1.0);
        let a = g.insert(DVec3::ZERO, 3.0);
        g.remove(a);
        assert!(g.query_sphere(DVec3::ZERO, 5.0).is_empty());
        assert_eq!(g.len(), 0);
    }

    /// The generation rule: a handle to a removed collider must not
    /// address whatever is inserted into its slot next.
    #[test]
    fn a_stale_handle_does_not_address_the_slots_next_occupant() {
        let mut g = SpatialGrid3D::new(10.0);
        let first = g.insert_tracked(DVec3::ZERO, 1.0, 0);
        g.remove(first.index);
        let second = g.insert_tracked(DVec3::ZERO, 1.0, 0);

        assert_eq!(first.index, second.index, "the slot should be reused");
        assert!(!g.contains(first), "the old handle must not resolve");
        assert!(g.contains(second));
    }

    /// `RandomState` seeds per process, so an unsorted pair list differs
    /// between a server and a client replaying the same tick — which for
    /// a deterministic simulation is a desync.
    #[test]
    fn pairs_come_back_sorted() {
        let mut g = SpatialGrid3D::new(100.0);
        for i in 0..8 {
            g.insert(DVec3::new(i as f64 * 0.1, 0.0, 0.0), 1.0);
        }
        let pairs = g.query_pairs();
        let mut sorted = pairs.clone();
        sorted.sort_unstable();
        assert_eq!(pairs, sorted);
    }

    /// A zero or negative cell size divides by zero in `cell`, hashing
    /// every collider into one bucket and silently turning the broadphase
    /// into an O(n^2) scan.
    #[test]
    fn a_nonsense_cell_size_is_corrected_rather_than_dividing_by_zero() {
        let g = SpatialGrid3D::new(0.0);
        assert!(g.cell_size > 0.0);
        let g = SpatialGrid3D::new(-5.0);
        assert!(g.cell_size > 0.0);
    }

    #[test]
    fn clear_empties_the_grid_but_keeps_its_buckets() {
        let mut g = SpatialGrid3D::new(10.0);
        g.insert(DVec3::ZERO, 1.0);
        let cells_before = g.occupied_cells();
        g.clear();
        assert_eq!(g.len(), 0);
        assert!(g.query_pairs().is_empty());
        assert_eq!(g.occupied_cells(), cells_before, "buckets should be reused");
    }
}
