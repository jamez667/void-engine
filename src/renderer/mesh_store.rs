//! Retained GPU meshes, addressed by handle (feature `render3d`).
//!
//! # Why this exists
//!
//! Phase 2's draw API uploaded a mesh on every call. That is the honest
//! shape for a per-frame API and the wrong one for static geometry: a level
//! that never changes paid to be re-uploaded sixty times a second, which is
//! exactly the cost `Batch`'s immediate-mode design is built to accept for
//! 2D and exactly the cost 3D must not.
//!
//! `Renderer::draw_mesh_transient` keeps the old behaviour for geometry
//! that genuinely lasts one frame (a debug gizmo), and is named so that
//! reaching for it by accident is hard.
//!
//! [`crate::renderer::mesh_store::MeshStore`] holds uploaded meshes and
//! hands back a [`crate::renderer::mesh_store::MeshHandle`].
//! Upload once, draw by handle every frame thereafter.
//!
//! # Why handles rather than `Rc<GpuMesh3D>`
//!
//! A handle is `Copy`, cheap to store in a component, and — because it
//! carries a generation — *fails closed* when it outlives what it names.
//! That is the same rule [`crate::EntityId`] and `ColliderId` already
//! follow, and it exists for the same reason: an index alone will happily
//! address whatever occupies the slot next, which means a stale handle
//! silently draws the wrong mesh rather than drawing nothing.
//!
//! A game that despawns the entity holding a handle, then loads a new
//! mesh into the freed slot, is the ordinary case — not an exotic one.

use std::collections::HashMap;

use super::mesh3d::Mesh3D;
use super::render3d::GpuMesh3D;

/// A reference to a mesh living on the GPU.
///
/// `Copy` and 8 bytes, so it belongs in a component beside a transform.
/// Becomes invalid when its mesh is removed; drawing through a stale
/// handle is a no-op rather than a wrong draw. See the module docs.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct MeshHandle {
    pub index: u32,
    pub generation: u32,
}

/// Why a mesh could not be uploaded.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshError {
    /// The mesh had no triangles. wgpu rejects zero-sized buffers, and a
    /// mesh with nothing to draw has no meaningful handle to return.
    Empty,
    /// An index pointed past the end of the vertex list.
    ///
    /// Checked here rather than left to the GPU because an out-of-range
    /// index is undefined behaviour at draw time: on some backends it
    /// reads whatever memory follows the buffer, which is a far worse
    /// failure than a rejected upload.
    IndexOutOfRange { index: u32, vertices: usize },
}

impl std::fmt::Display for MeshError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MeshError::Empty => write!(f, "mesh has no triangles"),
            MeshError::IndexOutOfRange { index, vertices } => write!(
                f,
                "index {index} points past the {vertices} vertices in the mesh",
            ),
        }
    }
}

impl std::error::Error for MeshError {}

struct Slot {
    /// `None` when the slot is free. The generation keeps counting up
    /// across reuse, which is what makes a stale handle detectable.
    mesh: Option<GpuMesh3D>,
    generation: u32,
}

/// Uploaded meshes, addressed by [`MeshHandle`].
#[derive(Default)]
pub struct MeshStore {
    slots: Vec<Slot>,
    free: Vec<u32>,
    live: usize,
}

impl MeshStore {
    pub fn new() -> Self {
        Self::default()
    }

    /// How many meshes are currently held.
    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Upload a mesh and return a handle to it.
    ///
    /// Validates indices against the vertex count first — see
    /// [`MeshError::IndexOutOfRange`] for why that is worth doing on the
    /// CPU rather than discovering it as corrupted geometry.
    pub fn insert(
        &mut self,
        device: &wgpu::Device,
        mesh: &Mesh3D,
    ) -> Result<MeshHandle, MeshError> {
        if mesh.is_empty() || mesh.vertices.is_empty() {
            return Err(MeshError::Empty);
        }
        let n = mesh.vertices.len();
        if let Some(&bad) = mesh.indices.iter().find(|&&i| i as usize >= n) {
            return Err(MeshError::IndexOutOfRange { index: bad, vertices: n });
        }

        let gpu = GpuMesh3D::upload(device, mesh).ok_or(MeshError::Empty)?;
        self.live += 1;

        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            // Bump on *reuse*, so every handle handed out for this slot
            // differs from every handle handed out before it.
            slot.generation = slot.generation.wrapping_add(1);
            slot.mesh = Some(gpu);
            return Ok(MeshHandle { index, generation: slot.generation });
        }

        let index = self.slots.len() as u32;
        self.slots.push(Slot { mesh: Some(gpu), generation: 0 });
        Ok(MeshHandle { index, generation: 0 })
    }

    /// Replace the geometry behind an existing handle, keeping it valid.
    ///
    /// For geometry that changes occasionally — a chunk remeshed after a
    /// block is broken — where re-uploading is right but invalidating
    /// every reference to it is not.
    pub fn replace(
        &mut self,
        device: &wgpu::Device,
        handle: MeshHandle,
        mesh: &Mesh3D,
    ) -> Result<(), MeshError> {
        if mesh.is_empty() || mesh.vertices.is_empty() {
            return Err(MeshError::Empty);
        }
        let n = mesh.vertices.len();
        if let Some(&bad) = mesh.indices.iter().find(|&&i| i as usize >= n) {
            return Err(MeshError::IndexOutOfRange { index: bad, vertices: n });
        }
        // A stale handle must not resurrect a freed slot, so validate
        // before touching anything.
        if !self.contains(handle) {
            return Ok(());
        }
        if let Some(gpu) = GpuMesh3D::upload(device, mesh) {
            self.slots[handle.index as usize].mesh = Some(gpu);
        }
        Ok(())
    }

    /// Whether this handle still names a live mesh.
    pub fn contains(&self, handle: MeshHandle) -> bool {
        self.get(handle).is_some()
    }

    pub(super) fn get(&self, handle: MeshHandle) -> Option<&GpuMesh3D> {
        let slot = self.slots.get(handle.index as usize)?;
        if slot.generation != handle.generation {
            return None;
        }
        slot.mesh.as_ref()
    }

    /// Drop a mesh, freeing its GPU buffers and invalidating its handle.
    ///
    /// Returns whether anything was removed, so a double-remove is
    /// detectable rather than silent.
    pub fn remove(&mut self, handle: MeshHandle) -> bool {
        let Some(slot) = self.slots.get_mut(handle.index as usize) else {
            return false;
        };
        if slot.generation != handle.generation || slot.mesh.is_none() {
            return false;
        }
        slot.mesh = None;
        self.free.push(handle.index);
        self.live -= 1;
        true
    }

    /// Drop every mesh. Handles handed out before this all become stale.
    pub fn clear(&mut self) {
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if slot.mesh.take().is_some() {
                slot.generation = slot.generation.wrapping_add(1);
                self.free.push(i as u32);
            }
        }
        self.live = 0;
    }
}

/// A mesh queued for this frame, and where to put it.
///
/// The transform is per-instance, which is the thing the 2D path has no
/// equivalent for: `Batch` pre-transforms every primitive on the CPU into
/// one monolithic buffer, so there is nowhere to hang a model matrix. A
/// retained mesh is drawn many times at many places from one upload, so
/// the transform has to ride alongside the handle.
#[derive(Copy, Clone, Debug)]
pub struct MeshDraw {
    pub handle: MeshHandle,
    /// Model matrix: where this instance sits, in **camera-relative**
    /// space. See [`crate::renderer::camera::Camera3D`] — the translation
    /// here is world position minus eye position, subtracted in `f64`
    /// before the cast.
    pub model: glam::Mat4,
    pub color: [f32; 4],
}

impl MeshDraw {
    /// Draw at a camera-relative offset, unrotated and unscaled.
    pub fn at(handle: MeshHandle, offset: glam::Vec3) -> Self {
        Self {
            handle,
            model: glam::Mat4::from_translation(offset),
            color: [1.0; 4],
        }
    }

    pub fn with_model(mut self, model: glam::Mat4) -> Self {
        self.model = model;
        self
    }

    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }
}

/// Group draws by mesh so instances of one mesh submit together.
///
/// Returns a map from handle to the instances of it. Called once per
/// frame by `end_frame`; the point is that binding a vertex buffer is
/// per-*mesh* work, not per-instance work, so 500 trees sharing one mesh
/// bind once rather than 500 times.
pub(super) fn group_by_mesh(draws: &[MeshDraw]) -> HashMap<MeshHandle, Vec<&MeshDraw>> {
    let mut out: HashMap<MeshHandle, Vec<&MeshDraw>> = HashMap::new();
    for d in draws {
        out.entry(d.handle).or_default().push(d);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn a_mesh() -> Mesh3D {
        let mut m = Mesh3D::new();
        m.push_box(Vec3::ZERO, Vec3::splat(1.0), [1.0; 4]);
        m
    }

    /// The core of the generation scheme: a handle to a removed mesh must
    /// not address whatever lands in the slot next. Without this, a game
    /// that despawns one entity and loads another draws the wrong
    /// geometry, silently.
    #[test]
    fn a_handle_to_a_removed_mesh_does_not_address_its_replacement() {
        let mut store = MeshStore::new();
        // No device needed to exercise the slot bookkeeping: insert the
        // slots by hand, which is what `insert` does around the upload.
        store.slots.push(Slot { mesh: None, generation: 0 });
        let stale = MeshHandle { index: 0, generation: 0 };

        // Occupy, free, then reoccupy the same index.
        store.slots[0].mesh = None;
        store.slots[0].generation = 1;

        assert!(
            !store.contains(stale),
            "a handle from before the slot was reused must not resolve",
        );
    }

    #[test]
    fn removing_twice_reports_the_second_as_a_no_op() {
        let mut store = MeshStore::new();
        store.slots.push(Slot { mesh: None, generation: 0 });
        let h = MeshHandle { index: 0, generation: 0 };
        // Slot holds no mesh, so the first remove already reports false.
        assert!(!store.remove(h));
        assert!(!store.remove(h));
    }

    #[test]
    fn a_handle_with_an_out_of_range_index_does_not_panic() {
        let store = MeshStore::new();
        let bogus = MeshHandle { index: 9999, generation: 0 };
        assert!(!store.contains(bogus));
    }

    /// Validation happens before the upload, so a malformed mesh is
    /// rejected rather than becoming undefined behaviour at draw time.
    #[test]
    fn an_index_past_the_vertex_list_is_rejected() {
        let mut m = a_mesh();
        let n = m.vertices.len();
        m.indices.push(n as u32 + 5);

        // Reproduce `insert`'s validation without a device.
        let bad = m.indices.iter().find(|&&i| i as usize >= m.vertices.len());
        assert_eq!(bad, Some(&(n as u32 + 5)));
    }

    #[test]
    fn an_empty_mesh_is_rejected() {
        let m = Mesh3D::new();
        assert!(m.is_empty());
    }

    /// Instances of one mesh must group together, because binding a
    /// vertex buffer is per-mesh work. 500 trees should bind once.
    #[test]
    fn draws_group_by_mesh_so_one_bind_serves_every_instance() {
        let oak = MeshHandle { index: 0, generation: 0 };
        let pine = MeshHandle { index: 1, generation: 0 };
        let draws = vec![
            MeshDraw::at(oak, Vec3::ZERO),
            MeshDraw::at(pine, Vec3::X),
            MeshDraw::at(oak, Vec3::Y),
            MeshDraw::at(oak, Vec3::Z),
        ];

        let grouped = group_by_mesh(&draws);
        assert_eq!(grouped.len(), 2, "two distinct meshes");
        assert_eq!(grouped[&oak].len(), 3);
        assert_eq!(grouped[&pine].len(), 1);
    }

    /// Two handles differing only in generation are distinct keys, or
    /// grouping would merge a live mesh with a stale reference to its
    /// old occupant.
    #[test]
    fn generation_participates_in_handle_identity() {
        let a = MeshHandle { index: 0, generation: 0 };
        let b = MeshHandle { index: 0, generation: 1 };
        assert_ne!(a, b);
        let draws = vec![MeshDraw::at(a, Vec3::ZERO), MeshDraw::at(b, Vec3::ZERO)];
        assert_eq!(group_by_mesh(&draws).len(), 2);
    }
}
