//! Capturing a `World` and putting it back exactly as it was.
//!
//! # What "exactly" means
//!
//! Not "an equivalent world" — the *same* world. Entity ids, generations
//! and the free list are reproduced verbatim, because a saved id that
//! renumbers on load is a dangling reference in every other record that
//! mentioned it. Restoring by replaying `spawn()` would do exactly that,
//! so the allocator is captured as data instead.
//!
//! Component columns are stored with their holes intact for the same
//! reason: a column compacted on save puts every later component at the
//! wrong slot on load.
//!
//! # What is not here
//!
//! No file IO — that is phase 2's atomic-write checkpoint. A [`Snapshot`]
//! is bytes in memory; where they go is the caller's business.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ecs::World;
use crate::rng::Pcg32;

use super::registry::{Persist, Registry};

/// Bumped when the *envelope* changes shape — the header, the column
/// framing, the ordering. Per-component schema evolution is versioned
/// separately, per entry in the registry.
pub const FORMAT_VERSION: u32 = 1;

/// A captured world, ready to encode.
#[derive(Serialize, Deserialize)]
pub struct Snapshot {
    pub format_version: u32,
    /// Simulation tick this was taken at. A checkpoint may not claim a
    /// tick ahead of what durable storage has acknowledged; phase 3 is
    /// where that matters, but recording it starts now so old snapshots
    /// are not useless when it does.
    pub tick: u64,
    /// Named RNG streams, captured as `(state, inc)` positions rather
    /// than seeds. A seed replays from the beginning of the stream; a
    /// position resumes where the simulation actually was.
    pub rng: Vec<(String, u64, u64)>,
    generations: Vec<u32>,
    alive: Vec<bool>,
    free_list: Vec<u32>,
    /// One entry per persisted component type, keyed by its registry
    /// *name* — never a `TypeId` or a type path. See `registry`.
    columns: Vec<Column>,
}

#[derive(Serialize, Deserialize)]
struct Column {
    name: String,
    /// The component's schema version at save time, so the load path can
    /// migrate forward.
    version: u32,
    bytes: Vec<u8>,
}

/// Why a snapshot could not be taken or restored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnapshotError {
    /// The envelope is from a newer engine. Refused deliberately: loading
    /// it would mean silently discarding fields this build cannot see,
    /// and a downgraded server writing that back is data loss.
    FutureFormat { found: u32, supported: u32 },
    /// A component in the snapshot is not registered in this build, and
    /// no alias covers it.
    UnknownComponent(String),
    /// A registered component's schema is older or newer than the
    /// snapshot's and no migration bridges the gap.
    SchemaMismatch { name: String, found: u32, expected: u32 },
    /// The allocator arrays contradict each other.
    Corrupt(String),
    /// bincode failed.
    Codec(String),
}

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SnapshotError::FutureFormat { found, supported } => write!(
                f, "snapshot format {found} is newer than supported {supported}"),
            SnapshotError::UnknownComponent(n) => write!(
                f, "snapshot names component {n:?}, which is not registered"),
            SnapshotError::SchemaMismatch { name, found, expected } => write!(
                f, "component {name:?} schema {found} != expected {expected}"),
            SnapshotError::Corrupt(why) => write!(f, "snapshot is inconsistent: {why}"),
            SnapshotError::Codec(e) => write!(f, "codec: {e}"),
        }
    }
}

impl std::error::Error for SnapshotError {}

/// Named RNG streams to capture alongside the world.
///
/// A simulation typically has several — worldgen, loot, AI — and each
/// must resume at its own position. Keyed by name so adding a stream does
/// not invalidate existing snapshots.
#[derive(Default)]
pub struct RngStreams<'a> {
    streams: Vec<(&'a str, &'a Pcg32)>,
}

impl<'a> RngStreams<'a> {
    pub fn new() -> Self {
        Self { streams: Vec::new() }
    }

    pub fn add(mut self, name: &'a str, rng: &'a Pcg32) -> Self {
        self.streams.push((name, rng));
        self
    }
}

/// Capture `world` into a [`Snapshot`].
///
/// Walks the registry rather than the world's storages: a component the
/// world holds but nobody registered is silently *not* saved, which is
/// precisely the bug [`Registry::audit`] exists to catch at startup.
/// Calling `audit` first is what makes this safe.
///
/// [`Registry::audit`]: super::registry::Registry::audit
pub fn capture(
    world: &World,
    registry: &Registry,
    tick: u64,
    rng: &RngStreams<'_>,
) -> Result<Snapshot, SnapshotError> {
    let (generations, alive, free_list) = world.allocator_state();

    let mut columns = Vec::new();
    for entry in registry.entries() {
        if !entry.persist.in_checkpoint() {
            continue;
        }
        let Some(codec) = entry.codec else { continue };
        // A registered component the world happens to hold none of is not
        // an error — it simply has no column yet.
        let Some(column) = world.column_for(entry.type_id) else { continue };
        let bytes = (codec.encode)(column).map_err(SnapshotError::Codec)?;
        columns.push(Column { name: entry.name.to_string(), version: entry.version, bytes });
    }

    Ok(Snapshot {
        format_version: FORMAT_VERSION,
        tick,
        rng: rng.streams.iter()
            .map(|(n, r)| { let (s, i) = r.state(); (n.to_string(), s, i) })
            .collect(),
        generations: generations.to_vec(),
        alive: alive.to_vec(),
        free_list: free_list.to_vec(),
        columns,
    })
}

/// Rebuild a world from a snapshot.
///
/// The world is replaced wholesale, not merged into: a restore is "this
/// is the state now", and leaving stale entities behind would resurrect
/// things the snapshot says are gone.
pub fn restore(snapshot: &Snapshot, registry: &Registry) -> Result<World, SnapshotError> {
    if snapshot.format_version > FORMAT_VERSION {
        return Err(SnapshotError::FutureFormat {
            found: snapshot.format_version,
            supported: FORMAT_VERSION,
        });
    }

    let mut world = World::new();
    world
        .restore_allocator(
            snapshot.generations.clone(),
            snapshot.alive.clone(),
            snapshot.free_list.clone(),
        )
        .map_err(SnapshotError::Corrupt)?;

    for column in &snapshot.columns {
        // Resolves through `rename` aliases, so a component saved under an
        // old name still loads.
        let entry = registry
            .by_name(&column.name)
            .ok_or_else(|| SnapshotError::UnknownComponent(column.name.clone()))?;

        if entry.version != column.version {
            return Err(SnapshotError::SchemaMismatch {
                name: column.name.clone(),
                found: column.version,
                expected: entry.version,
            });
        }

        let Some(codec) = entry.codec else {
            return Err(SnapshotError::UnknownComponent(column.name.clone()));
        };
        let decoded = (codec.decode)(&column.bytes).map_err(SnapshotError::Codec)?;
        world
            .install_column(entry.type_id, decoded, codec.make_storage)
            .map_err(SnapshotError::Corrupt)?;
    }

    Ok(world)
}

/// Restore the named RNG streams captured by [`capture`].
///
/// Returned by name so a caller can put each back where it belongs;
/// streams the caller does not recognise are ignored rather than
/// rejected, so adding one is backwards compatible.
pub fn restore_rng(snapshot: &Snapshot) -> HashMap<String, Pcg32> {
    snapshot
        .rng
        .iter()
        .map(|(n, s, i)| (n.clone(), Pcg32::from_state((*s, *i))))
        .collect()
}

/// Encode a snapshot to bytes.
pub fn to_bytes(snapshot: &Snapshot) -> Result<Vec<u8>, SnapshotError> {
    bincode::serialize(snapshot).map_err(|e| SnapshotError::Codec(e.to_string()))
}

/// Decode a snapshot from bytes.
pub fn from_bytes(bytes: &[u8]) -> Result<Snapshot, SnapshotError> {
    bincode::deserialize(bytes).map_err(|e| SnapshotError::Codec(e.to_string()))
}

/// Components the engine itself defines, registered under stable names.
///
/// A game calls this, then adds its own. The names are engine API: they
/// are what every existing save file already says, so they may not change
/// without a `rename`.
pub fn register_engine_components(registry: &mut Registry) -> Result<(), super::RegistryError> {
    use crate::components::*;
    registry.register::<Transform2D>("transform2d", Persist::Volatile)?;
    registry.register::<Velocity>("velocity", Persist::Volatile)?;
    registry.register::<Collider>("collider", Persist::Volatile)?;
    registry.register::<Destructible2D>("destructible2d", Persist::Volatile)?;
    // Particles are cosmetic and short-lived; restoring mid-flight sparks
    // is worse than letting them lapse.
    registry.register_transient::<Particle>("particle")?;
    // Client-side markers: which entity the camera follows is a property
    // of a session, not of the world.
    registry.register_transient::<PlayerTag>("player_tag")?;
    registry.register_transient::<CameraTarget>("camera_target")?;
    Ok(())
}
