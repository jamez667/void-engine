//! The component name registry: the one place that decides what a save
//! file calls each component type.
//!
//! # Why names and not `TypeId`
//!
//! `TypeId` is an opaque 128-bit hash with no cross-build stability
//! guarantee — a recompile may change it, silently, and a save file keyed
//! by one would stop matching with no error. `type_name` is no better:
//! `core::any` says its output "is not specified", is "intended for
//! diagnostic use", and "may change between versions of the compiler", and
//! it embeds the module path, so moving a component into a submodule
//! silently changes every key.
//!
//! # Why names and not integers
//!
//! Hand-assigned numbers need a registry of which values are taken, two
//! branches both picking `1047` collide silently at merge, and a save file
//! full of bare numbers is hostile to read while debugging a bad
//! migration. A duplicate *name* fails loudly at startup instead.
//!
//! # The rule
//!
//! **Renaming a Rust type must never change what a save file says.** The
//! name is data you control, declared here; the type is free to move.
//! Once a name has been written to a save file it is a commitment, but it
//! is a commitment in this file rather than in the type definition, so
//! refactoring stays free and [`Registry::rename`] recovers a bad choice.
//!
//! # Changing a component's shape
//!
//! Renaming the Rust type is free; changing its *fields* is not, because
//! a save file full of the old shape already exists. The sequence is:
//!
//! 1. Bump the component's `version` at its registration.
//! 2. Register a [`Migration`] for that step with
//!    [`Registry::register_migration`], which takes the old bytes and
//!    returns new ones.
//!
//! Without step 2 the bump is a hard error on every existing save, by
//! design — reading old bytes as the new shape is how a save file turns
//! into corrupt state that surfaces much later. With it, the load path
//! chains steps forward from whatever version the snapshot recorded.
//!
//! Migrations run **forward only**. A save from a newer build is refused
//! rather than downgraded.
//!
//! # Cost
//!
//! A name appears once per component type in a snapshot header, never per
//! entity. Names intern to a [`NameId`] (a `u16`) at registration, so the
//! per-entity encoding — and, later, replication's per-entity wire format
//! — pays two bytes rather than a string.

use std::any::TypeId;
use std::collections::{BTreeMap, HashMap};

use serde::de::DeserializeOwned;
use serde::Serialize;

/// How a component participates in persistence.
///
/// Chosen per component at registration. The classes are deliberately
/// coarse: the interesting question is not "how do I store this" but "what
/// happens if it is lost", and there are only really three answers.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Persist {
    /// Value a player would petition support about: currency, items,
    /// structures, claims. Never stored as a mutable number — the
    /// component is a cache of an append-only ledger.
    ///
    /// Phase 1 records the classification but has no ledger to enforce it;
    /// that arrives with the `ledger` feature. Registering a component as
    /// `Ledgered` today means "a snapshot must not treat this as the
    /// source of truth", which [`Registry`] checks for.
    Ledgered,
    /// Cheap to lose and reconstructible: position, velocity, health, AI
    /// state. Written to checkpoints, restored from them, re-derived on
    /// spawn if no checkpoint exists.
    Volatile,
    /// Never serialised: particles, floaty text, client-side markers.
    Transient,
}

impl Persist {
    /// Whether a checkpoint should carry this component's data.
    ///
    /// `Ledgered` is included: a checkpoint caches it for a fast restart,
    /// but the ledger remains authoritative and reconciliation is what
    /// settles a disagreement.
    pub fn in_checkpoint(self) -> bool {
        matches!(self, Persist::Ledgered | Persist::Volatile)
    }
}

/// An interned component name. Two bytes on the wire and in a snapshot's
/// per-entity encoding, instead of a string.
///
/// Only meaningful relative to the [`Registry`] that issued it — ids are
/// assigned in registration order and are *not* stable across runs. The
/// stable thing is the name; a snapshot header maps names to the ids used
/// in its body, so a reader re-interns against its own registry.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct NameId(pub u16);

/// Whether a component is sent to clients, and to which ones.
///
/// Deliberately separate from [`Persist`], which answers a different
/// question. Durability and visibility look similar and are not: a
/// particle is [`Persist::Transient`] — never worth saving — yet is
/// precisely the sort of thing every nearby client must see, while a
/// ledgered inventory is the most durable thing in the world and must
/// reach *only* its owner. Deriving one from the other gets both of those
/// backwards.
///
/// The default is [`Replicate::Never`]: a component reaches the network
/// because someone said so, not because they forgot to say otherwise.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum Replicate {
    /// Server-side only. Never encoded into a snapshot for a client.
    #[default]
    Never,
    /// Sent to every client that can see the entity, subject to
    /// area-of-interest.
    ToAll,
    /// Sent only to the client that owns the entity. Inventory, currency,
    /// quest state — things whose contents are an advantage if leaked to
    /// anyone standing nearby.
    ToOwner,
}

impl Replicate {
    /// Whether this component reaches the wire at all.
    pub fn on_the_wire(self) -> bool {
        matches!(self, Replicate::ToAll | Replicate::ToOwner)
    }
}

/// One version step's upgrade, as a pure function over encoded bytes.
///
/// # Why bytes rather than values
///
/// A migration exists precisely because the *old* shape is no longer a
/// Rust type this build has — that is what "the fields changed" means. A
/// value-level migration would need both the old and new types in scope,
/// so every historical shape would have to be kept compiled forever.
/// Working on bytes lets a game deserialise into whatever local struct it
/// likes (usually a small `#[derive(Deserialize)]` mirror of the old
/// shape, kept beside the migration) and re-encode in the new one.
///
/// The column is a `Vec<Option<T>>`, holes intact, as `codec_for` writes
/// it — a migration that decodes to `Vec<Option<Old>>` and encodes a
/// `Vec<Option<New>>` is doing the right thing.
pub type Migration = fn(&[u8]) -> Result<Vec<u8>, String>;

/// A component's registration.
#[derive(Clone, Debug)]
pub struct Entry {
    pub name: &'static str,
    pub id: NameId,
    pub persist: Persist,
    /// Who, if anyone, this component is replicated to. See [`Replicate`].
    pub replicate: Replicate,
    /// Schema version for this component's encoding. Bump when the
    /// component's fields change shape, and register a [`Migration`] for
    /// the step; the load path then runs migrations forward from the
    /// version recorded in the snapshot.
    ///
    /// Without a migration for every step between the saved version and
    /// this one, loading that save is a hard [`SchemaMismatch`] rather
    /// than a silent misread — see [`Registry::migrate`].
    ///
    /// [`SchemaMismatch`]: crate::persist::snapshot::SnapshotError::SchemaMismatch
    pub version: u32,
    /// Byte-level upgrades, one per version step, keyed by the version
    /// they read.
    ///
    /// `migrations[&3]` takes bytes written at version 3 and returns bytes
    /// readable at version 4. See [`Registry::register_migration`] for why
    /// these work on bytes rather than on values.
    pub migrations: BTreeMap<u32, Migration>,
    /// The Rust type this name refers to *in this build*. Used to look up
    /// a registration from a generic call site; never written to a file.
    pub type_id: TypeId,
    /// Encode/decode for this component, captured monomorphised at
    /// registration.
    ///
    /// This is why registration takes the serde bounds rather than the
    /// snapshot code reflecting over storages: `bincode::serialize` is
    /// generic over `T: Serialize` and `deserialize_from` needs
    /// `DeserializeOwned`, so neither can be called through a `&dyn`
    /// boundary. Capturing a function pointer per type at `register` time
    /// is what makes a type-erased snapshot possible at all.
    ///
    /// `None` for [`Persist::Transient`], which is never encoded.
    pub codec: Option<Codec>,
}

/// A decoded component column, type-erased for the ECS to install.
///
/// Concretely a `Box<Vec<Option<T>>>` for the `T` the codec was built
/// from; the ECS downcasts it back inside `install_column`.
pub type DecodedColumn = Box<dyn std::any::Any + Send + Sync>;

/// Monomorphised encode/decode for one component type.
///
/// Both sides work in terms of `Vec<Option<T>>` — a whole storage column,
/// including the holes — so a restored world reproduces slot indices
/// exactly rather than compacting them and renumbering every entity.
#[derive(Copy, Clone)]
pub struct Codec {
    /// Encode a `&ComponentStorage<T>`'s column, given as `&dyn Any`.
    pub encode: fn(&dyn std::any::Any) -> Result<Vec<u8>, String>,
    /// Decode into a fresh boxed `Vec<Option<T>>` the ECS can install.
    pub decode: fn(&[u8]) -> Result<DecodedColumn, String>,
    /// Build an empty storage for this component type.
    ///
    /// Needed because a restore may name a component the target world has
    /// never held — a fresh process loading a save. `World` cannot
    /// construct one itself: it would have to name `T`, which is the
    /// whole thing type erasure gave up.
    pub make_storage: fn() -> DecodedColumn,
}

impl std::fmt::Debug for Codec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Codec")
    }
}

/// Build a [`Codec`] for `T`. Free function rather than a method so the
/// monomorphisation happens at the `register` call site.
fn codec_for<T: Serialize + DeserializeOwned + Send + Sync + 'static>() -> Codec {
    Codec {
        encode: |any| {
            let column = any
                .downcast_ref::<Vec<Option<T>>>()
                .ok_or_else(|| "codec applied to the wrong storage type".to_string())?;
            bincode::serialize(column).map_err(|e| e.to_string())
        },
        decode: |bytes| {
            let column: Vec<Option<T>> =
                bincode::deserialize(bytes).map_err(|e| e.to_string())?;
            Ok(Box::new(column))
        },
        make_storage: crate::ecs::world::empty_storage_of::<T>,
    }
}

/// What went wrong registering a component.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegistryError {
    /// Two components claimed the same name. Fails loudly here rather
    /// than silently corrupting whichever one loads second.
    DuplicateName(&'static str),
    /// One type registered twice, under different names.
    DuplicateType(&'static str, &'static str),
    /// A name was rejected as unusable in a save file.
    InvalidName(&'static str, &'static str),
    /// `rename` was given an old name nothing is registered under.
    UnknownRename(&'static str),
    /// More than `u16::MAX` component types.
    TooManyComponents,
    /// A migration was registered for a step at or above the component's
    /// current version, so it could never run. Rejected rather than
    /// ignored: a typo here leaves a registration that looks complete.
    UselessMigration {
        name: &'static str,
        from: u32,
        current: u32,
    },
}

/// Why a column could not be brought forward to the current schema.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MigrateError {
    /// No migration registered for a step in the chain. The save cannot
    /// be loaded, which is the honest answer — reading bytes in the wrong
    /// shape would corrupt state rather than fail.
    MissingStep {
        name: &'static str,
        from: u32,
        to: u32,
    },
    /// A migration ran and failed.
    Failed {
        name: &'static str,
        from: u32,
        detail: String,
    },
    /// The save was written by a newer build. Migrations run forward
    /// only, so there is nothing to do but refuse.
    FromTheFuture {
        name: &'static str,
        found: u32,
        supported: u32,
    },
}

impl std::fmt::Display for MigrateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MigrateError::MissingStep { name, from, to } => write!(
                f,
                "no migration registered for {name:?} from version {from} to {to}",
            ),
            MigrateError::Failed { name, from, detail } => write!(
                f,
                "migrating {name:?} from version {from} failed: {detail}",
            ),
            MigrateError::FromTheFuture { name, found, supported } => write!(
                f,
                "{name:?} was saved at version {found}, newer than this build's {supported}",
            ),
        }
    }
}

impl std::error::Error for MigrateError {}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryError::DuplicateName(n) =>
                write!(f, "component name {n:?} registered twice"),
            RegistryError::DuplicateType(a, b) =>
                write!(f, "one type registered under two names: {a:?} and {b:?}"),
            RegistryError::InvalidName(n, why) =>
                write!(f, "component name {n:?} is invalid: {why}"),
            RegistryError::UnknownRename(n) =>
                write!(f, "rename from {n:?}, which is not registered"),
            RegistryError::TooManyComponents =>
                write!(f, "more than {} component types registered", u16::MAX),
            RegistryError::UselessMigration { name, from, current } => write!(
                f,
                "migration for {name:?} from version {from} can never run: \
                 the component is already at version {current}",
            ),
        }
    }
}

impl std::error::Error for RegistryError {}

/// Maps component types to the names a save file knows them by.
///
/// Built once at startup, then read-only. A game registers every component
/// it persists; [`Registry::audit`] then catches anything present in the
/// world but never classified.
#[derive(Default)]
pub struct Registry {
    entries: Vec<Entry>,
    by_name: HashMap<&'static str, NameId>,
    by_type: HashMap<TypeId, NameId>,
    /// Old name -> current name, for snapshots written before a rename.
    aliases: HashMap<String, &'static str>,
}

impl Registry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `T` under `name`.
    ///
    /// The name is what a save file records, so it must outlive any
    /// refactoring of `T`. Prefer a short, lowercase, stable noun —
    /// `"transform2d"`, `"wallet"` — over anything that echoes the current
    /// module path, which is exactly what this design avoids depending on.
    /// Panics if `persist` is [`Persist::Transient`] — use
    /// [`register_transient`] for that, which needs no serde bounds.
    ///
    /// [`register_transient`]: Registry::register_transient
    pub fn register<T: Send + Sync + Serialize + DeserializeOwned + 'static>(
        &mut self,
        name: &'static str,
        persist: Persist,
    ) -> Result<NameId, RegistryError> {
        self.register_versioned::<T>(name, persist, 1)
    }

    /// Register a component that is never written to a save file.
    ///
    /// Separate from [`register`] so a `Transient` component — a particle,
    /// a client-side marker — does not have to implement serde merely to
    /// be classified. Registering it at all is still worthwhile: it is
    /// what stops [`audit`] flagging it as forgotten.
    ///
    /// [`register`]: Registry::register
    /// [`audit`]: Registry::audit
    pub fn register_transient<T: Send + Sync + 'static>(
        &mut self,
        name: &'static str,
    ) -> Result<NameId, RegistryError> {
        self.insert_entry(name, Persist::Transient, Replicate::Never, 1, TypeId::of::<T>(), None)
    }

    /// [`register`] with an explicit schema version.
    ///
    /// [`register`]: Registry::register
    pub fn register_versioned<T: Send + Sync + Serialize + DeserializeOwned + 'static>(
        &mut self,
        name: &'static str,
        persist: Persist,
        version: u32,
    ) -> Result<NameId, RegistryError> {
        assert!(
            persist != Persist::Transient,
            "use register_transient for {name:?}: a Transient component needs no serde bounds",
        );
        self.insert_entry(
            name, persist, Replicate::Never, version, TypeId::of::<T>(), Some(codec_for::<T>()),
        )
    }

    /// Register a component that is also sent to clients.
    ///
    /// A separate entry point rather than a parameter on [`register`] for
    /// the same reason [`register_transient`] is one: the overwhelming
    /// majority of components are server-side only, and making every call
    /// site name a `Replicate` to say "no" is noise that obscures the few
    /// that say yes.
    ///
    /// [`register`]: Registry::register
    /// [`register_transient`]: Registry::register_transient
    pub fn register_replicated<T: Send + Sync + Serialize + DeserializeOwned + 'static>(
        &mut self,
        name: &'static str,
        persist: Persist,
        replicate: Replicate,
    ) -> Result<NameId, RegistryError> {
        self.register_replicated_versioned::<T>(name, persist, replicate, 1)
    }

    /// [`register_replicated`] with an explicit schema version.
    ///
    /// [`register_replicated`]: Registry::register_replicated
    pub fn register_replicated_versioned<
        T: Send + Sync + Serialize + DeserializeOwned + 'static,
    >(
        &mut self,
        name: &'static str,
        persist: Persist,
        replicate: Replicate,
        version: u32,
    ) -> Result<NameId, RegistryError> {
        // A replicated component still needs an encoding, and `Transient`
        // is the one class that has none. The pairing is legitimate —
        // a particle is Transient and replicated — but it has to go
        // through `register_replicated_transient`, which does not demand
        // the serde bounds this path captures.
        assert!(
            persist != Persist::Transient,
            "use register_replicated_transient for {name:?}: a Transient component needs no \
             serde bounds",
        );
        self.insert_entry(
            name, persist, replicate, version, TypeId::of::<T>(), Some(codec_for::<T>()),
        )
    }

    /// Register a component that is replicated but never saved.
    ///
    /// The particle case: cosmetic, not worth a byte of disk, and still
    /// something every nearby client must see. This is the pairing that
    /// deriving replication from [`Persist`] would get wrong.
    pub fn register_replicated_transient<T: Send + Sync + 'static>(
        &mut self,
        name: &'static str,
        replicate: Replicate,
    ) -> Result<NameId, RegistryError> {
        self.insert_entry(name, Persist::Transient, replicate, 1, TypeId::of::<T>(), None)
    }

    /// The shared body of every registration path.
    fn insert_entry(
        &mut self,
        name: &'static str,
        persist: Persist,
        replicate: Replicate,
        version: u32,
        type_id: TypeId,
        codec: Option<Codec>,
    ) -> Result<NameId, RegistryError> {
        validate_name(name)?;
        if self.by_name.contains_key(name) {
            return Err(RegistryError::DuplicateName(name));
        }
        if let Some(existing) = self.by_type.get(&type_id) {
            return Err(RegistryError::DuplicateType(
                self.entries[existing.0 as usize].name,
                name,
            ));
        }
        if self.entries.len() >= u16::MAX as usize {
            return Err(RegistryError::TooManyComponents);
        }

        let id = NameId(self.entries.len() as u16);
        self.entries.push(Entry {
            name, id, persist, replicate, version, type_id, codec,
            migrations: BTreeMap::new(),
        });
        self.by_name.insert(name, id);
        self.by_type.insert(type_id, id);
        Ok(id)
    }

    /// Record that a component previously saved as `old` is now `new`.
    ///
    /// This is what makes a badly-chosen name recoverable rather than
    /// permanent: the load path resolves `old` through the alias, so
    /// existing save files keep working while new ones use `new`. Call it
    /// after registering `new`.
    pub fn rename(&mut self, old: &str, new: &'static str) -> Result<(), RegistryError> {
        if !self.by_name.contains_key(new) {
            return Err(RegistryError::UnknownRename(new));
        }
        self.aliases.insert(old.to_string(), new);
        Ok(())
    }

    /// Register the upgrade from `from_version` to `from_version + 1`.
    ///
    /// Call once per step. To take a component from version 1 to 3, a game
    /// registers the 1→2 and 2→3 steps and `migrate` chains them; there is
    /// deliberately no way to declare a 1→3 jump, because a chain of small
    /// steps is what lets a save written at *any* intermediate version
    /// load.
    ///
    /// # Errors
    ///
    /// [`RegistryError::UnknownRename`] if `name` is not registered —
    /// reusing that variant rather than adding one, since it already means
    /// "you named a component that does not exist".
    ///
    /// A step at or above the component's current version is rejected: it
    /// could never run, because `migrate` stops when it reaches the
    /// current version, and accepting it would let a typo sit silently in
    /// a registration that looks complete.
    pub fn register_migration(
        &mut self,
        name: &'static str,
        from_version: u32,
        migration: Migration,
    ) -> Result<(), RegistryError> {
        let Some(&id) = self.by_name.get(name) else {
            return Err(RegistryError::UnknownRename(name));
        };
        let entry = &mut self.entries[id.0 as usize];
        if from_version >= entry.version {
            return Err(RegistryError::UselessMigration {
                name,
                from: from_version,
                current: entry.version,
            });
        }
        entry.migrations.insert(from_version, migration);
        Ok(())
    }

    /// Walk `bytes` forward from `from` to this entry's current version.
    ///
    /// Returns the bytes unchanged when already current, so the common
    /// case costs nothing. A gap in the chain is an error rather than a
    /// skip: silently loading bytes in the wrong shape is how a save file
    /// turns into corrupt state that only surfaces much later.
    ///
    /// A version *newer* than this build's is also an error. Migrations
    /// only run forward; a save from a future build cannot be downgraded,
    /// and pretending otherwise would misread it.
    pub fn migrate(&self, entry: &Entry, from: u32, bytes: &[u8]) -> Result<Vec<u8>, MigrateError> {
        if from > entry.version {
            return Err(MigrateError::FromTheFuture {
                name: entry.name,
                found: from,
                supported: entry.version,
            });
        }
        let mut current = from;
        let mut buf = bytes.to_vec();
        while current < entry.version {
            let Some(step) = entry.migrations.get(&current) else {
                return Err(MigrateError::MissingStep {
                    name: entry.name,
                    from: current,
                    to: current + 1,
                });
            };
            buf = step(&buf).map_err(|e| MigrateError::Failed {
                name: entry.name,
                from: current,
                detail: e,
            })?;
            current += 1;
        }
        Ok(buf)
    }

    /// Look up by the name a snapshot recorded, following any alias.
    pub fn by_name(&self, name: &str) -> Option<&Entry> {
        let current = self.aliases.get(name).copied().unwrap_or(name);
        let id = self.by_name.get(current)?;
        self.entries.get(id.0 as usize)
    }

    /// Look up the registration for a Rust type in this build.
    ///
    /// Deliberately free of the serde bounds `register` carries, so a
    /// caller can ask about a `Transient` component too.
    pub fn by_type<T: Send + Sync + 'static>(&self) -> Option<&Entry> {
        let id = self.by_type.get(&TypeId::of::<T>())?;
        self.entries.get(id.0 as usize)
    }

    pub fn by_id(&self, id: NameId) -> Option<&Entry> {
        self.entries.get(id.0 as usize)
    }

    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Fail if any component type present in `world` was never registered.
    ///
    /// The failure this prevents: adding a component, forgetting to
    /// classify it, and discovering after launch that it was silently
    /// absent from every save. Call once at startup, after the world is
    /// populated and before the first checkpoint.
    ///
    /// Returns the offending `type_name`s — diagnostic only, which is the
    /// one job `type_name` is actually fit for.
    pub fn audit(&self, present: &[(TypeId, &'static str)]) -> Result<(), Vec<&'static str>> {
        let missing: Vec<&'static str> = present
            .iter()
            .filter(|(tid, _)| !self.by_type.contains_key(tid))
            .map(|(_, name)| *name)
            .collect();
        if missing.is_empty() { Ok(()) } else { Err(missing) }
    }
}

/// Names go into files and error messages, so keep them boring.
///
/// Rejecting the empty string and whitespace is not pedantry: a name is a
/// map key in a snapshot header, and one that round-trips badly through a
/// hand edit or a diff is a silent load failure later.
fn validate_name(name: &'static str) -> Result<(), RegistryError> {
    if name.is_empty() {
        return Err(RegistryError::InvalidName(name, "must not be empty"));
    }
    if name.len() > 64 {
        return Err(RegistryError::InvalidName(name, "must be 64 bytes or fewer"));
    }
    if !name.is_ascii() {
        return Err(RegistryError::InvalidName(name, "must be ASCII"));
    }
    if name.chars().any(|c| c.is_whitespace()) {
        return Err(RegistryError::InvalidName(name, "must not contain whitespace"));
    }
    if name.chars().any(|c| c.is_ascii_uppercase()) {
        return Err(RegistryError::InvalidName(
            name,
            "must be lowercase, so a case-insensitive filesystem cannot merge two names",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod replicate_tests {
    use super::*;

    #[derive(Clone, Serialize, serde::Deserialize)]
    struct Pos(f32);
    #[derive(Clone, Serialize, serde::Deserialize)]
    struct Purse(u64);
    struct Spark;

    /// Nothing reaches the network by accident.
    #[test]
    fn registration_defaults_to_never_replicated() {
        let mut r = Registry::new();
        r.register::<Pos>("pos", Persist::Volatile).unwrap();
        r.register_transient::<Spark>("spark").unwrap();

        assert_eq!(r.by_name("pos").unwrap().replicate, Replicate::Never);
        assert_eq!(r.by_name("spark").unwrap().replicate, Replicate::Never);
        assert!(!Replicate::Never.on_the_wire());
    }

    /// The pairing that deriving replication from `Persist` would get
    /// wrong: never saved, always sent.
    #[test]
    fn a_transient_component_can_still_be_replicated() {
        let mut r = Registry::new();
        r.register_replicated_transient::<Spark>("spark", Replicate::ToAll).unwrap();

        let e = r.by_name("spark").unwrap();
        assert_eq!(e.persist, Persist::Transient, "a particle is not worth saving");
        assert_eq!(e.replicate, Replicate::ToAll, "and is still worth sending");
        assert!(!e.persist.in_checkpoint(), "durability and visibility are independent");
        assert!(e.replicate.on_the_wire());
    }

    /// And the mirror case: the most durable thing in the world, visible
    /// only to its owner.
    #[test]
    fn a_durable_component_can_be_owner_only() {
        let mut r = Registry::new();
        r.register_replicated::<Purse>("purse", Persist::Ledgered, Replicate::ToOwner).unwrap();

        let e = r.by_name("purse").unwrap();
        assert!(e.persist.in_checkpoint());
        assert_eq!(e.replicate, Replicate::ToOwner);
        assert!(e.replicate.on_the_wire(), "owner-only still reaches the wire");
    }

    /// A replicated component keeps its codec: replication needs an
    /// encoding just as much as a checkpoint does.
    #[test]
    fn a_replicated_component_carries_a_codec() {
        let mut r = Registry::new();
        r.register_replicated::<Pos>("pos", Persist::Volatile, Replicate::ToAll).unwrap();
        assert!(r.by_name("pos").unwrap().codec.is_some());

        // ...except the transient path, which has none to give.
        r.register_replicated_transient::<Spark>("spark", Replicate::ToAll).unwrap();
        assert!(r.by_name("spark").unwrap().codec.is_none());
    }

    /// Ids stay registration-ordered regardless of which entry point was
    /// used — replication interning depends on it.
    #[test]
    fn replicated_registrations_share_the_id_space() {
        let mut r = Registry::new();
        let a = r.register::<Pos>("pos", Persist::Volatile).unwrap();
        let b = r.register_replicated_transient::<Spark>("spark", Replicate::ToAll).unwrap();
        let c = r.register_replicated::<Purse>("purse", Persist::Ledgered, Replicate::ToOwner)
            .unwrap();

        assert_eq!((a, b, c), (NameId(0), NameId(1), NameId(2)));
    }

    /// Duplicate detection does not care which path registered first.
    #[test]
    fn a_replicated_registration_still_rejects_a_duplicate_name() {
        let mut r = Registry::new();
        r.register::<Pos>("thing", Persist::Volatile).unwrap();
        let err = r
            .register_replicated::<Purse>("thing", Persist::Volatile, Replicate::ToAll)
            .unwrap_err();
        assert!(matches!(err, RegistryError::DuplicateName("thing")));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use serde::{Deserialize, Serialize};

    #[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
    struct Transform(f64, f64);
    #[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
    struct Vel(f32);
    #[derive(Clone, Serialize, Deserialize, PartialEq, Debug)]
    struct Wallet;
    /// Deliberately *not* serde-capable: a Transient component must be
    /// registrable without implementing serde at all.
    #[derive(Clone)]
    struct Particle;

    #[test]
    fn register_and_look_up_by_name_and_type() {
        let mut r = Registry::new();
        let id = r.register::<Transform>("transform2d", Persist::Volatile).unwrap();
        assert_eq!(r.by_name("transform2d").unwrap().id, id);
        assert_eq!(r.by_type::<Transform>().unwrap().name, "transform2d");
        assert_eq!(r.by_id(id).unwrap().persist, Persist::Volatile);
    }

    #[test]
    fn ids_are_assigned_in_registration_order() {
        let mut r = Registry::new();
        let a = r.register::<Transform>("transform2d", Persist::Volatile).unwrap();
        let b = r.register::<Vel>("velocity", Persist::Volatile).unwrap();
        assert_eq!((a, b), (NameId(0), NameId(1)));
    }

    /// The collision this design exists to make loud. Two branches both
    /// picking a name must fail at startup, not corrupt whichever loads
    /// second.
    #[test]
    fn a_duplicate_name_is_rejected() {
        let mut r = Registry::new();
        r.register::<Transform>("thing", Persist::Volatile).unwrap();
        assert_eq!(
            r.register::<Vel>("thing", Persist::Volatile),
            Err(RegistryError::DuplicateName("thing")),
        );
    }

    #[test]
    fn one_type_under_two_names_is_rejected() {
        let mut r = Registry::new();
        r.register::<Transform>("transform2d", Persist::Volatile).unwrap();
        assert!(matches!(
            r.register::<Transform>("transform_v2", Persist::Volatile),
            Err(RegistryError::DuplicateType("transform2d", "transform_v2")),
        ));
    }

    /// The point of names over integers: the type may move or be renamed,
    /// and the save file is unaffected.
    #[test]
    fn renaming_the_rust_type_does_not_change_the_saved_name() {
        // `Transform` here stands in for a type that was later renamed;
        // the registration is what a save file sees.
        let mut r = Registry::new();
        r.register::<Transform>("transform2d", Persist::Volatile).unwrap();
        assert_eq!(r.by_type::<Transform>().unwrap().name, "transform2d");
    }

    #[test]
    fn an_alias_resolves_an_old_saved_name() {
        let mut r = Registry::new();
        r.register::<Transform>("transform2d", Persist::Volatile).unwrap();
        r.rename("xform", "transform2d").unwrap();
        assert_eq!(r.by_name("xform").unwrap().name, "transform2d",
            "a snapshot written under the old name must still load");
    }

    #[test]
    fn renaming_to_an_unregistered_name_is_rejected() {
        let mut r = Registry::new();
        assert_eq!(r.rename("old", "nope"), Err(RegistryError::UnknownRename("nope")));
    }

    #[test]
    fn unknown_names_do_not_resolve() {
        let r = Registry::new();
        assert!(r.by_name("nothing").is_none());
    }

    #[test]
    fn invalid_names_are_rejected() {
        let mut r = Registry::new();
        assert!(matches!(r.register::<Transform>("", Persist::Volatile),
            Err(RegistryError::InvalidName("", _))));
        assert!(matches!(r.register::<Vel>("has space", Persist::Volatile),
            Err(RegistryError::InvalidName("has space", _))));
        assert!(matches!(r.register::<Wallet>("MixedCase", Persist::Volatile),
            Err(RegistryError::InvalidName("MixedCase", _))));
    }

    #[test]
    fn transient_components_stay_out_of_checkpoints() {
        assert!(!Persist::Transient.in_checkpoint());
        assert!(Persist::Volatile.in_checkpoint());
        assert!(Persist::Ledgered.in_checkpoint());
    }

    /// Adding a component and forgetting to classify it must fail at
    /// startup, not silently vanish from every save.
    #[test]
    fn the_audit_catches_an_unregistered_component() {
        let mut r = Registry::new();
        r.register::<Transform>("transform2d", Persist::Volatile).unwrap();

        let present = [
            (TypeId::of::<Transform>(), "Transform"),
            (TypeId::of::<Wallet>(), "Wallet"),
        ];
        assert_eq!(r.audit(&present), Err(vec!["Wallet"]));

        let only_registered = [(TypeId::of::<Transform>(), "Transform")];
        assert_eq!(r.audit(&only_registered), Ok(()));
    }

    #[test]
    fn schema_version_defaults_to_one_and_can_be_set() {
        let mut r = Registry::new();
        r.register::<Transform>("transform2d", Persist::Volatile).unwrap();
        r.register_versioned::<Vel>("velocity", Persist::Volatile, 3).unwrap();
        assert_eq!(r.by_name("transform2d").unwrap().version, 1);
        assert_eq!(r.by_name("velocity").unwrap().version, 3);
    }

    /// A Transient component needs no serde bounds — that split is the
    /// whole point of `register_transient`. `Particle` above implements
    /// neither Serialize nor Deserialize; this would not compile if the
    /// bound-free path were removed.
    #[test]
    fn a_transient_component_registers_without_serde() {
        let mut r = Registry::new();
        let id = r.register_transient::<Particle>("particle").unwrap();
        let e = r.by_id(id).unwrap();
        assert_eq!(e.persist, Persist::Transient);
        assert!(e.codec.is_none(), "a Transient component must carry no codec");
        assert!(!e.persist.in_checkpoint());
    }

    /// Registering a Transient through the serde path is a programming
    /// error, not a silent no-op: it would attach a codec to something
    /// that must never be encoded.
    #[test]
    #[should_panic(expected = "use register_transient")]
    fn registering_transient_through_the_serde_path_panics() {
        let mut r = Registry::new();
        let _ = r.register::<Transform>("transform2d", Persist::Transient);
    }

    /// The codec is what makes a type-erased snapshot possible, so prove
    /// it round-trips a whole storage column — holes included, since slot
    /// indices must survive exactly.
    #[test]
    fn a_codec_round_trips_a_column_with_holes() {
        let mut r = Registry::new();
        let id = r.register::<Transform>("transform2d", Persist::Volatile).unwrap();
        let codec = r.by_id(id).unwrap().codec.expect("Volatile must carry a codec");

        let column: Vec<Option<Transform>> =
            vec![Some(Transform(1.0, 2.0)), None, None, Some(Transform(9.0, 9.5))];
        let bytes = (codec.encode)(&column as &dyn std::any::Any).unwrap();
        let back = (codec.decode)(&bytes).unwrap();
        let back = back.downcast_ref::<Vec<Option<Transform>>>().unwrap();

        assert_eq!(&column, back, "column must survive encode/decode unchanged");
        assert_eq!(back.len(), 4, "holes must be preserved, not compacted");
    }

    /// Applying a codec to the wrong column is an error, not a panic or a
    /// silent misread.
    #[test]
    fn a_codec_rejects_the_wrong_storage_type() {
        let mut r = Registry::new();
        let id = r.register::<Transform>("transform2d", Persist::Volatile).unwrap();
        let codec = r.by_id(id).unwrap().codec.unwrap();
        let wrong: Vec<Option<Vel>> = vec![Some(Vel(1.0))];
        assert!((codec.encode)(&wrong as &dyn std::any::Any).is_err());
    }

    /// A Transient component still participates in the audit — that is why
    /// registering it is worthwhile even though it is never saved.
    #[test]
    fn a_registered_transient_passes_the_audit() {
        let mut r = Registry::new();
        r.register_transient::<Particle>("particle").unwrap();
        let present = [(TypeId::of::<Particle>(), "Particle")];
        assert_eq!(r.audit(&present), Ok(()));
    }
}
