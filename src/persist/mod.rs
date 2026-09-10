//! Saving and loading a `World` — the `persist` feature.
//!
//! Tier 2 of three (see `TODO.md`, R2). Tier 1 is no persistence at all,
//! which is the default and what a puzzle game or a demo wants. Tier 3
//! adds an append-only ledger and Postgres for MMO-scale value tracking,
//! where "who created this item" has to be answerable months later.
//!
//! What is here in phase 1:
//!
//! - [`registry`] — the component name registry. Decides what a save file
//!   calls each component type, and classifies each one as ledgered,
//!   volatile or transient.
//!
//! Still to come in this tier: `World::snapshot`/`restore`, the on-disk
//! checkpoint with its atomic write, and the schema-migration chain.
//!
//! # The one rule worth repeating
//!
//! A save file commits to a component's *name*, never to its Rust type
//! path or a hand-assigned integer. Renaming a type or moving a module
//! must not change what a save file says. See [`registry`] for why the
//! alternatives all fail.

pub mod checkpoint;
/// Append-only value tracking — the `ledger` feature (tier 3).
///
/// Only exists in an MMO-scale build. A single-player game takes tier 2
/// (`persist`) and carries none of this.
#[cfg(feature = "ledger")]
pub mod ledger;
pub mod registry;
pub mod snapshot;

pub use checkpoint::{
    clear, load, save, CheckpointConfig, CheckpointError, DEFAULT_KEEP,
};
#[cfg(feature = "ledger")]
pub use ledger::{
    Account, Amount, Asset, Discrepancy, Entry as LedgerEntry, IdemKey, Ledger, LedgerError,
    Receipt, ReservationId, TransferRequest,
};
pub use registry::{Codec, DecodedColumn, NameId, Persist, Registry, RegistryError};
pub use snapshot::{
    capture, from_bytes, register_engine_components, restore, restore_rng, to_bytes,
    RngStreams, Snapshot, SnapshotError, FORMAT_VERSION,
};
