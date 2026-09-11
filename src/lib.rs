//! A 2D game engine: ECS, fixed-timestep loop, wgpu renderer, netcode
//! primitives.
//!
//! # Client and headless builds
//!
//! The `client` feature (on by default) carries everything that needs a
//! window or a GPU: the renderer, the immediate-mode UI, `fx`, bitmap text,
//! and the winit event loop behind `run`. A dedicated server turns it off:
//!
//! ```toml
//! void_engine = { version = "0.1", default-features = false }
//! ```
//!
//! and drives the simulation with `run_headless` instead. What survives is
//! the whole simulation surface — [`World`], `collision`, `pathfind`,
//! `terrain`, `physics`, `time`, `rng`, `sector`, `tilegrid` and (with the
//! `net` feature) `net`. Game logic written against [`App`] compiles into
//! both; only `ClientApp` needs the GPU.

/// A read-only operator status page over HTTP — the `admin` feature.
///
/// Surfaces the ledger audits, the reservation backlog and the loop's own
/// tick health, none of which had a caller before it existed. Binds
/// nothing unless a game calls `admin::serve`.
#[cfg(feature = "admin")]
pub mod admin;
pub mod app;
/// The headless fixed-tick driver. Available in every build.
pub mod app_headless;
#[cfg(feature = "audio")]
pub mod audio;
pub mod collision;
pub mod components;
pub mod ecs;
/// Visual effects drawn into a `Batch` — client-only.
#[cfg(feature = "client")]
pub mod fx;
pub mod input;
pub mod log;
pub mod math;
#[cfg(feature = "net")]
pub mod net;
pub mod pathfind;
pub mod perf;
/// Saving and loading a `World` — component name registry, snapshots,
/// checkpoints. Off by default; a game that never saves carries none of it.
#[cfg(feature = "persist")]
pub mod persist;
pub mod physics;
pub mod render_math;
/// wgpu renderer — client-only.
#[cfg(feature = "client")]
pub mod renderer;
pub mod rng;
pub mod sector;
/// Bitmap text, drawn into a `Batch` — client-only.
#[cfg(feature = "client")]
pub mod text;
pub mod terrain;
pub mod tile_collide;
pub mod tilegrid;
pub mod time;
/// Immediate-mode UI widgets, drawn into a `Batch` — client-only.
#[cfg(feature = "client")]
pub mod ui;
pub mod util;
pub mod walk;
pub mod world;

pub use app::{App, SimCtx};
pub use app_headless::{Exit, HeadlessConfig, TickHealth, run_headless, run_headless_with};
#[cfg(feature = "client")]
pub use app::{ClientApp, run};
pub use ecs::{EntityId, World};
pub use input::InputState;
pub use perf::PerfSnapshot;
#[cfg(feature = "client")]
pub use renderer::Renderer;
pub use math::*;
