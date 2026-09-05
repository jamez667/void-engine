//! Visual effects that run on top of `renderer::batch::Batch`. Each
//! submodule is a self-contained effect (starfield, floaty text, warp
//! bubble, …) with no game-crate coupling.
//!
//! Most are screen-space overlays; `bubble` is world-space and takes a
//! world-to-screen projection from the caller.

pub mod bubble;
pub mod floaty_text;
pub mod particles;
pub mod rings;
pub mod starfield;
