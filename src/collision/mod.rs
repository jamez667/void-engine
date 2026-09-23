//! Broad-phase spatial grid + narrow-phase SAT primitives for 2D
//! collision.
//!
//! The grid ([`SpatialGrid`]) buckets colliders into fixed-size cells
//! and yields the set of possibly-overlapping index pairs. The SAT
//! helpers ([`circle_vs_circle`], [`obb_vs_obb`], [`circle_vs_obb`],
//! [`obb_axes`]) return `Some((normal, overlap))` when two shapes
//! penetrate.
//!
//! Everything here is index-based and game-agnostic: callers build a
//! flat `Vec<Collider>` of their own, insert into the grid, iterate
//! the returned pair list, and dispatch the correct SAT branch based
//! on their own shape flags. Filter logic (ship-vs-chunk pass-through,
//! per-pad occupancy, docked-lock, etc.) stays in the caller — the
//! engine gives you the pair list, not the resolution.
//!
//! The two halves are split along that seam: [`grid`] is the stateful
//! spatial index, [`narrow`] is pure geometry with no state at all.
//! Both are re-exported here, so `collision::SpatialGrid` and
//! `collision::circle_vs_circle` name the same items they always did.

pub mod grid;
pub mod narrow;
/// 3D narrowphase: sphere and oriented-box tests.
///
/// **Not** re-exported flat, unlike [`narrow`]. `obb_axes` and
/// `obb_vs_obb` exist in both halves with different signatures, so
/// flattening them here would either shadow the 2D names two shipped
/// games already call or fail to compile. Reach for these as
/// `collision::narrow3d::obb_vs_obb`.
pub mod narrow3d;

pub use grid::{AoiScratch, ColliderId, SpatialGrid};
pub use narrow::{
    circle_vs_circle, circle_vs_obb, obb_axes, obb_vs_obb, segment_vs_circle,
    segment_vs_circle_t,
};
