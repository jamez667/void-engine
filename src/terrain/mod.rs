//! Building blocks for procedural terrain.
//!
//! This module deliberately does *not* generate a world. There is no `World` or
//! `Terrain` type here to configure, because the interesting part of a
//! generator — where the mountains go, what shape the coast is, which of that
//! is ocean — is the part that makes one game's world look like itself, and it
//! belongs in that game.
//!
//! What is shared is everything underneath: seeded noise that samples the same
//! way every run, the ramps and distance functions a heightfield is assembled
//! from, and rivers, which are worth having in common because a river couples
//! to the land in both directions — it must run downhill, and the ground around
//! it must be carved into a valley for that to look right.
//!
//! A generator is then a function from a point to a height, written by the
//! game, calling into these:
//!
//! ```
//! use glam::Vec2;
//! use void_engine::terrain::{field, noise};
//!
//! // An island: land in the middle, sea at the edges, with a wobbly coast.
//! let seed = 0xC0FFEE;
//! let radius = 10_000.0;
//! let elevation = |p: Vec2| {
//!     let coast = radius * (0.8 + 0.1 * noise::fbm(p * 0.00004, 4, seed));
//!     let mask = field::radial_mask(p.length(), coast * 0.85, coast);
//!     let relief = noise::ridged(p * 0.0003, 5, seed) * 900.0;
//!     -600.0 + mask * (600.0 + relief)
//! };
//!
//! assert!(elevation(Vec2::ZERO) > 0.0, "the middle is land");
//! assert!(elevation(Vec2::new(radius, 0.0)) < 0.0, "the edge is sea");
//! ```
//!
//! Rivers are then traced down that field with [`river::trace`], and the field
//! is carved around them with [`field::carve_valley`].

pub mod field;
pub mod noise;
pub mod river;

pub use field::{apply_delta, carve_valley, hillshade, point_seg_dist, point_seg_dist2, slope, smoothstep};
pub use noise::{fbm, ridged, value_noise, warped};
pub use river::{catmull_rom, River, RiverNetwork, TraceParams};
