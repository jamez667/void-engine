//! Generic 2D-game-engine components. These are intentionally minimal —
//! anything an engine consumer might need to place, move, or collide
//! entities lives here. Domain components (Ship, Asteroid, Npc, …) belong
//! in the game crate.

use glam::{DVec2, DVec3, Quat};

// Serde derives are gated on `persist` so a single-player game that never
// saves does not compile them. `DVec2` carries serde support via glam's
// feature, enabled alongside ours.
#[cfg(feature = "persist")]
use serde::{Deserialize, Serialize};

/// World-space pose. Position is `f64` so the world can span light-hours
/// without losing precision; rotation is `f32` in radians.
///
/// The game/engine convention is that render code casts to `f32` *only
/// after* subtracting the camera position — see `void_engine::renderer`
/// notes — so `Transform2D` itself never needs to be `f32`.
#[cfg_attr(feature = "persist", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct Transform2D {
    pub pos: DVec2,
    pub rot: f32,
}

/// Linear + angular velocity, integrated by `void_engine::physics::integrate`.
#[cfg_attr(feature = "persist", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct Velocity {
    pub linear: DVec2,
    pub angular: f32,
}

/// Body-collision shape. Either a circle (legacy: `size = [0, 0]`,
/// `radius` used) or an oriented box (`size = [hw, hh]`, the entity's
/// `Transform2D.rot` provides the orientation; `radius` becomes the
/// bounding-circle for the spatial grid). The collision pair loop picks
/// circle-vs-circle, circle-vs-OBB, or OBB-vs-OBB math per pair from the
/// two shape flags.
#[cfg_attr(feature = "persist", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct Collider {
    /// Bounding-circle radius. Always populated so the spatial-grid
    /// insert (which is circle-based) works for both shapes.
    pub radius: f32,
    /// Oriented-box half-extents. `[0.0, 0.0]` means "I'm a circle, use
    /// `radius`." Anything else means "I'm a box; the collision pair
    /// loop should do box-vs-box or box-vs-circle."
    pub size: [f32; 2],
}

impl Collider {
    /// Circle collider — the classic shape every existing spawn site
    /// uses. `size` defaults to zero so the collision loop falls into
    /// the circle branch.
    pub fn circle(radius: f32) -> Self {
        Self { radius, size: [0.0, 0.0] }
    }
    /// Oriented-box collider with half-extents `(hw, hh)`. The
    /// bounding-circle radius is derived from the box corner so the
    /// spatial-grid insert stays correct.
    pub fn box2d(hw: f32, hh: f32) -> Self {
        Self {
            radius: (hw * hw + hh * hh).sqrt(),
            size: [hw, hh],
        }
    }
    /// True when this collider is a real box (non-zero half-extents).
    /// False = circle (fall back to `radius`).
    pub fn is_box(&self) -> bool {
        self.size[0] > 0.0 || self.size[1] > 0.0
    }
}

/// Short-lived particle: colour + size lerped over `lifetime`. Movement
/// comes from the sibling `Velocity` component (integrated separately).
/// `tag` lets a compact network snapshot pick a palette without shipping
/// full RGBA (0 = default spark, 1 = shield flash — extend per game).
#[cfg_attr(feature = "persist", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct Particle {
    pub lifetime: f32,
    pub max_lifetime: f32,
    pub color_start: [f32; 4],
    pub color_end: [f32; 4],
    pub size_start: f32,
    pub size_end: f32,
    pub tag: u8,
}

/// Shared physical properties for any destructible object in 2D space —
/// asteroid, salvage wreck, breakable prop, etc. Embed in a game component
/// to get common size / mass / health tracking. Mass is caller-supplied so
/// each game picks its own density model.
#[cfg_attr(feature = "persist", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct Destructible2D {
    pub radius: f32,
    pub mass: f32,
    pub health: f32,
    pub max_health: f32,
}

impl Destructible2D {
    /// Convenience constructor: `mass_for_radius` lets the caller plug in
    /// its own density function (kg per m³, spherical volume, etc.).
    pub fn new(radius: f32, mass: f32) -> Self {
        let health = radius * 3.0;
        Self { radius, mass, health, max_health: health }
    }
    pub fn health_frac(&self) -> f32 { self.health / self.max_health }
    pub fn is_dead(&self) -> bool { self.health <= 0.0 }
    pub fn take_damage(&mut self, amount: f32) {
        self.health = (self.health - amount).max(0.0);
    }
    pub fn set_health_frac(&mut self, frac: f32) {
        self.health = (self.max_health * frac).max(0.0);
    }
}

// ── 3D components ───────────────────────────────────────────────────────
//
// These sit *beside* the 2D ones, never replacing them. Two shipped games
// use `Transform2D` (void-claim alone has ~490 references and 1,284 uses
// of `DVec2`), and widening it in place would break both — as well as
// every existing save file, since the registry's schema check is a hard
// error with no migration path behind it. See `docs/3d-spec.md` §4-5.
//
// A game is 2D or 3D; nothing stops it holding both component sets, but
// the engine's own systems operate on one or the other.

/// World-space pose in three dimensions.
///
/// Position is `f64` for the same reason [`Transform2D`] uses it: a world
/// can span light-hours, and `f32` loses sub-metre precision long before
/// that. Render code casts to `f32` only after subtracting the camera —
/// see `renderer::camera::Camera3D`, which builds its view matrix
/// camera-relative for exactly this.
///
/// Rotation is a [`Quat`] rather than the scalar radians `Transform2D`
/// carries. A single angle describes every 2D orientation and no 3D one:
/// the two axes a 2D rotation leaves fixed are precisely what a 3D
/// rotation moves.
#[cfg_attr(feature = "persist", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Transform3D {
    pub pos: DVec3,
    pub rot: Quat,
}

impl Transform3D {
    /// At `pos`, unrotated.
    pub fn at(pos: DVec3) -> Self {
        Self { pos, rot: Quat::IDENTITY }
    }

    /// The model matrix for this pose, relative to `camera_pos`.
    ///
    /// Subtracts in `f64` before casting, which is the precision contract
    /// the whole 3D path follows. Hands back exactly what
    /// `renderer::mesh_store::MeshDraw` wants for `model`.
    pub fn model_matrix(&self, camera_pos: DVec3) -> glam::Mat4 {
        glam::Mat4::from_rotation_translation(
            self.rot,
            (self.pos - camera_pos).as_vec3(),
        )
    }
}

impl Default for Transform3D {
    fn default() -> Self {
        Self { pos: DVec3::ZERO, rot: Quat::IDENTITY }
    }
}

/// Linear + angular velocity in three dimensions.
///
/// `angular` is an axis-angle vector: its direction is the axis, its
/// magnitude the rate in radians per second. That is the form that
/// integrates cleanly — scaling by `dt` and converting to a quaternion is
/// exact — where storing a quaternion here would need a logarithm every
/// step.
#[cfg_attr(feature = "persist", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Velocity3D {
    pub linear: DVec3,
    pub angular: DVec3,
}

/// Body-collision shape in three dimensions.
///
/// Sphere when `half_extents` is zero (use `radius`), oriented box
/// otherwise — the same discriminator [`Collider`] uses, so the 2D and 3D
/// broadphases behave alike. `radius` is always populated because the
/// spatial grid inserts by bounding sphere for both shapes.
#[cfg_attr(feature = "persist", derive(Serialize, Deserialize))]
#[derive(Clone, Debug, PartialEq)]
pub struct Collider3D {
    /// Bounding-sphere radius. Always meaningful.
    pub radius: f32,
    /// Oriented-box half-extents. `[0, 0, 0]` means "I'm a sphere".
    pub half_extents: [f32; 3],
}

impl Collider3D {
    pub fn sphere(radius: f32) -> Self {
        Self { radius, half_extents: [0.0; 3] }
    }

    /// Oriented box. The bounding-sphere radius is derived from the box
    /// corner so the grid insert stays correct for both shapes.
    pub fn box3d(hx: f32, hy: f32, hz: f32) -> Self {
        Self {
            radius: (hx * hx + hy * hy + hz * hz).sqrt(),
            half_extents: [hx, hy, hz],
        }
    }

    pub fn is_box(&self) -> bool {
        self.half_extents.iter().any(|&h| h > 0.0)
    }
}

/// Marker tag: this entity is the local player. Camera-follow, HUD, and
/// input systems key off it. Zero-sized.
#[cfg_attr(feature = "persist", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct PlayerTag;

/// Marker tag: the camera should follow this entity. Usually attached to
/// the same entity as `PlayerTag`, but the split lets you swap the
/// follow-target (spectate, cinematic) without moving the player marker.
#[cfg_attr(feature = "persist", derive(Serialize, Deserialize))]
#[derive(Clone)]
pub struct CameraTarget;
