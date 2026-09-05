//! Generic 2D-game-engine components. These are intentionally minimal —
//! anything an engine consumer might need to place, move, or collide
//! entities lives here. Domain components (Ship, Asteroid, Npc, …) belong
//! in the game crate.

use glam::DVec2;

/// World-space pose. Position is `f64` so the world can span light-hours
/// without losing precision; rotation is `f32` in radians.
///
/// The game/engine convention is that render code casts to `f32` *only
/// after* subtracting the camera position — see `void_engine::renderer`
/// notes — so `Transform2D` itself never needs to be `f32`.
#[derive(Clone)]
pub struct Transform2D {
    pub pos: DVec2,
    pub rot: f32,
}

/// Linear + angular velocity, integrated by `void_engine::physics::integrate`.
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

/// Marker tag: this entity is the local player. Camera-follow, HUD, and
/// input systems key off it. Zero-sized.
pub struct PlayerTag;

/// Marker tag: the camera should follow this entity. Usually attached to
/// the same entity as `PlayerTag`, but the split lets you swap the
/// follow-target (spectate, cinematic) without moving the player marker.
pub struct CameraTarget;
