//! Radial-burst particle spawn + lifetime tick. Particles are ECS
//! entities: `Transform2D` + `Velocity` + `Particle`. Movement is
//! integrated by whoever owns the physics pass (server does it as part
//! of `physics::integrate`; the client calls `integrate` here so
//! particles still move on frames when only the local ship is
//! predicted). The engine ships no rendering — game code walks
//! `world.iter::<Particle>()` and paints them however it wants.
//!
//! Randomness is injected via a `FnMut() -> f32` returning values in
//! `[0.0, 1.0)`. Callers hand in a seedable RNG (e.g. `Lcg`) or a
//! thread-local shim.

use glam::DVec2;
use crate::{World, EntityId};
use crate::components::{Transform2D, Velocity, Particle};

/// Spawn `count` sparks radially around `pos` with alternating
/// color_a/color_b. Speeds randomised in `[speed_min, speed_min +
/// speed_jitter)`, lifetimes in `[life_min, life_min + life_jitter)`.
/// Use for impact FX, hit confirms, etc.
#[allow(clippy::too_many_arguments)]
pub fn burst(
    world: &mut World,
    rand: &mut dyn FnMut() -> f32,
    pos: DVec2,
    count: u32,
    speed_min: f64,
    speed_jitter: f64,
    life_min: f32,
    life_jitter: f32,
    color_a: [f32; 4],
    color_b: [f32; 4],
    tag: u8,
) {
    if count == 0 { return; }
    for i in 0..count {
        let angle = i as f64 * std::f64::consts::TAU / count as f64
            + rand() as f64 * 0.6;
        let speed = speed_min + rand() as f64 * speed_jitter;
        let vel   = DVec2::new(angle.cos(), angle.sin()) * speed;
        let color = if i % 2 == 0 { color_a } else { color_b };
        let life  = life_min + rand() * life_jitter;
        spawn_tagged(world, pos, vel, color, life, tag);
    }
}

pub fn spawn(world: &mut World, pos: DVec2, vel: DVec2, color: [f32; 4], lifetime: f32) {
    spawn_tagged(world, pos, vel, color, lifetime, 0);
}

pub fn spawn_tagged(world: &mut World, pos: DVec2, vel: DVec2, color: [f32; 4], lifetime: f32, tag: u8) {
    spawn_sized(world, pos, vel, color, lifetime, tag, 3.0, 0.5);
}

/// Full-control spawn with explicit start/end sizes. Used for explosion
/// "chunks" that want big fat pixels flying instead of the standard 3px
/// sparks.
pub fn spawn_sized(
    world: &mut World,
    pos: DVec2,
    vel: DVec2,
    color: [f32; 4],
    lifetime: f32,
    tag: u8,
    size_start: f32,
    size_end: f32,
) {
    // Default fade: darken toward the start hue and go fully
    // transparent. `spawn_full` when you need a different end colour.
    let color_end = [color[0] * 0.3, color[1] * 0.3, color[2] * 0.3, 0.0];
    spawn_full(world, pos, vel, color, color_end, lifetime, tag, size_start, size_end);
}

/// Most general spawn: explicit start *and* end colour on top of the
/// size curve. `spawn_sized` is this with the conventional
/// darken-and-fade end colour; effects that need to fade toward a
/// different hue (or hold their hue while fading out) call this.
#[allow(clippy::too_many_arguments)]
pub fn spawn_full(
    world: &mut World,
    pos: DVec2,
    vel: DVec2,
    color_start: [f32; 4],
    color_end: [f32; 4],
    lifetime: f32,
    tag: u8,
    size_start: f32,
    size_end: f32,
) {
    let id = world.spawn();
    world.insert(id, Transform2D { pos, rot: 0.0 });
    world.insert(id, Velocity { linear: vel, angular: 0.0 });
    world.insert(id, Particle {
        lifetime,
        max_lifetime: lifetime,
        color_start,
        color_end,
        size_start,
        size_end,
        tag,
    });
}

/// Sized variant of `burst`. Same radial-spread + speed-jitter recipe
/// but each particle uses `size_start` / `size_end` from the caller.
/// Used for explosion chunks (fat pixels tumbling out) vs the fine
/// spark trail (`burst`).
#[allow(clippy::too_many_arguments)]
pub fn burst_sized(
    world: &mut World,
    rand: &mut dyn FnMut() -> f32,
    pos: DVec2,
    count: u32,
    speed_min: f64,
    speed_jitter: f64,
    life_min: f32,
    life_jitter: f32,
    color_a: [f32; 4],
    color_b: [f32; 4],
    tag: u8,
    size_start: f32,
    size_end: f32,
) {
    if count == 0 { return; }
    for i in 0..count {
        let angle = i as f64 * std::f64::consts::TAU / count as f64
            + rand() as f64 * 0.6;
        let speed = speed_min + rand() as f64 * speed_jitter;
        let vel   = DVec2::new(angle.cos(), angle.sin()) * speed;
        let color = if i % 2 == 0 { color_a } else { color_b };
        let life  = life_min + rand() * life_jitter;
        spawn_sized(world, pos, vel, color, life, tag, size_start, size_end);
    }
}

pub fn update(world: &mut World, dt: f32) {
    // Typed iter over `Particle` — the old `world.entities().collect()`
    // walked every asteroid/chunk/ship/projectile just to `get::<Particle>`
    // on each. Collect ids first so we can borrow mut later without
    // holding the immutable iter.
    let ids: Vec<EntityId> = world.iter::<Particle>().map(|(id, _)| id).collect();
    let mut dead = Vec::new();
    for id in ids {
        if let Some(p) = world.get_mut::<Particle>(id) {
            p.lifetime -= dt;
            if p.lifetime <= 0.0 { dead.push(id); }
        }
    }
    for id in dead { world.despawn(id); }
}

/// Client-only: advance particle positions by their velocity. The server
/// gets this for free from `physics::integrate`, but the client only
/// integrates the player ship — particles need a dedicated pass.
pub fn integrate(world: &mut World, dt: f32) {
    let ids: Vec<EntityId> = world.iter::<Particle>().map(|(id, _)| id).collect();
    for id in ids {
        let vel = match world.get::<Velocity>(id) { Some(v) => v.linear, None => continue };
        if let Some(t) = world.get_mut::<Transform2D>(id) {
            t.pos += vel * dt as f64;
        }
    }
}
