//! Generic 2D physics integrator. Semi-implicit Euler + per-tick drag.
//!
//! Drag constants are exposed as parameters because the game-friendly
//! defaults (0.999 linear, 0.995 angular) are the void_sim baseline;
//! other consumers may want cleaner conservation or heavier damping.
//! The zero-arg convenience wrappers in `void_sim::physics` supply the
//! sim's canonical values.

use crate::{EntityId, World};
use crate::components::{Transform2D, Velocity};

/// Integrate one entity for `dt`. `lin_drag` and `ang_drag` are per-tick
/// multipliers on `Velocity.linear` / `Velocity.angular` (1.0 = no drag,
/// 0.0 = instant stop).
pub fn integrate_entity(world: &mut World, id: EntityId, dt: f32, lin_drag: f64, ang_drag: f32) {
    let vel = match world.get::<Velocity>(id) { Some(v) => v.clone(), None => return };
    if let Some(t) = world.get_mut::<Transform2D>(id) {
        t.pos += vel.linear * dt as f64;
        t.rot += vel.angular * dt;
    }
    if let Some(v) = world.get_mut::<Velocity>(id) {
        v.linear *= lin_drag;
        v.angular *= ang_drag;
    }
}

/// Integrate every entity that has a `Velocity` for `dt`. Typed iteration
/// avoids the O(N_total) miss-heavy full-world walk — only real movers
/// (ships, projectiles, particles) get touched.
///
/// The collect-into-Vec dance is intentional: we drop the immutable
/// `iter::<Velocity>` borrow before taking `get_mut` twice per mover.
pub fn integrate(world: &mut World, dt: f32, lin_drag: f64, ang_drag: f32) {
    let movers: Vec<(EntityId, Velocity)> = world.iter::<Velocity>()
        .map(|(id, v)| (id, v.clone()))
        .collect();
    for (id, vel) in movers {
        if let Some(t) = world.get_mut::<Transform2D>(id) {
            t.pos += vel.linear * dt as f64;
            t.rot += vel.angular * dt;
        }
        if let Some(v) = world.get_mut::<Velocity>(id) {
            v.linear *= lin_drag;
            v.angular *= ang_drag;
        }
    }
}
