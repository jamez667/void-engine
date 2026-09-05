//! World-level utilities that operate on the ECS as a whole.
//!
//! Currently: origin rebasing. When the camera drifts far enough from
//! `DVec2::ZERO` that f32 render deltas would start losing sub-metre
//! precision, shift every entity's `Transform2D.pos` by the camera
//! offset and zero the camera. Consumers pass the same threshold their
//! game wants (the void_sim default = 1 Tm).

use glam::DVec2;
use crate::World;
use crate::components::Transform2D;

/// If `camera_pos` is farther than `threshold` from the origin, shift
/// every `Transform2D` by `-camera_pos` and zero `camera_pos`. Anything
/// caching absolute positions must subscribe to a rebase and re-anchor
/// itself — this function only touches the ECS.
pub fn origin_rebase(world: &mut World, camera_pos: &mut DVec2, threshold: f64) {
    if camera_pos.length() < threshold { return; }
    let offset = *camera_pos;
    for (_, t) in world.iter_mut::<Transform2D>() {
        t.pos -= offset;
    }
    *camera_pos = DVec2::ZERO;
}
