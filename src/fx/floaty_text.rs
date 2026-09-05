//! Rising floaty text. Each entry is a world-space anchor + a string
//! that fades and drifts upward over its lifetime. The game feeds new
//! entries in (damage numbers, XP pops, salvage credits, …) and calls
//! `tick` + `draw` each frame.

use glam::{DVec2, Vec2};
use crate::renderer::batch::Batch;
use crate::renderer::camera::world_to_screen_offset;
use crate::text::draw_text_centered;

pub struct FloatyText {
    pub spawn_world_pos: DVec2,
    pub age:             f32,
    pub text:            String,
    pub color:           [f32; 4],
}

const LIFETIME:        f32 = 1.0;
const RISE_PX_PER_SEC: f32 = 60.0;
const SPAWN_OFFSET_Y:  f32 = 25.0;
const TEXT_SCALE:      f32 = 1.8;

/// Advance each entry's `age` and drop any that have exceeded `LIFETIME`.
pub fn tick(numbers: &mut Vec<FloatyText>, dt: f32) {
    for n in numbers.iter_mut() { n.age += dt; }
    numbers.retain(|n| n.age < LIFETIME);
}

/// Draw all live entries into `batch`. `camera_pos` is the world-space
/// camera; `render_scale` converts world units to screen pixels
/// (typically `camera.zoom`).
pub fn draw(batch: &mut Batch, camera_pos: DVec2, render_scale: f32, numbers: &[FloatyText]) {
    for dn in numbers {
        let t = (dn.age / LIFETIME).clamp(0.0, 1.0);
        let alpha = 1.0 - t;
        let rise = dn.age * RISE_PX_PER_SEC + SPAWN_OFFSET_Y;
        let world_screen = world_to_screen_offset(dn.spawn_world_pos, camera_pos, render_scale);
        let pos = world_screen + Vec2::new(0.0, rise);
        let mut color = dn.color;
        color[3] *= alpha;
        draw_text_centered(batch, &dn.text, pos, TEXT_SCALE, color);
    }
}
