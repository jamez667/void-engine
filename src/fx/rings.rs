//! Ring-based FX overlays — pulsing charge ring, expanding drop-out
//! ring, transit tint, small chip below the compass. Every one used
//! to live in `void_claim::hyperspace_fx`; extracted here as generic
//! FX so any 2D game with a "charging + travel + arrival" sequence
//! (portal, teleport, dive, jump) can reuse them.
//!
//! The `warmup` / `decel_max` seconds are passed in — the engine
//! doesn't know about the game's hyperspace constants.

use std::f32::consts::TAU;
use glam::Vec2;
use crate::renderer::batch::Batch;
use crate::text::draw_text_centered;
use crate::ui::{Anchor, UiRect, style};

/// Pulsing charging rings during the warmup countdown. `timer` counts
/// down from `warmup_max` → 0; `target_name` is the label shown under
/// the "JUMP DRIVE CHARGING" header.
pub fn draw_warmup(
    batch: &mut Batch,
    _viewport: Vec2,
    timer: f32,
    warmup_max: f32,
    target_name: &str,
) {
    let progress = (1.0 - timer / warmup_max).clamp(0.0, 1.0);
    let ring_r   = 60.0 + progress * 140.0;
    let pulse    = (timer * TAU * 2.0).sin() * 0.3 + 0.7;

    const SEGS: u32 = 48;
    for (r, a) in [(ring_r, 0.50 * pulse), (ring_r + 8.0, 0.15_f32)] {
        for i in 0..SEGS {
            let a0 = (i     as f32 / SEGS as f32) * TAU;
            let a1 = ((i+1) as f32 / SEGS as f32) * TAU;
            batch.line(
                Vec2::new(a0.cos() * r, a0.sin() * r),
                Vec2::new(a1.cos() * r, a1.sin() * r),
                2.0, [0.25, 0.55, 1.0, a],
            );
        }
    }

    draw_text_centered(batch, "JUMP DRIVE CHARGING",
        Vec2::new(0.0, ring_r + 18.0), style::FONT_HUD, [0.4, 0.7, 1.0, 0.9]);
    draw_text_centered(batch,
        &format!("-> {}   T-{:.1}s", target_name, timer),
        Vec2::new(0.0, ring_r + 4.0), style::FONT_HINT, [0.6, 0.8, 1.0, 0.75]);
}

/// Subtle blue tint during warp transit — the actual streaks are drawn
/// by whatever starfield the game runs, this just adds the ambient
/// wash over the whole viewport.
pub fn draw_warp_streaks(batch: &mut Batch, viewport: Vec2) {
    batch.rect(Vec2::ZERO, viewport, [0.04, 0.08, 0.22, 0.20]);
}

/// Reverse-warmup ring as the ship drops out of warp. `timer` counts
/// down from `decel_max` → 0.
pub fn draw_exit_anim(batch: &mut Batch, _viewport: Vec2, timer: f32, decel_max: f32) {
    let progress = 1.0 - (timer / decel_max).clamp(0.0, 1.0);
    let ring_r = 200.0 * progress;
    let ring_a = (1.0 - progress) * 0.70;
    if ring_a > 0.0 {
        const SEGS: u32 = 48;
        for i in 0..SEGS {
            let a0 = (i     as f32 / SEGS as f32) * TAU;
            let a1 = ((i+1) as f32 / SEGS as f32) * TAU;
            batch.line(
                Vec2::new(a0.cos() * ring_r, a0.sin() * ring_r),
                Vec2::new(a1.cos() * ring_r, a1.sin() * ring_r),
                2.0, [0.40, 0.75, 1.0, ring_a],
            );
        }
    }
    draw_text_centered(batch, "DROPPING OUT OF WARP",
        Vec2::new(0.0, 50.0), style::FONT_HUD, [0.5, 0.8, 1.0, (1.0 - progress * 2.0).max(0.0)]);
}

/// Small chip near the bottom compass showing "DRIVE COOLDOWN Ns".
pub fn draw_cooldown_chip(batch: &mut Batch, viewport: Vec2, timer: f32) {
    let label   = format!("DRIVE COOLDOWN  {:.0}s", timer.ceil());
    let chip_sz = Vec2::new(200.0, 20.0);
    let center  = Anchor::Bottom.inset(viewport, style::PAD + 42.0);
    let rect    = UiRect::from_center(Vec2::new(center.x, center.y), chip_sz);
    rect.fill(batch, style::PANEL_BG);
    rect.outline(batch, 1.0, [0.3, 0.55, 1.0, 0.65]);
    draw_text_centered(batch, &label, rect.center(), style::FONT_HINT, [0.4, 0.65, 1.0, 0.9]);
}
