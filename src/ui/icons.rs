//! Pixel-art vital-icon glyphs rendered directly into a `Batch`. Kept
//! in one module so the visual language stays consistent — every HUD
//! panel that shows a "stat with an icon" pulls from this palette
//! rather than inventing new glyphs.
//!
//! Add a variant + arm in `draw` to extend. Every recipe is composed
//! from `Batch::circle` / `Batch::triangle` / `Batch::line` primitives
//! plus a `UiRect::fill` rectangle where useful, sized by the incoming
//! `size` so the same icon reads at both small (12 px) and big (16 px)
//! HUD variants.

use glam::Vec2;
use crate::renderer::batch::Batch;
use super::{UiRect, style};

/// Every vital-icon glyph the HUD knows how to draw.
#[derive(Clone, Copy)]
pub enum VIcon {
    Heart, ShieldBlue, ShieldYellow, Gear, Lightning,
    Box, Wrench, Fuel, Jump, Moon, Food, Coin,
}

pub fn draw(batch: &mut Batch, icon: VIcon, c: Vec2, size: f32, color: [f32; 4]) {
    let s = size;
    match icon {
        VIcon::Heart => {
            let r = s * 0.22;
            batch.circle(c + Vec2::new(-s*0.18,  s*0.10), r, color, 12);
            batch.circle(c + Vec2::new( s*0.18,  s*0.10), r, color, 12);
            batch.triangle(
                c + Vec2::new(-s*0.42,  s*0.05),
                c + Vec2::new( s*0.42,  s*0.05),
                c + Vec2::new( 0.0,    -s*0.42),
                color,
            );
        }
        VIcon::ShieldBlue | VIcon::ShieldYellow => {
            let top    = c + Vec2::new(0.0,  s*0.40);
            let l_top  = c + Vec2::new(-s*0.34,  s*0.30);
            let r_top  = c + Vec2::new( s*0.34,  s*0.30);
            let l_mid  = c + Vec2::new(-s*0.34, -s*0.05);
            let r_mid  = c + Vec2::new( s*0.34, -s*0.05);
            let bot    = c + Vec2::new(0.0, -s*0.45);
            batch.triangle(top, l_top, l_mid, color);
            batch.triangle(top, l_mid, r_mid, color);
            batch.triangle(top, r_mid, r_top, color);
            batch.triangle(l_mid, r_mid, bot, color);
        }
        VIcon::Gear => {
            let r = s * 0.36;
            batch.circle(c, r, color, 14);
            let tooth = Vec2::new(s * 0.18, s * 0.18);
            UiRect::from_center(c + Vec2::new(0.0, -s*0.42), tooth).fill(batch, color);
            UiRect::from_center(c + Vec2::new(0.0,  s*0.42), tooth).fill(batch, color);
            UiRect::from_center(c + Vec2::new(-s*0.42, 0.0), tooth).fill(batch, color);
            UiRect::from_center(c + Vec2::new( s*0.42, 0.0), tooth).fill(batch, color);
            batch.circle(c, s * 0.13, style::PANEL_BG, 10);
        }
        VIcon::Lightning => {
            let p1 = c + Vec2::new(-s*0.10, -s*0.45);
            let p2 = c + Vec2::new( s*0.30, -s*0.10);
            let p3 = c + Vec2::new(-s*0.05, -s*0.05);
            let p4 = c + Vec2::new( s*0.10,  s*0.45);
            let p5 = c + Vec2::new(-s*0.30,  s*0.10);
            let p6 = c + Vec2::new( s*0.05,  s*0.05);
            batch.triangle(p1, p2, p3, color);
            batch.triangle(p4, p5, p6, color);
        }
        VIcon::Box => {
            let outer = Vec2::new(s * 0.80, s * 0.70);
            UiRect::from_center(c, outer).outline(batch, 1.5, color);
            batch.line(
                c + Vec2::new(-s*0.40,  s*0.10),
                c + Vec2::new( s*0.40,  s*0.10),
                1.5, color,
            );
            UiRect::from_center(c + Vec2::new(0.0,  s*0.10), Vec2::new(s*0.16, s*0.12)).fill(batch, color);
        }
        VIcon::Wrench => {
            batch.line(
                c + Vec2::new(-s*0.30, -s*0.30),
                c + Vec2::new( s*0.20,  s*0.20),
                2.0, color,
            );
            batch.circle(c + Vec2::new(-s*0.30, -s*0.30), s*0.18, color, 8);
            batch.circle(c + Vec2::new( s*0.25,  s*0.25), s*0.16, color, 8);
        }
        VIcon::Fuel => {
            batch.circle(c + Vec2::new(0.0, -s*0.10), s*0.28, color, 12);
            batch.triangle(
                c + Vec2::new(-s*0.22,  s*0.00),
                c + Vec2::new( s*0.22,  s*0.00),
                c + Vec2::new( 0.0,     s*0.45),
                color,
            );
        }
        VIcon::Jump => {
            let p1 = c + Vec2::new(-s*0.30, -s*0.30);
            let p2 = c + Vec2::new( s*0.30,  0.0);
            let p3 = c + Vec2::new(-s*0.30,  s*0.30);
            let p4 = c + Vec2::new(-s*0.10,  0.0);
            batch.triangle(p1, p2, p4, color);
            batch.triangle(p2, p3, p4, color);
        }
        VIcon::Moon => {
            batch.circle(c + Vec2::new(-s*0.05, 0.0), s * 0.40, color, 16);
            batch.circle(c + Vec2::new( s*0.18, -s*0.08), s * 0.32, style::PANEL_BG, 16);
        }
        VIcon::Food => {
            let bowl_r = s * 0.38;
            batch.circle(c + Vec2::new(0.0, -s*0.05), bowl_r, color, 14);
            UiRect::from_center(
                c + Vec2::new(0.0, s*0.20),
                Vec2::new(s * 0.90, s * 0.40),
            ).fill(batch, style::PANEL_BG);
            batch.line(
                c + Vec2::new(-s*0.44, -s*0.02),
                c + Vec2::new( s*0.44, -s*0.02),
                2.0, color,
            );
            batch.line(
                c + Vec2::new(-s*0.10, s*0.10),
                c + Vec2::new(-s*0.05, s*0.32),
                1.5, color,
            );
            batch.line(
                c + Vec2::new( s*0.10, s*0.10),
                c + Vec2::new( s*0.15, s*0.32),
                1.5, color,
            );
        }
        VIcon::Coin => {
            batch.circle(c, s * 0.40, color, 16);
            batch.circle(c, s * 0.28, style::PANEL_BG, 16);
            UiRect::from_center(c, Vec2::new(s * 0.14, s * 0.14)).fill(batch, color);
        }
    }
}
