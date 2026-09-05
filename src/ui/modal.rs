//! Generic list-modal primitive: a centred panel with a title, N
//! enabled/disabled rows the user picks from, and an "Esc to cancel"
//! footer. Every "press E and pick something" popup in a 2D game reuses
//! this shape — bar, elevator, shop, pause menu, spawn-picker, …

use glam::Vec2;
use crate::renderer::batch::Batch;
use crate::text::draw_text_centered;
use super::{UiInput, UiRect, style, widgets};

/// One row inside a walkup modal (bar, elevator, shop, pause menu …).
pub struct WalkupRow<Id> {
    pub id:      Id,
    pub text:    String,
    pub enabled: bool,
}

/// Enabled-or-disabled action row. Enabled rows delegate to the standard
/// `button`; disabled rows draw the "greyed" chrome and eat clicks.
pub fn draw_action_row(
    batch:   &mut Batch,
    ui:      &mut UiInput,
    rect:    UiRect,
    text:    &str,
    enabled: bool,
) -> bool {
    if enabled {
        widgets::button(batch, ui, rect, text)
    } else {
        rect.fill(batch, style::ROW_DISABLED_BG);
        rect.outline(batch, 1.0, style::ROW_DISABLED_BORDER);
        let scale = widgets::fit_scale(text, rect.size().x, style::FONT_HUD);
        draw_text_centered(batch, text, rect.center(), scale, style::TEXT_DISABLED);
        false
    }
}

/// Centred walk-up modal used by every "press E and pick something"
/// popup. Reusable on ship or station. Returns the picked row id, if any.
pub fn draw_walkup_modal<Id: Copy>(
    batch: &mut Batch,
    ui:    &mut UiInput,
    title: &str,
    rows:  &[WalkupRow<Id>],
) -> Option<Id> {
    const PANEL_W: f32 = 380.0;
    const ROW_H:   f32 = 46.0;
    const PAD:     f32 = 12.0;
    let panel_h = PAD * 2.0 + 36.0 + (rows.len() as f32) * (ROW_H + 6.0) + 22.0;
    let panel = UiRect::from_center(Vec2::ZERO, Vec2::new(PANEL_W, panel_h));
    panel.fill(batch, style::PANEL_BG);
    panel.outline(batch, 2.0, style::PANEL_BORDER);

    let mut cursor_y = panel.max.y - PAD - 20.0;
    draw_text_centered(batch, title, Vec2::new(0.0, cursor_y),
                       2.0, style::PANEL_HEADER);
    cursor_y -= 32.0;

    let mut clicked: Option<Id> = None;
    for row in rows.iter() {
        let row_rect = UiRect::from_min_size(
            Vec2::new(panel.min.x + PAD, cursor_y - ROW_H),
            Vec2::new(PANEL_W - PAD * 2.0, ROW_H),
        );
        if draw_action_row(batch, ui, row_rect, &row.text, row.enabled) {
            clicked = Some(row.id);
        }
        cursor_y -= ROW_H + 6.0;
    }

    draw_text_centered(batch, "Esc to cancel",
                       Vec2::new(0.0, panel.min.y + 14.0),
                       1.0, style::PANEL_FOOTER);
    clicked
}
