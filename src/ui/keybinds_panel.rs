//! Collapsible bottom-right key-hint panel. Collapsed: a single dim
//! "F1: keybinds" hint. Expanded: a two-column-ish panel of
//! key→action lines. Games hand in the current line set as a
//! `&[&str]` (view-mode dispatch is a game concern), the shortcut
//! label (defaults to "F1"), and a bool for whether the panel is open.

use glam::Vec2;
use crate::renderer::batch::Batch;
use crate::text::{draw_text, text_width};
use super::{Anchor, UiInput, UiRect, style};

pub fn draw_keybinds_panel(
    batch: &mut Batch,
    viewport: Vec2,
    ui: &mut UiInput,
    lines: &[&str],
    open: bool,
    toggle_label: &str,
) {
    let scale = style::FONT_HINT;
    if !open {
        // Single-line dim hint, no panel chrome.
        let hint = format!("{}: keybinds", toggle_label);
        let tw = text_width(&hint, scale);
        let br = Anchor::BottomRight.inset(viewport, style::PAD);
        draw_text(batch, &hint, Vec2::new(br.x - tw, br.y + 3.0 * scale), scale, style::TEXT_DIM);
        return;
    }

    let max_w = lines.iter().map(|l| text_width(l, scale)).fold(0.0_f32, f32::max);
    let row_h = 14.0;

    let panel_w = style::PAD_INNER * 2.0 + max_w;
    let panel_h = style::PAD_INNER * 2.0 + row_h * lines.len() as f32;

    let br = Anchor::BottomRight.inset(viewport, style::PAD);
    let rect = UiRect::from_min_size(
        Vec2::new(br.x - panel_w, br.y),
        Vec2::new(panel_w, panel_h),
    );
    let hovered = ui.consume_if_hovered(rect);
    rect.fill(batch, style::HUD_BG);
    let border = if hovered {
        [0.65, 0.70, 0.85, 0.80]
    } else {
        style::PANEL_BORDER
    };
    rect.outline(batch, 1.0, border);

    let inner = rect.inset(style::PAD_INNER);
    for (i, line) in lines.iter().enumerate() {
        let cy = inner.max.y - row_h * 0.5 - row_h * i as f32;
        draw_text(batch, line, Vec2::new(inner.min.x, cy + 3.0 * scale), scale, style::TEXT_DIM);
    }
}
