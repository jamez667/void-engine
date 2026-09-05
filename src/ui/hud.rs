//! Small HUD chrome helpers — panel fill/outline, icon+bar row, and
//! anchored bullet list. These are the reusable pieces every vitals /
//! hotbar / notification panel is built from. Game-specific vitals-panel
//! layouts (ship stats, character sheet, etc.) live in the game crate;
//! only the primitive row + panel + list shapes are engine.

use glam::Vec2;
use crate::renderer::batch::Batch;
use crate::text::{draw_text, draw_text_centered};
use super::{UiRect, style};
use super::icons::{self, VIcon};

pub const WARN_THRESHOLD: f32 = 0.25;
pub const FLASH_HZ:       f32 = 4.0;
pub const HOVER_BORDER:   [f32; 4] = [0.65, 0.70, 0.85, 0.80];

/// Row-layout dimensions shared by ship-vitals and character-vitals
/// panels — they must match visually because the two panels swap in
/// and out of the same HUD slot.
pub struct RowMetrics {
    pub small_bar:  Vec2,
    pub big_bar:    Vec2,
    pub small_icon: f32,
    pub big_icon:   f32,
    pub small_row:  f32,
    pub big_row:    f32,
    pub small_val:  f32,
    pub big_val:    f32,
    pub row_gap:    f32,
    pub sep_h:      f32,
}

impl RowMetrics {
    pub const fn default() -> Self {
        Self {
            small_bar:  Vec2::new(220.0, 14.0),
            big_bar:    Vec2::new(240.0, 22.0),
            small_icon: 22.0,
            big_icon:   30.0,
            small_row:  22.0,
            big_row:    32.0,
            small_val:  1.2,
            big_val:    1.6,
            row_gap:    4.0,
            sep_h:      6.0,
        }
    }

    pub fn row_height(&self, big: bool) -> f32 {
        if big { self.big_row } else { self.small_row }
    }
}

pub fn panel(batch: &mut Batch, rect: UiRect, hovered: bool) {
    panel_bg(batch, rect, hovered, style::PANEL_BG);
}

pub fn panel_bg(batch: &mut Batch, rect: UiRect, hovered: bool, bg: [f32; 4]) {
    rect.fill(batch, bg);
    let border = if hovered { HOVER_BORDER } else { style::PANEL_BORDER };
    rect.outline(batch, 1.0, border);
}

/// One horizontal bar: icon on the left, filled progress bar on the
/// right with a centered value string overlaid.
#[allow(clippy::too_many_arguments)]
pub fn draw_icon_row(
    batch: &mut Batch,
    x_min: f32,
    cy: f32,
    icon_w: f32,
    bar_size: Vec2,
    value_scale: f32,
    icon: VIcon,
    frac: f32,
    value_text: &str,
    bg: [f32; 4],
    fill: [f32; 4],
    icon_color: [f32; 4],
    bar_border: [f32; 4],
    small: bool,
) {
    let icon_size = if small { 12.0 } else { 16.0 };
    let icon_center = Vec2::new(x_min + icon_w * 0.5, cy);
    icons::draw(batch, icon, icon_center, icon_size, icon_color);

    let bar_min_x = x_min + icon_w;
    let bar_center = Vec2::new(bar_min_x + bar_size.x * 0.5, cy);
    UiRect::from_center(bar_center, bar_size).fill(batch, bg);
    if frac > 0.0 {
        let fill_w = bar_size.x * frac;
        let fill_center = Vec2::new(bar_min_x + fill_w * 0.5, cy);
        UiRect::from_center(fill_center, Vec2::new(fill_w, bar_size.y)).fill(batch, fill);
    }
    UiRect::from_center(bar_center, bar_size).outline(batch, 1.0, bar_border);

    draw_text_centered(batch, value_text, bar_center, value_scale, style::TEXT);
}

/// Small header + bullet list anchored to the left edge of the screen.
/// Used by nearby-entities panels — same visual language, same colour,
/// same offsets. `rows` beyond `max_rows` collapse to a "+N more" line.
pub fn draw_left_edge_list(
    batch:    &mut Batch,
    viewport: Vec2,
    base_y:   f32,
    header:   &str,
    rows:     &[String],
    max_rows: usize,
) {
    if rows.is_empty() { return; }

    const HEADER_C: [f32; 4] = [0.70, 0.90, 1.00, 1.0];
    const ROW_C:    [f32; 4] = [0.70, 0.90, 1.00, 0.85];
    const MORE_C:   [f32; 4] = [0.70, 0.90, 1.00, 0.65];
    const LINE_H:   f32 = 14.0;

    let header_x = -viewport.x * 0.5 + 16.0;
    let row_x    = -viewport.x * 0.5 + 24.0;
    draw_text(batch, header, Vec2::new(header_x, base_y), style::FONT_HUD, HEADER_C);

    let shown = rows.iter().take(max_rows);
    for (i, r) in shown.enumerate() {
        let y = base_y - 16.0 - (i as f32) * LINE_H;
        draw_text(batch, r, Vec2::new(row_x, y), style::FONT_HUD, ROW_C);
    }
    if rows.len() > max_rows {
        let y = base_y - 16.0 - (max_rows as f32) * LINE_H;
        let overflow = format!("- +{} more", rows.len() - max_rows);
        draw_text(batch, &overflow, Vec2::new(row_x, y), style::FONT_HUD, MORE_C);
    }
}
