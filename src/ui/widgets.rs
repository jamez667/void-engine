//! Generic immediate-mode widgets: button, slider, backdrop, modal
//! shell, horizontal rule, progress bar, vertical scrollbar, filter
//! chips, sub-tab strip, labelled slider row, and `fit_scale`. All
//! colours source from `void_engine::ui::style`; the game crate's own
//! style module re-exports those constants and adds its palette on top.

use glam::Vec2;
use crate::renderer::batch::Batch;
use crate::text::{draw_text, draw_text_centered, text_width};
use super::{UiInput, UiRect, style};

const BTN_BG_HOVER: [f32; 4] = [0.10, 0.13, 0.20, 0.95];
const BTN_BORDER_HOVER: [f32; 4] = [0.70, 0.75, 0.92, 0.90];

/// Pick the largest font scale that keeps `label` inside `width` (with
/// a small padding). Caps at the supplied `preferred` scale.
pub fn fit_scale(label: &str, width: f32, preferred: f32) -> f32 {
    if label.is_empty() { return preferred; }
    let avail = (width - 12.0).max(8.0);
    let wanted = text_width(label, preferred);
    if wanted <= avail { preferred } else {
        (preferred * (avail / wanted)).max(0.8)
    }
}

pub fn button(batch: &mut Batch, ui: &mut UiInput, rect: UiRect, label: &str) -> bool {
    let hovered = ui.hovered(rect);
    let clicked = ui.clicked(rect);

    let bg     = if hovered { BTN_BG_HOVER        } else { style::PANEL_BG     };
    let border = if hovered { BTN_BORDER_HOVER     } else { style::PANEL_BORDER };
    let text   = if hovered { style::TEXT          } else { style::TEXT_DIM     };

    rect.fill(batch, bg);
    rect.outline(batch, 1.0, border);
    let scale = fit_scale(label, rect.size().x, style::FONT_HUD);
    draw_text_centered(batch, label, rect.center(), scale, text);

    clicked
}

/// Horizontal slider, value in 0..=1. Returns true if the value was
/// changed this frame. Dragging while lmb_down updates `value` to
/// wherever the mouse is along the rect. Clicking also jumps to that
/// position.
pub fn slider(batch: &mut Batch, ui: &mut UiInput, rect: UiRect, value: &mut f32) -> bool {
    let hovered = ui.hovered(rect);
    rect.fill(batch, [0.05, 0.06, 0.10, 0.85]);
    rect.outline(batch, 1.0, if hovered { BTN_BORDER_HOVER } else { style::PANEL_BORDER });

    let frac = value.clamp(0.0, 1.0);
    let fill_w = (rect.max.x - rect.min.x) * frac;
    if fill_w > 0.0 {
        let fill_rect = UiRect::from_min_size(rect.min, Vec2::new(fill_w, rect.max.y - rect.min.y));
        fill_rect.fill(batch, [0.30, 0.55, 0.90, 0.85]);
    }
    let thumb_x = rect.min.x + fill_w;
    let thumb_h = rect.max.y - rect.min.y + 6.0;
    let thumb = UiRect::from_center(
        Vec2::new(thumb_x, (rect.min.y + rect.max.y) * 0.5),
        Vec2::new(8.0, thumb_h),
    );
    thumb.fill(batch, [0.85, 0.90, 1.0, 1.0]);

    let mut changed = false;
    if hovered && ui.lmb_down() && !ui.consumed {
        let mx = ui.mouse_pos().x;
        let new_frac = ((mx - rect.min.x) / (rect.max.x - rect.min.x)).clamp(0.0, 1.0);
        if (new_frac - *value).abs() > 0.001 {
            *value = new_frac;
            changed = true;
        }
        ui.consume_if_hovered(rect);
    }
    changed
}

// ── shared modal / list primitives ───────────────────────────────────────

/// Full-screen dim rect. Every modal fades the world behind it; alpha
/// varies from 0.55 (light) to 0.94 (near-opaque, used by inventory /
/// loadout when the world underneath is distracting).
pub fn draw_backdrop(batch: &mut Batch, viewport: Vec2, alpha: f32) {
    batch.rect(Vec2::ZERO, viewport, [0.0, 0.0, 0.06, alpha]);
}

/// Standard centred modal chrome: backdrop + panel fill/outline +
/// centred title + top-right "X" close button. Returns the inner rect
/// the caller should render its content inside (title / X button
/// already excluded) and a bool that's true iff the close button was
/// clicked this frame.
pub fn draw_modal_shell(
    batch:          &mut Batch,
    ui:             &mut UiInput,
    viewport:       Vec2,
    size:           Vec2,
    title:          &str,
    backdrop_alpha: f32,
    opaque_panel:   bool,
) -> (UiRect, bool) {
    draw_backdrop(batch, viewport, backdrop_alpha);
    let panel = UiRect::from_center(Vec2::ZERO, size);
    let bg = if opaque_panel { [0.05, 0.06, 0.09, 1.0] } else { style::PANEL_BG };
    panel.fill(batch, bg);
    panel.outline(batch, 1.0, style::PANEL_BORDER);

    let title_y = panel.max.y - 28.0;
    draw_text_centered(batch, title, Vec2::new(panel.center().x, title_y),
        style::FONT_HUD * 1.2, style::TEXT);

    let close_rect = UiRect::from_min_size(
        Vec2::new(panel.max.x - 40.0, panel.max.y - 32.0),
        Vec2::new(32.0, 24.0),
    );
    let closed = button(batch, ui, close_rect, "X");

    let head_y = panel.max.y - 50.0;
    draw_hrule(batch, panel.min.x + 16.0, panel.max.x - 16.0, head_y, None);

    let inner = UiRect {
        min: Vec2::new(panel.min.x + 16.0, panel.min.y + 16.0),
        max: Vec2::new(panel.max.x - 16.0, head_y - 8.0),
    };
    (inner, closed)
}

/// Horizontal separator line spanning `min_x..max_x` at `y`. Colour
/// defaults to `style::PANEL_BORDER` when `color` is `None`.
pub fn draw_hrule(batch: &mut Batch, min_x: f32, max_x: f32, y: f32, color: Option<[f32; 4]>) {
    batch.line(Vec2::new(min_x, y), Vec2::new(max_x, y), 1.0,
               color.unwrap_or(style::PANEL_BORDER));
}

/// Horizontal progress bar: background rect + fill rect scaled by
/// `frac` (clamped to 0..=1).
pub fn draw_progress_bar(
    batch: &mut Batch,
    rect:  UiRect,
    frac:  f32,
    fg:    [f32; 4],
    bg:    [f32; 4],
) {
    rect.fill(batch, bg);
    let frac = frac.clamp(0.0, 1.0);
    if frac > 0.0 {
        let fill = UiRect::from_min_size(
            rect.min,
            Vec2::new(rect.size().x * frac, rect.size().y),
        );
        fill.fill(batch, fg);
    }
}

/// Vertical scrollbar for a scrollable list. Draws a track spanning
/// `track` and a thumb sized by `visible_frac` (visible_rows /
/// total_rows) positioned by `scroll / max_scroll`. Read-only — the
/// caller handles scroll input via `ctx.input.scroll_delta`.
pub fn draw_vscrollbar(
    batch:         &mut Batch,
    track:         UiRect,
    scroll:        f32,
    max_scroll:    f32,
    visible_frac:  f32,
) {
    track.fill(batch, style::ROW_DISABLED_BG);
    track.outline(batch, 1.0, style::ROW_DISABLED_BORDER);
    let track_h = track.size().y.max(1.0);
    let thumb_frac = visible_frac.clamp(0.10, 1.0);
    let thumb_h = (track_h * thumb_frac).max(20.0);
    let pos_frac = if max_scroll > 0.0 {
        (scroll / max_scroll).clamp(0.0, 1.0)
    } else { 0.0 };
    let thumb_top = track.max.y - pos_frac * (track_h - thumb_h);
    let thumb = UiRect::from_min_size(
        Vec2::new(track.min.x + 1.0, thumb_top - thumb_h),
        Vec2::new(track.size().x - 2.0, thumb_h),
    );
    thumb.fill(batch, [0.45, 0.55, 0.80, 0.95]);
}

/// One row of filter chips (labelled). Each chip is a small
/// active/inactive rect + centred label; clicking sets `*active` to the
/// clicked chip's id. `label` is drawn to the left of the chip row;
/// leave empty for a naked row. `origin` = top-left of the row.
pub fn draw_filter_chips<T: Copy + Eq>(
    batch:  &mut Batch,
    ui:     &mut UiInput,
    origin: Vec2,
    label:  &str,
    chips:  &[(T, &str)],
    chip_w: f32,
    active: &mut T,
) {
    const ROW_H: f32 = 24.0;
    const GAP:   f32 = 8.0;
    if !label.is_empty() {
        draw_text(batch, label,
            Vec2::new(origin.x, origin.y + 18.0),
            style::FONT_HINT, style::TEXT_DIM);
    }
    let mut x = origin.x + if label.is_empty() { 0.0 } else { 64.0 };
    for &(id, chip_label) in chips {
        let r = UiRect::from_min_size(Vec2::new(x, origin.y), Vec2::new(chip_w, ROW_H));
        let is_active = *active == id;
        let bg     = if is_active { style::TAB_ACTIVE_BG     } else { style::TAB_INACTIVE_BG     };
        let border = if is_active { style::TAB_ACTIVE_BORDER } else { style::TAB_INACTIVE_BORDER };
        r.fill(batch, bg);
        r.outline(batch, 1.0, border);
        draw_text_centered(batch, chip_label,
            Vec2::new(r.center().x, r.center().y + style::FONT_HINT * 3.0),
            style::FONT_HINT,
            if is_active { style::TEXT } else { style::TEXT_DIM });
        if ui.clicked(r) { *active = id; }
        x += chip_w + GAP;
    }
}

/// Horizontal sub-tab strip: N equal-width tabs across `rect`, active
/// tab highlighted. Returns `Some(id)` if the user clicked a non-active
/// tab.
pub fn draw_subtab_strip<T: Copy + Eq>(
    batch:  &mut Batch,
    ui:     &mut UiInput,
    rect:   UiRect,
    tabs:   &[(T, &str)],
    active: T,
    gap:    f32,
) -> Option<T> {
    let count = tabs.len().max(1) as f32;
    let total_w = rect.size().x;
    let tab_w = ((total_w - gap * (count - 1.0).max(0.0)) / count).max(60.0);
    let h = rect.size().y;
    let mut clicked = None;
    for (i, &(id, label)) in tabs.iter().enumerate() {
        let x = rect.min.x + (tab_w + gap) * i as f32;
        let r = UiRect::from_min_size(Vec2::new(x, rect.min.y), Vec2::new(tab_w, h));
        let is_active = id == active;
        let bg     = if is_active { style::TAB_ACTIVE_BG     } else { style::TAB_INACTIVE_BG     };
        let border = if is_active { style::TAB_ACTIVE_BORDER } else { style::TAB_INACTIVE_BORDER };
        r.fill(batch, bg);
        r.outline(batch, 1.0, border);
        let s = fit_scale(label, r.size().x, style::FONT_HUD);
        draw_text_centered(batch, label, r.center(), s,
            if is_active { style::PANEL_HEADER } else { style::TEXT_DIM });
        if !is_active && ui.clicked(r) { clicked = Some(id); }
    }
    clicked
}

/// A row containing: label (left), horizontal `slider` (mid), qty text
/// (right of slider), optional action button (far right). `value` is
/// 0..=1 fraction; the caller converts to whatever qty range it needs.
/// Returns `(slider_changed, button_clicked)`.
pub fn draw_labelled_slider_row(
    batch:      &mut Batch,
    ui:         &mut UiInput,
    row:        UiRect,
    label:      &str,
    qty_text:   &str,
    value:      &mut f32,
    action_btn: Option<&str>,
) -> (bool, bool) {
    const SLIDER_W:  f32 = 200.0;
    const QTY_LBL_W: f32 = 80.0;
    const ACTION_W:  f32 = 130.0;
    let row_w = row.size().x;
    let row_h = row.size().y;
    let want_action = action_btn.is_some();
    let action_take = if want_action { ACTION_W + 8.0 } else { 0.0 };
    let label_w = (row_w - SLIDER_W - QTY_LBL_W - action_take - 8.0).max(60.0);

    let label_rect = UiRect::from_min_size(row.min, Vec2::new(label_w, row_h));
    let slider_rect = UiRect::from_min_size(
        Vec2::new(row.min.x + label_w + 4.0, row.min.y + row_h * 0.5 - 9.0),
        Vec2::new(SLIDER_W, 18.0),
    );
    let qty_lbl_rect = UiRect::from_min_size(
        Vec2::new(row.min.x + label_w + SLIDER_W + 8.0, row.min.y),
        Vec2::new(QTY_LBL_W, row_h),
    );

    label_rect.fill(batch, style::TAB_INACTIVE_BG);
    label_rect.outline(batch, 1.0, style::TAB_INACTIVE_BORDER);
    let s = fit_scale(label, label_rect.size().x, style::FONT_HUD);
    draw_text_centered(batch, label, label_rect.center(), s, [0.75, 0.85, 0.95, 1.0]);

    let changed = slider(batch, ui, slider_rect, value);

    draw_text_centered(batch, qty_text, qty_lbl_rect.center(),
        style::FONT_HUD, [0.85, 0.95, 1.0, 1.0]);

    let clicked = if let Some(btn) = action_btn {
        let action_rect = UiRect::from_min_size(
            Vec2::new(row.min.x + row_w - ACTION_W, row.min.y),
            Vec2::new(ACTION_W, row_h),
        );
        button(batch, ui, action_rect, btn)
    } else { false };

    (changed, clicked)
}
