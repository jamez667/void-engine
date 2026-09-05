//! Debug-panel primitives: a labelled-rows overlay and a
//! hold-to-confirm popup.
//!
//! [`StatPanel`] renders a sized backdrop and a stack of monospace text
//! rows anchored at a fixed origin — the shape used by every "F3
//! diagnostics" overlay. The caller supplies the row list (one string
//! per line, optional colour override, optional section-heading blank
//! gap); the panel figures out sizing.
//!
//! [`draw_hold_confirm_popup`] renders a centred "hold X to fire"
//! dialog with a progress bar filling as `held_secs / total_secs`.
//!
//! Both helpers are pure UI — game code owns the data struct and any
//! rate limiting.

use glam::Vec2;

use crate::renderer::batch::Batch;
use crate::text::{draw_text, draw_text_centered};
use crate::ui::{style, widgets, UiRect};

/// One row of a [`StatPanel`]. `text` is drawn as-is; `color` overrides
/// the default text colour when `Some`; `blank_before` inserts a
/// vertical gap above this row (used for section headings).
pub struct StatRow<'a> {
    pub text:         &'a str,
    pub color:        Option<[f32; 4]>,
    pub blank_before: bool,
}

impl<'a> StatRow<'a> {
    pub fn plain(text: &'a str) -> Self {
        Self { text, color: None, blank_before: false }
    }
    pub fn colored(text: &'a str, color: [f32; 4]) -> Self {
        Self { text, color: Some(color), blank_before: false }
    }
    pub fn heading(text: &'a str, color: [f32; 4]) -> Self {
        Self { text, color: Some(color), blank_before: true }
    }
}

/// Draw a labelled-rows overlay panel anchored at `origin` (upper-left
/// corner of the first row's baseline area). Row spacing = `line_h`,
/// glyph scale = `font_scale`. Panel width is `panel_w`; rows overflow
/// visually if their strings exceed it — caller sizes accordingly.
///
/// The rows are drawn top-down from `origin.y`. Each `blank_before`
/// row adds an extra `line_h` gap above itself; that gap counts in the
/// backdrop's height.
pub fn draw_stat_panel(
    batch:      &mut Batch,
    origin:     Vec2,
    rows:       &[StatRow],
    line_h:     f32,
    font_scale: f32,
    panel_w:    f32,
) {
    if rows.is_empty() { return; }
    let mut total_rows = 0f32;
    for r in rows {
        if r.blank_before { total_rows += 1.0; }
        total_rows += 1.0;
    }
    // Panel: centred vertically on the mid-span of the rows, with a
    // half-line pad above + below. Panel horizontal centre sits at
    // `origin.x + panel_w * 0.5 - 10.0` so a 10 px left margin lines up
    // with the leftmost glyph.
    let panel_h = line_h * (total_rows + 1.0);
    let panel_center = Vec2::new(
        origin.x + panel_w * 0.5 - 10.0,
        origin.y - line_h * (total_rows * 0.5 + 0.5),
    );
    UiRect::from_center(panel_center, Vec2::new(panel_w, panel_h))
        .fill(batch, style::PANEL_BG);

    let mut y = origin.y;
    for r in rows {
        if r.blank_before { y -= line_h; }
        let colour = r.color.unwrap_or(style::TEXT);
        draw_text(batch, r.text, Vec2::new(origin.x, y), font_scale, colour);
        y -= line_h;
    }
}

/// Centred "hold X to confirm" popup with a progress bar. Renders
/// nothing when `held_secs < 0.1` (below the debounce threshold that
/// tells us the key is actively held).
///
/// `title` reads across the top; `subtitle_fmt` is called with the
/// remaining seconds to fill the row below the bar.
pub fn draw_hold_confirm_popup(
    batch:        &mut Batch,
    held_secs:    f32,
    total_secs:   f32,
    title:        &str,
    subtitle:     &str,
    panel_color:  [f32; 4],
    border_color: [f32; 4],
    bar_fill:     [f32; 4],
    bar_bg:       [f32; 4],
    title_color:  [f32; 4],
    detail_color: [f32; 4],
) {
    if held_secs < 0.1 { return; }
    let frac = (held_secs / total_secs).clamp(0.0, 1.0);

    let rect = UiRect::from_center(Vec2::ZERO, Vec2::new(420.0, 72.0));
    rect.fill(batch, panel_color);
    rect.outline(batch, 2.0, border_color);

    let bar_rect = UiRect::from_center(Vec2::new(0.0, -22.0), Vec2::new(380.0, 8.0));
    widgets::draw_progress_bar(batch, bar_rect, frac, bar_fill, bar_bg);

    draw_text_centered(batch, title, Vec2::new(0.0, 10.0), 2.0, title_color);
    draw_text_centered(batch, subtitle, Vec2::new(0.0, -6.0), 1.5, detail_color);
}
