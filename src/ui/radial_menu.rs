//! Hold-to-open radial (pie) menu with a centre dead-zone that cancels.
//!
//! Screen-centred wheel; each slot occupies an equal wedge starting at
//! 12 o'clock and going clockwise. Mouse angle from screen centre picks
//! the highlighted slot; mouse inside `inner_r` returns `None`
//! (cancel).
//!
//! State machine — driven by the caller:
//!   - `active=false`, key press → `open()`
//!   - while active, mouse move → `update_selection(mouse_ui)`
//!   - key release → `release()`; if a slot was highlighted its index
//!     latches into `pending`, drained with `take_pending()`.
//!
//! The widget is purely UI — it does not read input or fire game
//! actions. Game code owns the emote table / callback and dispatches
//! from the latched index.

use glam::Vec2;

use crate::renderer::batch::Batch;
use crate::text;
use crate::ui::style;

/// Visual + hit-test tuning for a radial menu.
#[derive(Clone, Copy)]
pub struct RadialStyle {
    /// Ring outer radius in pixels — where the wedges end.
    pub outer_r: f32,
    /// Inner dead-zone radius — mouse inside cancels.
    pub inner_r: f32,
    /// Highlight-glow width x height for the selected slot's label.
    pub highlight_size: Vec2,
}

impl Default for RadialStyle {
    fn default() -> Self {
        Self {
            outer_r: 140.0,
            inner_r: 40.0,
            highlight_size: Vec2::new(88.0, 34.0),
        }
    }
}

/// Radial-menu widget. Slot count is dictated by the label slice passed
/// to `draw()` / `update_selection()`; the caller keeps the payload
/// list alongside and dereferences the latched index on `take_pending`.
#[derive(Default)]
pub struct RadialMenu {
    pub active: bool,
    /// Index of the highlighted slot, or `None` inside the dead-zone.
    pub selected: Option<usize>,
    /// Index latched by the most recent `release()` with a selection.
    /// `None` until then; drained by `take_pending()`.
    pending: Option<usize>,
}

impl RadialMenu {
    /// Open the wheel — call on the hold-key press.
    pub fn open(&mut self) {
        self.active = true;
        self.selected = None;
    }

    /// Close without firing.
    pub fn cancel(&mut self) {
        self.active = false;
        self.selected = None;
    }

    /// Close and, if a slot was highlighted, latch its index for
    /// `take_pending`. Call on the hold-key release.
    pub fn release(&mut self) {
        if self.active {
            if let Some(idx) = self.selected {
                self.pending = Some(idx);
            }
        }
        self.active = false;
        self.selected = None;
    }

    /// Drain the latched slot index, if any.
    pub fn take_pending(&mut self) -> Option<usize> {
        self.pending.take()
    }

    /// Update `selected` from the mouse position in UI space
    /// (origin at wheel centre, +y up). `slot_count` = number of
    /// equally-sized wedges; slot 0 is centred at 12 o'clock and
    /// numbering proceeds clockwise.
    pub fn update_selection(&mut self, mouse_ui: Vec2, slot_count: usize, style: &RadialStyle) {
        if !self.active || slot_count == 0 { return; }
        let r = mouse_ui.length();
        if r < style.inner_r {
            self.selected = None;
            return;
        }
        // atan2 returns angle from +x, CCW. We want angle from +y CW, in [0..1).
        let n = slot_count as f32;
        let ang = mouse_ui.x.atan2(mouse_ui.y); // 0 at top, +ve to the right
        let mut norm = ang / std::f32::consts::TAU; // [-0.5..0.5]
        if norm < 0.0 { norm += 1.0; }
        let idx = (norm * n + 0.5).floor() as usize % slot_count;
        self.selected = Some(idx);
    }

    /// Render the wheel. Wedge labels are drawn from `labels` (index in
    /// slot order). `centre_hint_active` / `_inactive` control the
    /// text shown in the dead-zone based on whether a slot is
    /// currently selected.
    ///
    /// Coordinates are UI space (origin at wheel centre, +y up).
    pub fn draw(
        &self,
        batch: &mut Batch,
        labels: &[&str],
        style: &RadialStyle,
        centre_hint_active:   &str,
        centre_hint_inactive: &str,
    ) {
        if !self.active || labels.is_empty() { return; }
        let centre = Vec2::ZERO;
        let n = labels.len();

        // Backdrop ring — dark disc with a subtle outline.
        batch.circle(centre, style.outer_r + 4.0, [0.00, 0.00, 0.00, 0.55], 48);
        batch.circle(centre, style.outer_r, [0.04, 0.07, 0.12, 0.90], 48);

        // Inner dead-zone — brighter when the mouse is currently in it.
        let dead_col = if self.selected.is_none() {
            [0.55, 0.22, 0.22, 0.90]
        } else {
            [0.08, 0.10, 0.14, 0.90]
        };
        batch.circle(centre, style.inner_r, dead_col, 32);

        // Segment dividers — thin radial lines between slots.
        let slot_span = std::f32::consts::TAU / n as f32;
        for i in 0..n {
            let a = std::f32::consts::FRAC_PI_2 - slot_span * (i as f32 + 0.5);
            let dir = Vec2::new(a.cos(), a.sin());
            batch.line(
                centre + dir * style.inner_r,
                centre + dir * style.outer_r,
                1.0,
                [0.30, 0.40, 0.55, 0.80],
            );
        }

        // Labels + highlight fill for the selected slot.
        for (i, label) in labels.iter().enumerate() {
            let a = std::f32::consts::FRAC_PI_2 - slot_span * i as f32;
            let dir = Vec2::new(a.cos(), a.sin());
            let label_r = (style.outer_r + style.inner_r) * 0.5;
            let label_pos = centre + dir * label_r;
            let is_sel = self.selected == Some(i);
            if is_sel {
                batch.rect(label_pos, style.highlight_size, [0.20, 0.35, 0.65, 0.85]);
            }
            let colour = if is_sel {
                [1.00, 0.95, 0.55, 1.0]
            } else {
                style::TEXT
            };
            text::draw_text_centered(batch, label, label_pos, 2.0, colour);
        }

        // Centre hint text.
        let hint = if self.selected.is_some() { centre_hint_active } else { centre_hint_inactive };
        let hint_col = if self.selected.is_some() {
            [0.70, 0.90, 1.00, 0.95]
        } else {
            [0.95, 0.65, 0.60, 0.95]
        };
        text::draw_text_centered(batch, hint, centre, 1.5, hint_col);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dead_zone_returns_none() {
        let mut m = RadialMenu::default();
        m.open();
        m.update_selection(Vec2::new(5.0, 5.0), 4, &RadialStyle::default());
        assert!(m.selected.is_none());
    }

    #[test]
    fn top_selects_slot_zero() {
        let mut m = RadialMenu::default();
        m.open();
        m.update_selection(Vec2::new(0.0, 100.0), 4, &RadialStyle::default());
        assert_eq!(m.selected, Some(0));
    }

    #[test]
    fn right_selects_slot_one_of_four() {
        let mut m = RadialMenu::default();
        m.open();
        m.update_selection(Vec2::new(100.0, 0.0), 4, &RadialStyle::default());
        assert_eq!(m.selected, Some(1));
    }

    #[test]
    fn release_latches_pending() {
        let mut m = RadialMenu::default();
        m.open();
        m.update_selection(Vec2::new(0.0, 100.0), 4, &RadialStyle::default());
        m.release();
        assert_eq!(m.take_pending(), Some(0));
        assert_eq!(m.take_pending(), None);
    }

    #[test]
    fn cancel_drops_pending() {
        let mut m = RadialMenu::default();
        m.open();
        m.update_selection(Vec2::new(0.0, 100.0), 4, &RadialStyle::default());
        m.cancel();
        assert_eq!(m.take_pending(), None);
    }
}
