//! Thin wrapper over `InputState` that works in UI space (center-origin,
//! +y-up) and tracks whether any UI element has consumed mouse input
//! this frame.
//!
//! `lmb_pressed` / `rmb_pressed` are cached flags rather than read from
//! `InputState` directly, because `InputState::begin_frame()` clears
//! pressed flags before `render` runs. The game caches the press in
//! `fixed_update` and passes it here via the `with_*_pressed` builders.

use glam::Vec2;
use winit::event::MouseButton;
use crate::input::InputState;
use super::UiRect;

pub struct UiInput<'a> {
    inner:       &'a InputState,
    viewport:    Vec2,
    lmb_pressed: bool,
    rmb_pressed: bool,
    pub consumed: bool,
}

impl<'a> UiInput<'a> {
    pub fn new(inner: &'a InputState, viewport: Vec2) -> Self {
        Self { inner, viewport, lmb_pressed: false, rmb_pressed: false, consumed: false }
    }

    pub fn with_lmb_pressed(mut self, pressed: bool) -> Self {
        self.lmb_pressed = pressed;
        self
    }

    pub fn with_rmb_pressed(mut self, pressed: bool) -> Self {
        self.rmb_pressed = pressed;
        self
    }

    /// Mouse position in UI space.
    pub fn mouse_pos(&self) -> Vec2 {
        let px = self.inner.mouse_pos;
        Vec2::new(
            px.x - self.viewport.x * 0.5,
            -(px.y - self.viewport.y * 0.5),
        )
    }

    pub fn hovered(&self, rect: UiRect) -> bool {
        rect.contains(self.mouse_pos())
    }

    /// Returns true and marks consumed if the mouse left-clicked inside `rect`.
    pub fn clicked(&mut self, rect: UiRect) -> bool {
        if self.consumed { return false; }
        if rect.contains(self.mouse_pos()) && self.lmb_pressed {
            self.consumed = true;
            true
        } else {
            false
        }
    }

    /// Marks consumed (blocking game input) whenever mouse is over `rect`.
    pub fn consume_if_hovered(&mut self, rect: UiRect) -> bool {
        if rect.contains(self.mouse_pos()) {
            self.consumed = true;
            true
        } else {
            false
        }
    }

    /// Returns true and marks consumed if the mouse right-clicked inside `rect`.
    pub fn right_clicked(&mut self, rect: UiRect) -> bool {
        if self.consumed { return false; }
        if rect.contains(self.mouse_pos()) && self.rmb_pressed {
            self.consumed = true;
            true
        } else {
            false
        }
    }

    pub fn scroll_delta(&self) -> f32 { self.inner.scroll_delta }
    pub fn lmb_down(&self) -> bool { self.inner.mouse_down(MouseButton::Left) }
    pub fn lmb_pressed(&self) -> bool { self.lmb_pressed }
    pub fn rmb_pressed(&self) -> bool { self.rmb_pressed }
    pub fn mmb_pressed(&self) -> bool { self.inner.mouse_pressed(MouseButton::Middle) }
}
