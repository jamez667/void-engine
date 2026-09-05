pub mod keybinds;

use glam::{DVec2, Vec2};
use winit::keyboard::KeyCode;
use winit::event::MouseButton;

/// Generic 2D movement input. Every 2D top-down game shares this bag:
/// thrust/reverse + turn + brake for a ship or vehicle, WASD walk + shift
/// sprint for a character, plus an aim vector and an interact button.
/// Game-specific action bits (mine, tractor, dock, sell_ore, …) do NOT
/// belong here — they live in the game crate's own input struct, which
/// may embed this one.
#[derive(Clone, Debug, Default)]
pub struct Movement2DInput {
    // Vehicle / ship
    pub thrust:     bool,
    pub reverse:    bool,
    pub turn_left:  bool,
    pub turn_right: bool,
    pub brake:      bool,
    // Character walk (on-foot)
    pub walk_n: bool,
    pub walk_s: bool,
    pub walk_e: bool,
    pub walk_w: bool,
    /// Hold-to-sprint modifier for walking.
    pub walk_sprint: bool,
    /// E — interact with nearest prompt.
    pub interact: bool,
    /// World-space aim direction (pre-computed by client from mouse +
    /// camera). Length-1 vector when live; zero when no aim source.
    pub aim_world: DVec2,
}

#[derive(Default, Clone)]
pub struct InputState {
    keys_down: [u64; 4],
    keys_pressed: [u64; 4],
    keys_released: [u64; 4],
    pub mouse_pos: Vec2,
    pub mouse_delta: Vec2,
    pub mouse_buttons_down: u8,
    pub mouse_buttons_pressed: u8,
    pub mouse_buttons_released: u8,
    pub scroll_delta: f32,
}

fn key_bit(key: KeyCode) -> Option<(usize, u64)> {
    let k = key as u32;
    if k >= 256 {
        return None;
    }
    Some(((k / 64) as usize, 1u64 << (k % 64)))
}

impl InputState {
    pub fn begin_frame(&mut self) {
        self.keys_pressed = [0; 4];
        self.keys_released = [0; 4];
        self.mouse_delta = Vec2::ZERO;
        self.mouse_buttons_pressed = 0;
        self.mouse_buttons_released = 0;
        self.scroll_delta = 0.0;
    }

    pub fn on_scroll(&mut self, delta: f32) {
        self.scroll_delta += delta;
    }

    pub fn on_key_down(&mut self, key: KeyCode) {
        if let Some((slot, bit)) = key_bit(key) {
            if self.keys_down[slot] & bit == 0 {
                self.keys_pressed[slot] |= bit;
            }
            self.keys_down[slot] |= bit;
        }
    }

    pub fn on_key_up(&mut self, key: KeyCode) {
        if let Some((slot, bit)) = key_bit(key) {
            self.keys_down[slot] &= !bit;
            self.keys_released[slot] |= bit;
        }
    }

    pub fn on_mouse_move(&mut self, pos: Vec2, delta: Vec2) {
        self.mouse_pos = pos;
        self.mouse_delta += delta;
    }

    pub fn on_mouse_down(&mut self, btn: MouseButton) {
        let bit = mouse_bit(btn);
        if self.mouse_buttons_down & bit == 0 {
            self.mouse_buttons_pressed |= bit;
        }
        self.mouse_buttons_down |= bit;
    }

    pub fn on_mouse_up(&mut self, btn: MouseButton) {
        let bit = mouse_bit(btn);
        self.mouse_buttons_down &= !bit;
        self.mouse_buttons_released |= bit;
    }

    pub fn key_down(&self, key: KeyCode) -> bool {
        key_bit(key)
            .map(|(s, b)| self.keys_down[s] & b != 0)
            .unwrap_or(false)
    }

    pub fn key_pressed(&self, key: KeyCode) -> bool {
        key_bit(key)
            .map(|(s, b)| self.keys_pressed[s] & b != 0)
            .unwrap_or(false)
    }

    pub fn mouse_down(&self, btn: MouseButton) -> bool {
        self.mouse_buttons_down & mouse_bit(btn) != 0
    }

    pub fn mouse_pressed(&self, btn: MouseButton) -> bool {
        self.mouse_buttons_pressed & mouse_bit(btn) != 0
    }
}

fn mouse_bit(btn: MouseButton) -> u8 {
    match btn {
        MouseButton::Left => 1,
        MouseButton::Right => 2,
        MouseButton::Middle => 4,
        _ => 0,
    }
}

#[cfg(test)]
mod edge_flag_tests {
    use super::*;

    /// A key press must be visible to exactly ONE fixed step, however many
    /// steps the frame runs.
    ///
    /// `App::frame` clears the edge flags after the first step of the
    /// catch-up loop. It used to clear after the whole loop, so when the
    /// renderer fell behind and `advance` returned several steps, every
    /// step saw the same press — one keystroke typed five characters into
    /// the login field at ~25 fps against a 60 Hz step.
    #[test]
    fn a_press_is_consumed_by_exactly_one_step() {
        let mut input = InputState::default();
        input.on_key_down(KeyCode::KeyA);

        // Simulate a frame that runs 5 catch-up steps, clearing after the
        // first exactly as `App::frame` does.
        let mut seen = 0;
        for step in 0..5 {
            if input.key_pressed(KeyCode::KeyA) { seen += 1; }
            if step == 0 { input.begin_frame(); }
        }
        assert_eq!(seen, 1, "one press was seen by {seen} steps");
    }

    /// ...and must NOT be dropped by a frame that runs no steps at all.
    ///
    /// The render loop runs at ~62 Hz against a 1/60 fixed step, so frames
    /// with zero steps are routine. Clearing unconditionally at frame top
    /// swallowed presses that arrived in them — the original bug, whose
    /// fix caused the duplicate above.
    #[test]
    fn a_press_survives_a_frame_that_runs_no_steps() {
        let mut input = InputState::default();
        input.on_key_down(KeyCode::KeyA);

        // Frame with steps == 0: the loop body never runs, so nothing clears.
        for step in 0..0 {
            if step == 0 { input.begin_frame(); }
        }
        assert!(input.key_pressed(KeyCode::KeyA),
            "a press must survive until a fixed step actually consumes it");
    }

    /// Holding a key must not re-fire the edge. `on_key_down` arrives
    /// repeatedly while a key is held (OS key repeat).
    #[test]
    fn holding_a_key_does_not_re_fire_the_edge() {
        let mut input = InputState::default();
        input.on_key_down(KeyCode::KeyA);
        assert!(input.key_pressed(KeyCode::KeyA));
        input.begin_frame();

        input.on_key_down(KeyCode::KeyA); // still held / OS repeat
        assert!(!input.key_pressed(KeyCode::KeyA), "a held key must not re-press");
        assert!(input.key_down(KeyCode::KeyA), "but it is still down");
    }
}
