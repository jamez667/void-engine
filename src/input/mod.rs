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

    /// True on the frame a key went down.
    ///
    /// # A press may already be over
    ///
    /// This reports the *edge*, not the current state: a tap that goes
    /// down and up inside one frame sets this and leaves
    /// [`key_down`](Self::key_down) false. That is deliberate — the press
    /// must not be lost — but logic shaped as
    /// `if key_pressed(k) { start_hold() }` followed by
    /// `while key_down(k) { ... }` then never starts the hold. At 62 Hz
    /// polling a fast tap or a replayed input pair can land both events
    /// in the same frame.
    pub fn key_pressed(&self, key: KeyCode) -> bool {
        key_bit(key)
            .map(|(s, b)| self.keys_pressed[s] & b != 0)
            .unwrap_or(false)
    }

    /// True on the frame a key came up.
    ///
    /// The release edge was recorded and cleared on the same one-step
    /// schedule as [`key_pressed`](Self::key_pressed) since edges existed,
    /// but had no accessor until 2026-09-11 — `keys_released` was
    /// write-only state, so hold-to-charge/release-to-fire could not be
    /// expressed through this API at all. The mouse half
    /// (`mouse_buttons_released`) was readable the whole time, being a
    /// public field.
    pub fn key_released(&self, key: KeyCode) -> bool {
        key_bit(key)
            .map(|(s, b)| self.keys_released[s] & b != 0)
            .unwrap_or(false)
    }

    /// True on the frame a mouse button came up. Mirrors
    /// [`key_released`](Self::key_released).
    pub fn mouse_released(&self, btn: MouseButton) -> bool {
        self.mouse_buttons_released & mouse_bit(btn) != 0
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
mod release_edge_tests {
    use super::*;

    /// `keys_released` was written and cleared correctly but had no
    /// accessor, so a release edge could not be read at all.
    #[test]
    fn a_release_is_visible_for_one_frame() {
        let mut i = InputState::default();
        i.on_key_down(KeyCode::KeyF);
        assert!(!i.key_released(KeyCode::KeyF), "holding is not releasing");

        i.on_key_up(KeyCode::KeyF);
        assert!(i.key_released(KeyCode::KeyF), "the release edge must be readable");
        assert!(!i.key_down(KeyCode::KeyF), "and the key is no longer held");

        i.begin_frame();
        assert!(!i.key_released(KeyCode::KeyF), "the edge lasts exactly one frame");
    }

    /// The mouse half was already a public field; the accessor just makes
    /// the two halves symmetric.
    #[test]
    fn mouse_releases_read_the_same_way() {
        let mut i = InputState::default();
        i.on_mouse_down(MouseButton::Left);
        assert!(!i.mouse_released(MouseButton::Left));
        i.on_mouse_up(MouseButton::Left);
        assert!(i.mouse_released(MouseButton::Left));
        i.begin_frame();
        assert!(!i.mouse_released(MouseButton::Left));
    }

    /// A tap inside one frame reports press *and* release together, with
    /// `key_down` never true — the case the `key_pressed` docs warn about.
    #[test]
    fn a_sub_frame_tap_reports_both_edges_and_no_hold() {
        let mut i = InputState::default();
        i.on_key_down(KeyCode::Space);
        i.on_key_up(KeyCode::Space);

        assert!(i.key_pressed(KeyCode::Space), "the press must not be lost");
        assert!(i.key_released(KeyCode::Space), "nor the release");
        assert!(!i.key_down(KeyCode::Space), "but it is not held at any point a caller can see");
    }

    /// Keys past the bitset's range are dropped rather than aliasing onto
    /// another key — the same contract `key_pressed` has.
    #[test]
    fn releases_of_unmapped_keys_are_not_reported() {
        let i = InputState::default();
        assert!(!i.key_released(KeyCode::KeyA), "nothing has been released");
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

        // Mirror `app.rs`'s catch-up loop for a frame the timestep gave no
        // steps. `steps` is a binding rather than a literal `0..0` so the
        // range is not statically empty — clippy denies that by default,
        // and the point here is that the loop body never runs, not how the
        // bound is spelled.
        let steps = 0;
        for step in 0..steps {
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
