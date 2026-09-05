//! Generic chat window widget + rendering.
//!
//! Owns the scrollback buffer, the input row, per-channel tab bar, and
//! keyboard handling (T to focus, Tab to cycle channels, Enter to send,
//! Backspace, alphanumeric + punctuation input).
//!
//! Game code supplies:
//! - A channel enum implementing [`ChatChannel`] (label, colour, all-channels
//!   iteration order, whisper-flag, cycle target).
//! - The wire-protocol translation for sent messages (returned from
//!   [`ChatWindow::handle_input`]).
//!
//! The `CHAR_MAP` keycode-to-char table is game-agnostic ASCII and lives
//! here.

use std::collections::VecDeque;
use std::time::Instant;
use winit::keyboard::KeyCode;

use glam::Vec2;

use crate::input::InputState;
use crate::renderer::batch::Batch;
use crate::text::{draw_text, draw_text_centered, text_width};
use crate::ui::{style, Anchor, UiInput, UiRect};

/// Per-channel presentation + cycling / whisper behaviour supplied by
/// the game crate. Channel enums implement this to plug into the
/// generic [`ChatWindow`].
pub trait ChatChannel: Copy + PartialEq + std::fmt::Debug + 'static {
    /// Bottom-of-panel tab label — short, all caps by convention.
    fn label(self) -> &'static str;
    /// Text colour for this channel's messages.
    fn color(self) -> [f32; 4];
    /// Colour for the active-tab outline. Default: same as `color`
    /// with alpha 0.75.
    fn tab_color(self) -> [f32; 4] {
        let [r, g, b, _] = self.color();
        [r, g, b, 0.75]
    }
    /// Every channel, in tab-bar order.
    fn all() -> &'static [Self];
    /// Cycle-target for Tab. Typically wraps around `all()`.
    fn next(self) -> Self;
    /// Whisper channels use the input prefix `[W>target]` and let
    /// Backspace pop the target when the input is empty. Others return
    /// false.
    fn is_whisper(self) -> bool { false }
}

// ── message ───────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct ChatMessage<C: ChatChannel> {
    pub channel: C,
    pub sender:  String,
    pub text:    String,
    /// System notice (mining rig, station, hyperspace, etc.). Rendered with
    /// a grey `[SYS]` tag instead of a channel tag so it stands apart from
    /// player chatter — but lives in the same scrollback so it persists.
    pub system:  bool,
}

// ── window ────────────────────────────────────────────────────────────────────

const MAX_MESSAGES: usize = 100;
const MAX_INPUT:    usize = 120;

pub struct ChatWindow<C: ChatChannel> {
    pub messages:        VecDeque<ChatMessage<C>>,
    pub input:           String,
    pub active_channel:  C,
    pub focused:         bool,
    pub whisper_target:  String,
    pub cursor_timer:    f32,
    /// Wall-clock time of last activity (send / receive / focus). Drives the
    /// idle fade in [`draw`] so chat dims out of the way when nothing's
    /// happening.
    pub last_activity:   Instant,
}

impl<C: ChatChannel> ChatWindow<C> {
    /// Fresh window seeded with `initial` as the active channel and a
    /// single system notice.
    pub fn new(initial: C, welcome: &str) -> Self {
        let mut w = Self {
            messages:       VecDeque::new(),
            input:          String::new(),
            active_channel: initial,
            focused:        false,
            whisper_target: String::new(),
            cursor_timer:   0.0,
            last_activity:  Instant::now(),
        };
        w.push_system(welcome);
        w
    }

    /// Push a message received from the network or system.
    pub fn push(&mut self, channel: C, sender: impl Into<String>, text: impl Into<String>) {
        self.messages.push_back(ChatMessage {
            channel, sender: sender.into(), text: text.into(), system: false,
        });
        while self.messages.len() > MAX_MESSAGES { self.messages.pop_front(); }
        self.last_activity = Instant::now();
    }

    /// Push a system notice — station terminals, mining rig, hyperspace
    /// state, etc. Renders with a grey `[SYS]` tag.
    pub fn push_system(&mut self, text: impl Into<String>) {
        self.messages.push_back(ChatMessage {
            channel: self.active_channel, sender: String::new(),
            text: text.into(), system: true,
        });
        while self.messages.len() > MAX_MESSAGES { self.messages.pop_front(); }
        self.last_activity = Instant::now();
    }

    /// Process keyboard input each fixed tick.
    ///
    /// Returns `(channel, target, text)` on Enter with non-empty input;
    /// `target` is the whisper target when the active channel returns
    /// true from `is_whisper`, empty string otherwise.
    pub fn handle_input(&mut self, input: &InputState, dt: f32) -> Option<(C, String, String)> {
        self.cursor_timer += dt;

        let shift = input.key_down(KeyCode::ShiftLeft) || input.key_down(KeyCode::ShiftRight);

        // T focuses (when not focused)
        if !self.focused && input.key_pressed(KeyCode::KeyT) {
            self.focused = true;
            self.last_activity = Instant::now();
            return None;
        }

        if !self.focused { return None; }
        // Focused = active; keep the fade timer reset every tick.
        self.last_activity = Instant::now();

        // Esc unfocus is owned by the game's shared Esc priority chain.

        // Tab cycles channel
        if input.key_pressed(KeyCode::Tab) {
            self.active_channel = self.active_channel.next();
            return None;
        }

        // Backspace
        if input.key_pressed(KeyCode::Backspace) {
            if self.active_channel.is_whisper() && self.input.is_empty() {
                self.whisper_target.pop();
            } else {
                self.input.pop();
            }
        }

        // Enter → send
        if input.key_pressed(KeyCode::Enter) {
            let text = self.input.trim().to_string();
            if !text.is_empty() {
                let target = if self.active_channel.is_whisper() {
                    self.whisper_target.clone()
                } else {
                    String::new()
                };
                self.push(self.active_channel, "You", text.clone());
                self.input.clear();
                self.cursor_timer = 0.0;
                return Some((self.active_channel, target, text));
            }
            return None;
        }

        // Character input
        let typing_target = self.active_channel.is_whisper() && self.input.is_empty();
        for &(code, lower, upper) in CHAR_MAP {
            if input.key_pressed(code) {
                let ch = if shift { upper } else { lower };
                if typing_target {
                    if self.whisper_target.len() < 32 { self.whisper_target.push(ch); }
                } else if self.input.len() < MAX_INPUT {
                    self.input.push(ch);
                }
                self.cursor_timer = 0.0;
            }
        }

        None
    }

    #[allow(dead_code)]
    pub fn messages_for(&self, channel: C) -> impl Iterator<Item = &ChatMessage<C>> {
        self.messages.iter().filter(move |m| m.channel == channel)
    }

    pub fn all_messages(&self) -> impl Iterator<Item = &ChatMessage<C>> {
        self.messages.iter()
    }
}

// ── keycode → char table ──────────────────────────────────────────────────────

#[rustfmt::skip]
pub const CHAR_MAP: &[(KeyCode, char, char)] = &[
    (KeyCode::KeyA, 'a', 'A'), (KeyCode::KeyB, 'b', 'B'), (KeyCode::KeyC, 'c', 'C'),
    (KeyCode::KeyD, 'd', 'D'), (KeyCode::KeyE, 'e', 'E'), (KeyCode::KeyF, 'f', 'F'),
    (KeyCode::KeyG, 'g', 'G'), (KeyCode::KeyH, 'h', 'H'), (KeyCode::KeyI, 'i', 'I'),
    (KeyCode::KeyJ, 'j', 'J'), (KeyCode::KeyK, 'k', 'K'), (KeyCode::KeyL, 'l', 'L'),
    (KeyCode::KeyM, 'm', 'M'), (KeyCode::KeyN, 'n', 'N'), (KeyCode::KeyO, 'o', 'O'),
    (KeyCode::KeyP, 'p', 'P'), (KeyCode::KeyQ, 'q', 'Q'), (KeyCode::KeyR, 'r', 'R'),
    (KeyCode::KeyS, 's', 'S'), (KeyCode::KeyT, 't', 'T'), (KeyCode::KeyU, 'u', 'U'),
    (KeyCode::KeyV, 'v', 'V'), (KeyCode::KeyW, 'w', 'W'), (KeyCode::KeyX, 'x', 'X'),
    (KeyCode::KeyY, 'y', 'Y'), (KeyCode::KeyZ, 'z', 'Z'),

    (KeyCode::Digit0, '0', ')'), (KeyCode::Digit1, '1', '!'), (KeyCode::Digit2, '2', '@'),
    (KeyCode::Digit3, '3', '#'), (KeyCode::Digit4, '4', '$'), (KeyCode::Digit5, '5', '%'),
    (KeyCode::Digit6, '6', '^'), (KeyCode::Digit7, '7', '&'), (KeyCode::Digit8, '8', '*'),
    (KeyCode::Digit9, '9', '('),

    (KeyCode::Space,        ' ',  ' '),
    (KeyCode::Minus,        '-',  '_'), (KeyCode::Equal,     '=', '+'),
    (KeyCode::BracketLeft,  '[',  '{'), (KeyCode::BracketRight, ']', '}'),
    (KeyCode::Semicolon,    ';',  ':'), (KeyCode::Quote,     '\'', '"'),
    (KeyCode::Comma,        ',',  '<'), (KeyCode::Period,    '.',  '>'),
    (KeyCode::Slash,        '/',  '?'), (KeyCode::Backslash, '\\', '|'),
];

// ── rendering ────────────────────────────────────────────────────────────────

const PANEL_W:   f32 = 510.0;
const PANEL_H:   f32 = 210.0;
const TAB_H:     f32 = 18.0;
const INPUT_H:   f32 = 18.0;
const MSG_SCALE: f32 = 1.3;
const MSG_LINE:  f32 = 12.0;
/// Bottom margin for the chat panel. Tighter than the global `style::PAD` so
/// chat sits close to the bottom edge without touching it.
const CHAT_BOTTOM_PAD: f32 = 6.0;

/// Draw the chat window and handle tab-click focus. Combines
/// [`clicked_channel`] + [`draw`] in the correct order — always call
/// this from the game's chat draw path unless you have a reason to
/// split them.
pub fn draw_and_focus<C: ChatChannel>(
    batch: &mut Batch,
    viewport: Vec2,
    ui: &mut UiInput,
    window: &mut ChatWindow<C>,
) {
    // Tab click test must run BEFORE `draw` — `draw` calls
    // `ui.consume_if_hovered(panel)` for the whole chat rect, which sets
    // `ui.consumed = true`. `UiInput::clicked` short-circuits on `consumed`,
    // so if the order were reversed the tab clicks would never register.
    if let Some(ch) = clicked_channel::<C>(ui, viewport) {
        window.active_channel = ch;
        window.focused = true;
        window.last_activity = Instant::now();
    }
    draw(batch, viewport, ui, window);
}

pub fn draw<C: ChatChannel>(
    batch: &mut Batch,
    viewport: Vec2,
    ui: &mut UiInput,
    window: &ChatWindow<C>,
) {
    let mut bl = Anchor::BottomLeft.inset(viewport, style::PAD);
    bl.y = -viewport.y * 0.5 + CHAT_BOTTOM_PAD;
    let panel = UiRect::from_min_size(bl, Vec2::new(PANEL_W, PANEL_H));

    ui.consume_if_hovered(panel);

    // Idle fade: full alpha until FADE_START, then lerp to FADE_MIN over
    // FADE_DUR. Focused chat always renders at full alpha.
    const FADE_START: f32 = 5.0;
    const FADE_DUR:   f32 = 5.0;
    const FADE_MIN:   f32 = 0.15 / 0.65;
    let idle = window.last_activity.elapsed().as_secs_f32();
    let alpha_mul = if window.focused {
        1.0
    } else {
        let t = ((idle - FADE_START) / FADE_DUR).clamp(0.0, 1.0);
        1.0 + (FADE_MIN - 1.0) * t
    };

    let fade = |mut c: [f32; 4]| { c[3] *= alpha_mul; c };

    panel.fill(batch, fade(style::HUD_BG));
    panel.outline(batch, 1.0, fade(style::PANEL_BORDER));

    // ── channel tabs ──────────────────────────────────────────────────────────
    let all = C::all();
    let tab_count = all.len() as f32;
    let tab_w = (PANEL_W - 2.0) / tab_count;

    for (i, ch) in all.iter().copied().enumerate() {
        let tab_min = Vec2::new(panel.min.x + 1.0 + i as f32 * tab_w, panel.max.y - TAB_H);
        let tab_rect = UiRect::from_min_size(tab_min, Vec2::new(tab_w - 1.0, TAB_H));

        let active = window.active_channel == ch;
        let tab_bg = if active {
            let [r, g, b, _] = ch.color();
            [r * 0.25, g * 0.25, b * 0.25, 0.95]
        } else {
            [0.05, 0.06, 0.09, 0.85]
        };
        tab_rect.fill(batch, fade(tab_bg));

        let border = if active { ch.tab_color() } else { style::PANEL_BORDER };
        tab_rect.outline(batch, 1.0, fade(border));

        let label_color = if active { ch.color() } else { style::TEXT_DIM };
        draw_text_centered(batch, ch.label(), tab_rect.center(), MSG_SCALE, fade(label_color));
    }

    // ── message area ──────────────────────────────────────────────────────────
    let msg_top    = panel.max.y - TAB_H - 2.0;
    let msg_bottom = panel.min.y + INPUT_H + MSG_LINE;
    let msg_area   = UiRect {
        min: Vec2::new(panel.min.x + 4.0, msg_bottom),
        max: Vec2::new(panel.max.x - 4.0, msg_top),
    };
    let max_lines  = ((msg_area.size().y) / MSG_LINE) as usize;

    let msgs: Vec<_> = window.all_messages().collect();
    let start = msgs.len().saturating_sub(max_lines);
    let visible = &msgs[start..];

    for (i, msg) in visible.iter().enumerate() {
        let y = msg_bottom + MSG_LINE * (visible.len() - 1 - i) as f32 + 3.0 * MSG_SCALE;

        let (tag, tag_color) = if msg.system {
            ("[SYS]".to_string(), [0.62, 0.66, 0.72, 1.0])
        } else {
            (format!("[{}]", msg.channel.label().chars().next().unwrap_or('?')), msg.channel.color())
        };
        draw_text(batch, &tag, Vec2::new(msg_area.min.x, y), MSG_SCALE, fade(tag_color));
        let tag_w = text_width(&tag, MSG_SCALE) + 2.0;

        let sender_w = if msg.system {
            0.0
        } else {
            let sender_str = format!("{}: ", msg.sender);
            let sender_color = [0.75, 0.75, 0.80, 1.0];
            draw_text(batch, &sender_str, Vec2::new(msg_area.min.x + tag_w, y), MSG_SCALE, fade(sender_color));
            text_width(&sender_str, MSG_SCALE)
        };

        let text_x = msg_area.min.x + tag_w + sender_w;
        let avail_w = msg_area.max.x - text_x;
        let text_color = if msg.system { [0.82, 0.85, 0.90, 1.0] } else { msg.channel.color() };
        let truncated = fit_text(&msg.text, avail_w, MSG_SCALE);
        draw_text(batch, &truncated, Vec2::new(text_x, y), MSG_SCALE, fade(text_color));
    }

    // ── input row ─────────────────────────────────────────────────────────────
    let input_rect = UiRect::from_min_size(
        Vec2::new(panel.min.x + 1.0, panel.min.y + 1.0),
        Vec2::new(PANEL_W - 2.0, INPUT_H),
    );

    let input_bg = if window.focused {
        [0.08, 0.10, 0.16, 0.98]
    } else {
        [0.04, 0.05, 0.08, 0.90]
    };
    input_rect.fill(batch, fade(input_bg));
    let input_border = if window.focused { window.active_channel.tab_color() } else { style::PANEL_BORDER };
    input_rect.outline(batch, 1.0, fade(input_border));

    let ix = input_rect.min.x + 4.0;
    let iy = input_rect.min.y + INPUT_H * 0.5 + 3.0 * MSG_SCALE;

    if window.focused {
        let prefix = if window.active_channel.is_whisper() {
            let to = if window.whisper_target.is_empty() { "<name>" } else { &window.whisper_target };
            format!("[W>{}] ", to)
        } else {
            format!("[{}] ", window.active_channel.label().chars().next().unwrap_or(' '))
        };
        draw_text(batch, &prefix, Vec2::new(ix, iy), MSG_SCALE, fade(window.active_channel.color()));
        let prefix_w = text_width(&prefix, MSG_SCALE);

        draw_text(batch, &window.input, Vec2::new(ix + prefix_w, iy), MSG_SCALE, fade(style::TEXT));

        let cursor_visible = ((window.cursor_timer * 2.0) as u32).is_multiple_of(2);
        if cursor_visible {
            let cx = ix + prefix_w + text_width(&window.input, MSG_SCALE) + 1.0;
            batch.line(
                Vec2::new(cx, input_rect.min.y + 2.0),
                Vec2::new(cx, input_rect.max.y - 2.0),
                1.0,
                fade(window.active_channel.color()),
            );
        }
    } else {
        draw_text(batch, "Press T to chat", Vec2::new(ix, iy), MSG_SCALE, fade(style::TEXT_DIM));
    }
}

/// Return the channel the user clicked (for tab switching via mouse).
pub fn clicked_channel<C: ChatChannel>(ui: &mut UiInput, viewport: Vec2) -> Option<C> {
    let mut bl = Anchor::BottomLeft.inset(viewport, style::PAD);
    bl.y = -viewport.y * 0.5 + CHAT_BOTTOM_PAD;
    let panel_min = bl;
    let all = C::all();
    let tab_w = (PANEL_W - 2.0) / all.len() as f32;

    for (i, ch) in all.iter().copied().enumerate() {
        let tab_min = Vec2::new(panel_min.x + 1.0 + i as f32 * tab_w, panel_min.y + PANEL_H - TAB_H);
        let tab_rect = UiRect::from_min_size(tab_min, Vec2::new(tab_w - 1.0, TAB_H));
        if ui.clicked(tab_rect) {
            return Some(ch);
        }
    }
    None
}

/// Truncate text to fit within `max_w` pixels at `scale`.
fn fit_text(text: &str, max_w: f32, scale: f32) -> String {
    let char_w = 9.0 * scale;
    let max_chars = (max_w / char_w).floor() as usize;
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        let truncated: String = text.chars().take(max_chars.saturating_sub(2)).collect();
        format!("{}...", truncated)
    }
}
