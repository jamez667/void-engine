//! User-rebindable key bindings.
//!
//! Two pieces:
//! - Label / parse / scan helpers ([`key_label`], [`parse_key_label`],
//!   [`first_pressed`]) — the parts every game reimplements identically.
//! - [`KeyBinds<A>`] — a map from a game-supplied [`Action`] enum to
//!   [`KeyCode`], with a serializable label-based form for JSON round-trips.
//!
//! The *set* of rebindable actions is game-specific: the game defines an
//! `Action` enum, implements [`Action`], and uses `KeyBinds<GameAction>`.

use std::collections::HashMap;
use std::hash::Hash;
use serde::{Deserialize, Serialize};
use winit::keyboard::KeyCode;
use crate::input::InputState;

/// Parse a key label produced by [`key_label`] back into a `KeyCode`.
/// Returns `None` if the label is unknown.
pub fn parse_key_label(s: &str) -> Option<KeyCode> {
    let k = match s {
        "A" => KeyCode::KeyA, "B" => KeyCode::KeyB, "C" => KeyCode::KeyC,
        "D" => KeyCode::KeyD, "E" => KeyCode::KeyE, "F" => KeyCode::KeyF,
        "G" => KeyCode::KeyG, "H" => KeyCode::KeyH, "I" => KeyCode::KeyI,
        "J" => KeyCode::KeyJ, "K" => KeyCode::KeyK, "L" => KeyCode::KeyL,
        "M" => KeyCode::KeyM, "N" => KeyCode::KeyN, "O" => KeyCode::KeyO,
        "P" => KeyCode::KeyP, "Q" => KeyCode::KeyQ, "R" => KeyCode::KeyR,
        "S" => KeyCode::KeyS, "T" => KeyCode::KeyT, "U" => KeyCode::KeyU,
        "V" => KeyCode::KeyV, "W" => KeyCode::KeyW, "X" => KeyCode::KeyX,
        "Y" => KeyCode::KeyY, "Z" => KeyCode::KeyZ,
        "Space" => KeyCode::Space,
        "LShift" => KeyCode::ShiftLeft, "RShift" => KeyCode::ShiftRight,
        "LCtrl"  => KeyCode::ControlLeft, "RCtrl"  => KeyCode::ControlRight,
        "LAlt"   => KeyCode::AltLeft, "RAlt"   => KeyCode::AltRight,
        _ => return None,
    };
    Some(k)
}

/// Human-friendly key name. Falls back to the winit debug name for
/// keys not in the explicit table.
pub fn key_label(k: KeyCode) -> String {
    match k {
        KeyCode::KeyA => "A".into(), KeyCode::KeyB => "B".into(), KeyCode::KeyC => "C".into(),
        KeyCode::KeyD => "D".into(), KeyCode::KeyE => "E".into(), KeyCode::KeyF => "F".into(),
        KeyCode::KeyG => "G".into(), KeyCode::KeyH => "H".into(), KeyCode::KeyI => "I".into(),
        KeyCode::KeyJ => "J".into(), KeyCode::KeyK => "K".into(), KeyCode::KeyL => "L".into(),
        KeyCode::KeyM => "M".into(), KeyCode::KeyN => "N".into(), KeyCode::KeyO => "O".into(),
        KeyCode::KeyP => "P".into(), KeyCode::KeyQ => "Q".into(), KeyCode::KeyR => "R".into(),
        KeyCode::KeyS => "S".into(), KeyCode::KeyT => "T".into(), KeyCode::KeyU => "U".into(),
        KeyCode::KeyV => "V".into(), KeyCode::KeyW => "W".into(), KeyCode::KeyX => "X".into(),
        KeyCode::KeyY => "Y".into(), KeyCode::KeyZ => "Z".into(),
        KeyCode::Space => "Space".into(),
        KeyCode::ShiftLeft => "LShift".into(), KeyCode::ShiftRight => "RShift".into(),
        KeyCode::ControlLeft => "LCtrl".into(), KeyCode::ControlRight => "RCtrl".into(),
        KeyCode::AltLeft => "LAlt".into(), KeyCode::AltRight => "RAlt".into(),
        _ => format!("{:?}", k),
    }
}

/// Scan the standard set of bindable keys (letters + Space + modifiers)
/// and return the first one pressed this frame. Used by settings menus
/// to capture a new binding.
pub fn first_pressed(input: &InputState) -> Option<KeyCode> {
    const CANDIDATES: &[KeyCode] = &[
        KeyCode::KeyA, KeyCode::KeyB, KeyCode::KeyC, KeyCode::KeyD, KeyCode::KeyE,
        KeyCode::KeyF, KeyCode::KeyG, KeyCode::KeyH, KeyCode::KeyI, KeyCode::KeyJ,
        KeyCode::KeyK, KeyCode::KeyL, KeyCode::KeyM, KeyCode::KeyN, KeyCode::KeyO,
        KeyCode::KeyP, KeyCode::KeyQ, KeyCode::KeyR, KeyCode::KeyS, KeyCode::KeyT,
        KeyCode::KeyU, KeyCode::KeyV, KeyCode::KeyW, KeyCode::KeyX, KeyCode::KeyY,
        KeyCode::KeyZ,
        KeyCode::Space,
        KeyCode::ShiftLeft, KeyCode::ShiftRight,
        KeyCode::ControlLeft, KeyCode::ControlRight,
        KeyCode::AltLeft, KeyCode::AltRight,
    ];
    CANDIDATES.iter().copied().find(|&k| input.key_pressed(k))
}

// ── generic KeyBinds<A> ──────────────────────────────────────────────────────

/// Trait implemented by the game's action enum so the engine can drive the
/// keybinds map, defaults, and serialization.
pub trait Action: Copy + Eq + Hash + 'static {
    /// Every rebindable variant. Order controls the settings-UI row order.
    fn all() -> &'static [Self];
    /// Default `KeyCode` for this action.
    fn default_key(self) -> KeyCode;
    /// Stable string used as the JSON key (snake_case field name).
    fn serialized_name(self) -> &'static str;
}

/// Map from `A` → `KeyCode`. Clone-cheap; missing entries fall back to the
/// action's `default_key`.
#[derive(Clone, Debug)]
pub struct KeyBinds<A: Action> {
    map: HashMap<A, KeyCode>,
}

impl<A: Action> Default for KeyBinds<A> {
    fn default() -> Self {
        let mut map = HashMap::with_capacity(A::all().len());
        for &a in A::all() { map.insert(a, a.default_key()); }
        Self { map }
    }
}

impl<A: Action> KeyBinds<A> {
    /// Empty binds. Every `get` falls back to `default_key`. Rarely useful;
    /// prefer `Default::default()`.
    pub fn empty() -> Self { Self { map: HashMap::new() } }

    pub fn get(&self, a: A) -> KeyCode {
        self.map.get(&a).copied().unwrap_or_else(|| a.default_key())
    }

    pub fn set(&mut self, a: A, k: KeyCode) { self.map.insert(a, k); }

    /// Serialize to a `{ action_name: key_label, ... }` map for JSON storage.
    pub fn to_serialized(&self) -> SerializableKeyBinds {
        let mut out = HashMap::with_capacity(A::all().len());
        for &a in A::all() {
            out.insert(a.serialized_name().to_string(), key_label(self.get(a)));
        }
        SerializableKeyBinds(out)
    }

    /// Rebuild from a serialized map. Unknown labels fall back to the
    /// action's default — a hand-edited settings file with a bad key name
    /// doesn't soft-brick the input system.
    pub fn from_serialized(raw: &SerializableKeyBinds) -> Self {
        let mut map = HashMap::with_capacity(A::all().len());
        for &a in A::all() {
            let k = raw.0.get(a.serialized_name())
                .and_then(|s| parse_key_label(s))
                .unwrap_or_else(|| a.default_key());
            map.insert(a, k);
        }
        Self { map }
    }
}

/// Serialized form: `{ "thrust": "W", ... }`. Kept opaque so the game can
/// serde-embed it in its own settings struct without knowing the layout.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SerializableKeyBinds(pub HashMap<String, String>);
