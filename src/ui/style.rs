//! Generic UI style constants — layout, font scales, generic text /
//! panel / tab / status-bar palettes. Every value here is game-agnostic.
//! Game-specific palette entries (ore colours, tool-mode tints, warning
//! rings, …) live in the game crate's own `style` module.

// ── Layout ─────────────────────────────────────────────────────────────
pub const PAD:        f32 = 16.0;
pub const PAD_INNER:  f32 = 8.0;
pub const LINE:       f32 = 20.0;
pub const LINE_SMALL: f32 = 14.0;

// ── Font scales ────────────────────────────────────────────────────────
pub const FONT_HUD:   f32 = 2.0;
pub const FONT_HINT:  f32 = 1.5;
pub const FONT_DEBUG: f32 = 1.5;

// ── Text ───────────────────────────────────────────────────────────────
pub const TEXT:     [f32; 4] = [0.85, 0.85, 0.90, 1.0];
pub const TEXT_DIM: [f32; 4] = [0.55, 0.55, 0.60, 0.9];

// ── Modal panel shell ──────────────────────────────────────────────────
// One style for every popup (spawn ship, refinery, mission board,
// resupply, repair, market, ship store). Kept opaque enough (alpha ~0.95)
// that world HUD prompts underneath don't bleed through.
pub const PANEL_BG:      [f32; 4] = [0.04, 0.07, 0.12, 1.00];
/// Background for lightweight HUD chrome (vitals, minimap, chat, keybind
/// hint) — semi-transparent so world underneath stays readable.
pub const HUD_BG:        [f32; 4] = [0.04, 0.07, 0.12, 0.65];
pub const PANEL_BORDER:  [f32; 4] = [0.45, 0.70, 1.00, 0.90];
pub const PANEL_HEADER:  [f32; 4] = [0.85, 0.95, 1.00, 1.0];
pub const PANEL_FOOTER:  [f32; 4] = [0.55, 0.65, 0.80, 0.85];
pub const BACKDROP:      [f32; 4] = [0.00, 0.00, 0.06, 0.55];

// ── Tab strip ──────────────────────────────────────────────────────────
// Single accent so tabs read identically wherever they appear.
pub const TAB_ACTIVE_BG:      [f32; 4] = [0.20, 0.30, 0.55, 0.95];
pub const TAB_ACTIVE_BORDER:  [f32; 4] = [0.65, 0.80, 1.00, 1.0];
pub const TAB_INACTIVE_BG:    [f32; 4] = [0.05, 0.08, 0.14, 0.85];
pub const TAB_INACTIVE_BORDER:[f32; 4] = [0.20, 0.30, 0.45, 0.85];

// ── Disabled row ───────────────────────────────────────────────────────
pub const ROW_DISABLED_BG:     [f32; 4] = [0.06, 0.10, 0.14, 0.85];
pub const ROW_DISABLED_BORDER: [f32; 4] = [0.20, 0.30, 0.40, 0.85];
pub const TEXT_DISABLED:       [f32; 4] = [0.40, 0.50, 0.60, 1.0];

// ── Status bars ────────────────────────────────────────────────────────
// Health + energy meters. Games extend this palette with additional
// bar colours (shield/armour/comp/cargo) in their own style module.
pub const HP_LABEL: [f32; 4] = [0.8, 0.4, 0.4, 1.0];
pub const HP_BG:    [f32; 4] = [0.2, 0.1, 0.1, 0.8];

pub const ENERGY_LABEL: [f32; 4] = [0.3, 0.5, 0.8, 1.0];
pub const ENERGY_BG:    [f32; 4] = [0.1, 0.1, 0.2, 0.8];
pub const ENERGY_FILL:  [f32; 4] = [0.1, 0.4, 0.8, 1.0];

// ── Net status ─────────────────────────────────────────────────────────
pub const CONNECTED:    [f32; 4] = [0.3, 1.0, 0.4, 1.0];
pub const DISCONNECTED: [f32; 4] = [1.0, 0.3, 0.3, 1.0];
