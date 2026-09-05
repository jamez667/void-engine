//! Immediate-mode UI toolkit. Style constants, layout helpers, input
//! filtering, generic widgets, and modal-shell primitives. Every symbol
//! here is game-agnostic — game-specific palette entries (ore colours,
//! tool-mode tints, etc.) belong in the game crate's own style module,
//! which typically `pub use`'s these constants and adds its own.

pub mod anchor;
pub mod chat;
pub mod hud;
pub mod icons;
pub mod input;
pub mod keybinds_panel;
pub mod modal;
pub mod modal_stack;
pub mod radial_menu;
pub mod rect;
pub mod stat_panel;
pub mod style;
pub mod widgets;

pub use anchor::Anchor;
pub use input::UiInput;
pub use modal_stack::ModalStack;
pub use rect::UiRect;
