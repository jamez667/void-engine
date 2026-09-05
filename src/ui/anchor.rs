//! Nine-point anchor for placing HUD chrome relative to a viewport.
//! `pivot` returns the anchor's absolute screen point; `inward` returns
//! a unit vector pointing into the viewport, and `inset(viewport, pad)`
//! is the common one-liner used to place items just inside a corner.
//!
//! Also home to [`screen_to_ui`], which converts a window-space cursor
//! position into the same origin-centre / +y-up frame every anchor and
//! batch draw uses.

use glam::Vec2;

/// Window-space position (origin top-left, +y down — what the platform
/// hands you for the cursor) → UI space (origin centre, +y up), the frame
/// [`Anchor::pivot`] and every batch draw already work in.
///
/// Two-line conversion, but it was written inline at five call sites and a
/// dropped minus sign flips the y axis in a way that only shows up as
/// "hover targets are mirrored vertically" — worth a named function.
#[inline]
pub fn screen_to_ui(viewport: Vec2, screen_pos: Vec2) -> Vec2 {
    Vec2::new(
        screen_pos.x - viewport.x * 0.5,
        -(screen_pos.y - viewport.y * 0.5),
    )
}

#[derive(Clone, Copy, Debug)]
pub enum Anchor {
    TopLeft,    Top,    TopRight,
    Left,       Center, Right,
    BottomLeft, Bottom, BottomRight,
}

impl Anchor {
    pub fn pivot(self, viewport: Vec2) -> Vec2 {
        let h = viewport * 0.5;
        let (sx, sy) = self.signs();
        Vec2::new(h.x * sx, h.y * sy)
    }

    pub fn inward(self) -> Vec2 {
        let (sx, sy) = self.signs();
        Vec2::new(-sx, -sy)
    }

    pub fn inset(self, viewport: Vec2, pad: f32) -> Vec2 {
        self.pivot(viewport) + self.inward() * pad
    }

    fn signs(self) -> (f32, f32) {
        match self {
            Self::TopLeft     => (-1.0,  1.0),
            Self::Top         => ( 0.0,  1.0),
            Self::TopRight    => ( 1.0,  1.0),
            Self::Left        => (-1.0,  0.0),
            Self::Center      => ( 0.0,  0.0),
            Self::Right       => ( 1.0,  0.0),
            Self::BottomLeft  => (-1.0, -1.0),
            Self::Bottom      => ( 0.0, -1.0),
            Self::BottomRight => ( 1.0, -1.0),
        }
    }
}
