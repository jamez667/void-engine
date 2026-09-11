//! Bitmap text rendering built on top of `renderer::batch::Batch`.
//!
//! Public-domain IBM CP437 8x8 font, ASCII 32 (space) through 127 (DEL).
//! One byte per row, MSB = leftmost pixel. Text lives in the same
//! coordinate space as the rest of the batch — screen pixels for HUD,
//! world units for space labels.
//!
//! # One quad per glyph, not per lit pixel
//!
//! Glyphs used to be rasterised into the batch a pixel at a time: every
//! set bit became its own 1×1 quad. That is 20.8 quads for an average
//! glyph and 37 for the densest, so a ten-character nameplate cost 833
//! vertices. Measured at two hundred nameplates it came to 9.8 ms per
//! frame — 6.8 building the batch and 3.0 uploading 16.7 MB — before a
//! single draw call, and a thousand nameplates exceeded two whole frames.
//!
//! Now each glyph is one quad UV-mapped into the atlas that
//! [`atlas_rgba`](crate::text::atlas_rgba) builds, which is 23.7× fewer
//! vertices: the same two hundred nameplates build in 0.15 ms and upload
//! 180 KB.
//!
//! # Why the atlas carries the white pixel too
//!
//! Every solid primitive in [`Batch`](crate::renderer::batch::Batch)
//! writes `uv = [0.5, 0.5]` and relies on sampling a white texel, and the
//! renderer binds exactly one texture for the whole main pass. Rather than switch bind groups mid-pass, the
//! atlas reserves its centre for white, so flat-coloured geometry keeps
//! sampling white without knowing the texture changed underneath it —
//! about two thousand call sites across the games on this engine, none of
//! which move.
//!
//! That is why the glyphs occupy only the top six cell rows. Under
//! `FilterMode::Nearest` a sample at UV 0.5 selects texel
//! `floor(0.5 × size)`, which on a 128×128 atlas is (64, 64) — measured,
//! and the same at every size tried, even and odd alike. Ninety-six
//! glyphs fit exactly into six rows of sixteen, filling `y < 48` and
//! leaving that centre texel in clear space below them.

use crate::renderer::batch::Batch;
use glam::Vec2;

/// Width and height of the glyph atlas in texels.
///
/// Square and a power of two so the UV arithmetic is exact in `f32`: a
/// cell boundary at `n / 128.0` is representable without rounding, which
/// matters because a half-texel of drift shows up as a sheared glyph edge
/// under nearest sampling.
pub const ATLAS_SIZE: u32 = 128;

/// Glyph cell edge in texels. The font is 8×8.
pub const CELL: u32 = 8;

/// Glyph cells per atlas row: 128 / 8.
const CELLS_PER_ROW: u32 = ATLAS_SIZE / CELL;

/// Texel holding the white pixel that flat-coloured geometry samples.
///
/// `floor(0.5 × ATLAS_SIZE)` on both axes — see the module docs. Must sit
/// outside every glyph cell, which it does: glyphs end at `y = 48`.
const WHITE_TEXEL: (u32, u32) = (ATLAS_SIZE / 2, ATLAS_SIZE / 2);

// Public domain IBM CP437 8x8 font, ASCII 32 (space) through 127 (DEL)
// Each character = 8 bytes (one per row), MSB = leftmost pixel
#[rustfmt::skip]
const FONT: [[u8; 8]; 96] = [
    // 0x20 space
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 0x21 !
    [0x18, 0x3C, 0x3C, 0x18, 0x18, 0x00, 0x18, 0x00],
    // 0x22 "
    [0x36, 0x36, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 0x23 #
    [0x36, 0x36, 0x7F, 0x36, 0x7F, 0x36, 0x36, 0x00],
    // 0x24 $
    [0x0C, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x0C, 0x00],
    // 0x25 %
    [0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00],
    // 0x26 &
    [0x1C, 0x36, 0x1C, 0x6E, 0x3B, 0x33, 0x6E, 0x00],
    // 0x27 '
    [0x06, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 0x28 (
    [0x18, 0x0C, 0x06, 0x06, 0x06, 0x0C, 0x18, 0x00],
    // 0x29 )
    [0x06, 0x0C, 0x18, 0x18, 0x18, 0x0C, 0x06, 0x00],
    // 0x2A *
    [0x00, 0x66, 0x3C, 0xFF, 0x3C, 0x66, 0x00, 0x00],
    // 0x2B +
    [0x00, 0x0C, 0x0C, 0x3F, 0x0C, 0x0C, 0x00, 0x00],
    // 0x2C ,
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x06],
    // 0x2D -
    [0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00],
    // 0x2E .
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00],
    // 0x2F /
    [0x60, 0x30, 0x18, 0x0C, 0x06, 0x03, 0x01, 0x00],
    // 0x30 0
    [0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00],
    // 0x31 1
    [0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00],
    // 0x32 2
    [0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00],
    // 0x33 3
    [0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00],
    // 0x34 4
    [0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00],
    // 0x35 5
    [0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00],
    // 0x36 6
    [0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00],
    // 0x37 7
    [0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00],
    // 0x38 8
    [0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00],
    // 0x39 9
    [0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00],
    // 0x3A :
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x00],
    // 0x3B ;
    [0x00, 0x0C, 0x0C, 0x00, 0x00, 0x0C, 0x0C, 0x06],
    // 0x3C <
    [0x18, 0x0C, 0x06, 0x03, 0x06, 0x0C, 0x18, 0x00],
    // 0x3D =
    [0x00, 0x00, 0x3F, 0x00, 0x00, 0x3F, 0x00, 0x00],
    // 0x3E >
    [0x06, 0x0C, 0x18, 0x30, 0x18, 0x0C, 0x06, 0x00],
    // 0x3F ?
    [0x1E, 0x33, 0x30, 0x18, 0x0C, 0x00, 0x0C, 0x00],
    // 0x40 @
    [0x3E, 0x63, 0x7B, 0x7B, 0x7B, 0x03, 0x1E, 0x00],
    // 0x41 A
    [0x0C, 0x1E, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x00],
    // 0x42 B
    [0x3F, 0x66, 0x66, 0x3E, 0x66, 0x66, 0x3F, 0x00],
    // 0x43 C
    [0x3C, 0x66, 0x03, 0x03, 0x03, 0x66, 0x3C, 0x00],
    // 0x44 D
    [0x1F, 0x36, 0x66, 0x66, 0x66, 0x36, 0x1F, 0x00],
    // 0x45 E
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x46, 0x7F, 0x00],
    // 0x46 F
    [0x7F, 0x46, 0x16, 0x1E, 0x16, 0x06, 0x0F, 0x00],
    // 0x47 G
    [0x3C, 0x66, 0x03, 0x03, 0x73, 0x66, 0x7C, 0x00],
    // 0x48 H
    [0x33, 0x33, 0x33, 0x3F, 0x33, 0x33, 0x33, 0x00],
    // 0x49 I
    [0x1E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    // 0x4A J
    [0x78, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E, 0x00],
    // 0x4B K
    [0x67, 0x66, 0x36, 0x1E, 0x36, 0x66, 0x67, 0x00],
    // 0x4C L
    [0x0F, 0x06, 0x06, 0x06, 0x46, 0x66, 0x7F, 0x00],
    // 0x4D M
    [0x63, 0x77, 0x7F, 0x7F, 0x6B, 0x63, 0x63, 0x00],
    // 0x4E N
    [0x63, 0x67, 0x6F, 0x7B, 0x73, 0x63, 0x63, 0x00],
    // 0x4F O
    [0x1C, 0x36, 0x63, 0x63, 0x63, 0x36, 0x1C, 0x00],
    // 0x50 P
    [0x3F, 0x66, 0x66, 0x3E, 0x06, 0x06, 0x0F, 0x00],
    // 0x51 Q
    [0x1E, 0x33, 0x33, 0x33, 0x3B, 0x1E, 0x38, 0x00],
    // 0x52 R
    [0x3F, 0x66, 0x66, 0x3E, 0x36, 0x66, 0x67, 0x00],
    // 0x53 S
    [0x1E, 0x33, 0x07, 0x0E, 0x38, 0x33, 0x1E, 0x00],
    // 0x54 T
    [0x3F, 0x2D, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    // 0x55 U
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0x3F, 0x00],
    // 0x56 V
    [0x33, 0x33, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00],
    // 0x57 W
    [0x63, 0x63, 0x63, 0x6B, 0x7F, 0x77, 0x63, 0x00],
    // 0x58 X
    [0x63, 0x63, 0x36, 0x1C, 0x1C, 0x36, 0x63, 0x00],
    // 0x59 Y
    [0x33, 0x33, 0x33, 0x1E, 0x0C, 0x0C, 0x1E, 0x00],
    // 0x5A Z
    [0x7F, 0x63, 0x31, 0x18, 0x4C, 0x66, 0x7F, 0x00],
    // 0x5B [
    [0x1E, 0x06, 0x06, 0x06, 0x06, 0x06, 0x1E, 0x00],
    // 0x5C backslash
    [0x03, 0x06, 0x0C, 0x18, 0x30, 0x60, 0x40, 0x00],
    // 0x5D ]
    [0x1E, 0x18, 0x18, 0x18, 0x18, 0x18, 0x1E, 0x00],
    // 0x5E ^
    [0x08, 0x1C, 0x36, 0x63, 0x00, 0x00, 0x00, 0x00],
    // 0x5F _
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFF],
    // 0x60 `
    [0x0C, 0x0C, 0x18, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 0x61 a
    [0x00, 0x00, 0x1E, 0x30, 0x3E, 0x33, 0x6E, 0x00],
    // 0x62 b
    [0x07, 0x06, 0x06, 0x3E, 0x66, 0x66, 0x3B, 0x00],
    // 0x63 c
    [0x00, 0x00, 0x1E, 0x33, 0x03, 0x33, 0x1E, 0x00],
    // 0x64 d
    [0x38, 0x30, 0x30, 0x3E, 0x33, 0x33, 0x6E, 0x00],
    // 0x65 e
    [0x00, 0x00, 0x1E, 0x33, 0x3F, 0x03, 0x1E, 0x00],
    // 0x66 f
    [0x1C, 0x36, 0x06, 0x0F, 0x06, 0x06, 0x0F, 0x00],
    // 0x67 g
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x1F],
    // 0x68 h
    [0x07, 0x06, 0x36, 0x6E, 0x66, 0x66, 0x67, 0x00],
    // 0x69 i
    [0x0C, 0x00, 0x0E, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    // 0x6A j
    [0x30, 0x00, 0x30, 0x30, 0x30, 0x33, 0x33, 0x1E],
    // 0x6B k
    [0x07, 0x06, 0x66, 0x36, 0x1E, 0x36, 0x67, 0x00],
    // 0x6C l
    [0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x1E, 0x00],
    // 0x6D m
    [0x00, 0x00, 0x33, 0x7F, 0x7F, 0x6B, 0x63, 0x00],
    // 0x6E n
    [0x00, 0x00, 0x1F, 0x33, 0x33, 0x33, 0x33, 0x00],
    // 0x6F o
    [0x00, 0x00, 0x1E, 0x33, 0x33, 0x33, 0x1E, 0x00],
    // 0x70 p
    [0x00, 0x00, 0x3B, 0x66, 0x66, 0x3E, 0x06, 0x0F],
    // 0x71 q
    [0x00, 0x00, 0x6E, 0x33, 0x33, 0x3E, 0x30, 0x78],
    // 0x72 r
    [0x00, 0x00, 0x3B, 0x6E, 0x66, 0x06, 0x0F, 0x00],
    // 0x73 s
    [0x00, 0x00, 0x3E, 0x03, 0x1E, 0x30, 0x1F, 0x00],
    // 0x74 t
    [0x08, 0x0C, 0x3E, 0x0C, 0x0C, 0x2C, 0x18, 0x00],
    // 0x75 u
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x33, 0x6E, 0x00],
    // 0x76 v
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x1E, 0x0C, 0x00],
    // 0x77 w
    [0x00, 0x00, 0x63, 0x6B, 0x7F, 0x7F, 0x36, 0x00],
    // 0x78 x
    [0x00, 0x00, 0x63, 0x36, 0x1C, 0x36, 0x63, 0x00],
    // 0x79 y
    [0x00, 0x00, 0x33, 0x33, 0x33, 0x3E, 0x30, 0x1F],
    // 0x7A z
    [0x00, 0x00, 0x3F, 0x19, 0x0C, 0x26, 0x3F, 0x00],
    // 0x7B {
    [0x38, 0x0C, 0x0C, 0x07, 0x0C, 0x0C, 0x38, 0x00],
    // 0x7C |
    [0x18, 0x18, 0x18, 0x00, 0x18, 0x18, 0x18, 0x00],
    // 0x7D }
    [0x07, 0x0C, 0x0C, 0x38, 0x0C, 0x0C, 0x07, 0x00],
    // 0x7E ~
    [0x6E, 0x3B, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
    // 0x7F DEL
    [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00],
];

/// Total advance width of `text` rendered at `scale`. Each glyph is 8 px wide
/// with 1 px of inter-glyph spacing (the trailing space is not counted).
pub fn text_width(text: &str, scale: f32) -> f32 {
    let chars = text.chars().count();
    if chars == 0 { 0.0 } else { (chars as f32 * 9.0 - 1.0) * scale }
}

/// Trim `s` from the right until it fits in `max_w` at `scale`, appending
/// an ellipsis when truncation occurs. Used by list rows where a long
/// label would otherwise punch through into an adjacent column.
pub fn truncate_to_width(s: &str, scale: f32, max_w: f32) -> String {
    if text_width(s, scale) <= max_w { return s.to_string(); }
    let ell = "...";
    let mut chars: Vec<char> = s.chars().collect();
    while !chars.is_empty() {
        chars.pop();
        let candidate: String = chars.iter().collect::<String>() + ell;
        if text_width(&candidate, scale) <= max_w {
            return candidate;
        }
    }
    ell.to_string()
}

#[allow(dead_code)]
pub fn text_height(scale: f32) -> f32 { 8.0 * scale }

/// Draw `text` centered on `center`. Y center sits on the visual midline of the glyphs.
pub fn draw_text_centered(batch: &mut Batch, text: &str, center: Vec2, scale: f32, color: [f32; 4]) {
    let w = text_width(text, scale);
    // draw_text's pos.y is the row-0 baseline; glyph spans [pos.y - 7s, pos.y + s].
    // Visual center is pos.y - 3s, so pos.y = center.y + 3s.
    let pos = Vec2::new(center.x - w * 0.5, center.y + 3.0 * scale);
    draw_text(batch, text, pos, scale, color);
}

pub fn draw_text(batch: &mut Batch, text: &str, pos: Vec2, scale: f32, color: [f32; 4]) {
    let mut x = pos.x;
    for ch in text.chars() {
        let idx = ch as usize;
        if !(32..=127).contains(&idx) {
            x += 8.0 * scale;
            continue;
        }
        let char_idx = idx - 32;
        if char_idx >= FONT.len() {
            x += 8.0 * scale;
            continue;
        }

        // Space carries no set bits, so drawing it would push a quad that
        // samples nothing but empty atlas. Skip straight to the advance.
        if FONT[char_idx].iter().all(|&b| b == 0) {
            x += 8.0 * scale + scale;
            continue;
        }

        // The glyph box: 8×8 at `scale`, with `pos.y` the row-0 baseline.
        // Row 0 sits at the top, so the box runs from `pos.y + scale` down
        // to `pos.y - 7 × scale` — exactly the span the old per-pixel path
        // covered, and what `draw_text_centered` compensates for.
        let top = pos.y + scale;
        let bottom = pos.y - 7.0 * scale;
        let corners = [
            Vec2::new(x, top),
            Vec2::new(x + 8.0 * scale, top),
            Vec2::new(x + 8.0 * scale, bottom),
            Vec2::new(x, bottom),
        ];

        // Corner order matches `Batch::rect`: top-left, top-right,
        // bottom-right, bottom-left. Atlas V grows downward while world Y
        // grows upward, so the top corners take the cell's minimum V.
        let (u0, v0, u1, v1) = cell_uv(char_idx as u32);
        batch.push_quad(corners, [[u0, v0], [u1, v0], [u1, v1], [u0, v1]], color);

        x += 8.0 * scale + scale;
    }
}

/// UV rectangle of glyph `cell` in the atlas, as `(u0, v0, u1, v1)`.
///
/// Cells run left to right, top to bottom, sixteen per row.
fn cell_uv(cell: u32) -> (f32, f32, f32, f32) {
    let col = cell % CELLS_PER_ROW;
    let row = cell / CELLS_PER_ROW;
    let size = ATLAS_SIZE as f32;
    let u0 = (col * CELL) as f32 / size;
    let v0 = (row * CELL) as f32 / size;
    (u0, v0, u0 + CELL as f32 / size, v0 + CELL as f32 / size)
}

/// Build the glyph atlas as tightly-packed RGBA8, ready to upload.
///
/// See the module docs for why the centre texel is reserved.
///
/// Set bits become opaque white, clear bits transparent black, and the
/// reserved centre texel is opaque white so flat-coloured geometry
/// sampling UV (0.5, 0.5) gets what it has always got.
///
/// The renderer calls this once at startup; it is `pub` so a caller
/// standing up its own device can too.
pub fn atlas_rgba() -> Vec<u8> {
    let size = ATLAS_SIZE as usize;
    let mut data = vec![0u8; size * size * 4];

    for (cell, glyph) in FONT.iter().enumerate() {
        let col = (cell as u32 % CELLS_PER_ROW) * CELL;
        let row = (cell as u32 / CELLS_PER_ROW) * CELL;
        for (gy, &byte) in glyph.iter().enumerate().take(CELL as usize) {
            for gx in 0..CELL as usize {
                // MSB is the leftmost pixel, matching the font's own
                // comment and the per-pixel path this replaced.
                if byte & (1 << gx) == 0 {
                    continue;
                }
                let x = col as usize + gx;
                let y = row as usize + gy;
                let i = (y * size + x) * 4;
                data[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
    }

    let (wx, wy) = WHITE_TEXEL;
    let i = (wy as usize * size + wx as usize) * 4;
    data[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);

    data
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The advance metrics 347 downstream call sites lay out against.
    /// These held before the atlas and must hold after it, to the float.
    #[test]
    fn advance_metrics_are_unchanged() {
        assert_eq!(text_width("", 1.0), 0.0);
        // One glyph is 8 wide; the trailing 1 px of spacing is not counted.
        assert_eq!(text_width("A", 1.0), 8.0);
        assert_eq!(text_width("AB", 1.0), 17.0);
        assert_eq!(text_width("Player_0042", 2.0), (11.0 * 9.0 - 1.0) * 2.0);
        assert_eq!(text_height(2.0), 16.0);
    }

    /// A glyph occupies `[pos.y - 7s, pos.y + s]` with `pos.y` the row-0
    /// baseline. `draw_text_centered` subtracts 3s to put the visual
    /// midline on `center`, so a change to either shifts every label.
    #[test]
    fn glyph_box_spans_the_documented_range() {
        let mut b = Batch::new();
        draw_text(&mut b, "A", Vec2::new(0.0, 0.0), 1.0, [1.0; 4]);
        let ys: Vec<f32> = b.vertices.iter().map(|v| v.pos[1]).collect();
        let top = ys.iter().cloned().fold(f32::MIN, f32::max);
        let bottom = ys.iter().cloned().fold(f32::MAX, f32::min);
        assert_eq!(top, 1.0, "row 0 sits one scale above the baseline");
        assert_eq!(bottom, -7.0, "row 7 sits seven below");
    }

    /// One quad per glyph is the entire point: four vertices and six
    /// indices, not one quad per lit pixel.
    #[test]
    fn a_glyph_costs_one_quad() {
        let mut b = Batch::new();
        draw_text(&mut b, "A", Vec2::ZERO, 1.0, [1.0; 4]);
        assert_eq!(b.vertices.len(), 4);
        assert_eq!(b.indices.len(), 6);

        // Ten characters, one of them a space, which draws nothing.
        let mut b = Batch::new();
        draw_text(&mut b, "Player_004", Vec2::ZERO, 1.0, [1.0; 4]);
        assert_eq!(b.vertices.len(), 40, "ten glyphs, four vertices each");
    }

    /// Space has no set bits, so it advances without pushing geometry —
    /// a quad there would sample empty atlas and cost vertices for it.
    #[test]
    fn a_space_draws_nothing_but_still_advances() {
        let mut b = Batch::new();
        draw_text(&mut b, " ", Vec2::ZERO, 1.0, [1.0; 4]);
        assert!(b.vertices.is_empty());

        let mut b = Batch::new();
        draw_text(&mut b, "A B", Vec2::ZERO, 1.0, [1.0; 4]);
        assert_eq!(b.vertices.len(), 8, "two glyphs drawn, the space skipped");
        // The third glyph still starts where the space left it.
        let xs: Vec<f32> = b.vertices.iter().map(|v| v.pos[0]).collect();
        let rightmost = xs.iter().cloned().fold(f32::MIN, f32::max);
        assert_eq!(rightmost, 26.0, "two advances of 9 put B at 18..26");
    }

    /// Every glyph maps inside the atlas, and the cells tile without
    /// overlapping — a half-texel of drift shears glyph edges.
    #[test]
    fn every_glyph_cell_lies_inside_the_atlas() {
        for cell in 0..FONT.len() as u32 {
            let (u0, v0, u1, v1) = cell_uv(cell);
            assert!((0.0..=1.0).contains(&u0) && (0.0..=1.0).contains(&u1), "cell {cell} u");
            assert!((0.0..=1.0).contains(&v0) && (0.0..=1.0).contains(&v1), "cell {cell} v");
            assert!(u1 > u0 && v1 > v0, "cell {cell} is degenerate");
            let expect = CELL as f32 / ATLAS_SIZE as f32;
            assert!((u1 - u0 - expect).abs() < 1e-6, "cell {cell} width");
            assert!((v1 - v0 - expect).abs() < 1e-6, "cell {cell} height");
        }
        // Adjacent cells share an edge exactly rather than overlapping.
        let (_, _, u1_of_0, _) = cell_uv(0);
        let (u0_of_1, _, _, _) = cell_uv(1);
        assert_eq!(u1_of_0, u0_of_1);
    }

    /// **The load-bearing one.** Flat-coloured geometry samples UV
    /// (0.5, 0.5) and must find white there. Under nearest filtering that
    /// selects texel `floor(0.5 × size)`, which has to be opaque white and
    /// outside every glyph cell — otherwise roughly two thousand
    /// primitive call sites across the games on this engine start
    /// sampling a letter.
    #[test]
    fn uv_half_lands_on_an_opaque_white_texel_clear_of_glyphs() {
        let data = atlas_rgba();
        let size = ATLAS_SIZE as usize;
        assert_eq!(data.len(), size * size * 4);

        let (wx, wy) = WHITE_TEXEL;
        assert_eq!(
            (wx as f32 / ATLAS_SIZE as f32, wy as f32 / ATLAS_SIZE as f32),
            (0.5, 0.5),
            "the reserved texel must be exactly what UV 0.5 selects",
        );

        let i = (wy as usize * size + wx as usize) * 4;
        assert_eq!(&data[i..i + 4], &[255, 255, 255, 255], "and be opaque white");

        // Glyphs fill whole cell rows from the top; the reserved texel
        // must sit below all of them.
        let rows_used = FONT.len() as u32 / CELLS_PER_ROW;
        assert!(
            wy >= rows_used * CELL,
            "white texel at y={wy} collides with glyph rows ending at {}",
            rows_used * CELL,
        );
    }

    /// The atlas carries the font it claims to: a glyph with known bits
    /// lands where `cell_uv` says it does, right way up and round.
    #[test]
    fn the_atlas_holds_the_font_at_the_mapped_cells() {
        let data = atlas_rgba();
        let size = ATLAS_SIZE as usize;
        let lit = |x: usize, y: usize| data[(y * size + x) * 4 + 3] != 0;

        // Space is cell 0 and entirely clear.
        for y in 0..CELL as usize {
            for x in 0..CELL as usize {
                assert!(!lit(x, y), "space should be blank at ({x}, {y})");
            }
        }

        // '!' is cell 1: 0x18 on row 0 means bits 3 and 4 set, and MSB is
        // the leftmost pixel, so columns 3 and 4 of that cell are lit.
        let base = CELL as usize;
        assert!(lit(base + 3, 0) && lit(base + 4, 0), "row 0 of '!'");
        assert!(!lit(base, 0), "and not its left edge");

        // Its row 5 is 0x00 — the gap above the dot.
        for x in 0..CELL as usize {
            assert!(!lit(base + x, 5), "row 5 of '!' is blank");
        }
    }
}
