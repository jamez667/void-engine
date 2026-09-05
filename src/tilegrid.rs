//! Generic 2D tile grid — a flat `Vec<T>` sized `w*h` with cached
//! dimensions and the "grid centered on origin, row 0 = top" geometry
//! that the ships/stations in this project all share.
//!
//! Extracted from `void_sim::station_interior::Floor` (and the
//! `void_sim::module::Module` inner grid, which shares the same shape).
//! Callers that already stored `grid: Vec<T>`, `cached_w`, `cached_h`
//! now delegate the arithmetic here and keep only their game overlays
//! (apartment ownership, ACL doors, module composites, ...).
//!
//! Coord conventions:
//! - `w` / `h` in tiles are `u32` (never negative, and the grid backing
//!   store is sized from them).
//! - Column / row reads take `i32` so callers can pass unclamped
//!   values from the collision / pathfinding hot paths without a
//!   pre-check. Out-of-bounds `tile_at` returns `T::default()` — for
//!   `TileKind` that's `Empty`, which is what every caller already
//!   treated an off-grid read as.
//! - `set` is bounds-checked and silently no-ops on out-of-range
//!   coords, matching `Floor::set_tile`.

use glam::DVec2;

pub use crate::tile_collide::{pos_to_tile_default, tile_center_default};

/// Owned tile grid + cached dimensions. `T` is the per-tile value —
/// usually an enum like `TileKind` — and needs to be `Copy + Default`
/// so out-of-bounds reads can synthesise a safe fill without touching
/// the backing store.
#[derive(Clone, Debug, Default)]
pub struct TileGrid<T: Copy + Default> {
    grid: Vec<T>,
    w: u32,
    h: u32,
}

impl<T: Copy + Default> TileGrid<T> {
    /// Empty grid — no storage, `dims() == (0, 0)`. Callers rebuild it
    /// from source data via [`TileGrid::rebuild_from_rows`] or by
    /// filling with [`TileGrid::new_filled`].
    pub fn empty() -> Self { Self { grid: Vec::new(), w: 0, h: 0 } }

    /// Allocate `w*h` tiles all initialised to `val`. `T: Default` is
    /// only required at the type level; `val` can be any value.
    pub fn new_filled(w: u32, h: u32, val: T) -> Self {
        Self { grid: vec![val; (w * h) as usize], w, h }
    }

    /// (Width, height) in tiles.
    #[inline]
    pub fn dims(&self) -> (u32, u32) { (self.w, self.h) }

    /// Width in tiles.
    #[inline]
    pub fn width(&self) -> u32 { self.w }

    /// Height in tiles.
    #[inline]
    pub fn height(&self) -> u32 { self.h }

    /// True when `dims()` is `(0, 0)`. Cheap sentinel used by fixture
    /// code paths that construct a value via `..Default::default()`
    /// and never call `rebuild_from_rows`.
    #[inline]
    pub fn is_empty(&self) -> bool { self.grid.is_empty() }

    /// Read at `(col, row)`. Out-of-bounds (including negative) reads
    /// return `T::default()` so callers can walk past the edge without
    /// a pre-check — collision, pathfinding, and neighbour scans rely
    /// on this.
    #[inline]
    pub fn tile_at(&self, col: i32, row: i32) -> T {
        if col < 0 || row < 0 || self.grid.is_empty() { return T::default(); }
        let (c, r) = (col as u32, row as u32);
        if c >= self.w || r >= self.h { return T::default(); }
        self.grid[(r * self.w + c) as usize]
    }

    /// True if `(col, row)` is inside the grid bounds.
    #[inline]
    pub fn in_bounds(&self, col: i32, row: i32) -> bool {
        col >= 0 && row >= 0 && (col as u32) < self.w && (row as u32) < self.h
    }

    /// Write at `(col, row)`. Silently no-ops when out of bounds; the
    /// editor's `set_tile` mutator relies on this to be safe against
    /// stray clicks past the grid edge.
    #[inline]
    pub fn set(&mut self, col: i32, row: i32, val: T) {
        if !self.in_bounds(col, row) || self.grid.is_empty() { return; }
        let idx = (row as u32 * self.w + col as u32) as usize;
        self.grid[idx] = val;
    }

    /// Fill every tile with `val`, keeping the current dimensions.
    pub fn fill(&mut self, val: T) {
        for c in self.grid.iter_mut() { *c = val; }
    }

    /// Borrow the flat backing buffer, row-major `r*w + c`.
    #[inline]
    pub fn as_slice(&self) -> &[T] { &self.grid }

    /// Iterate every cell as `(col, row, val)` in row-major scan order.
    pub fn iter_cells(&self) -> impl Iterator<Item = (u32, u32, T)> + '_ {
        let w = self.w;
        self.grid.iter().enumerate().map(move |(i, &v)| {
            let i = i as u32;
            (i % w, i / w, v)
        })
    }

    /// Rebuild storage from a row-major provider. `w` / `h` are set to
    /// the passed dims and every cell is filled by calling `read(c, r)`.
    /// The standard use is decoding a `Vec<String>` of glyphs into
    /// enum tiles — pass a closure that indexes into the strings, or
    /// prefer [`TileGrid::rebuild_from_glyphs`] which handles that
    /// pattern directly.
    pub fn rebuild_from_rows<F>(&mut self, w: u32, h: u32, mut read: F)
    where
        F: FnMut(u32, u32) -> T,
    {
        self.w = w;
        self.h = h;
        let mut buf = Vec::with_capacity((w * h) as usize);
        for r in 0..h {
            for c in 0..w {
                buf.push(read(c, r));
            }
        }
        self.grid = buf;
    }

    /// Rebuild from a `&[String]` of glyph rows via a `char -> T`
    /// decoder. Dims are derived from the string vec (height = row
    /// count, width = first row's byte length). Bytes past a short
    /// row and unmapped chars fall through to `T::default()`. Every
    /// tile-file loader in the project (station floors, ship
    /// interiors, module tiles) shares this exact decode path.
    pub fn rebuild_from_glyphs<F>(&mut self, tiles: &[String], mut decode: F)
    where
        F: FnMut(char) -> T,
    {
        let h = tiles.len() as u32;
        let w = tiles.first().map(|r| r.len() as u32).unwrap_or(0);
        self.rebuild_from_rows(w, h, |c, r| {
            tiles.get(r as usize)
                .and_then(|row| row.as_bytes().get(c as usize).copied())
                .map(|b| decode(b as char))
                .unwrap_or_default()
        });
    }

    /// Read a `(col, row)` from a `&[String]` of glyph rows without a
    /// prebuilt grid. Returns `T::default()` for out-of-range or short
    /// rows. Used by the `tile_at` fallback path on fresh-from-`serde`
    /// instances where `rebuild_from_glyphs` hasn't run yet.
    pub fn tile_at_glyphs<F>(tiles: &[String], col: i32, row: i32, mut decode: F) -> T
    where
        F: FnMut(char) -> T,
    {
        if col < 0 || row < 0 { return T::default(); }
        tiles.get(row as usize)
            .and_then(|s| s.as_bytes().get(col as usize).copied())
            .map(|b| decode(b as char))
            .unwrap_or_default()
    }

    /// World-space centre of tile `(col, row)`. Standard "grid centered
    /// on origin, row 0 at top" layout. Forwards to
    /// [`tile_center_default`] — kept as a method so callers reach it
    /// through the grid struct without a separate import.
    #[inline]
    pub fn tile_center(&self, col: u32, row: u32, tile_size_m: f32) -> DVec2 {
        tile_center_default(col as i32, row as i32, self.w, self.h, tile_size_m)
    }

    /// World-space position → `(col, row)`. Inverse of
    /// [`TileGrid::tile_center`]. Caller should range-check.
    #[inline]
    pub fn pos_to_tile(&self, pos: DVec2, tile_size_m: f32) -> (i32, i32) {
        pos_to_tile_default(pos, self.w, self.h, tile_size_m)
    }
}

/// Rotate a row-major `w * h` tile buffer by `rot` quarter-turns
/// clockwise (taken modulo 4), returning `(new_w, new_h, tiles)`.
///
/// Odd quarter-turns transpose the dimensions. `rot == 0` is a straight
/// copy. Reads go through `src`, a `(col, row) -> T` closure, rather
/// than a slice, so callers whose canonical storage is glyph rows or a
/// lazily-populated cache can rotate without materialising a flat buffer
/// first.
///
/// Extracted because `void_sim::module::Module::rotated_tiles` and the
/// tilemap editor's `Clipboard::rotated` carried byte-identical copies
/// of this kernel, and a divergence between them would silently desync
/// the editor preview from the runtime composite.
pub fn rotate_tiles<T, F>(w: u32, h: u32, rot: u8, mut src: F) -> (u32, u32, Vec<T>)
where
    T: Copy + Default,
    F: FnMut(u32, u32) -> T,
{
    let rot = rot % 4;
    let (sw, sh) = (w as usize, h as usize);
    let (nw, nh) = match rot {
        1 | 3 => (sh, sw),
        _     => (sw, sh),
    };
    let mut out = vec![T::default(); nw * nh];
    for r in 0..sh {
        for c in 0..sw {
            let val = src(c as u32, r as u32);
            let (nc, nr) = match rot {
                0 => (c, r),
                1 => (sh - 1 - r, c),          // 90° CW
                2 => (sw - 1 - c, sh - 1 - r), // 180°
                3 => (r, sw - 1 - c),          // 270° CW (= 90° CCW)
                _ => unreachable!(),
            };
            out[nr * nw + nc] = val;
        }
    }
    (nw as u32, nh as u32, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Copy, Clone, Debug, Default, PartialEq)]
    enum Kind {
        #[default] Empty,
        Wall,
        Floor,
    }

    #[test]
    fn rotate_tiles_turns_clockwise_and_transposes() {
        // 3x2 grid, distinct values so orientation is unambiguous:
        //   1 2 3
        //   4 5 6
        let src = |c: u32, r: u32| -> u8 { (r * 3 + c) as u8 + 1 };

        // rot 0 is identity, dims unchanged.
        let (w, h, t) = rotate_tiles(3, 2, 0, src);
        assert_eq!((w, h), (3, 2));
        assert_eq!(t, vec![1, 2, 3, 4, 5, 6]);

        // 90° CW → 2x3:
        //   4 1
        //   5 2
        //   6 3
        let (w, h, t) = rotate_tiles(3, 2, 1, src);
        assert_eq!((w, h), (2, 3));
        assert_eq!(t, vec![4, 1, 5, 2, 6, 3]);

        // 180° → 3x2, fully reversed.
        let (w, h, t) = rotate_tiles(3, 2, 2, src);
        assert_eq!((w, h), (3, 2));
        assert_eq!(t, vec![6, 5, 4, 3, 2, 1]);

        // 270° CW → 2x3, the mirror of the 90° case.
        let (w, h, t) = rotate_tiles(3, 2, 3, src);
        assert_eq!((w, h), (2, 3));
        assert_eq!(t, vec![3, 6, 2, 5, 1, 4]);

        // Four quarter-turns returns to the original.
        let mut cur = (3u32, 2u32, vec![1u8, 2, 3, 4, 5, 6]);
        for _ in 0..4 {
            let (cw, _, ref ct) = cur;
            let snapshot = ct.clone();
            let width = cw;
            cur = rotate_tiles(cur.0, cur.1, 1, |c, r| snapshot[(r * width + c) as usize]);
        }
        assert_eq!(cur, (3, 2, vec![1, 2, 3, 4, 5, 6]));

        // rot wraps modulo 4.
        assert_eq!(rotate_tiles(3, 2, 5, src), rotate_tiles(3, 2, 1, src));
    }

    #[test]
    fn empty_grid_reads_default() {
        let g: TileGrid<Kind> = TileGrid::empty();
        assert_eq!(g.dims(), (0, 0));
        assert_eq!(g.tile_at(0, 0), Kind::Empty);
        assert_eq!(g.tile_at(-1, -1), Kind::Empty);
    }

    #[test]
    fn new_filled_and_read_write() {
        let mut g = TileGrid::new_filled(3, 2, Kind::Floor);
        assert_eq!(g.dims(), (3, 2));
        assert_eq!(g.tile_at(1, 1), Kind::Floor);
        g.set(1, 1, Kind::Wall);
        assert_eq!(g.tile_at(1, 1), Kind::Wall);
    }

    #[test]
    fn out_of_bounds_reads_default_writes_noop() {
        let mut g = TileGrid::new_filled(2, 2, Kind::Floor);
        assert_eq!(g.tile_at(5, 5), Kind::Empty);
        assert_eq!(g.tile_at(-1, 0), Kind::Empty);
        g.set(5, 5, Kind::Wall);
        assert_eq!(g.tile_at(5, 5), Kind::Empty);
        assert_eq!(g.tile_at(1, 1), Kind::Floor);
    }

    #[test]
    fn rebuild_from_rows_populates_grid() {
        let rows = ["Wf", "fW"];
        let mut g: TileGrid<Kind> = TileGrid::empty();
        g.rebuild_from_rows(2, 2, |c, r| {
            match rows[r as usize].as_bytes()[c as usize] as char {
                'W' => Kind::Wall,
                'f' => Kind::Floor,
                _ => Kind::Empty,
            }
        });
        assert_eq!(g.tile_at(0, 0), Kind::Wall);
        assert_eq!(g.tile_at(1, 0), Kind::Floor);
        assert_eq!(g.tile_at(0, 1), Kind::Floor);
        assert_eq!(g.tile_at(1, 1), Kind::Wall);
    }

    #[test]
    fn iter_cells_row_major() {
        let mut g = TileGrid::new_filled(2, 2, Kind::Empty);
        g.set(0, 0, Kind::Wall);
        g.set(1, 1, Kind::Floor);
        let cells: Vec<_> = g.iter_cells().collect();
        assert_eq!(cells, vec![
            (0, 0, Kind::Wall),
            (1, 0, Kind::Empty),
            (0, 1, Kind::Empty),
            (1, 1, Kind::Floor),
        ]);
    }

    #[test]
    fn tile_center_matches_floor_layout() {
        let g = TileGrid::<Kind>::new_filled(3, 3, Kind::Floor);
        // Center of grid = origin: tile (1, 1) center is (0, 0).
        let c = g.tile_center(1, 1, 1.0);
        assert!((c.x).abs() < 1e-9 && (c.y).abs() < 1e-9);
    }
}
