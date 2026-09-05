//! Circle-vs-tile push resolution for a top-down walker on a
//! grid-aligned floor.
//!
//! Two-pass sweep across a 5×5 neighbourhood around the character:
//! each blocking tile pushes the circle out along its shortest axis.
//! Two passes give stable corner behaviour without recursion.
//!
//! The floor geometry assumed here matches [`tile_center_default`]:
//! centered on origin, row 0 at top (highest y), grid width×height
//! tiles of `tile_size_m` metres each. Callers whose floors match this
//! convention (`void_sim::station_interior::Floor`) can use
//! [`tile_collide`] directly. Callers with a different tile layout can
//! call the pushing kernel themselves and supply their own tile-center
//! function — see the source.

use glam::DVec2;

/// Standard "grid centered on origin, row 0 = top" tile-center formula.
/// Matches `Floor::tile_center` in the shipping game.
#[inline]
pub fn tile_center_default(col: i32, row: i32, width: u32, height: u32, tile_size_m: f32) -> DVec2 {
    let w = width as f64;
    let h = height as f64;
    let s = tile_size_m as f64;
    DVec2::new(
        (col as f64 - w * 0.5 + 0.5) * s,
        ((h - 1.0) * 0.5 - row as f64) * s,
    )
}

/// Standard "grid centered on origin" tile-index lookup for a world
/// position. Inverse of [`tile_center_default`].
#[inline]
pub fn pos_to_tile_default(pos: DVec2, width: u32, height: u32, tile_size_m: f32) -> (i32, i32) {
    let w = width as f64;
    let h = height as f64;
    let s = tile_size_m as f64;
    let col = (pos.x / s + w * 0.5).floor() as i32;
    let row = ((h - 1.0) * 0.5 - pos.y / s + 0.5).floor() as i32;
    (col, row)
}

/// Two-pass circle-vs-blocking-tile push. `blocks(col, row)` returns
/// true for any tile the walker cannot enter — the caller layers plain
/// wall/blocking-tile checks and any per-walker ACL (locked doors, pad
/// ownership, etc.) into a single predicate.
///
/// `dims = (width, height)` and `tile_size_m` are used to derive the
/// tile centers. Assumes the standard grid-centered-on-origin layout;
/// call the tile-center formula in [`tile_center_default`] to match.
pub fn tile_collide<F>(
    pos: &mut DVec2,
    char_radius: f64,
    dims: (u32, u32),
    tile_size_m: f32,
    blocks: F,
) where
    F: Fn(i32, i32) -> bool,
{
    let (width, height) = dims;
    let tile_s = tile_size_m as f64;
    let half = tile_s * 0.5;
    for _ in 0..2 {
        let (cx, cy) = pos_to_tile_default(*pos, width, height, tile_size_m);
        for dy in -2..=2 {
            for dx in -2..=2 {
                let col = cx + dx;
                let row = cy + dy;
                if !blocks(col, row) { continue; }
                let center = tile_center_default(col, row, width, height, tile_size_m);

                let closest = DVec2::new(
                    pos.x.clamp(center.x - half, center.x + half),
                    pos.y.clamp(center.y - half, center.y + half),
                );
                let delta = *pos - closest;
                let dist2 = delta.length_squared();
                if dist2 == 0.0 {
                    let dx_axis = pos.x - center.x;
                    let dy_axis = pos.y - center.y;
                    if dx_axis.abs() > dy_axis.abs() {
                        pos.x = center.x + dx_axis.signum() * (half + char_radius);
                    } else {
                        pos.y = center.y + dy_axis.signum() * (half + char_radius);
                    }
                    continue;
                }
                if dist2 < char_radius * char_radius {
                    let dist = dist2.sqrt();
                    let push = char_radius - dist;
                    let dir = delta / dist;
                    *pos += dir * push;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_op_in_open_space() {
        let mut p = DVec2::new(0.1, 0.2);
        tile_collide(&mut p, 0.4, (5, 5), 1.0, |_c, _r| false);
        assert!((p - DVec2::new(0.1, 0.2)).length() < 1e-9);
    }

    #[test]
    fn pushes_off_wall_tile() {
        // 5x5 floor, only col=0 blocks. Character starts inside col 1
        // pressed against the col-0 wall.
        let mut p = DVec2::new(-1.4, 0.0);
        tile_collide(&mut p, 0.4, (5, 5), 1.0, |c, _r| c == 0);
        // col 0 center = (-2, 0), right edge = -1.5, so radius=0.4 must sit
        // at x >= -1.5 + 0.4 = -1.1.
        assert!(p.x >= -1.1 - 1e-6, "x = {}", p.x);
    }
}
