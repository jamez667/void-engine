//! Generic 2D sector-address type + world→sector wrapping.
//!
//! A sector is a fixed-size square tile of world space. A `Sector2D`
//! identifies one tile by integer `(x, y)`. As an entity's local
//! position drifts past ±SECTOR_SIZE/2 the caller wraps it back into
//! range and bumps the sector coordinate — same trick as origin
//! rebasing but at a coarser granularity.
//!
//! Game-side sector types (with interior/station flags, roles, etc.)
//! wrap this primitive. See `void_sim::sector::SectorAddr` for the
//! shipping game type.

use glam::DVec2;

/// Bare 2D integer sector address. No interior/exterior flag baked in —
/// game-side code can steal the sign bit or high bits of `x` for tags.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Sector2D {
    pub x: i64,
    pub y: i64,
}

/// Wrap a local position back into `±sector_size/2` and bump the sector
/// coordinate for each full sector crossed. Both axes independently.
///
/// Caller supplies the sector size (world units per sector). The check
/// runs unconditionally — callers that need to skip wrapping (e.g. for
/// interior sectors that keep pos in tile coords) should short-circuit
/// before calling.
pub fn wrap_pos(pos: &mut DVec2, sx: &mut i64, sy: &mut i64, sector_size: f64) {
    let half = sector_size * 0.5;
    while pos.x >  half { pos.x -= sector_size; *sx += 1; }
    while pos.x < -half { pos.x += sector_size; *sx -= 1; }
    while pos.y >  half { pos.y -= sector_size; *sy += 1; }
    while pos.y < -half { pos.y += sector_size; *sy -= 1; }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wraps_positive_x_and_bumps_sector() {
        let mut p = DVec2::new(1500.0, 0.0);
        let (mut sx, mut sy) = (0i64, 0i64);
        wrap_pos(&mut p, &mut sx, &mut sy, 1000.0);
        assert_eq!(sx, 1);
        assert_eq!(sy, 0);
        assert!(p.x > -500.0 && p.x <= 500.0);
    }

    #[test]
    fn wraps_negative_y() {
        let mut p = DVec2::new(0.0, -1500.0);
        let (mut sx, mut sy) = (0i64, 0i64);
        wrap_pos(&mut p, &mut sx, &mut sy, 1000.0);
        assert_eq!(sy, -1);
    }

    #[test]
    fn in_range_no_change() {
        let mut p = DVec2::new(10.0, -10.0);
        let (mut sx, mut sy) = (5i64, -3i64);
        wrap_pos(&mut p, &mut sx, &mut sy, 1000.0);
        assert_eq!((sx, sy), (5, -3));
    }
}
