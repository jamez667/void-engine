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
/// # Why this is arithmetic and not a loop
///
/// This subtracted one sector per iteration until 2026-09-11, which is
/// correct for the one-crossing case every caller actually has and
/// unbounded for anything else. A teleport, a bad spawn, or a physics
/// blow-up puts a single entity into a loop the tick cannot leave:
/// measured at sector size 1000, `pos.x = 1e9` cost **0.368 ms** and
/// `pos.x = 1e12` cost **381 ms** — one entity stalling a 60 Hz tick for
/// twenty-three frames.
///
/// An infinite coordinate never terminated at all: `inf > half` stays
/// true and `inf - sector_size` is still `inf`. A NaN exited immediately
/// instead, since every comparison against NaN is false — so the two
/// non-finite cases failed in opposite directions, one hanging and one
/// silently passing NaN through to the sector address.
///
/// `div_euclid` does the whole job in constant time. Non-finite input is
/// refused rather than propagated: a NaN sector coordinate is not a
/// position the caller can recover from, and `as i64` on a NaN is 0,
/// which would silently teleport the entity to the origin sector.
pub fn wrap_pos(pos: &mut DVec2, sx: &mut i64, sy: &mut i64, sector_size: f64) {
    debug_assert!(sector_size > 0.0, "sector size must be positive, got {sector_size}");
    // `is_nan() || <= 0.0` rather than `!(> 0.0)`: same set, but a negated
    // comparison on a partially-ordered type reads as though NaN were an
    // afterthought when it is the case that matters.
    if !pos.x.is_finite() || !pos.y.is_finite() || sector_size.is_nan() || sector_size <= 0.0 {
        return;
    }
    let half = sector_size * 0.5;
    // Transcribed from the two loops rather than re-derived, because the
    // obvious single-expression forms all get this wrong.
    //
    // The loops run in *sequence*: `while v > half` first, then
    // `while v < -half`. So they are not two halves of one symmetric
    // operation — a value below `-half` skips the first loop entirely.
    // Both `+half` and `-half` are fixed points, making the resting
    // window `[-half, half]`, closed at both ends.
    //
    // Worked: at size 1000, `v = 1500` takes one step down to exactly
    // `+500` and stops (`500 > 500` is false), giving `k = 1`.
    // `v = -1500` takes one step up to `-500`, giving `k = -1`. A single
    // `floor((v + half) / size)` returns 2 for the first; a `ceil() - 1`
    // returns -2 for the second. Three cases is what the loops actually
    // are.
    let wrap = |v: f64, s: &mut i64| -> f64 {
        let crossed = if v > half {
            ((v - half) / sector_size).ceil()
        } else if v < -half {
            -(((-v) - half) / sector_size).ceil()
        } else {
            0.0
        };
        *s += crossed as i64;
        v - crossed * sector_size
    };
    pos.x = wrap(pos.x, sx);
    pos.y = wrap(pos.y, sy);
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

    // ── the arithmetic rewrite ───────────────────────────────────────
    //
    // This was four `while` loops until 2026-09-11. Replacing them with
    // arithmetic took three wrong attempts, every one of which passed
    // *some* of the two tests above — so these pin the exact behaviour
    // the loops had, not an approximation of it.

    /// Both boundaries are fixed points: the loops stop at `> half` and
    /// `< -half`, so a position sitting exactly on either stays put.
    #[test]
    fn both_boundaries_are_fixed_points() {
        for v in [500.0, -500.0] {
            let mut p = DVec2::new(v, v);
            let (mut sx, mut sy) = (7i64, 7i64);
            wrap_pos(&mut p, &mut sx, &mut sy, 1000.0);
            assert_eq!((sx, sy), (7, 7), "pos {v} must not cross a sector");
            assert_eq!((p.x, p.y), (v, v), "pos {v} must not move");
        }
    }

    /// The asymmetry that broke every single-expression attempt: the two
    /// loops run in sequence, so `-1500` skips the first one entirely.
    #[test]
    fn positive_and_negative_cross_by_the_same_count() {
        let mut p = DVec2::new(1500.0, -1500.0);
        let (mut sx, mut sy) = (0i64, 0i64);
        wrap_pos(&mut p, &mut sx, &mut sy, 1000.0);
        assert_eq!((sx, sy), (1, -1));
        assert_eq!((p.x, p.y), (500.0, -500.0));
    }

    /// Many sectors in one call, which is the case the loops made O(n).
    #[test]
    fn a_far_position_wraps_in_one_step() {
        let mut p = DVec2::new(1_000_000_000.0, -1_000_000_000.0);
        let (mut sx, mut sy) = (0i64, 0i64);
        wrap_pos(&mut p, &mut sx, &mut sy, 1000.0);
        assert_eq!((sx, sy), (1_000_000, -1_000_000));
        assert!(p.x.abs() <= 500.0 && p.y.abs() <= 500.0, "got {p:?}");
    }

    /// An infinite coordinate used to loop forever: `inf > half` stays
    /// true and `inf - size` is still `inf`. A NaN exited immediately
    /// instead, since every comparison against it is false — so the two
    /// non-finite cases failed in opposite directions.
    #[test]
    fn non_finite_positions_are_refused_rather_than_propagated() {
        for bad in [f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let mut p = DVec2::new(bad, 0.0);
            let (mut sx, mut sy) = (3i64, 4i64);
            wrap_pos(&mut p, &mut sx, &mut sy, 1000.0);
            assert_eq!((sx, sy), (3, 4), "{bad} must not move the sector address");
        }
    }

    /// Every wrap must land inside the resting window, whatever it started
    /// at — the property the loops guaranteed by construction and an
    /// arithmetic form has to be checked for.
    #[test]
    fn every_result_lands_in_the_resting_window() {
        let size = 256.0;
        let half = size * 0.5;
        for i in -50i32..=50 {
            let v = i as f64 * 37.5;
            let mut p = DVec2::new(v, -v);
            let (mut sx, mut sy) = (0i64, 0i64);
            wrap_pos(&mut p, &mut sx, &mut sy, size);
            assert!(p.x >= -half && p.x <= half, "x {} from {v}", p.x);
            assert!(p.y >= -half && p.y <= half, "y {} from {v}", p.y);
            // And the sector bump accounts for exactly the distance moved.
            assert!(
                (v - (p.x + sx as f64 * size)).abs() < 1e-9,
                "x lost track: {v} -> pos {} sector {sx}",
                p.x,
            );
        }
    }
}
