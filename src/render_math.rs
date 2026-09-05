//! Small stateless helpers used by every render module — viewport
//! culling, LOD segment count, colour + scalar lerp. Kept out of the
//! renderer proper so they can be `use`d without pulling in wgpu types.

use glam::DVec2;

/// True if a circle at world position `pos` with `radius` overlaps the
/// axis-aligned view box `[view_min, view_max]`.
#[inline]
pub fn in_view(pos: DVec2, radius: f32, view_min: DVec2, view_max: DVec2) -> bool {
    let r = radius as f64;
    pos.x + r >= view_min.x && pos.x - r <= view_max.x
        && pos.y + r >= view_min.y && pos.y - r <= view_max.y
}

/// Choose a circle-segment count for LOD based on the object's on-screen
/// size and distance. Falls off with distance past 150 px, clamped to
/// the caller-supplied `[min_segs, max_segs]`.
#[inline]
pub fn lod_segments(
    world_radius: f32,
    dist_from_camera: f32,
    zoom: f32,
    min_segs: u32,
    max_segs: u32,
) -> u32 {
    let screen_px   = world_radius * zoom;
    let dist_px     = dist_from_camera * zoom;
    let dist_factor = (1.0 - (dist_px - 150.0).max(0.0) / 500.0).clamp(0.0, 1.0);
    ((screen_px * 0.6 * dist_factor) as u32).clamp(min_segs, max_segs)
}

/// Component-wise linear interpolation between two RGBA colours.
pub fn lerp_color(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    std::array::from_fn(|i| a[i] + (b[i] - a[i]) * t)
}

/// Scalar linear interpolation.
pub fn lerp_f32(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }
