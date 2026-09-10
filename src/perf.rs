//! Frame-timing rollup, shared between the loop that measures it and the
//! renderer that surfaces it.
//!
//! This lives in its own module rather than in `app` because `renderer`
//! stores a `PerfSnapshot` (so `App::render` can draw an F3 overlay showing
//! exactly what the `[perf]` log line says). With the type in `app`, the
//! dependency ran `renderer -> app`, and `app` is one of the modules that
//! has to survive with the `client` feature off. A headless build has no
//! renderer, so the cycle would have forced `PerfSnapshot` to be
//! conditionally compiled purely to satisfy a module it does not use.

/// Per-second rollup of render-loop timings + vertex count. Published on
/// `Renderer::last_perf` after each rollup tick (~1 Hz) so `App::render`
/// can surface the same numbers the `[perf]` log line shows. Defaults to
/// zero on the first frame.
///
/// Every field is milliseconds except `fps` and `vertex_count`.
#[derive(Default, Clone, Copy, Debug)]
pub struct PerfSnapshot {
    pub fps: f32,
    pub avg_frame_ms: f32,
    pub p50_frame_ms: f32,
    pub p95_frame_ms: f32,
    pub p99_frame_ms: f32,
    pub worst_frame_ms: f32,
    pub avg_update_ms: f32,
    pub avg_batch_ms: f32,
    pub avg_present_ms: f32,
    pub vertex_count: u32,
}
