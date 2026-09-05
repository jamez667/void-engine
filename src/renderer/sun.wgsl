// Global directional-sun pass.
//
// The sun sits at infinity — parallel rays. For each screen pixel we
// march toward the sun through the `wall_mask` texture (the same
// occluder mask the on-foot light pass uses). If any tap hits a wall,
// the pixel is in shadow and receives no sun contribution. If the ray
// exits the mask without hitting anything, the pixel is lit and picks
// up `color * intensity`.
//
// Wall pixels themselves are handled specially: if the pixel is INSIDE
// a wall we still let it read as sunlit at a reduced fraction, so
// outer wall tiles glow on the sunlit side rather than reading as
// pitch-black silhouettes. Walls with more wall behind them (i.e. the
// march immediately hits another wall) still resolve to shadow.
//
// The composite runs additive on top of the main pass, so the shader
// output is just the sun contribution (or black in shadow).

struct SunUniform {
    // Per-step delta in UV space (toward the sun). Aspect-corrected
    // by the caller so shapes stay right regardless of surface size.
    sun_dir_uv: vec2<f32>,
    // Number of march steps (encoded as f32).
    taps:       f32,
    // Multiplier applied to `color * intensity` when the lit pixel
    // sits inside a wall (so outer walls glow half-strength rather
    // than jumping out at full sun).
    wall_lit_scale: f32,
    // Sun colour in linear RGB (already tinted daylight-warm).
    color:      vec3<f32>,
    // Peak intensity for a fully lit pixel.
    intensity:  f32,
    // Frame time (seconds) so dust motes can drift.
    time:       f32,
    _pad0:      f32,
    _pad1:      f32,
    _pad2:      f32,
};

@group(0) @binding(0) var<uniform> u:      SunUniform;
@group(0) @binding(1) var          t_mask: texture_2d<f32>;
@group(0) @binding(2) var          s_mask: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
};

// Rim-light depth in march-taps. Wall pixels within this many taps of
// their sun-facing edge get progressively brighter; anything deeper
// stays dark. Kept small so only a thin edge on the outer wall glows.
const RIM_TAPS: i32 = 3;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    var xs = array<f32, 3>(-1.0,  3.0, -1.0);
    var ys = array<f32, 3>(-1.0, -1.0,  3.0);
    var us = array<f32, 3>( 0.0,  2.0,  0.0);
    var vs = array<f32, 3>( 1.0,  1.0, -1.0); // flip: uv.y = 0 at top
    let i = i32(vid);
    out.pos = vec4<f32>(xs[i], ys[i], 0.0, 1.0);
    out.uv  = vec2<f32>(us[i], vs[i]);
    return out;
}

// wall_mask coverage vs. the on-screen surface. Must match
// `shadow::MASK_SCALE`. All wall_mask samples in this shader run through
// `surface_uv_to_mask_uv` first because the mask is larger than the
// surface (so off-screen walls still block sun rays that leave the
// viewport before they'd hit the sun).
const MASK_SCALE: f32 = 2.0;

fn surface_uv_to_mask_uv(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(0.5, 0.5) + (uv - vec2<f32>(0.5, 0.5)) * (1.0 / MASK_SCALE);
}

@fragment
fn fs_sun(in: VsOut) -> @location(0) vec4<f32> {
    // Translate fragment UV (surface space) into wall_mask UV. All
    // subsequent samples of `t_mask` in this fragment use mask-UV.
    // `u.sun_dir_uv` is already pre-scaled to mask-UV per step.
    let start_uv = surface_uv_to_mask_uv(in.uv);
    // Mask channels: .r = opaque wall, .g = window (transparent). Empty
    // tiles have both channels 0.
    let here = textureSample(t_mask, s_mask, start_uv);

    // 1. Window pixel: lit only if the sun ray from this window's tile
    // reaches OPEN SPACE without hitting a solid wall. That means the
    // window is on the sun-facing outer hull of the station. Windows on
    // the shaded side will hit an outer wall on their way to the sun.
    if (here.g > 0.5) {
        let taps = i32(u.taps);
        var uv = start_uv;
        for (var i = 0; i < taps; i = i + 1) {
            uv = uv + u.sun_dir_uv;
            if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
                // Sun-facing window: brighter than the wall tint so glass
                // reads as glowing daylight.
                return vec4<f32>(u.color * (u.intensity * 2.0), 1.0);
            }
            let s = textureSample(t_mask, s_mask, uv);
            if (s.r > 0.5) {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        }
        return vec4<f32>(u.color * (u.intensity * 2.0), 1.0);
    }

    // Wall pixel: check the sun-side neighbour.
    if (here.r > 0.5) {
        let uv0 = start_uv + u.sun_dir_uv;
        if (uv0.x < 0.0 || uv0.x > 1.0 || uv0.y < 0.0 || uv0.y > 1.0) {
            // Outer edge of surface, unoccluded → sun-lit.
            return vec4<f32>(u.color * u.intensity, 1.0);
        }
        let n0 = textureSample(t_mask, s_mask, uv0);
        // 2. Sun-side neighbour is a window → check that the window is
        // on the sun-facing hull by marching from the window tile
        // toward the sun. If the ray reaches open space (or another
        // window) without hitting a wall, this is a sunlit window and
        // we're the wall pixel just inside it → green. Otherwise stay
        // dark.
        if (n0.g > 0.5) {
            let taps = i32(u.taps);
            var uv = uv0;
            var reached_sun = false;
            for (var i = 0; i < taps; i = i + 1) {
                uv = uv + u.sun_dir_uv;
                if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
                    reached_sun = true;
                    break;
                }
                let s = textureSample(t_mask, s_mask, uv);
                if (s.r > 0.5) { break; }
            }
            if (reached_sun) {
                // Wall pixel adjacent to a sun-lit window — bright warm
                // "the light comes through here" edge.
                return vec4<f32>(u.color * (u.intensity * 1.5), 1.0);
            }
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        // If neighbour is still opaque wall, we're not on the outer
        // face → shaded.
        if (n0.r > 0.5) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        // Neighbour is empty. Continue marching to check for occluders
        // further out.
        let taps = i32(u.taps);
        var uv = uv0;
        for (var i = 1; i < taps; i = i + 1) {
            uv = uv + u.sun_dir_uv;
            if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
                return vec4<f32>(u.color * u.intensity, 1.0);
            }
            let s = textureSample(t_mask, s_mask, uv);
            if (s.r > 0.5) {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        }
        return vec4<f32>(u.color * u.intensity, 1.0);
    }

    // Empty-space handled by a separate god-ray pass (see godray.wgsl):
    // the sun shader only writes the wall-surface contribution. God-ray
    // beams are rendered from lit-window pixels into an offscreen buffer,
    // blurred, and additively composited so they read as soft volumetric
    // shafts instead of per-pixel raycast bands.
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
}

// Composite: additive blend onto the main pass. Blend state on the
// pipeline (src=One, dst=One) does the add — this shader just outputs
// the sun contribution verbatim.
@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(t_mask, s_mask, in.uv);
    return vec4<f32>(c.rgb, 1.0);
}
