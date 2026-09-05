// Wall-occluding sun-shadow raycast.
//
// The caller renders solid walls (and character silhouettes) into a
// single-channel `wall_mask` texture at surface size. This shader marches
// each screen pixel toward the sun; if any tap hits a wall, the pixel is
// shadowed. Output is written into `shadow_map`, which the main pass
// then multiply-blends onto the composited scene so shadowed pixels
// darken but lit pixels are left untouched.
//
// Uniform layout:
//   sun_dir_uv : normalized march direction in UV space (toward the sun,
//                pre-scaled so `sun_dir_uv * step` moves one march step)
//   taps       : number of march steps (int, encoded as f32)
//   darkness   : output value for a shadowed pixel (e.g. 0.5)

struct ShadowUniform {
    sun_dir_uv: vec2<f32>, // per-step delta in UV space
    taps:       f32,
    darkness:   f32,
};

@group(0) @binding(0) var<uniform> u:      ShadowUniform;
@group(0) @binding(1) var          t_mask: texture_2d<f32>;
@group(0) @binding(2) var          s_mask: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
};

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

// wall_mask coverage vs. the on-screen surface. Must match `MASK_SCALE`
// in `shadow.rs`. Fragment UV (surface space) is translated into
// wall_mask UV via `mask_uv = 0.5 + (uv - 0.5) / MASK_SCALE` before any
// texture sample of the mask.
const MASK_SCALE: f32 = 2.0;

fn surface_uv_to_mask_uv(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(0.5, 0.5) + (uv - vec2<f32>(0.5, 0.5)) * (1.0 / MASK_SCALE);
}

@fragment
fn fs_shadow(in: VsOut) -> @location(0) vec4<f32> {
    // Translate fragment (surface) UV into wall_mask UV — the mask is
    // MASK_SCALE larger than the surface so off-screen occluders still
    // register. sun_dir_uv is already pre-scaled to mask-UV units.
    let start_uv = surface_uv_to_mask_uv(in.uv);
    // Skip the fragment's own texel — solid tiles shouldn't shadow themselves.
    let self_hit = textureSample(t_mask, s_mask, start_uv).r;
    if (self_hit > 0.5) {
        // Inside a wall — draw fully lit so the wall body doesn't self-darken
        // (walls read as opaque architecture, shadows fall on the floor).
        return vec4<f32>(1.0, 1.0, 1.0, 1.0);
    }
    let taps = i32(u.taps);
    var uv = start_uv;
    for (var i = 0; i < taps; i = i + 1) {
        uv = uv + u.sun_dir_uv;
        // Clamp: outside the mask is "no wall" — stop marching gracefully.
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            break;
        }
        let s = textureSample(t_mask, s_mask, uv).r;
        if (s > 0.5) {
            return vec4<f32>(u.darkness, u.darkness, u.darkness, 1.0);
        }
    }
    return vec4<f32>(1.0, 1.0, 1.0, 1.0);
}

// Composite: multiply-blend shadow_map onto the main pass. The blend state
// on the pipeline (dst_factor=Zero, src_factor=Dst) does the multiply — we
// just output the shadow value verbatim.
@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let s = textureSample(t_mask, s_mask, in.uv).r;
    return vec4<f32>(s, s, s, 1.0);
}
