// Per-light additive falloff with wall-mask occlusion.
//
// Called once per light emitter (one-pass-per-light). The shader emits a
// fullscreen triangle, computes the radial falloff toward the given light
// position (in UV space), then marches from the fragment toward the light
// through the wall_mask. Any solid tap along the march zeroes the light's
// contribution for that pixel. Blend state on the pipeline is additive so
// overlapping lights sum into the target light-map texture.
//
// A separate composite pass multiplies the accumulated light-map onto the
// main scene at the caller's chosen split index (mirror of the shadow +
// blur composite pattern).

struct LightUniform {
    // Light position in UV space (0..1 across the surface).
    pos_uv:    vec2<f32>,
    // 1 / surface pixel width / height. Used to convert UV distance into
    // pixels so `radius_px` is honoured 1:1 with the screen.
    px_uv:     vec2<f32>,
    // Colour * intensity, pre-multiplied so the shader only does falloff.
    color:     vec3<f32>,
    intensity: f32,
    // Radial: falloff hits zero at this pixel distance.
    // Rectangle: along-axis reach in pixels.
    radius_px: f32,
    // Number of raycast taps for wall occlusion.
    taps:      f32,
    // 0.0 = radial (existing behaviour), 1.0 = rectangle beam.
    kind:      f32,
    _pad0:     f32,
    // Rectangle-only: unit facing direction in pixel space.
    dir_px:         vec2<f32>,
    // Rectangle-only: perpendicular half-width in pixels.
    half_width_px:  f32,
    _pad1:          f32,
    _pad2:          f32,
    _pad3:          f32,
};

// Per-light passes bind wall_mask here; the composite pass binds the
// accumulated light_map. Same layout, different data.
@group(0) @binding(0) var<uniform> u:      LightUniform;
@group(0) @binding(1) var          t_in:   texture_2d<f32>;
@group(0) @binding(2) var          s_in:   sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
};

// wall_mask coverage vs. the on-screen surface. Must match
// `shadow::MASK_SCALE`. Occlusion samples of `t_in` (per-light passes)
// remap from surface UV into mask UV so off-screen walls occlude
// on-screen lights. Composite pass samples the surface-sized light_map
// so it skips the remap.
const MASK_SCALE: f32 = 2.0;

fn surface_uv_to_mask_uv(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(0.5, 0.5) + (uv - vec2<f32>(0.5, 0.5)) * (1.0 / MASK_SCALE);
}

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

@fragment
fn fs_light(in: VsOut) -> @location(0) vec4<f32> {
    // Delta from fragment to light, in pixel space (so shapes stay
    // aspect-correct regardless of surface aspect ratio).
    let d_uv = u.pos_uv - in.uv;
    let d_px = vec2<f32>(d_uv.x / u.px_uv.x, d_uv.y / u.px_uv.y);

    var fall: f32;
    if (u.kind < 0.5) {
        // ── Radial pool ────────────────────────────────────────────────
        let dist_px = length(d_px);
        if (dist_px >= u.radius_px) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        // Quadratic falloff: (1 - d/r)^2. Soft edges, bright core.
        let t = 1.0 - dist_px / u.radius_px;
        fall = t * t;
    } else {
        // ── Rectangle beam ─────────────────────────────────────────────
        // `dir_px` points *away* from the light (toward where the beam
        // shines). `d_px = light - pixel`, so the "toward pixel" vector
        // is `-d_px`; we project that onto `dir_px`.
        let to_pixel_px = -d_px;
        let along = dot(to_pixel_px, u.dir_px);
        if (along < 0.0 || along > u.radius_px) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        let perp_vec = to_pixel_px - u.dir_px * along;
        let perp = length(perp_vec);
        if (perp > u.half_width_px) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
        // Linear falloff along beam length only — hard edges on sides.
        fall = 1.0 - along / u.radius_px;
    }

    // Raycast occlusion: march from fragment toward the light. If any
    // sample lands on a wall (mask > 0.5), the light is blocked.
    // Skip the first + last taps (start = self, end = the light source
    // itself — often a wall tile like a Window or ceiling fixture).
    let taps = i32(u.taps);
    if (taps > 0) {
        // wall_mask lives in a larger texture than the surface. Remap
        // both the ray origin (fragment) and the light position into
        // mask-UV space and interpolate there — the ray's world span is
        // preserved because both endpoints scale the same way.
        let start_mask = surface_uv_to_mask_uv(in.uv);
        let end_mask   = surface_uv_to_mask_uv(u.pos_uv);
        let dir_mask   = end_mask - start_mask;
        let inv = 1.0 / f32(taps + 1);
        for (var i = 1; i <= taps; i = i + 1) {
            let s_uv = start_mask + dir_mask * (f32(i) * inv);
            if (s_uv.x < 0.0 || s_uv.x > 1.0 || s_uv.y < 0.0 || s_uv.y > 1.0) {
                break;
            }
            let m = textureSample(t_in, s_in, s_uv).r;
            if (m > 0.5) {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        }
    }

    let contrib = u.color * (u.intensity * fall);
    return vec4<f32>(contrib, 1.0);
}

// Composite: multiply the accumulated light-map onto the main scene. The
// blend state on the pipeline (src_factor=Dst, dst_factor=Zero) does the
// multiply — this shader just outputs the light value verbatim.
@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(t_in, s_in, in.uv);
    return vec4<f32>(c.rgb, 1.0);
}
