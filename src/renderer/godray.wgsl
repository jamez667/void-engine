// Offscreen god-ray pipeline.
//
// Two fragment stages sharing one uniform + wall_mask + seed sampler:
//
//   fs_seed    : reads wall_mask, emits 1.0 at LIT WINDOW pixels
//                (mask.g > 0.5 whose sun-side neighbour is empty/off-surface),
//                0.0 everywhere else. Writes into godray_seed.
//
//   fs_march   : reads godray_seed + wall_mask together, marches N taps
//                in `-sun_dir_uv` direction (away from the sun), accumulates
//                a decayed sum of the seed at each tap, aborts when a tap
//                hits an opaque wall (mask.r > 0.5) — so beams get blocked
//                by interior geometry. Outputs colored beam contribution.
//
//   fs_composite : additive blend the beam texture onto the main pass.
//
// The offscreen radial-blur approach avoids the tile-aligned banding the
// per-pixel raycast produced — samples are summed with a smooth kernel at
// surface resolution instead of resolving to a boolean threshold per row.

struct GodrayUniform {
    // Per-step delta in UV space TOWARD the sun (positive = toward sun).
    // The march uses the negative to walk away from the sun (light travel
    // direction from windows into interior spaces).
    sun_dir_uv: vec2<f32>,
    // Number of march taps.
    taps:       f32,
    // Per-tap scale on `sun_dir_uv` — controls beam length. 1.0 = one
    // uniform delta per tap; larger = longer beams, coarser sampling.
    step_scale: f32,
    // Sun colour in linear RGB.
    color:      vec3<f32>,
    // Peak scalar multiplier on the accumulated beam.
    intensity:  f32,
    // Game time (s) — drives dust-mote drift along the beam.
    time:       f32,
    _pad0:      f32,
    _pad1:      f32,
    _pad2:      f32,
};

@group(0) @binding(0) var<uniform> u:      GodrayUniform;
@group(0) @binding(1) var          t_src:  texture_2d<f32>;
@group(0) @binding(2) var          s_src:  sampler;

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
    var vs = array<f32, 3>( 1.0,  1.0, -1.0);
    let i = i32(vid);
    out.pos = vec4<f32>(xs[i], ys[i], 0.0, 1.0);
    out.uv  = vec2<f32>(us[i], vs[i]);
    return out;
}

// ---- Seed pass ----------------------------------------------------------
// `t_src` is wall_mask. Emit 1.0 at pixels sitting on the sun-facing outer
// hull — i.e. a window pixel (.g > 0.5) whose neighbour ONE STEP TOWARD
// the sun is empty or off-surface. Any other pixel: 0.

@fragment
fn fs_seed(in: VsOut) -> @location(0) vec4<f32> {
    let here = textureSample(t_src, s_src, in.uv);
    if (here.g < 0.5) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    // A window is "sun-lit" if its ray toward the sun escapes the mask
    // without hitting a wall. Empirically `+sun_dir_uv` IS the direction
    // toward the sun in this shader (verified against the wall-lit sun
    // shader that works). Beams *travel* the other way in fs_march, and
    // the flip we saw earlier was a rendering-flip elsewhere.
    let taps = i32(u.taps);
    var uv = in.uv;
    for (var i: i32 = 0; i < taps; i = i + 1) {
        uv = uv + u.sun_dir_uv * u.step_scale;
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            return vec4<f32>(1.0, 0.0, 0.0, 1.0);
        }
        let s = textureSample(t_src, s_src, uv);
        if (s.r > 0.5) {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    }
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}

// ---- Radial-march / blur pass ------------------------------------------
// Bindings for this pass use a DIFFERENT bind group that binds both the
// seed texture AND wall_mask. To keep the pipeline layout uniform we use
// a two-texture layout for this stage — see the second bind group in
// godray.rs where t_src=seed, t_mask=wall_mask.

@group(0) @binding(3) var          t_mask: texture_2d<f32>;

// wall_mask + seed textures are MASK_SCALE larger than the surface so
// off-screen occluders / off-screen lit windows still influence on-
// screen beams. Must match `shadow::MASK_SCALE`. Only the march pass
// remaps here: the seed pass already runs at mask resolution, and the
// composite samples the surface-sized beam texture.
const MASK_SCALE: f32 = 2.0;

fn surface_uv_to_mask_uv(uv: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(0.5, 0.5) + (uv - vec2<f32>(0.5, 0.5)) * (1.0 / MASK_SCALE);
}

// Cheap integer hash for dust-mote flicker (IQ-style).
fn hash21(p: vec2<f32>) -> f32 {
    var q = fract(p * vec2<f32>(123.34, 456.21));
    q = q + dot(q, q + 45.32);
    return fract(q.x * q.y);
}

@fragment
fn fs_march(in: VsOut) -> @location(0) vec4<f32> {
    // Beams stream FROM the window INTO the room — same direction as
    // `sun_dir_uv` here. (The client convention we settled on has
    // sun_dir_uv pointing away from the sun on-screen, so + is beam
    // travel direction.)
    let step = u.sun_dir_uv * u.step_scale;

    // Occluder check at the start: if we're INSIDE a wall we still let a
    // small contribution through, but the accumulated beam intensity is
    // effectively zero after the first tap hits. So just start marching.
    let taps = i32(u.taps);
    var accum: f32 = 0.0;
    var weight_sum: f32 = 0.0;
    // Start in mask-UV so seed + wall_mask samples land at the right
    // world position. `u.sun_dir_uv` is already pre-scaled to a mask-UV
    // step, so the per-tap real-world reach is unchanged.
    var uv = surface_uv_to_mask_uv(in.uv);
    for (var i: i32 = 0; i < taps; i = i + 1) {
        // Weight = linear falloff — tap 0 fully counted, last tap zero.
        let t = f32(i) / f32(taps);
        let w = 1.0 - t;

        // Occluder check at this tap. If we hit a solid wall, the beam
        // is blocked from here onward — stop accumulating.
        let m = textureSample(t_mask, s_src, uv);
        if (m.r > 0.5 && i > 0) {
            break;
        }

        // Off-surface: stop — no source to sample beyond the frame edge.
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
            break;
        }

        let seed = textureSample(t_src, s_src, uv).r;
        accum = accum + seed * w;
        weight_sum = weight_sum + w;
        uv = uv + step;
    }

    if (weight_sum <= 0.0) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    var beam = accum / weight_sum;

    // Dust-mote flicker: sparse hash-based modulation along the beam so
    // shafts have some texture. Only kicks in where beam already exists.
    // Cell size ~1/400 UV. Time drifts along the beam direction.
    if (beam > 0.001) {
        let drift = u.sun_dir_uv * u.time * 0.05;
        let cell = floor((in.uv + drift) * 400.0);
        let n = hash21(cell);
        // Bias toward 1.0 so most cells pass through unchanged; only the
        // top ~15% of hashes darken slightly, giving a subtle sparkle.
        let mote = mix(1.0, n, 0.35);
        beam = beam * mote;
    }

    let col = u.color * (u.intensity * beam);
    return vec4<f32>(col, 1.0);
}

// ---- Composite ---------------------------------------------------------
@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let c = textureSample(t_src, s_src, in.uv);
    return vec4<f32>(c.rgb, 1.0);
}
