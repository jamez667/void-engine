// Separable Gaussian blur used by the post-process pipeline.
//
// A fullscreen triangle is emitted from `vertex_index` — no vertex buffer
// needs to be bound. The uniform packs:
//   axis   : 0 = horizontal, 1 = vertical
//   radius : blur radius in pixels
//   px_x   : 1 / texture_width  (in u.v — matched to the input texture)
//   px_y   : 1 / texture_height
//
// Kernel is 9 taps (center + 4 on each side) with hand-tuned Gaussian
// weights. Ample softness for the "roof-of-lower-floors" pass without
// paying for a bloom-grade 13/17-tap kernel.

struct BlurUniform {
    axis:   f32,
    radius: f32,
    px_x:   f32,
    px_y:   f32,
};

@group(0) @binding(0) var<uniform> u: BlurUniform;
@group(0) @binding(1) var t_in:    texture_2d<f32>;
@group(0) @binding(2) var s_in:    sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)       uv:  vec2<f32>,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    // 3-vertex oversized triangle covering the [-1, 1] clip rect.
    var out: VsOut;
    var xs = array<f32, 3>(-1.0,  3.0, -1.0);
    var ys = array<f32, 3>(-1.0, -1.0,  3.0);
    var us = array<f32, 3>( 0.0,  2.0,  0.0);
    var vs = array<f32, 3>( 1.0,  1.0, -1.0); // flip so uv.y=0 at top
    let i = i32(vid);
    out.pos = vec4<f32>(xs[i], ys[i], 0.0, 1.0);
    out.uv  = vec2<f32>(us[i], vs[i]);
    return out;
}

// 9-tap Gaussian, sigma≈2. Weights sum to 1.
const W0: f32 = 0.20236;
const W1: f32 = 0.17904;
const W2: f32 = 0.12395;
const W3: f32 = 0.06649;
const W4: f32 = 0.02777;

@fragment
fn fs_blur(in: VsOut) -> @location(0) vec4<f32> {
    let step = select(
        vec2<f32>(0.0, u.px_y),
        vec2<f32>(u.px_x, 0.0),
        u.axis < 0.5,
    ) * u.radius;
    var acc = textureSample(t_in, s_in, in.uv) * W0;
    acc = acc + textureSample(t_in, s_in, in.uv + step * 1.0) * W1;
    acc = acc + textureSample(t_in, s_in, in.uv - step * 1.0) * W1;
    acc = acc + textureSample(t_in, s_in, in.uv + step * 2.0) * W2;
    acc = acc + textureSample(t_in, s_in, in.uv - step * 2.0) * W2;
    acc = acc + textureSample(t_in, s_in, in.uv + step * 3.0) * W3;
    acc = acc + textureSample(t_in, s_in, in.uv - step * 3.0) * W3;
    acc = acc + textureSample(t_in, s_in, in.uv + step * 4.0) * W4;
    acc = acc + textureSample(t_in, s_in, in.uv - step * 4.0) * W4;
    return acc;
}

// Composite pass: straight copy of the (already blurred) texture.
@fragment
fn fs_copy(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(t_in, s_in, in.uv);
}
