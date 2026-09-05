struct CameraUniform {
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> camera: CameraUniform;

@group(1) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(1) @binding(1)
var s_diffuse: sampler;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    // Position in pattern space (world metres / feature size).
    @location(3) pattern: vec2<f32>,
    // Material id in the low 16 bits, strength 0-255 in the high 16.
    @location(4) material: u32,
    // Pattern cells per pixel, for anti-aliasing the hatch away before it moires.
    @location(5) scale: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) pattern: vec2<f32>,
    @location(3) @interpolate(flat) material: u32,
    @location(4) @interpolate(flat) scale: f32,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.pattern = in.pattern;
    out.material = in.material;
    out.scale = in.scale;
    return out;
}

// --- Procedural pattern support ---------------------------------------------
//
// Everything below is a function of pattern-space position. Nothing is sampled
// and no image exists: these patterns cost memory nothing, never tile visibly,
// and stay sharp at any zoom because they are evaluated per fragment.

fn hash2(p: vec2<f32>) -> f32 {
    // Cheap, stable, and good enough for scatter and jitter.
    let h = dot(p, vec2<f32>(127.1, 311.7));
    return fract(sin(h) * 43758.5453123);
}

fn hash2v(p: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(hash2(p), hash2(p + vec2<f32>(19.19, 7.71)));
}

fn value_noise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    // Smoothstep weights, so the field has no grid-aligned creases.
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash2(i);
    let b = hash2(i + vec2<f32>(1.0, 0.0));
    let c = hash2(i + vec2<f32>(0.0, 1.0));
    let d = hash2(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

// Turf: short strokes at varied angles, clumping into denser tufts.
fn pat_grass(p: vec2<f32>) -> f32 {
    // Clumps: a slow field that decides where the grass is thick.
    let clump = value_noise(p * 0.7);

    var blade = 0.0;
    // Three passes of jittered cells, so blades overlap and tuft rather than
    // sitting on a visible grid.
    for (var k = 0; k < 3; k = k + 1) {
        let off = vec2<f32>(f32(k) * 0.37, f32(k) * 0.61);
        let cell = floor(p + off);
        let local = fract(p + off) - 0.5;
        let r = hash2v(cell);
        // Each blade leans its own way, mostly upright.
        let ang = (r.x - 0.5) * 1.9 + 1.57;
        let dir = vec2<f32>(cos(ang), sin(ang));
        // A long stroke through the cell, offset so blades are not concentric.
        let base = local - (r - 0.5) * 0.55;
        let along = clamp(dot(base, dir), -0.42, 0.42);
        let d = length(base - dir * along);
        // Thin and crisp: a blade is a line, not a smudge.
        blade = max(blade, 1.0 - smoothstep(0.015, 0.075, d));
    }
    return blade * clamp(clump * 1.6, 0.35, 1.0);
}

// Bare earth: fine irregular stipple with occasional coarser grains.
fn pat_dirt(p: vec2<f32>) -> f32 {
    // Fine grain: sharpened well past noise so it reads as grit, not blur.
    let n = value_noise(p * 6.0);
    let fine = smoothstep(0.55, 0.78, n);

    // Pebbles: distinct rounded grains scattered through the earth. Two
    // offset lattices so they do not fall into rows.
    var pebble = 0.0;
    for (var k = 0; k < 2; k = k + 1) {
        let q = p * 2.1 + vec2<f32>(f32(k) * 0.5, f32(k) * 0.27);
        let cell = floor(q);
        let local = fract(q) - 0.5;
        let r = hash2v(cell);
        // Only some cells carry a pebble, and they vary in size.
        let has = step(0.45, hash2(cell + vec2<f32>(3.3, 1.7)));
        let radius = 0.10 + r.y * 0.16;
        let d = length(local - (r - 0.5) * 0.55);
        pebble = max(pebble, has * (1.0 - smoothstep(radius * 0.6, radius, d)));
    }
    return clamp(fine * 0.5 + pebble, 0.0, 1.0);
}

// Broken rock: angular cracks between mottled blocks.
fn pat_stone(p: vec2<f32>) -> f32 {
    // Worley-style cells give the blocky structure rock actually breaks into.
    let base = floor(p);
    var d1 = 8.0;
    var d2 = 8.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let cell = base + vec2<f32>(f32(x), f32(y));
            let site = cell + hash2v(cell);
            let d = length(site - p);
            if (d < d1) {
                d2 = d1;
                d1 = d;
            } else if (d < d2) {
                d2 = d;
            }
        }
    }
    // The gap between nearest and next-nearest site is the crack between
    // blocks. Kept narrow and hard-edged: a fracture in rock is a line.
    let crack = 1.0 - smoothstep(0.01, 0.06, d2 - d1);
    // Faint mottling across each block face, so the rock is not dead flat
    // between cracks -- but well below the cracks, which are the figure.
    let mottle = smoothstep(0.45, 0.85, value_noise(p * 2.5)) * 0.22;
    return clamp(crack + mottle, 0.0, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(t_diffuse, s_diffuse, in.uv);
    var out = in.color * tex_color;

    let id = in.material & 0xFFFFu;
    if (id == 0u) {
        return out;
    }

    // Fade the pattern out as its features approach pixel size. Past that point
    // a hatch stops reading as texture and starts aliasing into moire, so it is
    // better to show honest flat colour -- which is also what a survey drawing
    // does when the scale no longer supports the hatch.
    let fade = 1.0 - smoothstep(0.18, 0.55, in.scale);
    if (fade <= 0.001) {
        return out;
    }

    var v = 0.0;
    if (id == 1u) {
        v = pat_grass(in.pattern);
    } else if (id == 2u) {
        v = pat_dirt(in.pattern);
    } else if (id == 3u) {
        v = pat_stone(in.pattern);
    }

    let strength = f32((in.material >> 16u) & 0xFFu) / 255.0;
    // The pattern darkens and lightens the vertex colour rather than replacing
    // it, so a material and a palette colour compose: one grass shader serves
    // spring green and dry summer buff alike.
    let amount = v * strength * fade;
    let shaded = out.rgb * (1.0 - 0.62 * amount) + vec3<f32>(0.05) * amount;
    return vec4<f32>(shaded, out.a);
}
