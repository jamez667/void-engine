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
    // Base layer: id in bits 0-15, strength in 16-23, blend mode in 24-27.
    @location(4) material: u32,
    // Overlay layer, packed the same way. Zero means none.
    @location(5) overlay: u32,
    // Ink colour the pattern draws in; alpha is the overlay's scale ratio.
    @location(6) ink: vec4<f32>,
    // Pattern cells per pixel, for anti-aliasing the hatch away before it moires.
    @location(7) scale: f32,
    // Position within the tile (-1..1 along the blend direction), and where the
    // overlay crosses over as a fraction of the half-width.
    @location(8) local: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) pattern: vec2<f32>,
    @location(3) @interpolate(flat) material: u32,
    @location(4) @interpolate(flat) overlay: u32,
    @location(5) ink: vec4<f32>,
    @location(6) @interpolate(flat) scale: f32,
    // Interpolated across the quad: this is what lets the blend happen inside
    // a tile rather than uniformly over it.
    @location(7) local: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.pattern = in.pattern;
    out.material = in.material;
    out.overlay = in.overlay;
    out.ink = in.ink;
    out.scale = in.scale;
    out.local = in.local;
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
        // Wide enough to survive. A blade at 0.015 of a cell is under a pixel
        // wherever the cell is small enough to hold several blades, so it
        // averages away and the cover comes out as flat colour -- which is
        // exactly what turf did on the map before this was widened.
        blade = max(blade, 1.0 - smoothstep(0.05, 0.16, d));
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
    // A joint: narrow, hard-edged, and sparse. Rock is unbroken over metres
    // with the odd fracture running across it, which is exactly what tells it
    // apart from scree -- scree is all edges, rock is almost none.
    let joint = 1.0 - smoothstep(0.004, 0.022, d2 - d1);
    // Only some cell boundaries are jointed at all; the rest is solid face.
    let solid = step(0.45, hash2(floor(p) + vec2<f32>(3.1, 7.7)));
    // Broad, soft variation across the face: weathering, not fragments.
    let weather = smoothstep(0.35, 0.75, value_noise(p * 0.8)) * 0.30;
    return clamp(joint * solid + weather, 0.0, 1.0);
}

// --- Shared drawing helpers -------------------------------------------------

// Worley cells: nearest and next-nearest site distance. The basis for anything
// that breaks into blocks or fragments.
fn worley(p: vec2<f32>) -> vec2<f32> {
    let base = floor(p);
    var d1 = 8.0;
    var d2 = 8.0;
    for (var y = -1; y <= 1; y = y + 1) {
        for (var x = -1; x <= 1; x = x + 1) {
            let cell = base + vec2<f32>(f32(x), f32(y));
            let site = cell + hash2v(cell);
            let d = length(site - p);
            if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
        }
    }
    return vec2<f32>(d1, d2);
}

// A ruled line every unit in y. The basis for bedding, courses, and hatching.
fn rule_h(y: f32, thickness: f32) -> f32 {
    let d = abs(fract(y) - 0.5);
    return 1.0 - smoothstep(thickness * 0.5, thickness, d);
}

// Rotate into a hatch direction.
fn rot(p: vec2<f32>, a: f32) -> vec2<f32> {
    let c = cos(a);
    let s = sin(a);
    return vec2<f32>(p.x * c - p.y * s, p.x * s + p.y * c);
}

// Scattered dots: `density` of cells carry one, `size` sets the radius.
fn stipple(p: vec2<f32>, density: f32, size: f32, vary: f32) -> f32 {
    let cell = floor(p);
    let local = fract(p) - 0.5;
    let r = hash2v(cell);
    let has = step(1.0 - density, hash2(cell + vec2<f32>(5.2, 1.3)));
    let radius = size * (1.0 - vary + vary * r.y * 2.0);
    let d = length(local - (r - 0.5) * 0.6);
    return has * (1.0 - smoothstep(radius * 0.55, radius, d));
}

// --- Ground cover -----------------------------------------------------------

// Talus and mine waste: angular fragments of every size, jumbled.
fn pat_scree(p: vec2<f32>) -> f32 {
    // Every boundary, at three sizes: a talus slope is nothing but the edges
    // of broken rock, with no unbroken ground anywhere in it. Against stone,
    // which is mostly unbroken face, that reads as a different substance
    // rather than as the same one at a different size.
    let a = worley(p);
    let coarse = 1.0 - smoothstep(0.01, 0.08, a.y - a.x);
    let b = worley(p * 2.3 + vec2<f32>(11.3, 4.1));
    let mid = 1.0 - smoothstep(0.015, 0.09, b.y - b.x);
    let c = worley(p * 5.1 + vec2<f32>(3.7, 19.1));
    let fine = 1.0 - smoothstep(0.02, 0.11, c.y - c.x);
    // Each fragment takes its own tone, so the slope reads as a heap of
    // separate pieces rather than as a net of lines.
    let facet = hash2(floor(p * 2.3 + vec2<f32>(11.3, 4.1))) * 0.34;
    return clamp(max(coarse, max(mid * 0.85, fine * 0.6)) + facet, 0.0, 1.0);
}

// Creek bed: rounded, sorted, water-worn stones packed together.
fn pat_gravel(p: vec2<f32>) -> f32 {
    let w = worley(p);
    // Shade by distance from each cell centre, so stones read as domed rather
    // than as cracked.
    let dome = smoothstep(0.0, 0.55, w.x);
    let seam = 1.0 - smoothstep(0.02, 0.09, w.y - w.x);
    return clamp(dome * 0.55 + seam * 0.7, 0.0, 1.0);
}

// Wind-drifted sand: ripples, no fragments at all.
fn pat_sand(p: vec2<f32>) -> f32 {
    // Crests bent by a slow field, so they are not ruled lines.
    let bend = value_noise(p * 0.35) * 2.4;
    let ripple = sin((p.y + bend) * 6.2831) * 0.5 + 0.5;
    let grain = value_noise(p * 9.0) * 0.18;
    return clamp(smoothstep(0.55, 0.95, ripple) * 0.7 + grain, 0.0, 1.0);
}

// Dried mud: the polygon crack pattern of a playa.
fn pat_cracked(p: vec2<f32>) -> f32 {
    // One crack per plate boundary, thin and clean, over plates that are
    // otherwise smooth. Mud dries into large flat plates; the smoothness
    // between the cracks is what distinguishes it from broken rock.
    let w = worley(p * 0.55);
    let crack = 1.0 - smoothstep(0.008, 0.045, w.y - w.x);
    // Each plate curls up at its edge, so it is lighter in the middle and
    // darker where it lifts -- the shading a dried playa actually has.
    let curl = (1.0 - smoothstep(0.05, 0.45, w.x)) * 0.28;
    return clamp(crack + curl, 0.0, 1.0);
}

// Timber from the side: bark furrows running along the grain.
fn pat_timber(p: vec2<f32>) -> f32 {
    let wander = value_noise(p * vec2<f32>(0.6, 2.2)) * 0.55;
    let furrow = rule_h(p.y * 2.4 + wander, 0.35);
    // Broken along their length, because bark is not continuous.
    let breaks = smoothstep(0.35, 0.75, value_noise(p * vec2<f32>(3.5, 0.8)));
    return clamp(furrow * (0.45 + 0.55 * breaks), 0.0, 1.0);
}

// Sawn end grain: growth rings about the pith.
fn pat_endgrain(p: vec2<f32>) -> f32 {
    // Rings, unevenly spaced the way seasons actually are.
    let r = length(p) + value_noise(p * 1.7) * 0.25;
    let ring = rule_h(r * 3.6, 0.32);
    // Radial drying checks, running out from the centre.
    let ang = atan2(p.y, p.x);
    let check = step(0.86, value_noise(vec2<f32>(ang * 2.4, 0.0)) + 0.35)
        * (1.0 - smoothstep(0.1, 1.2, length(p)));
    return clamp(ring * 0.8 + check * 0.5, 0.0, 1.0);
}

// --- Lithology --------------------------------------------------------------

// Alluvium: scattered stipple of irregular size, loose and unbedded.
fn pat_alluvium(p: vec2<f32>) -> f32 {
    let big = stipple(p * 1.6, 0.55, 0.30, 0.7);
    let small = stipple(p * 3.1 + vec2<f32>(7.0, 7.0), 0.35, 0.16, 0.5);
    return clamp(big + small * 0.6, 0.0, 1.0);
}

// Shale: dense horizontal bedding, close spaced. The most ruled of the set.
fn pat_shale(p: vec2<f32>) -> f32 {
    // Slight waviness, because bedding is never dead straight.
    let wave = value_noise(p * vec2<f32>(0.4, 0.15)) * 0.2;
    let bed = rule_h(p.y * 3.2 + wave, 0.22);
    // Partings: occasional heavier lines where the rock splits.
    let parting = rule_h(p.y * 0.8 + wave * 0.5, 0.12) * 0.6;
    return clamp(bed * 0.75 + parting, 0.0, 1.0);
}

// Sandstone: uniform fine stipple. Deliberately plain -- it is the neutral rock
// the others are read against.
fn pat_sandstone(p: vec2<f32>) -> f32 {
    return clamp(stipple(p * 3.4, 0.7, 0.22, 0.25), 0.0, 1.0);
}

// Brick courses, shared by limestone and dolomite: horizontal beds with
// staggered vertical joints, the standard carbonate convention.
// A brick is `BRICK_W` wide and one unit tall in course space. Wide, because a
// carbonate course on a survey drawing is a long flat brick, and because a
// nearly square brick reads as woven cloth rather than as masonry.
const BRICK_W: f32 = 1.15;
/// Courses per pattern unit. Well below one: a carbonate bed is a broad band,
/// and drawing one per unit turns the rock into ruled paper on which the
/// dolomite tick cannot be seen at all.
const COURSE: f32 = 0.22;

fn courses(p: vec2<f32>) -> f32 {
    let course = floor(p.y);
    // Every other course offset half a brick, so joints never line up.
    let stagger = fract(course * 0.5);
    let bed = rule_h(p.y, 0.13);
    let joint_x = p.x / BRICK_W + stagger;
    let joint = 1.0 - smoothstep(0.010, 0.028, abs(fract(joint_x) - 0.5));
    // Joints run within their own course, never across the bedding.
    let inside = 1.0 - rule_h(p.y, 0.26);
    return clamp(bed + joint * inside, 0.0, 1.0);
}

fn pat_limestone(p: vec2<f32>) -> f32 {
    return courses(p * COURSE);
}

// Dolomite: the same courses, with the 45 degree tick in each that is the
// standard convention distinguishing it from limestone.
fn pat_dolomite(p: vec2<f32>) -> f32 {
    let q = p * COURSE;
    let base = courses(q);
    // One tick per brick, in the middle of its own course.
    let brick = vec2<f32>(q.x / BRICK_W + fract(floor(q.y) * 0.5), q.y);
    let local = vec2<f32>(fract(brick.x) - 0.5, fract(brick.y) - 0.5);
    // Undo the brick's aspect so the tick sits at a true 45 degrees.
    let square = vec2<f32>(local.x * BRICK_W, local.y);
    let d = rot(square, -0.7854);
    let tick = (1.0 - smoothstep(0.05, 0.10, abs(d.y)))
        * (1.0 - smoothstep(0.22, 0.32, abs(d.x)));
    return clamp(base + tick, 0.0, 1.0);
}

// Quartzite: fine cross-hatch at 45 and 135 degrees.
fn pat_quartzite(p: vec2<f32>) -> f32 {
    let a = rot(p, 0.7854);
    let b = rot(p, -0.7854);
    let h1 = rule_h(a.y * 2.6, 0.16);
    let h2 = rule_h(b.y * 2.6, 0.16);
    return clamp(max(h1, h2) * 0.85, 0.0, 1.0);
}

// Granite: randomised crosses and plus marks at low density -- the crystalline
// convention. Individual marks scattered on the field, not a hatch.
fn pat_granite(p: vec2<f32>) -> f32 {
    var mark = 0.0;
    // One lattice, at pattern scale, so each mark has room to be a drawn
    // symbol rather than a speck. Low density is the convention: granite is
    // mostly open ground with crosses scattered on it.
    for (var k = 0; k < 2; k = k + 1) {
        let q = p * 0.62 + vec2<f32>(f32(k) * 0.5, f32(k) * 0.27);
        let cell = floor(q);
        let local = fract(q) - 0.5;
        let r = hash2v(cell);
        if (hash2(cell + vec2<f32>(4.4, 8.8)) < 0.42) { continue; }
        // Half plus marks, half crosses, as a geological drawing has both.
        var ang = 0.0;
        if (r.x > 0.5) { ang = 0.7854; }
        let d = rot(local - (r - 0.5) * 0.35, ang);
        let arm = 0.20 + r.y * 0.10;
        let w = 0.035;
        let bar1 = (1.0 - smoothstep(w, w * 1.8, abs(d.y)))
            * (1.0 - smoothstep(arm * 0.85, arm, abs(d.x)));
        let bar2 = (1.0 - smoothstep(w, w * 1.8, abs(d.x)))
            * (1.0 - smoothstep(arm * 0.85, arm, abs(d.y)));
        mark = max(mark, max(bar1, bar2));
    }
    // A very light speckle between the marks, well under them.
    let speck = stipple(p * 2.6, 0.18, 0.07, 0.4) * 0.22;
    return clamp(mark + speck, 0.0, 1.0);
}

// Gossan: irregular blotches of iron oxide, pitted through with boxwork voids.
fn pat_gossan(p: vec2<f32>) -> f32 {
    let a = value_noise(p * 1.1);
    let b = value_noise(p * 2.7 + vec2<f32>(13.0, 13.0));
    let blotch = smoothstep(0.45, 0.75, a * 0.65 + b * 0.35);
    let pit = stipple(p * 3.2, 0.4, 0.20, 0.6) * 0.55;
    return clamp(blotch * 0.8 + pit, 0.0, 1.0);
}

// --- Mineralisation and workings --------------------------------------------

// Sulphide ore: bright cubic glints scattered through the rock.
fn pat_ore_sulphide(p: vec2<f32>) -> f32 {
    var glint = 0.0;
    for (var k = 0; k < 2; k = k + 1) {
        let q = p * 1.8 + vec2<f32>(f32(k) * 0.41, f32(k) * 0.67);
        let cell = floor(q);
        let local = fract(q) - 0.5;
        let r = hash2v(cell);
        if (hash2(cell + vec2<f32>(6.1, 2.2)) < 0.55) { continue; }
        // A little square, turned: galena breaks in cubes.
        let d = rot(local - (r - 0.5) * 0.45, r.x * 1.57);
        let s = 0.10 + r.y * 0.09;
        let cube = (1.0 - smoothstep(s * 0.75, s, abs(d.x)))
            * (1.0 - smoothstep(s * 0.75, s, abs(d.y)));
        glint = max(glint, cube);
    }
    return clamp(glint, 0.0, 1.0);
}

// Quartz: massive and glassy, faintly banded, with sparse curved fracture.
fn pat_quartz(p: vec2<f32>) -> f32 {
    let band = smoothstep(0.4, 0.6, value_noise(p * vec2<f32>(0.5, 1.8)));
    let w = worley(p * 0.8);
    let frac = (1.0 - smoothstep(0.03, 0.13, w.y - w.x)) * 0.5;
    return clamp(band * 0.30 + frac, 0.0, 1.0);
}

// Timbered ground: the regular marks of sets standing in a drift.
fn pat_timbered(p: vec2<f32>) -> f32 {
    let posts = rule_h(p.x * 1.0, 0.22);
    let lag = rule_h(p.y * 3.0, 0.10) * 0.4;
    return clamp(posts * 0.85 + lag, 0.0, 1.0);
}

// --- Material dispatch ------------------------------------------------------
//
// One place where an id becomes a pattern value. Everything else -- colouring,
// blending, layering, fading -- is downstream of this and applies to every
// material alike, so adding one means adding a pattern function and a case
// here, and nothing more.

fn pattern_of(id: u32, p: vec2<f32>) -> f32 {
    switch (id) {
        // Ground cover.
        case 1u: { return pat_grass(p); }
        case 2u: { return pat_dirt(p); }
        case 3u: { return pat_stone(p); }
        case 4u: { return pat_scree(p); }
        case 5u: { return pat_gravel(p); }
        case 6u: { return pat_sand(p); }
        case 7u: { return pat_cracked(p); }
        case 8u: { return pat_timber(p); }
        case 9u: { return pat_endgrain(p); }
        // Lithology.
        case 16u: { return pat_alluvium(p); }
        case 17u: { return pat_shale(p); }
        case 18u: { return pat_sandstone(p); }
        case 19u: { return pat_limestone(p); }
        case 20u: { return pat_dolomite(p); }
        case 21u: { return pat_quartzite(p); }
        case 22u: { return pat_granite(p); }
        case 23u: { return pat_gossan(p); }
        // Mineralisation and workings.
        case 32u: { return pat_ore_sulphide(p); }
        case 33u: { return pat_quartz(p); }
        case 34u: { return pat_timbered(p); }
        default: { return 0.0; }
    }
}

// --- Layer unpacking and compositing ----------------------------------------

fn layer_id(word: u32) -> u32 { return word & 0xFFFFu; }
fn layer_strength(word: u32) -> f32 { return f32((word >> 16u) & 0xFFu) / 255.0; }
fn layer_blend(word: u32) -> u32 { return (word >> 24u) & 0xFu; }

// Combine one layer's pattern into the colour under it.
//
// `ground` is what has been built so far, `ink` the colour this layer draws in,
// and `amount` how much of it lands here. Every blend mode is a function of
// exactly those, which is what lets layers stack without special cases.
fn composite(ground: vec3<f32>, ink: vec3<f32>, amount: f32, mode: u32) -> vec3<f32> {
    switch (mode) {
        // Ink on paper: the pattern is drawn in its own colour.
        case 0u: { return mix(ground, ink, amount); }
        // Shade: darken, keeping the ground's hue.
        case 1u: { return ground * (1.0 - 0.75 * amount); }
        // Lighten: quartz, efflorescence, frost.
        case 2u: { return mix(ground, ground + (vec3<f32>(1.0) - ground) * 0.85, amount); }
        // Stain: a true multiply, so the result takes the colour of both and
        // is never lighter than either. The doubling this once carried made
        // stain collapse onto plain ink at mid-grey, where ground * ink * 2
        // happens to equal ink exactly.
        case 3u: { return mix(ground, ground * ink, amount); }
        default: { return mix(ground, ink, amount); }
    }
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_color = textureSample(t_diffuse, s_diffuse, in.uv);
    let base = in.color * tex_color;

    let id = layer_id(in.material);
    if (id == 0u) {
        return base;
    }

    // Fade patterns out as their features approach pixel size. Past that point
    // a hatch stops reading as texture and starts aliasing into moire, so it is
    // better to show honest flat colour -- which is also what a survey drawing
    // does when the scale no longer supports the hatch. Applied once, to every
    // layer, so no material can forget it.
    let fade = 1.0 - smoothstep(0.18, 0.55, in.scale);
    if (fade <= 0.001) {
        return base;
    }

    var rgb = base.rgb;
    let ink = in.ink.rgb;

    // Base layer.
    let v = pattern_of(id, in.pattern);
    rgb = composite(rgb, ink, v * layer_strength(in.material) * fade, layer_blend(in.material));

    // Overlay, if there is one. Drawn at its own feature size so two layers
    // read as two rather than merging into one muddled pattern, and faded on
    // its own terms because a finer overlay aliases sooner than its base.
    let over_id = layer_id(in.overlay);
    if (over_id != 0u) {
        let ratio = max(in.ink.a, 0.01);
        let over_fade = 1.0 - smoothstep(0.18, 0.55, in.scale * ratio);
        if (over_fade > 0.001) {
            // How much of the overlay lands on *this fragment*.
            //
            // With a blend direction the overlay is absent at the far side of
            // the tile and full at the near one, crossing over a band around
            // the given depth. That band is what makes a boundary a gradation:
            // two tiles each blending toward the other meet in a shared strip
            // rather than changing over on the edge between them.
            //
            // depth of 1 means no direction was given, and the overlay applies
            // evenly across the whole tile as it always did.
            var edge = 1.0;
            let depth = in.local.z;
            if (depth < 0.999) {
                // local.x runs -1 at the far edge to +1 at the near one. The
                // crossover sits at `1 - 2 * depth`, so a depth of 0.5 puts it
                // at the centre and a smaller one nearer the edge.
                let at = 1.0 - 2.0 * depth;
                edge = smoothstep(at - depth, at + depth, in.local.x);
            }
            let ov = pattern_of(over_id, in.pattern * ratio);
            rgb = composite(
                rgb,
                ink,
                ov * layer_strength(in.overlay) * over_fade * edge,
                layer_blend(in.overlay),
            );
        }
    }

    return vec4<f32>(rgb, base.a);
}
