// The 3D geometry shader (feature `render3d`).
//
// Deliberately separate from `shader.wgsl` rather than an added branch in
// it. That file's fragment stage is a ~20-function procedural material
// system evaluated in *pattern space* — world metres on a flat plane —
// with an anti-aliasing fade driven by a per-vertex metres-per-pixel
// value. Under perspective there is no single metres-per-pixel for a
// frame, so those inputs stop meaning anything. Porting them is a real
// design question (see `docs/3d-spec.md` §8); drawing lit geometry is not,
// and this file does the second without blocking on the first.
//
// Position is camera-relative, matching `Camera3D::build_uniform`, which
// places the eye at the origin and subtracts in f64 before the f32 cast.
// A vertex handed to this shader in absolute world coordinates will lose
// precision far from the origin exactly as the 2D path would.

struct CameraUniform {
    // Same layout as the 2D path's uniform, so both share one bind group
    // layout, one buffer and one upload. `Camera3D` puts `proj * view`
    // here where `Camera2D` puts an orthographic projection alone.
    view_proj: mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> camera: CameraUniform;

// Per-instance transform, supplied through a dynamic-offset uniform so one
// bind group serves every instance in the frame — the same trick the 2D
// light pass uses for its per-light uniforms.
//
// This is what the 2D path has no equivalent for: `Batch` pre-transforms
// every primitive on the CPU into one monolithic buffer, so there is
// nowhere to hang a model matrix. A retained mesh is uploaded once and
// drawn at many places, so the transform has to arrive separately.
struct InstanceUniform {
    model: mat4x4<f32>,
    // Multiplied into the vertex colour, so one grey mesh can be drawn as
    // many differently tinted objects without re-uploading its vertices.
    tint: vec4<f32>,
};

@group(1) @binding(0) var<uniform> instance: InstanceUniform;

// Shadow map and the directional light that cast it.
//
// `light_view_proj` transforms a camera-relative position into the light's
// clip space, so a fragment can ask "what depth did the light record in
// my direction?" — see `shadow3d.rs` on why that is the technique here
// and why the 2D wall-mask march is not portable to 3D.
struct ShadowUniform {
    light_view_proj: mat4x4<f32>,
    // xyz = direction toward the light (normalised), w = ambient floor.
    light_dir_ambient: vec4<f32>,
};

@group(2) @binding(0) var<uniform> shadow: ShadowUniform;
// A *depth* texture with a *comparison* sampler: sampling returns the
// blended result of `depth <= reference` over four taps rather than a
// depth value, which is hardware PCF and gives a soft edge for free.
@group(2) @binding(1) var shadow_map: texture_depth_2d;
@group(2) @binding(2) var shadow_sampler: sampler_comparison;

// Point lights. Kept in one uniform array rather than the 2D path's
// one-fullscreen-pass-per-light, because 3D geometry is not fullscreen:
// a pass per light would re-rasterise every mesh per light, where a loop
// in the fragment shader costs only the shading.
const MAX_POINT_LIGHTS: u32 = 16u;

struct PointLight {
    // xyz = camera-relative position, w = radius. Packed into vec4s so
    // std140 needs no padding guesswork.
    pos_radius: vec4<f32>,
    // xyz = colour, w = intensity.
    color_intensity: vec4<f32>,
};

struct PointLights {
    // How many of `lights` are live this frame. A u32 padded to a full
    // 16-byte block, since std140 aligns what follows to 16 anyway.
    count: vec4<u32>,
    lights: array<PointLight, MAX_POINT_LIGHTS>,
};

@group(3) @binding(0) var<uniform> point_lights: PointLights;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) uv:       vec2<f32>,
    @location(3) color:    vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0)       normal:        vec3<f32>,
    @location(1)       uv:            vec2<f32>,
    @location(2)       color:         vec4<f32>,
    // Camera-relative world position, needed by the fragment stage for
    // both the point-light distances and the shadow lookup.
    @location(3)       world_pos:     vec3<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world = instance.model * vec4<f32>(in.position, 1.0);
    // The 2D path writes `vec4(in.position, 0.0, 1.0)` here, pinning every
    // vertex to z = 0. This is that line with a real z and a model
    // transform in front of it.
    out.clip_position = camera.view_proj * world;

    // Normals take the model's rotation but not its translation, hence
    // mat3. This is correct for the rigid and uniformly-scaled transforms
    // `MeshDraw` builds; a *non-uniform* scale would need the inverse
    // transpose, which is not computed here because nothing produces one
    // yet and doing it per vertex is the wrong place to pay for it.
    let rot = mat3x3<f32>(
        instance.model[0].xyz,
        instance.model[1].xyz,
        instance.model[2].xyz,
    );
    out.normal = rot * in.normal;
    out.uv = in.uv;
    out.color = in.color * instance.tint;
    out.world_pos = world.xyz;
    return out;
}

/// How much of the directional light reaches this fragment: 1 = fully
/// lit, 0 = fully shadowed.
///
/// Returns 1.0 for anything outside the shadow map rather than 0.0.
/// Unshadowed is the safe default: a scene larger than the map's covered
/// radius should have its distant geometry look ordinary, not plunged
/// into darkness at a hard circular boundary.
fn shadow_factor(world_pos: vec3<f32>) -> f32 {
    let light_clip = shadow.light_view_proj * vec4<f32>(world_pos, 1.0);
    // The light's projection is orthographic, so w is 1 and this divide
    // is a formality — but doing it keeps the function correct if the
    // projection is ever swapped for a perspective one (a spotlight).
    let ndc = light_clip.xyz / light_clip.w;

    // Clip space is x,y in -1..1 but texture UVs are 0..1 with y flipped.
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, ndc.y * -0.5 + 0.5);

    // Outside the map, or behind the light's near plane: unshadowed.
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0
        || ndc.z < 0.0 || ndc.z > 1.0) {
        return 1.0;
    }

    // `textureSampleCompare` returns the *filtered* comparison result —
    // four taps blended — rather than a single boolean, which is what
    // makes the shadow edge soft without a manual PCF loop. The depth
    // bias lives in the shadow pipeline's rasteriser state, not here.
    return textureSampleCompare(shadow_map, shadow_sampler, uv, ndc.z);
}

/// Accumulated contribution of every live point light.
///
/// Smooth radial falloff, clamped at each light's radius. Looping in the
/// fragment shader rather than taking a pass per light (what the 2D path
/// does) because 3D geometry is not fullscreen: a pass per light would
/// re-rasterise every mesh once per light, where this costs only shading.
fn point_light_contribution(world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    var total = vec3<f32>(0.0);
    let count = min(point_lights.count.x, MAX_POINT_LIGHTS);
    for (var i = 0u; i < count; i = i + 1u) {
        let l = point_lights.lights[i];
        let to_light = l.pos_radius.xyz - world_pos;
        let dist = length(to_light);
        let radius = l.pos_radius.w;
        if (dist >= radius || radius <= 0.0) {
            continue;
        }
        // Quadratic falloff, matching the 2D light pass's `(1 - d/r)^2`.
        let t = 1.0 - dist / radius;
        let falloff = t * t;
        let lambert = max(dot(n, to_light / max(dist, 1e-6)), 0.0);
        total = total + l.color_intensity.rgb * l.color_intensity.w * lambert * falloff;
    }
    return total;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Interpolating unit normals across a triangle denormalises them, so
    // renormalise before the dot or shading drifts across large faces.
    let n = normalize(in.normal);
    let ambient = shadow.light_dir_ambient.w;

    // Directional term, gated by the shadow map. Only the *direct* light
    // is shadowed: ambient and point lights are not, which is what keeps
    // a shadowed surface readable rather than black.
    let lambert = max(dot(n, normalize(shadow.light_dir_ambient.xyz)), 0.0);
    let direct = lambert * shadow_factor(in.world_pos);

    let lit = ambient + (1.0 - ambient) * direct;
    let shade = vec3<f32>(lit) + point_light_contribution(in.world_pos, n);

    return vec4<f32>(in.color.rgb * shade, in.color.a);
}
