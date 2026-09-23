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
    return out;
}

// Directional light, pointing down and somewhat to the side. +Z is up,
// the convention `terrain::field::hillshade` established by building its
// surface normal as (-dz/dx, -dz/dy, 1).
const LIGHT_DIR: vec3<f32> = vec3<f32>(0.32, 0.43, 0.84);
// Ambient floor, so a face pointing away from the light reads as shaded
// rather than as a black hole. Matches the spirit of the 2D light pass's
// ambient clear.
const AMBIENT: f32 = 0.25;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Interpolating unit normals across a triangle denormalises them, so
    // renormalise before the dot or shading drifts across large faces.
    let n = normalize(in.normal);
    let lambert = max(dot(n, normalize(LIGHT_DIR)), 0.0);
    let shade = AMBIENT + (1.0 - AMBIENT) * lambert;
    return vec4<f32>(in.color.rgb * shade, in.color.a);
}
