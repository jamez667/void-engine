// Depth-only pass from the light's point of view (feature `render3d`).
//
// Renders the same geometry as `shader3d.wgsl` with the light's
// view-projection substituted for the camera's, writing nothing but depth.
// There is no fragment stage: the pipeline sets `fragment: None`, so the
// rasteriser fills the depth buffer and stops. A colour target here would
// be bandwidth spent on a result nobody reads.
//
// The bind groups match the main 3D pipeline's first two exactly — camera
// at 0, per-instance transform at 1 — so the same `Vertex3D` layout and the
// same instance ring drive both passes without a second copy of either.

struct CameraUniform {
    // The *light's* view-projection in this pass, not the camera's. Same
    // layout, so the main camera bind group layout is reused verbatim.
    view_proj: mat4x4<f32>,
};

struct InstanceUniform {
    model: mat4x4<f32>,
    tint: vec4<f32>,
};

@group(0) @binding(0) var<uniform> light: CameraUniform;
@group(1) @binding(0) var<uniform> instance: InstanceUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) uv:       vec2<f32>,
    @location(3) color:    vec4<f32>,
};

// Only the clip position is produced. The other vertex attributes are
// declared because the vertex *buffer layout* is shared with the main
// pipeline and must describe every attribute the buffer holds, but nothing
// downstream consumes them.
@vertex
fn vs_shadow(in: VertexInput) -> @builtin(position) vec4<f32> {
    return light.view_proj * instance.model * vec4<f32>(in.position, 1.0);
}
