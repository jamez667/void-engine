//! Directional shadow mapping for the 3D path (feature `render3d`).
//!
//! # Why this is a replacement, not a port
//!
//! The 2D engine already has shadows (`shadow.rs`), and none of it applies
//! here. That pass renders occluders into a flat binary `wall_mask` and
//! marches each *screen pixel in UV space* toward the sun — a top-down
//! trick over a mask with no height information to give. There is no
//! version of it that generalises to geometry at different elevations.
//!
//! This is the standard alternative: render the scene from the light's
//! point of view into a depth texture, then in the main pass transform each
//! fragment into that same light space and compare its depth against what
//! the light recorded. Closer means lit; farther means something stood in
//! the way.
//!
//! # The two failure modes, and what is done about them
//!
//! **Acne** — a surface shadowing itself, because the depth it recorded and
//! the depth it tests against differ by a hair of floating-point and
//! rasterisation error. Addressed with a slope-scaled depth bias in the
//! shadow pipeline (`DepthBiasState`), which offsets steeply-lit surfaces
//! more than face-on ones, since those are where the error is largest.
//!
//! **Peter-panning** — the shadow detaching from the object's feet,
//! because the bias pushed it too far. The bias here is deliberately
//! modest for that reason; see
//! [`crate::renderer::shadow3d::DEPTH_BIAS_SLOPE`].
//!
//! Both are tuned against `tests/shadow3d.rs`, which renders a lit floor
//! with an occluder above it and asserts the floor is darker underneath
//! and not darker elsewhere.

use glam::{Mat4, Vec3};

/// Shadow map resolution, square.
///
/// 2048 is the usual starting point: enough that a character-scale scene
/// does not show obvious stair-stepping at the shadow edge, and 16 MB at
/// `Depth32Float` — real but not extravagant. A game wanting crisper
/// shadows over a larger area wants cascades rather than a bigger map,
/// because doubling this quadruples the memory for a linear gain in edge
/// quality.
pub const SHADOW_MAP_SIZE: u32 = 2048;

/// Constant depth bias, in depth-buffer units.
///
/// Small, because a constant bias applies equally to face-on surfaces
/// (which barely need it) and grazing ones (which need much more) — the
/// slope-scaled term below is what actually does the work.
pub const DEPTH_BIAS_CONSTANT: i32 = 2;

/// Slope-scaled depth bias.
///
/// Scales with the polygon's depth gradient, so a surface nearly edge-on
/// to the light — where a pixel's depth varies most across its own
/// footprint, and where acne therefore appears first — is offset more than
/// one facing it squarely.
///
/// 2.0 is chosen as the smallest value that cleared acne in
/// `tests/shadow3d.rs` on the test scene. Raising it hides acne at the
/// cost of peter-panning, so it wants lowering rather than raising if
/// shadows start to look detached.
pub const DEPTH_BIAS_SLOPE: f32 = 2.0;

/// The light's view-projection, plus its direction.
///
/// Laid out in 16-byte blocks so WGSL's std140 rules need no trailing pad
/// guesswork — `LightUniform` in `lights.rs` documents what goes wrong
/// otherwise (a `min_binding_size` mismatch at pipeline creation).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ShadowUniform {
    /// Transforms a camera-relative position into the light's clip space.
    pub light_view_proj: [[f32; 4]; 4],
    /// Direction *toward* the light, normalised. `w` carries the ambient
    /// floor, which keeps the block at 16 bytes without a pad field.
    pub light_dir_ambient: [f32; 4],
}

/// Build the light's view-projection for a directional light.
///
/// # Why orthographic
///
/// A directional light has no position — its rays are parallel — so the
/// projection that matches it is orthographic. A perspective projection
/// here would make shadows converge toward a point the light does not
/// have.
///
/// # Fitting the frustum
///
/// `radius` is the half-extent of the world the map covers, centred on
/// `center`. Everything outside is unshadowed, so this wants to be about
/// the size of the visible scene: too large wastes resolution on empty
/// space and makes shadow edges chunky, too small leaves visible geometry
/// unshadowed.
///
/// `center` is **camera-relative**, like everything else the 3D path
/// consumes — see [`crate::renderer::camera::Camera3D`].
pub fn light_view_proj(direction: Vec3, center: Vec3, radius: f32) -> Mat4 {
    // Guard a zero or non-finite direction: `look_at_rh` would return NaN
    // and every fragment would fail its comparison, blacking out the
    // scene rather than failing loudly.
    let dir = if direction.length_squared() > 0.0 && direction.is_finite() {
        direction.normalize()
    } else {
        Vec3::new(0.0, 0.0, -1.0)
    };

    // Place the eye *at* the light: `direction` points toward it, so the
    // eye goes along +dir and looks back down at the scene.
    //
    // The sign matters and is easy to get backwards. With `- dir` the eye
    // lands on the far side of the scene, looking up from underneath: the
    // floor then records a *smaller* depth than the box above it, so a
    // `LessEqual` comparison finds the floor closer to the light and
    // lights it. The result is a scene with no shadows at all and nothing
    // obviously wrong in the code — see `tests/shadow3d.rs`, which caught
    // exactly this.
    let eye = center + dir * radius * 2.0;

    // `look_at_rh` cannot resolve an up vector parallel to the gaze. +Z is
    // the engine's up (see `terrain::field::hillshade`), so a light
    // pointing straight down needs a different one.
    let up = if dir.z.abs() > 0.99 { Vec3::Y } else { Vec3::Z };

    let view = Mat4::look_at_rh(eye, center, up);
    // Depth range 0..4*radius: the eye sits 2*radius back, so this spans
    // from the eye to twice the far side of the covered sphere.
    let proj = Mat4::orthographic_rh(
        -radius,
        radius,
        -radius,
        radius,
        0.0,
        radius * 4.0,
    );
    proj * view
}

pub(super) struct Shadow3D {
    pub view: wgpu::TextureView,
    _tex: wgpu::Texture,
    pub pipeline: wgpu::RenderPipeline,
    pub uniform_buffer: wgpu::Buffer,
    /// Bound in the *shadow* pass: the light's matrix, for the depth-only
    /// render from the light's point of view.
    pub light_camera_bg: wgpu::BindGroup,
    pub light_camera_buffer: wgpu::Buffer,
    /// Bound in the *main* pass: the shadow map plus the light uniform.
    pub sample_bg: wgpu::BindGroup,
    pub sample_bgl: wgpu::BindGroupLayout,
}

impl Shadow3D {
    pub fn new(
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        instance_bgl: &wgpu::BindGroupLayout,
    ) -> Self {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow3d_map"),
            size: wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: super::depth::DEPTH_FORMAT,
            // TEXTURE_BINDING as well as RENDER_ATTACHMENT: unlike the
            // main depth buffer, this one is sampled.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        // A *comparison* sampler, not a filtering one. It returns the
        // result of `depth < reference` averaged over its taps rather
        // than a depth value, which is what gives hardware PCF: four
        // comparisons blended, so the shadow edge is soft for free.
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow3d_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let light_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow3d_light_camera"),
            size: std::mem::size_of::<super::camera::CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Reuses the main camera layout, so the depth-only pass can drive
        // the same vertex shader with the light's matrix in place of the
        // camera's — the same trick `ShadowPass::write_mask_camera` uses
        // in 2D.
        let light_camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow3d_light_camera_bg"),
            layout: camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_camera_buffer.as_entire_binding(),
            }],
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow3d_uniform"),
            size: std::mem::size_of::<ShadowUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow3d_sample_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
            ],
        });

        let sample_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow3d_sample_bg"),
            layout: &sample_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Depth-only pipeline: same vertex shader and vertex layout as the
        // main 3D pass, no fragment stage at all. Nothing is written but
        // depth, so a colour target would be wasted bandwidth.
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow3d.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shadow3d.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow3d_pl"),
            bind_group_layouts: &[camera_bgl, instance_bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow3d_pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_shadow",
                buffers: &[super::mesh3d::Vertex3D::desc()],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // Front-face culling, the opposite of the main pass.
                // Rendering back faces into the map puts the recorded
                // depth on the far side of each object, which moves the
                // self-shadowing comparison away from the lit surface and
                // removes most acne before the bias is even considered.
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: super::depth::DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: DEPTH_BIAS_CONSTANT,
                    slope_scale: DEPTH_BIAS_SLOPE,
                    clamp: 0.0,
                },
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            view,
            _tex: tex,
            pipeline,
            uniform_buffer,
            light_camera_bg,
            light_camera_buffer,
            sample_bg,
            sample_bgl,
        }
    }

    /// The depth attachment for the shadow pass.
    pub fn attachment(&self) -> wgpu::RenderPassDepthStencilAttachment<'_> {
        wgpu::RenderPassDepthStencilAttachment {
            view: &self.view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The light's projection must be orthographic: a directional light's
    /// rays are parallel, and a perspective projection would converge them
    /// toward a point the light does not have.
    ///
    /// Checked by confirming two points at different depths but the same
    /// lateral offset land at the same place — which is exactly what
    /// parallel projection means and what perspective would break.
    #[test]
    fn the_light_projection_is_parallel_not_perspective() {
        let m = light_view_proj(Vec3::new(0.0, 0.0, -1.0), Vec3::ZERO, 10.0);
        let near = m.project_point3(Vec3::new(3.0, 0.0, 2.0));
        let far = m.project_point3(Vec3::new(3.0, 0.0, -2.0));
        assert!(
            (near.x - far.x).abs() < 1e-4 && (near.y - far.y).abs() < 1e-4,
            "two points at the same lateral offset but different depths \
             landed apart ({near:?} vs {far:?}) — the projection is \
             converging, so it is not orthographic",
        );
    }

    /// Everything inside the covered radius must land inside the map, or
    /// geometry at the edge of the scene silently stops casting.
    #[test]
    fn the_covered_radius_lands_inside_clip_space() {
        let r = 10.0;
        let m = light_view_proj(Vec3::new(0.0, 0.0, -1.0), Vec3::ZERO, r);
        for p in [
            Vec3::new(r * 0.99, 0.0, 0.0),
            Vec3::new(-r * 0.99, 0.0, 0.0),
            Vec3::new(0.0, r * 0.99, 0.0),
            Vec3::new(0.0, -r * 0.99, 0.0),
        ] {
            let c = m.project_point3(p);
            assert!(
                c.x.abs() <= 1.0 && c.y.abs() <= 1.0,
                "{p:?} projected to {c:?}, outside the shadow map",
            );
            assert!(
                (0.0..=1.0).contains(&c.z),
                "{p:?} projected to depth {} — outside the 0..1 range the \
                 map stores, so it would be clipped away",
                c.z,
            );
        }
    }

    /// A light pointing straight down the up axis would make `look_at_rh`
    /// pick an up vector parallel to its gaze, which it cannot resolve.
    /// +Z is the engine's up, so this is the ordinary "sun overhead" case
    /// rather than an exotic one.
    #[test]
    fn a_light_straight_down_the_up_axis_does_not_produce_a_nan_matrix() {
        let m = light_view_proj(Vec3::new(0.0, 0.0, -1.0), Vec3::ZERO, 10.0);
        assert!(m.is_finite(), "overhead light produced {m:?}");
    }

    #[test]
    fn a_degenerate_light_direction_does_not_produce_a_nan_matrix() {
        let m = light_view_proj(Vec3::ZERO, Vec3::ZERO, 10.0);
        assert!(m.is_finite(), "zero light direction produced {m:?}");
    }

    /// The uniform's size must be a multiple of 16, or WGSL's std140 view
    /// of it disagrees with Rust's and wgpu rejects the pipeline with a
    /// `min_binding_size` mismatch. `LightUniform` in `lights.rs` carries
    /// four trailing pad floats for exactly this reason; this layout
    /// avoids needing them by using 16-byte blocks throughout.
    #[test]
    fn the_shadow_uniform_needs_no_trailing_padding() {
        let size = std::mem::size_of::<ShadowUniform>();
        assert_eq!(size % 16, 0, "ShadowUniform is {size} bytes, not a multiple of 16");
        assert_eq!(size, 80, "4x4 matrix (64) + vec4 (16)");
    }
}
