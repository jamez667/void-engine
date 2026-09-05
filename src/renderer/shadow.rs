//! Wall-occluding directional-sun shadow pipeline.
//!
//! Companion to `postprocess.rs` (blur). Structured the same way — a
//! rebuild-on-resize `ShadowPass` owning offscreen textures + pipelines,
//! driven by the renderer from `end_frame`. Runs three steps per frame:
//!
//!  1. Caller populates a "wall mask" batch (rects at every solid tile +
//!     each character silhouette). We rasterize it into `wall_mask` using
//!     the main camera bind group so mask uv == main-pass uv.
//!  2. A fullscreen raycast shader marches each pixel toward the sun; if
//!     any tap along the way hits a wall, the pixel is written dark into
//!     `shadow_map`, else lit (1.0).
//!  3. A composite pipeline multiply-blends `shadow_map` onto the main
//!     pass at the caller's chosen split index — shadowed pixels darken,
//!     lit pixels are untouched.
//!
//! The mask + shadow textures are `Rgba8Unorm` (not R8Unorm) because the
//! offscreen batch reuses the main vertex shader and writes a 4-channel
//! colour target; a single-channel target would need a dedicated shader
//! variant for little memory saving at 1080p.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec2};

use super::batch::Vertex;
use super::camera::CameraUniform;

/// Uniform driving the shadow raycast shader. Kept flat + `Pod` so a
/// single `write_buffer` copies it verbatim.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct ShadowUniform {
    sun_dir_uv: [f32; 2], // per-step delta in UV space (toward the sun)
    taps:       f32,      // number of march steps
    darkness:   f32,      // shadow output value (e.g. 0.5)
}

/// Number of raycast taps per pixel. 16 is comfortable at 1080p on
/// dedicated GPUs; drop to 8 if this bites on integrated hardware.
pub const SHADOW_TAPS: u32 = 16;

/// Shadow darkness [0..1]. Multiplied against the scene, so 0.5 = 50%
/// dim in shadow, 1.0 = no darkening.
pub const SHADOW_DARKNESS: f32 = 0.55;

/// Wall-mask coverage multiplier vs. the on-screen surface. `wall_mask`
/// (and `godray_seed`) are allocated at `MASK_SCALE * surface` on each
/// axis, so world content up to `(MASK_SCALE-1)/2` screens past the
/// viewport edges still occludes on-screen light + sun rays.
///
/// The mask-draw pass uses its own camera uniform with a viewport
/// enlarged by this factor so world (0,0) still maps to the mask centre
/// and geometry lands in the correct pixels of the bigger texture.
///
/// Consumer shaders (lights, shadow, sun, godray) remap their fragment
/// UV to mask UV via `mask_uv = 0.5 + (surface_uv - 0.5) / MASK_SCALE`
/// before sampling `wall_mask`, and scale their per-step UV march delta
/// by `1/MASK_SCALE` so a march step covers the same world distance.
pub const MASK_SCALE: f32 = 2.0;

/// Format used for the wall mask and shadow map. Uses the same 4-channel
/// SDR format the offscreen blur target uses so the offscreen batch can
/// draw into it without a shader variant.
const MASK_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub(super) struct ShadowPass {
    // Offscreen targets. `wall_mask` receives the caster rects; the
    // raycast writes into `shadow_map`, which the composite pass then
    // multiplies onto the main scene.
    pub wall_mask_view:  wgpu::TextureView,
    pub shadow_map_view: wgpu::TextureView,
    _wall_mask_tex:  wgpu::Texture,
    _shadow_map_tex: wgpu::Texture,

    // Pipeline for drawing the caller's mask batch into `wall_mask`. Reuses
    // the main shader (same vertex layout) but targets MASK_FORMAT with an
    // opaque overwrite so overlapping mask rects don't cancel out.
    pub mask_pipeline: wgpu::RenderPipeline,

    // Raycast shader: samples wall_mask, writes shadow_map. Composite
    // pipeline: samples shadow_map, multiplies onto the main target.
    pub raycast_pipeline:   wgpu::RenderPipeline,
    pub composite_pipeline: wgpu::RenderPipeline,

    _sample_bgl: wgpu::BindGroupLayout,
    pub bg_raycast:   wgpu::BindGroup, // reads wall_mask
    pub bg_composite: wgpu::BindGroup, // reads shadow_map

    pub uniform: wgpu::Buffer,
    _sampler:    wgpu::Sampler,

    // Dedicated camera uniform for the mask-draw pass. The projection
    // matches the main camera but with a viewport enlarged by
    // `MASK_SCALE`, so world (0,0) still lands at the mask centre and
    // world content within `(MASK_SCALE-1)/2` screens off-viewport still
    // rasterises into the enlarged wall_mask. Bind group uses the
    // shared `camera_bgl` layout so it drops into `mask_pipeline`
    // without a layout switch.
    pub mask_camera_buffer:     wgpu::Buffer,
    pub mask_camera_bind_group: wgpu::BindGroup,

    /// Width/height of the mask texture (surface × MASK_SCALE).
    pub mask_width:  u32,
    pub mask_height: u32,
}

impl ShadowPass {
    pub fn new(
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        texture_bgl: &wgpu::BindGroupLayout,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }

        // wall_mask is enlarged so off-screen occluders still register
        // (see MASK_SCALE docs). shadow_map is a consumer output and
        // stays surface-sized because the composite samples it at
        // surface UV.
        let mask_width  = ((width  as f32) * MASK_SCALE).round().max(1.0) as u32;
        let mask_height = ((height as f32) * MASK_SCALE).round().max(1.0) as u32;
        let (wall_mask_tex, wall_mask_view)   = make_offscreen(device, mask_width, mask_height, "wall_mask");
        let (shadow_map_tex, shadow_map_view) = make_offscreen(device, width, height, "shadow_map");

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let sample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow_sample_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("shadow_uniform"),
            size: std::mem::size_of::<ShadowUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg_raycast = make_sample_bg(
            device, &sample_bgl, &uniform, &wall_mask_view, &sampler, "bg_shadow_raycast",
        );
        let bg_composite = make_sample_bg(
            device, &sample_bgl, &uniform, &shadow_map_view, &sampler, "bg_shadow_composite",
        );

        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shadow.wgsl").into()),
        });

        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_pl"),
            bind_group_layouts: &[&sample_bgl],
            push_constant_ranges: &[],
        });

        let raycast_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_raycast_pipeline"),
            layout: Some(&shadow_layout),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: "vs_fullscreen",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shadow_shader,
                entry_point: "fs_shadow",
                targets: &[Some(wgpu::ColorTargetState {
                    format: MASK_FORMAT,
                    blend: None, // full overwrite
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Multiply blend: `out = src.rgb * dst.rgb`. `src_factor=Dst,
        // dst_factor=Zero, op=Add` collapses to `dst * src + 0 * dst`,
        // which is the multiply we want. Alpha is untouched (write_mask
        // drops it below anyway).
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_composite_pipeline"),
            layout: Some(&shadow_layout),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: "vs_fullscreen",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shadow_shader,
                entry_point: "fs_composite",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::Dst,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation:  wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::Zero,
                            operation:  wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::COLOR,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Mask pipeline: draw the caller's rects into wall_mask using the
        // shared main shader (same vertex layout, so `Batch::rect` etc.
        // work as-is). Overwrite blend keeps mask values crisp — no alpha
        // averaging.
        let main_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shadow_mask_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let mask_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow_mask_pl"),
            bind_group_layouts: &[camera_bgl, texture_bgl],
            push_constant_ranges: &[],
        });
        let mask_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_mask_pipeline"),
            layout: Some(&mask_layout),
            vertex: wgpu::VertexState {
                module: &main_shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &main_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: MASK_FORMAT,
                    blend: None, // opaque overwrite
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Dedicated camera uniform for the mask pass. Contents are
        // updated per-frame in `write_mask_camera`. Zero-init here; the
        // renderer overwrites it before the mask draw fires.
        let mask_camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mask_camera_buffer"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mask_camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mask_camera_bg"),
            layout: camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: mask_camera_buffer.as_entire_binding(),
            }],
        });

        Some(Self {
            wall_mask_view,
            shadow_map_view,
            _wall_mask_tex:  wall_mask_tex,
            _shadow_map_tex: shadow_map_tex,
            mask_pipeline,
            raycast_pipeline,
            composite_pipeline,
            _sample_bgl: sample_bgl,
            bg_raycast,
            bg_composite,
            uniform,
            _sampler: sampler,
            mask_camera_buffer,
            mask_camera_bind_group,
            mask_width,
            mask_height,
        })
    }

    /// Write the mask-pass camera uniform for this frame. Uses the same
    /// `position` and `zoom` as the main camera but with the viewport
    /// enlarged by `MASK_SCALE` — same units per world pixel, wider
    /// visible extent. World (0,0) maps to the middle of the enlarged
    /// mask texture so that the on-screen viewport occupies the central
    /// `1/MASK_SCALE` of each axis of the mask.
    pub fn write_mask_camera(&self, queue: &wgpu::Queue, position: Vec2, zoom: f32, viewport: Vec2) {
        let scaled_viewport = viewport * MASK_SCALE;
        let half_w = scaled_viewport.x * 0.5 / zoom;
        let half_h = scaled_viewport.y * 0.5 / zoom;
        let left   = position.x - half_w;
        let right  = position.x + half_w;
        let bottom = position.y - half_h;
        let top    = position.y + half_h;
        let proj = Mat4::orthographic_rh(left, right, bottom, top, -1.0, 1.0);
        let u = CameraUniform { view_proj: proj.to_cols_array_2d() };
        queue.write_buffer(&self.mask_camera_buffer, 0, bytemuck::bytes_of(&u));
    }

    /// Fill the raycast uniform for this frame. `sun_dir` is the direction
    /// shadows fall in screen pixel space (caller supplies whatever "sun"
    /// vector their game uses); we invert it here so the shader marches
    /// *toward* the light. `shadow_length_px` is the total march length in
    /// pixels.
    pub fn write_uniforms(&self, queue: &wgpu::Queue, sun_dir: Vec2, shadow_length_px: f32) {
        let n = SHADOW_TAPS.max(1) as f32;
        // Direction toward the sun, per-step, in UV space. Screen-y flips
        // to uv-y (uv.y=0 at top, screen-y grows downward), so the y
        // component negates during the pixel→uv mapping.
        let per_step_px = (-sun_dir.normalize_or_zero()) * (shadow_length_px / n);
        // sun_dir_uv is expressed in *mask* UV space (the enlarged
        // wall_mask). Dividing by mask_width/height keeps a march step
        // covering the same real distance as it did before scaling.
        let du =  per_step_px.x / self.mask_width  as f32;
        let dv = -per_step_px.y / self.mask_height as f32;
        let u = ShadowUniform {
            sun_dir_uv: [du, dv],
            taps:       n,
            darkness:   SHADOW_DARKNESS,
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&u));
    }
}

fn make_offscreen(
    device: &wgpu::Device, width: u32, height: u32, label: &str,
) -> (wgpu::Texture, wgpu::TextureView) {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: MASK_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}

fn make_sample_bg(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform: &wgpu::Buffer,
    tex_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    label: &str,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(tex_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}
