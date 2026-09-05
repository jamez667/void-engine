//! Per-light additive light-map pipeline (replaces the sun-shadow pass on
//! foot). Everything on the floor starts dark (ambient clear colour); each
//! emitter runs one fullscreen pass that additively writes its wall-occluded
//! radial falloff into `light_map`. A composite pass then multiply-blends
//! the light-map onto the main scene at the caller's split index — dark
//! pixels get squashed, lit pixels stay bright.
//!
//! Mirrors the shape of `postprocess.rs` / `shadow.rs`: rebuild-on-resize
//! `LightPass` owning offscreen textures + pipelines; the renderer drives
//! it from `end_frame`. The wall_mask (built by the same batch machinery
//! we used for the sun-shadow pass) is reused as the occlusion source —
//! same texture, different consumer.

use bytemuck::{Pod, Zeroable};

/// GPU uniform for one light. `_pad*` fields keep the struct 16-byte
/// aligned so std140 rules line up and the layout is unambiguous when
/// this ends up in a dynamic-offset ring buffer.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub(super) struct LightUniform {
    pub pos_uv:    [f32; 2],
    pub px_uv:     [f32; 2],
    // WGSL uniform layout pads vec3 to 16 bytes: `color` occupies its 12
    // data bytes followed by an implicit `intensity` at offset 28 (12+16=28
    // in the shader's std140 view). We keep the Rust ordering as-is (color
    // then intensity) and the pattern lines up.
    pub color:     [f32; 3],
    pub intensity: f32,
    pub radius_px: f32,
    pub taps:      f32,
    /// 0.0 = radial (default), 1.0 = rectangle (directional beam).
    pub kind:      f32,
    pub _pad0:     f32,
    /// Rectangle-light facing direction, unit vector in *pixel* space
    /// (aspect-correct). Shader compares against pixel-space delta so
    /// the beam stays undistorted regardless of surface aspect ratio.
    /// Ignored when `kind == 0.0`.
    pub dir_px:         [f32; 2],
    /// Perpendicular half-width of the rectangle beam, in pixels.
    /// Ignored when `kind == 0.0`.
    pub half_width_px:  f32,
    pub _pad1:          f32,
    // Trailing pads to bring the struct to a full 80 bytes. WGSL's std140
    // layout for the shader-side struct (which contains a vec3 + vec2)
    // rounds each 16-byte-aligned block up, so the shader expects 80 even
    // though the Rust field sum is 72. Match it here or wgpu rejects the
    // pipeline with a min_binding_size mismatch on device create.
    pub _tail0:         f32,
    pub _tail1:         f32,
    pub _tail2:         f32,
    pub _tail3:         f32,
}

/// Max lights per frame. Fullscreen passes are cheap at 1080p on dedicated
/// GPUs, but cap the fan-out so a runaway light emitter can't stall the
/// frame. Excess lights are dropped (LRU-nearest cull would be nicer;
/// realistic light counts on a station floor sit comfortably under this).
pub const MAX_LIGHTS_PER_FRAME: usize = 384;

/// Number of raycast taps used for wall occlusion per light. Cheaper than
/// the sun-shadow pass (16) because rays here are short (radius ~5 tiles,
/// not a full march to the horizon).
pub const LIGHT_TAPS: u32 = 12;

/// Ambient clear colour for the light-map. Almost-black with a slight cool
/// tint. Everything gets multiplied by this in unlit regions.
pub const AMBIENT_CLEAR: wgpu::Color = wgpu::Color {
    r: 0.06,
    g: 0.08,
    b: 0.12,
    a: 1.0,
};

/// Format used for the light-map. Rgba8UnormSrgb is fine for a soft "dark
/// interior" look — the lights don't push above 1.0 hard, so HDR isn't
/// worth the extra memory here. Matches the shadow / blur convention so
/// the composite blend factors work the same.
const LIGHT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub(super) struct LightPass {
    pub width:  u32,
    pub height: u32,

    // Accumulator: cleared to AMBIENT_CLEAR, each per-light pass adds its
    // wall-occluded radial contribution here.
    pub light_map_view: wgpu::TextureView,
    _light_map_tex:     wgpu::Texture,

    // One pipeline for per-light passes (additive blend), one for the
    // final multiply-composite onto the main scene.
    pub light_pipeline:     wgpu::RenderPipeline,
    pub composite_pipeline: wgpu::RenderPipeline,

    _sample_bgl: wgpu::BindGroupLayout,

    /// Ring buffer holding up to `MAX_LIGHTS_PER_FRAME` `LightUniform`
    /// slots, one per light this frame. Bound with a dynamic offset so
    /// every light draw uses the same bind group + buffer without
    /// per-light bind-group churn.
    pub light_uniforms: wgpu::Buffer,
    /// Bind group for per-light passes: wall_mask + dynamic-offset uniform.
    pub bg_light: wgpu::BindGroup,
    /// Bind group for the composite pass: samples the light_map. Uses a
    /// stubbed uniform slice (offset 0) since the composite shader ignores
    /// the uniform data.
    pub bg_composite: wgpu::BindGroup,

    /// Byte stride between light slots in `light_uniforms`. Alignment is
    /// device-dependent (usually 256).
    pub slot_stride: u32,

    _sampler: wgpu::Sampler,
}

impl LightPass {
    pub fn new(
        device: &wgpu::Device,
        wall_mask_view: &wgpu::TextureView,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }

        let (light_map_tex, light_map_view) = make_offscreen(device, width, height, "light_map");

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("light_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Round the uniform slot size up to the device's required alignment.
        // Nearly always 256 bytes; struct itself is 48 bytes.
        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let raw = std::mem::size_of::<LightUniform>() as u64;
        let slot_stride = raw.div_ceil(align) * align;

        let sample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("light_sample_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: std::num::NonZeroU64::new(raw),
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

        let light_uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("light_uniforms_ring"),
            size: slot_stride * MAX_LIGHTS_PER_FRAME as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Per-light bind group: samples wall_mask, reads a slot from the
        // ring at a dynamic offset. Only one bind group is needed for
        // every light this frame.
        let bg_light = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_light"),
            layout: &sample_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &light_uniforms,
                        offset: 0,
                        size: std::num::NonZeroU64::new(raw),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(wall_mask_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        // Composite bind group: samples the light_map. Uniform is unused
        // by the shader but still needs a valid dynamic-offset binding —
        // slot 0 of the ring is fine (contents don't matter).
        let bg_composite = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_light_composite"),
            layout: &sample_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &light_uniforms,
                        offset: 0,
                        size: std::num::NonZeroU64::new(raw),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&light_map_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("lights.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("lights.wgsl").into()),
        });

        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("light_pl"),
            bind_group_layouts: &[&sample_bgl],
            push_constant_ranges: &[],
        });

        // Per-light pass: additive blend into the light_map so overlapping
        // lights sum up rather than overwriting.
        let light_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("light_pipeline"),
            layout: Some(&pl_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_fullscreen",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_light",
                targets: &[Some(wgpu::ColorTargetState {
                    format: LIGHT_FORMAT,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
                            operation:  wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
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

        // Composite: multiply-blend the accumulated light-map onto the main
        // pass (same trick as the sun-shadow composite: src=Dst, dst=Zero).
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("light_composite_pipeline"),
            layout: Some(&pl_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_fullscreen",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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

        Some(Self {
            width,
            height,
            light_map_view,
            _light_map_tex: light_map_tex,
            light_pipeline,
            composite_pipeline,
            _sample_bgl: sample_bgl,
            light_uniforms,
            bg_light,
            bg_composite,
            slot_stride: slot_stride as u32,
            _sampler: sampler,
        })
    }

    /// Convert a screen-pixel light emitter into the packed uniform layout
    /// the shader expects. `pos_px` is centre relative to the top-left of
    /// the surface (matches the wall_mask uv origin: uv.y = 0 at top).
    pub fn make_uniform(
        &self,
        pos_px: [f32; 2],
        color: [f32; 3],
        radius_px: f32,
        intensity: f32,
    ) -> LightUniform {
        let px_uv = [1.0 / self.width as f32, 1.0 / self.height as f32];
        let pos_uv = [pos_px[0] * px_uv[0], pos_px[1] * px_uv[1]];
        LightUniform {
            pos_uv,
            px_uv,
            color,
            intensity,
            radius_px,
            taps: LIGHT_TAPS as f32,
            kind: 0.0,
            _pad0: 0.0,
            dir_px: [0.0, 0.0],
            half_width_px: 0.0,
            _pad1: 0.0,
            _tail0: 0.0,
            _tail1: 0.0,
            _tail2: 0.0,
            _tail3: 0.0,
        }
    }

    /// Build a rectangle (directional beam) light uniform. `dir_px` is a
    /// pixel-space facing vector (not required to be unit); it's normalised
    /// here. `length_px` sets the beam's reach along `dir_px`; the shader
    /// does linear falloff along that axis and hard-cuts at
    /// `+/- half_width_px` perpendicular.
    pub fn make_rect_uniform(
        &self,
        pos_px: [f32; 2],
        dir_px: [f32; 2],
        length_px: f32,
        half_width_px: f32,
        color: [f32; 3],
        intensity: f32,
    ) -> LightUniform {
        let px_uv = [1.0 / self.width as f32, 1.0 / self.height as f32];
        let pos_uv = [pos_px[0] * px_uv[0], pos_px[1] * px_uv[1]];
        let len = (dir_px[0] * dir_px[0] + dir_px[1] * dir_px[1]).sqrt().max(1e-6);
        let dir_unit = [dir_px[0] / len, dir_px[1] / len];
        LightUniform {
            pos_uv,
            px_uv,
            color,
            intensity,
            radius_px: length_px,
            taps: LIGHT_TAPS as f32,
            kind: 1.0,
            _pad0: 0.0,
            dir_px: dir_unit,
            half_width_px,
            _pad1: 0.0,
            _tail0: 0.0,
            _tail1: 0.0,
            _tail2: 0.0,
            _tail3: 0.0,
        }
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
        format: LIGHT_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}
