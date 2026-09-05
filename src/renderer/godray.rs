//! Offscreen god-ray pass.
//!
//! Sits alongside `sun.rs`. The sun pass writes lit-window + lit-wall
//! contribution; this pass writes the soft volumetric shafts streaming
//! from lit windows into interior spaces.
//!
//! Pipeline:
//!   1. `fs_seed`     — reads `wall_mask`, writes 1.0 into `seed_map` at
//!      lit-window pixels (sun-facing hull), 0 elsewhere.
//!   2. `fs_march`    — samples `seed_map` + `wall_mask` together, marches
//!      N taps away from the sun with decaying weight,
//!      respects opaque wall occluders. Writes coloured
//!      beam contribution into `beam_map`.
//!   3. `fs_composite`— additively blends `beam_map` onto the main pass
//!      at the caller's split index.
//!
//! Using an offscreen texture + weighted march avoids the tile-aligned
//! banding of the per-pixel raycast approach — sums are continuous, not
//! boolean thresholds.

use bytemuck::{Pod, Zeroable};
use glam::Vec2;

use super::shadow::MASK_SCALE;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct GodrayUniform {
    sun_dir_uv: [f32; 2],
    taps:       f32,
    step_scale: f32,
    color:      [f32; 3],
    intensity:  f32,
    time:       f32,
    _pad0:      f32,
    _pad1:      f32,
    _pad2:      f32,
}

/// March tap count. Higher = longer shafts and smoother edges at the cost
/// of texture bandwidth. 48 taps × 1080p ≈ 100M samples/frame — sub-ms on
/// any dedicated GPU.
pub const GODRAY_TAPS: u32 = 48;

/// Per-tap UV step as a multiplier on `sun_dir_uv` (which is one-tile per
/// step from the caller). 0.75 keeps taps tightly overlapping so beams
/// don't gap between samples but still covers 48 * 0.75 ≈ 36 tiles of
/// reach — plenty to cross an interior room.
pub const GODRAY_STEP_SCALE: f32 = 0.75;

const GODRAY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub(super) struct GodrayPass {
    pub width:  u32,
    pub height: u32,

    // Seed texture — 1.0 at lit-window pixels, 0 elsewhere.
    pub seed_view: wgpu::TextureView,
    _seed_tex:     wgpu::Texture,
    // Beam texture — output of the march. Composited onto the main pass.
    pub beam_view: wgpu::TextureView,
    _beam_tex:     wgpu::Texture,

    pub seed_pipeline:      wgpu::RenderPipeline,
    pub march_pipeline:     wgpu::RenderPipeline,
    pub composite_pipeline: wgpu::RenderPipeline,

    // Bind groups match the two layouts below.
    pub bg_seed:      wgpu::BindGroup, // reads wall_mask
    pub bg_march:     wgpu::BindGroup, // reads seed + wall_mask
    pub bg_composite: wgpu::BindGroup, // reads beam

    pub uniform: wgpu::Buffer,
    _sampler:    wgpu::Sampler,
    _bgl_single: wgpu::BindGroupLayout,
    _bgl_double: wgpu::BindGroupLayout,
}

impl GodrayPass {
    pub fn new(
        device: &wgpu::Device,
        wall_mask_view: &wgpu::TextureView,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Option<Self> {
        if width == 0 || height == 0 { return None; }

        // Seed texture is the same size as `wall_mask` (MASK_SCALE
        // × surface) so seed and mask can be sampled at identical UVs
        // during the march pass. beam_map is composited additively at
        // surface UV and stays surface-sized.
        let mask_width  = ((width  as f32) * MASK_SCALE).round().max(1.0) as u32;
        let mask_height = ((height as f32) * MASK_SCALE).round().max(1.0) as u32;
        let (seed_tex, seed_view) = make_offscreen(device, mask_width, mask_height, "godray_seed");
        let (beam_tex, beam_view) = make_offscreen(device, width, height, "godray_beam");

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("godray_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Layout A: uniform + one texture + sampler. Used by seed and
        // composite stages.
        let bgl_single = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("godray_bgl_single"),
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

        // Layout B: same as A plus a second texture at binding 3. Used by
        // the march stage — t_src=seed, t_mask=wall_mask.
        let bgl_double = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("godray_bgl_double"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
            ],
        });

        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("godray_uniform"),
            size: std::mem::size_of::<GodrayUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg_seed = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_godray_seed"),
            layout: &bgl_single,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(wall_mask_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });
        let bg_march = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_godray_march"),
            layout: &bgl_double,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&seed_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(wall_mask_view) },
            ],
        });
        let bg_composite = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_godray_composite"),
            layout: &bgl_single,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&beam_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("godray.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("godray.wgsl").into()),
        });

        let pl_single = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("godray_pl_single"),
            bind_group_layouts: &[&bgl_single],
            push_constant_ranges: &[],
        });
        let pl_double = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("godray_pl_double"),
            bind_group_layouts: &[&bgl_double],
            push_constant_ranges: &[],
        });

        let seed_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("godray_seed_pipeline"),
            layout: Some(&pl_single),
            vertex: wgpu::VertexState {
                module: &shader, entry_point: "vs_fullscreen",
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: "fs_seed",
                targets: &[Some(wgpu::ColorTargetState {
                    format: GODRAY_FORMAT,
                    blend: None,
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

        let march_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("godray_march_pipeline"),
            layout: Some(&pl_double),
            vertex: wgpu::VertexState {
                module: &shader, entry_point: "vs_fullscreen",
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: "fs_march",
                targets: &[Some(wgpu::ColorTargetState {
                    format: GODRAY_FORMAT,
                    blend: None,
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

        // Additive composite onto main pass.
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("godray_composite_pipeline"),
            layout: Some(&pl_single),
            vertex: wgpu::VertexState {
                module: &shader, entry_point: "vs_fullscreen",
                buffers: &[], compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader, entry_point: "fs_composite",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::One,
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
            width, height,
            seed_view, _seed_tex: seed_tex,
            beam_view, _beam_tex: beam_tex,
            seed_pipeline, march_pipeline, composite_pipeline,
            bg_seed, bg_march, bg_composite,
            uniform,
            _sampler: sampler,
            _bgl_single: bgl_single,
            _bgl_double: bgl_double,
        })
    }

    /// Populate the godray uniform for this frame. Uses the same `tile_px`
    /// step convention as the sun pass — one full-tile step per march
    /// unit — but the march multiplies this by `GODRAY_STEP_SCALE` so
    /// beams remain continuous rather than gapping between taps.
    pub fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        sun_dir_screen: Vec2,
        intensity: f32,
        color: [f32; 3],
        tile_px: f32,
        time: f32,
    ) {
        let d = sun_dir_screen.normalize_or_zero();
        // Match SunPass: one on-screen tile per uniform step, so beam
        // reach scales with zoom. Half-tile granularity keeps the march
        // fine enough to sample the seed texture smoothly.
        let step_px = (tile_px * 0.5).max(1.0);
        // sun_dir_uv is in *mask* UV space (wall_mask + seed are both
        // MASK_SCALE × surface). Match the sun pass so per-step real-
        // world reach is unchanged.
        let du = d.x * step_px / (self.width  as f32 * MASK_SCALE);
        let dv = d.y * step_px / (self.height as f32 * MASK_SCALE);
        let u = GodrayUniform {
            sun_dir_uv: [du, dv],
            taps:       GODRAY_TAPS as f32,
            step_scale: GODRAY_STEP_SCALE,
            color,
            intensity,
            time,
            _pad0: 0.0, _pad1: 0.0, _pad2: 0.0,
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
        format: GODRAY_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    (tex, view)
}
