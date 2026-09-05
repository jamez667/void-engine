//! Global directional-sun light pass.
//!
//! Sits alongside `shadow.rs` and `lights.rs`. Consumes the wall_mask
//! texture owned by `ShadowPass` (same occluders — solid tiles minus
//! windows, plus character silhouettes) and writes a full-viewport
//! `sun_map` of the sun contribution per pixel. A composite pass then
//! additively blends `sun_map` onto the main scene at the caller's
//! chosen split index, so pixels with line-of-sight to the sun pick
//! up a warm daylight tint while shadowed pixels are untouched.
//!
//! The sun is at infinity → parallel rays. The uniform gives a per-step
//! delta in UV space (pre-aspect-corrected by the caller). Wall pixels
//! that would otherwise self-occlude get a small `wall_lit_scale` dose
//! so outer walls glow rather than reading as pitch-black silhouettes.

use bytemuck::{Pod, Zeroable};
use glam::Vec2;

use super::shadow::MASK_SCALE;

/// Uniform driving the sun march shader. Kept `Pod` for a straight
/// `write_buffer` copy. std140-safe: vec3<color> is padded to 16 bytes
/// before `intensity`, matching Rust field order below.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct SunUniform {
    sun_dir_uv:     [f32; 2], // per-step delta in UV space (toward the sun)
    taps:           f32,      // number of march steps
    wall_lit_scale: f32,      // multiplier for lit-wall pixels
    color:          [f32; 3], // linear RGB
    intensity:      f32,      // scalar multiplier
    time:           f32,      // game time (s) — drives dust-mote drift
    _pad0:          f32,
    _pad1:          f32,
    _pad2:          f32,
}

/// Number of march steps per pixel. Enough to march from an interior
/// wall all the way to the outer hull — a big floor is ~50 tiles wide;
/// at 12 px per step (~half a tile) we need ~100 taps to cross it.
/// Perf-wise: 100 × 1080p ≈ 200M texture samples/frame. Fine on modern
/// GPUs.
pub const SUN_TAPS: u32 = 128;

/// Fraction of full sun applied to pixels inside a wall body. Keeps
/// outer walls readable as sunlit architecture without blowing out.
pub const SUN_WALL_LIT_SCALE: f32 = 0.5;

/// `Rgba16Float` — sun contribution is added on top of the main pass,
/// so keeping HDR headroom lets warm daylight lift the exposure a hair
/// without clipping. Fits into the same "offscreen surface-sized" mould
/// as the other passes; the composite additive blend handles it fine.
const SUN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub(super) struct SunPass {
    pub width:  u32,
    pub height: u32,

    pub sun_map_view: wgpu::TextureView,
    _sun_map_tex:     wgpu::Texture,

    pub sun_pipeline:       wgpu::RenderPipeline,
    pub composite_pipeline: wgpu::RenderPipeline,

    _sample_bgl: wgpu::BindGroupLayout,
    pub bg_sun:       wgpu::BindGroup, // reads wall_mask
    pub bg_composite: wgpu::BindGroup, // reads sun_map

    pub uniform: wgpu::Buffer,
    _sampler:    wgpu::Sampler,
}

impl SunPass {
    pub fn new(
        device: &wgpu::Device,
        wall_mask_view: &wgpu::TextureView,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Option<Self> {
        if width == 0 || height == 0 { return None; }

        let (sun_map_tex, sun_map_view) = make_offscreen(device, width, height, "sun_map");

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("sun_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let sample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("sun_sample_bgl"),
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
            label: Some("sun_uniform"),
            size: std::mem::size_of::<SunUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg_sun = make_sample_bg(
            device, &sample_bgl, &uniform, wall_mask_view, &sampler, "bg_sun",
        );
        let bg_composite = make_sample_bg(
            device, &sample_bgl, &uniform, &sun_map_view, &sampler, "bg_sun_composite",
        );

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sun.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("sun.wgsl").into()),
        });

        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("sun_pl"),
            bind_group_layouts: &[&sample_bgl],
            push_constant_ranges: &[],
        });

        let sun_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sun_pipeline"),
            layout: Some(&pl_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_fullscreen",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_sun",
                targets: &[Some(wgpu::ColorTargetState {
                    format: SUN_FORMAT,
                    blend: None, // full overwrite of sun_map
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

        // Additive composite: `out = dst + src`. Sun contribution is
        // laid over whatever the main pass painted.
        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("sun_composite_pipeline"),
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
            width,
            height,
            sun_map_view,
            _sun_map_tex: sun_map_tex,
            sun_pipeline,
            composite_pipeline,
            _sample_bgl: sample_bgl,
            bg_sun,
            bg_composite,
            uniform,
            _sampler: sampler,
        })
    }

    /// Populate the sun uniform for this frame. `sun_dir_screen` is the
    /// direction TOWARD the sun in screen pixel space (y grows downward
    /// to match wall_mask uv). We aspect-correct to a per-step UV delta
    /// sized so the march step clears roughly one tile.
    pub fn write_uniforms(
        &self,
        queue: &wgpu::Queue,
        sun_dir_screen: Vec2,
        intensity: f32,
        color: [f32; 3],
        tile_px: f32,
        time: f32,
    ) {
        let n = SUN_TAPS.max(1) as f32;
        let d = sun_dir_screen.normalize_or_zero();
        // Per-step size = half the on-screen tile size in surface pixels.
        // Client passes `tile_px` = one-tile in surface pixels at current
        // zoom, so this stays correct regardless of how zoomed in/out
        // the on-foot view is. Half a tile guarantees we hit the
        // neighbour tile without skipping over thin walls.
        let step_px = (tile_px * 0.5).max(1.0);
        let per_step_px_x = d.x * step_px;
        let per_step_px_y = d.y * step_px;
        // Express the step in *mask* UV space (wall_mask is MASK_SCALE
        // larger than the surface on each axis). Dividing by the mask
        // extent keeps the per-step real-world distance identical to
        // pre-scaling: `step_px / (surface * MASK_SCALE)` == the same
        // slice of world as `step_px / surface` was in the un-enlarged
        // texture. Consumer shader translates its starting `in.uv` into
        // mask-UV before marching.
        let du = per_step_px_x / (self.width  as f32 * MASK_SCALE);
        let dv = per_step_px_y / (self.height as f32 * MASK_SCALE);
        let u = SunUniform {
            sun_dir_uv:     [du, dv],
            taps:           n,
            wall_lit_scale: SUN_WALL_LIT_SCALE,
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
        format: SUN_FORMAT,
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
