//! Off-screen post-process pipeline.
//!
//! Exists so callers can render a sub-scene (currently: the "roof of lower
//! floors" pre-pass on the on-foot view) into an off-screen texture, run a
//! two-pass separable Gaussian blur over it, then composite the blurred
//! result back into the main pass. Structured as a full mini-pipeline (two
//! ping-pong textures, blur shader, composite draw) rather than a one-shot
//! helper so future effects (bloom, fog-of-war, aim-focus) can hang off the
//! same offscreen target without another surgery of the main frame loop.

use bytemuck::{Pod, Zeroable};

use super::batch::Vertex;

/// GPU uniform for the blur shader. Layout kept flat + `Pod` so a single
/// `write_buffer` copies it verbatim.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BlurUniform {
    axis:   f32, // 0 = horizontal, 1 = vertical
    radius: f32,
    px_x:   f32, // 1 / tex width
    px_y:   f32, // 1 / tex height
}

pub(super) struct PostProcess {
    pub width:  u32,
    pub height: u32,

    // Ping-pong render targets. blur_a is the offscreen batch target and
    // the final resolved image; blur_b is the intermediate H-blur output.
    pub blur_a_view: wgpu::TextureView,
    pub blur_b_view: wgpu::TextureView,
    _blur_a_tex: wgpu::Texture,
    _blur_b_tex: wgpu::Texture,

    // Pipeline for drawing the offscreen `Batch` into blur_a. Shares vertex
    // layout with the main pipeline but writes to Rgba8UnormSrgb.
    pub offscreen_pipeline: wgpu::RenderPipeline,

    // Blur (H+V) and composite pipelines. Composite draws blur_a into the
    // main pass with alpha blending.
    pub blur_pipeline:      wgpu::RenderPipeline,
    pub composite_pipeline: wgpu::RenderPipeline,

    // Bind groups feeding the blur/composite shaders. Two uniform buffers
    // (one per axis) because `queue.write_buffer` collapses to a single
    // upload at submit time — sharing one buffer between H and V would
    // make both passes read the last-written axis.
    _sample_bgl: wgpu::BindGroupLayout,
    pub bg_sample_a_h: wgpu::BindGroup, // reads blur_a with H uniform
    pub bg_sample_b_v: wgpu::BindGroup, // reads blur_b with V uniform
    pub bg_composite:  wgpu::BindGroup, // reads blur_a with copy uniform

    pub uniform_h: wgpu::Buffer,
    pub uniform_v: wgpu::Buffer,
    pub uniform_copy: wgpu::Buffer,
    _sampler: wgpu::Sampler,
}

pub(super) const OFFSCREEN_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

impl PostProcess {
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

        let (blur_a_tex, blur_a_view) = make_offscreen(device, width, height, "blur_a");
        let (blur_b_tex, blur_b_view) = make_offscreen(device, width, height, "blur_b");

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("blur_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let sample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("blur_sample_bgl"),
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

        let mk_uniform = |label: &str| device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: std::mem::size_of::<BlurUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_h    = mk_uniform("blur_uniform_h");
        let uniform_v    = mk_uniform("blur_uniform_v");
        let uniform_copy = mk_uniform("blur_uniform_copy");

        let bg_sample_a_h = make_sample_bg(
            device, &sample_bgl, &uniform_h, &blur_a_view, &sampler, "bg_sample_a_h",
        );
        let bg_sample_b_v = make_sample_bg(
            device, &sample_bgl, &uniform_v, &blur_b_view, &sampler, "bg_sample_b_v",
        );
        let bg_composite = make_sample_bg(
            device, &sample_bgl, &uniform_copy, &blur_a_view, &sampler, "bg_composite",
        );

        // Blur/composite share the same sample_bgl layout and vertex-less
        // fullscreen triangle.
        let blur_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("blur.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("blur.wgsl").into()),
        });

        let blur_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("blur_pl"),
            bind_group_layouts: &[&sample_bgl],
            push_constant_ranges: &[],
        });

        let blur_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blur_pipeline"),
            layout: Some(&blur_layout),
            vertex: wgpu::VertexState {
                module: &blur_shader,
                entry_point: "vs_fullscreen",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blur_shader,
                entry_point: "fs_blur",
                targets: &[Some(wgpu::ColorTargetState {
                    format: OFFSCREEN_FORMAT,
                    blend: None, // full overwrite between ping-pong passes
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

        let composite_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("blur_composite_pipeline"),
            layout: Some(&blur_layout),
            vertex: wgpu::VertexState {
                module: &blur_shader,
                entry_point: "vs_fullscreen",
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &blur_shader,
                entry_point: "fs_copy",
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        // Offscreen pipeline reuses the main shader (same vertex layout) but
        // targets OFFSCREEN_FORMAT and starts with a transparent clear so the
        // blur only reads pixels the caller actually wrote.
        let main_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("offscreen_main_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let offscreen_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("offscreen_pl"),
            bind_group_layouts: &[camera_bgl, texture_bgl],
            push_constant_ranges: &[],
        });

        let offscreen_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("offscreen_pipeline"),
            layout: Some(&offscreen_layout),
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
                    format: OFFSCREEN_FORMAT,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        Some(Self {
            width,
            height,
            blur_a_view,
            blur_b_view,
            _blur_a_tex: blur_a_tex,
            _blur_b_tex: blur_b_tex,
            offscreen_pipeline,
            blur_pipeline,
            composite_pipeline,
            _sample_bgl: sample_bgl,
            bg_sample_a_h,
            bg_sample_b_v,
            bg_composite,
            uniform_h,
            uniform_v,
            uniform_copy,
            _sampler: sampler,
        })
    }

    /// Populate the three per-axis uniform buffers for this frame's blur
    /// radius. Called once per frame before the blur passes are encoded.
    pub fn write_uniforms(&self, queue: &wgpu::Queue, radius: f32) {
        let px_x = 1.0 / self.width  as f32;
        let px_y = 1.0 / self.height as f32;
        let h = BlurUniform { axis: 0.0, radius, px_x, px_y };
        let v = BlurUniform { axis: 1.0, radius, px_x, px_y };
        let c = BlurUniform { axis: 0.0, radius: 0.0, px_x, px_y };
        queue.write_buffer(&self.uniform_h,    0, bytemuck::bytes_of(&h));
        queue.write_buffer(&self.uniform_v,    0, bytemuck::bytes_of(&v));
        queue.write_buffer(&self.uniform_copy, 0, bytemuck::bytes_of(&c));
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
        format: OFFSCREEN_FORMAT,
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
