//! `Renderer::new` — GPU context bring-up, pipeline + bind-group layout
//! creation, and the initial allocation of every offscreen pass.
//!
//! Split out of `renderer/mod.rs` (was 1351 lines): construction is a long
//! straight-line sequence that is read once and rarely touched, so keeping
//! it beside the per-frame hot path made both harder to navigate. `mod.rs`
//! retains the `Renderer` struct definition, `resize`, and the small
//! queue/accessor methods.
//!
//! This is a child module of `renderer`, so it reaches `Renderer`'s private
//! fields directly — no visibility widening was needed for the split.

use super::*;

const WHITE_PIXEL: &[u8] = &[255, 255, 255, 255];

impl Renderer {
    pub fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let gpu = GpuContext::new(window.clone());
        let viewport = Vec2::new(size.width as f32, size.height as f32);
        let camera = Camera2D::new(viewport);

        let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("main shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // Camera uniform buffer
        let camera_uniform = camera.build_uniform();
        let camera_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_buffer"),
            contents: bytemuck::bytes_of(&camera_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let camera_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("camera_bgl"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let camera_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_bg"),
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
        });

        // White 1x1 texture for colored primitives
        let white_tex = gpu.device.create_texture_with_data(
            &gpu.queue,
            &wgpu::TextureDescriptor {
                label: Some("white"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            WHITE_PIXEL,
        );
        let white_view = white_tex.create_view(&Default::default());
        let white_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let texture_bgl =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("texture_bgl"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float {
                                    filterable: true,
                                },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(
                                wgpu::SamplerBindingType::Filtering,
                            ),
                            count: None,
                        },
                    ],
                });
        let white_texture_bind_group =
            gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("white_bg"),
                layout: &texture_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&white_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&white_sampler),
                    },
                ],
            });

        let pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("pipeline_layout"),
                    bind_group_layouts: &[&camera_bgl, &texture_bgl],
                    push_constant_ranges: &[],
                });

        let pipeline =
            gpu.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("main_pipeline"),
                    layout: Some(&pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: "vs_main",
                        buffers: &[Vertex::desc()],
                        compilation_options: Default::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: "fs_main",
                        targets: &[Some(wgpu::ColorTargetState {
                            format: gpu.format,
                            blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: Default::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,
                        ..Default::default()
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                });

        let vertex_capacity = 65536;
        let index_capacity = 196608;
        let vertex_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vbuf"),
            size: (vertex_capacity * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let index_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ibuf"),
            size: (index_capacity * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Separate offscreen vertex/index buffers so the offscreen batch and
        // the main batch don't stomp on each other's GPU memory when both are
        // encoded into the same frame.
        let offscreen_vcap = 8192;
        let offscreen_icap = 24576;
        let offscreen_vbuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offscreen_vbuf"),
            size: (offscreen_vcap * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let offscreen_ibuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offscreen_ibuf"),
            size: (offscreen_icap * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let postprocess = PostProcess::new(
            &gpu.device,
            &camera_bgl,
            &texture_bgl,
            gpu.format,
            gpu.surface_config.width,
            gpu.surface_config.height,
        );
        if postprocess.is_none() {
            log::warn!("[renderer] postprocess pipeline unavailable — blur passes will be skipped");
        }

        let shadow = ShadowPass::new(
            &gpu.device,
            &camera_bgl,
            &texture_bgl,
            gpu.format,
            gpu.surface_config.width,
            gpu.surface_config.height,
        );
        if shadow.is_none() {
            log::warn!("[renderer] shadow pipeline unavailable — shadow passes will be skipped");
        }

        // Light pass shares the wall_mask texture with the (now-inert)
        // shadow pipeline — same rasterised solid-tile + character
        // silhouettes are the occluders for point lights.
        let lights_pass = shadow.as_ref().and_then(|sh| LightPass::new(
            &gpu.device,
            &sh.wall_mask_view,
            gpu.format,
            gpu.surface_config.width,
            gpu.surface_config.height,
        ));
        if lights_pass.is_none() {
            log::warn!("[renderer] light pipeline unavailable — on-foot lighting will be skipped");
        }

        // Sun pass — reuses the shadow pipeline's wall_mask texture so the
        // same solid-tile + character-silhouette rasterisation drives
        // sun occlusion. `None` when the underlying wall_mask isn't
        // available (allocation failure / zero surface).
        let sun_pass = shadow.as_ref().and_then(|sh| SunPass::new(
            &gpu.device,
            &sh.wall_mask_view,
            gpu.format,
            gpu.surface_config.width,
            gpu.surface_config.height,
        ));
        if sun_pass.is_none() {
            log::warn!("[renderer] sun pipeline unavailable — global sun pass will be skipped");
        }

        // Godray pass — same story as sun_pass; shares the wall_mask.
        let godray_pass = shadow.as_ref().and_then(|sh| GodrayPass::new(
            &gpu.device,
            &sh.wall_mask_view,
            gpu.format,
            gpu.surface_config.width,
            gpu.surface_config.height,
        ));
        if godray_pass.is_none() {
            log::warn!("[renderer] godray pipeline unavailable — god-ray beams will be skipped");
        }

        // Same growth pattern as `offscreen_vbuf`. Mask geometry = solid
        // tile rects + character rects on the visible floor; a modest
        // starting cap covers typical VisRange sizes and doubles on demand.
        let mask_vcap = 16384;
        let mask_icap = 49152;
        let mask_vbuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mask_vbuf"),
            size: (mask_vcap * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mask_ibuf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mask_ibuf"),
            size: (mask_icap * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            gpu,
            camera,
            batch: Batch::new(),
            pipeline,
            camera_buffer,
            camera_bind_group,
            vertex_buffer,
            index_buffer,
            white_texture_bind_group,
            vertex_capacity,
            index_capacity,
            shake_trauma: 0.0,
            shake_offset: Vec2::ZERO,
            screenshot_pending: false,
            screenshot_data:    None,
            last_perf: crate::app::PerfSnapshot::default(),
            window,
            postprocess,
            offscreen_batch: Batch::new(),
            offscreen_vbuf,
            offscreen_ibuf,
            offscreen_vcap,
            offscreen_icap,
            pending_blur_radius: None,
            composite_split_index: None,
            camera_bgl,
            texture_bgl,
            shadow,
            mask_batch: Batch::new(),
            mask_vbuf,
            mask_ibuf,
            mask_vcap,
            mask_icap,
            pending_shadow: None,
            shadow_split_index: None,
            lights_pass,
            pending_lights: Vec::with_capacity(MAX_LIGHTS_PER_FRAME),
            lights_split_index: None,
            lights_pending: false,
            sun_pass,
            pending_sun: None,
            sun_split_index: None,
            godray_pass,
            pending_godray: None,
            godray_split_index: None,
        }
    }
}
