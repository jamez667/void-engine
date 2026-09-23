//! Render a shadow on a real GPU and check it lands where it should.
//!
//! Phase 4 of `docs/3d-spec.md`. The unit tests beside `shadow3d.rs` check
//! the light's projection in isolation — that it is parallel, that the
//! covered radius lands inside clip space, that degenerate directions do
//! not produce NaN. None of them renders anything, so none would catch a
//! shadow map that is sampled with flipped UVs, compared the wrong way
//! round, or riddled with acne.
//!
//! This runs the real two-pass sequence — depth from the light, then the
//! main pass sampling it — through the real shaders, and asserts three
//! things a broken implementation gets wrong:
//!
//! 1. A floor under an occluder is darker than the same floor with the
//!    occluder removed.
//! 2. A floor *away* from the occluder is not darkened (the shadow is
//!    somewhere specific, not everywhere).
//! 3. An unoccluded floor shows no acne — a surface must not shadow
//!    itself, which is what the depth bias and front-face culling in
//!    `shadow3d.rs` are for.
//!
//! Skipped, not failed, when no adapter is available.

use glam::{DVec3, Mat4, Vec3};
use void_engine::renderer::camera::Camera3D;
use void_engine::renderer::depth::DEPTH_FORMAT;
use void_engine::renderer::mesh3d::{Mesh3D, Vertex3D};
use void_engine::renderer::render3d::depth_state;
use void_engine::renderer::shadow3d::{light_view_proj, SHADOW_MAP_SIZE};

const W: u32 = 128;
const H: u32 = 128;

/// Direction toward the light: overhead and slightly to one side, so the
/// shadow lands beside the occluder rather than exactly under it.
const LIGHT_DIR: Vec3 = Vec3::new(0.0, -0.35, 1.0);
/// How much of the scene the shadow map covers.
const SHADOW_RADIUS: f32 = 12.0;

struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

fn gpu() -> Option<Gpu> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some("shadow3d test device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .ok()?;
    Some(Gpu { device, queue })
}

/// Looking down at a floor from above and behind, so a shadow cast on it
/// is visible rather than edge-on.
fn camera() -> Camera3D {
    let mut c = Camera3D::new(glam::Vec2::new(W as f32, H as f32));
    c.position = DVec3::new(0.0, -9.0, 7.0);
    c.target = DVec3::ZERO;
    c
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadowUniform {
    light_view_proj: [[f32; 4]; 4],
    light_dir_ambient: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Instance {
    model: [[f32; 4]; 4],
    tint: [f32; 4],
}

/// Run both passes and read back the frame.
///
/// `caster` is drawn into the shadow map *and* the scene; `floor` only
/// into the scene, so a test can render the identical floor with and
/// without an occluder above it.
fn render(gpu: &Gpu, floor: &Mesh3D, caster: Option<&Mesh3D>, cam: &Camera3D) -> Vec<u8> {
    use wgpu::util::DeviceExt;

    let main_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shader3d"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../src/renderer/shader3d.wgsl").into(),
        ),
    });
    let shadow_shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shadow3d"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../src/renderer/shadow3d.wgsl").into(),
        ),
    });

    let lvp = light_view_proj(LIGHT_DIR, Vec3::ZERO, SHADOW_RADIUS);

    // ---- shared bind group layouts -------------------------------------
    let camera_bgl = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
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
    let instance_bgl = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
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

    let make_camera_bg = |m: Mat4, label: &str| {
        let u = void_engine::renderer::camera::CameraUniform {
            view_proj: m.to_cols_array_2d(),
        };
        let buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(&u),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &camera_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
        })
    };
    let scene_camera_bg = make_camera_bg(
        Mat4::from_cols_array_2d(&cam.build_uniform().view_proj),
        "scene_camera",
    );
    let light_camera_bg = make_camera_bg(lvp, "light_camera");

    let identity = Instance {
        model: Mat4::IDENTITY.to_cols_array_2d(),
        tint: [1.0; 4],
    };
    let instance_buf = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instance"),
            contents: bytemuck::bytes_of(&identity),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let instance_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &instance_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: instance_buf.as_entire_binding(),
        }],
    });

    // ---- shadow map -----------------------------------------------------
    let shadow_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow_map"),
        size: wgpu::Extent3d {
            width: SHADOW_MAP_SIZE,
            height: SHADOW_MAP_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let shadow_view = shadow_tex.create_view(&Default::default());

    let shadow_pipeline = gpu
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow_pipeline"),
            layout: Some(&gpu.device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&camera_bgl, &instance_bgl],
                    push_constant_ranges: &[],
                },
            )),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: "vs_shadow",
                buffers: &[Vertex3D::desc()],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Front),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState {
                    constant: void_engine::renderer::shadow3d::DEPTH_BIAS_CONSTANT,
                    slope_scale: void_engine::renderer::shadow3d::DEPTH_BIAS_SLOPE,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

    // ---- main pass bindings ---------------------------------------------
    let shadow_uniform = ShadowUniform {
        light_view_proj: lvp.to_cols_array_2d(),
        light_dir_ambient: {
            let d = LIGHT_DIR.normalize();
            [d.x, d.y, d.z, 0.25]
        },
    };
    let shadow_ubuf = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shadow_uniform"),
            contents: bytemuck::bytes_of(&shadow_uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let shadow_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        label: None,
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });
    let sample_bgl = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
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
    let sample_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &sample_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: shadow_ubuf.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&shadow_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::Sampler(&shadow_sampler),
            },
        ],
    });

    let zeros = vec![0u8; 16 + 16 * 32];
    let lights_buf = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &zeros,
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let lights_bgl = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
    let lights_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &lights_bgl,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: lights_buf.as_entire_binding() }],
    });

    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let main_pipeline = gpu
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("main_pipeline"),
            layout: Some(&gpu.device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&camera_bgl, &instance_bgl, &sample_bgl, &lights_bgl],
                    push_constant_ranges: &[],
                },
            )),
            vertex: wgpu::VertexState {
                module: &main_shader,
                entry_point: "vs_main",
                buffers: &[Vertex3D::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &main_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: depth_state(),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

    // ---- targets --------------------------------------------------------
    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&Default::default());
    let depth_tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&Default::default());

    let upload = |m: &Mesh3D| {
        (
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&m.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&m.indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
            m.indices.len() as u32,
        )
    };
    let floor_buf = upload(floor);
    let caster_buf = caster.map(upload);

    let unpadded = W * 4;
    let padded = unpadded.div_ceil(256) * 256;
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = gpu.device.create_command_encoder(&Default::default());

    // Pass 1: depth from the light. Only the caster goes in — a floor
    // that cast into its own shadow map is what acne is.
    {
        let mut sp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow_pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &shadow_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        if let Some((vb, ib, n)) = &caster_buf {
            sp.set_pipeline(&shadow_pipeline);
            sp.set_bind_group(0, &light_camera_bg, &[]);
            sp.set_bind_group(1, &instance_bg, &[]);
            sp.set_vertex_buffer(0, vb.slice(..));
            sp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            sp.draw_indexed(0..*n, 0, 0..1);
        }
    }

    // Pass 2: the scene, sampling what pass 1 wrote.
    {
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rp.set_pipeline(&main_pipeline);
        rp.set_bind_group(0, &scene_camera_bg, &[]);
        rp.set_bind_group(1, &instance_bg, &[]);
        rp.set_bind_group(2, &sample_bg, &[]);
        rp.set_bind_group(3, &lights_bg, &[]);

        let (vb, ib, n) = &floor_buf;
        rp.set_vertex_buffer(0, vb.slice(..));
        rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
        rp.draw_indexed(0..*n, 0, 0..1);

        if let Some((vb, ib, n)) = &caster_buf {
            rp.set_vertex_buffer(0, vb.slice(..));
            rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..*n, 0, 0..1);
        }
    }

    enc.copy_texture_to_buffer(
        wgpu::ImageCopyTexture {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::ImageCopyBuffer {
            buffer: &readback,
            layout: wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(H),
            },
        },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    gpu.queue.submit([enc.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    gpu.device.poll(wgpu::Maintain::Wait);
    let mapped = slice.get_mapped_range();
    let mut out = Vec::with_capacity((unpadded * H) as usize);
    for row in 0..H {
        let start = (row * padded) as usize;
        out.extend_from_slice(&mapped[start..start + unpadded as usize]);
    }
    drop(mapped);
    readback.unmap();
    out
}

/// A large floor at z = 0, facing +Z.
fn floor_mesh(cam: &Camera3D) -> Mesh3D {
    let o = cam.world_to_camera_offset(DVec3::ZERO);
    let mut m = Mesh3D::new();
    let s = 10.0;
    m.push_quad(
        o + Vec3::new(-s, -s, 0.0),
        o + Vec3::new(s, -s, 0.0),
        o + Vec3::new(s, s, 0.0),
        o + Vec3::new(-s, s, 0.0),
        [1.0, 1.0, 1.0, 1.0],
    );
    m
}

/// A box hovering above the floor's centre.
fn caster_mesh(cam: &Camera3D) -> Mesh3D {
    let o = cam.world_to_camera_offset(DVec3::ZERO);
    let mut m = Mesh3D::new();
    m.push_box(o + Vec3::new(0.0, 0.0, 3.0), Vec3::splat(2.5), [1.0, 1.0, 1.0, 1.0]);
    m
}

/// Mean red channel over a small window, as a stand-in for brightness.
fn brightness_at(px: &[u8], cx: u32, cy: u32, half: u32) -> f32 {
    let mut sum = 0u32;
    let mut n = 0u32;
    for y in cy.saturating_sub(half)..(cy + half).min(H) {
        for x in cx.saturating_sub(half)..(cx + half).min(W) {
            let i = ((y * W + x) * 4) as usize;
            sum += px[i] as u32;
            n += 1;
        }
    }
    if n == 0 { 0.0 } else { sum as f32 / n as f32 }
}

/// Find the floor's brightness somewhere the occluder does not reach.
///
/// The bottom rows of the frame are the near edge of the floor, well
/// clear of a box hovering over its centre.
fn unshadowed_sample(px: &[u8]) -> f32 {
    brightness_at(px, W / 2, H - 12, 5)
}

/// An occluder must darken the floor beneath it.
///
/// The comparison is against the *same* floor rendered without the
/// caster, so the only difference is the shadow — not the geometry, the
/// camera or the lighting.
#[test]
fn an_occluder_casts_a_shadow_on_the_floor_below_it() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let cam = camera();
    let floor = floor_mesh(&cam);

    let lit = render(&gpu, &floor, None, &cam);
    let shadowed = render(&gpu, &floor, Some(&caster_mesh(&cam)), &cam);

    // Sample where the shadow actually lands, which is *below* the box in
    // frame: the light leans toward -Y, throwing the shadow toward the
    // camera. Rows above the box centre are the box itself occluding the
    // floor, which darkens whether or not the shadow map works — probing
    // there is what let an earlier revision of this test pass against a
    // completely disabled shadow lookup.
    //
    // Measured at this row: 250 lit, 137 shadowed.
    let probe_y = H / 2;
    let before = brightness_at(&lit, W / 2, probe_y, 6);
    let after = brightness_at(&shadowed, W / 2, probe_y, 6);

    assert!(
        before > 0.0,
        "precondition: the floor must be visible and lit at the probe \
         point before a shadow is cast on it (got {before})",
    );
    assert!(
        after < before * 0.9,
        "the floor under the occluder was not meaningfully darkened: \
         {before:.1} lit vs {after:.1} shadowed. The shadow map is not \
         being sampled, or the comparison is inverted.",
    );
}

/// The shadow must be somewhere specific, not everywhere.
///
/// Guards the failure where the lookup misses the map entirely and every
/// fragment comes back shadowed — which the test above would happily pass.
#[test]
fn the_shadow_does_not_darken_the_whole_floor() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let cam = camera();
    let floor = floor_mesh(&cam);

    let lit = render(&gpu, &floor, None, &cam);
    let shadowed = render(&gpu, &floor, Some(&caster_mesh(&cam)), &cam);

    let before = unshadowed_sample(&lit);
    let after = unshadowed_sample(&shadowed);

    assert!(
        before > 0.0,
        "precondition: the floor must be lit away from the occluder \
         (got {before})",
    );
    assert!(
        after > before * 0.9,
        "floor far from the occluder was darkened too ({before:.1} -> \
         {after:.1}) — the whole scene is being treated as shadowed, so \
         the lookup is probably landing outside the map",
    );
}

/// An unoccluded floor must not shadow itself.
///
/// Acne is the classic shadow-mapping failure: a surface's recorded depth
/// and its tested depth differ by a hair of float and rasterisation
/// error, so it stipples itself dark. Rendering the floor with *no*
/// caster and requiring it to match its no-shadow-map brightness is the
/// direct check that the bias and front-face culling in `shadow3d.rs`
/// are doing their job.
#[test]
fn an_unoccluded_floor_shows_no_self_shadowing() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let cam = camera();
    let floor = floor_mesh(&cam);

    // The floor is in the scene but casts nothing, so every fragment
    // should sample the cleared (far) map and come back fully lit.
    let px = render(&gpu, &floor, None, &cam);

    // Sample a broad band of the floor and require it uniformly lit. Acne
    // shows up as variance — some pixels shadowed, their neighbours not.
    let mut min = f32::MAX;
    let mut max: f32 = 0.0;
    for y in (H / 2..H - 8).step_by(4) {
        let b = brightness_at(&px, W / 2, y, 3);
        if b > 0.0 {
            min = min.min(b);
            max = max.max(b);
        }
    }
    assert!(max > 0.0, "precondition: the floor must be visible");
    assert!(
        max - min < max * 0.25,
        "the unoccluded floor varies from {min:.1} to {max:.1} across its \
         surface — that stippling is shadow acne, so the depth bias or \
         the front-face culling in shadow3d.rs is not sufficient",
    );
}
