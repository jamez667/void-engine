//! Draw a 2D overlay on top of a 3D scene, in one pass, on a real GPU.
//!
//! Until this worked, a frame was 2D *or* 3D: both cameras shared one
//! uniform buffer, so setting a 3D camera overwrote the 2D one and any
//! `Batch` geometry in the same frame was transformed by a perspective
//! matrix expecting camera-relative metres rather than the pixel offsets
//! `Batch` produces. A HUD over a 3D scene was impossible.
//!
//! The fix is a second camera uniform. That is a one-line claim and an
//! easy one to get subtly wrong — the obvious failure is that the 2D
//! overlay *is* drawn but through the wrong matrix, which puts it
//! somewhere unpredictable rather than making it vanish. So this test
//! checks all three things that have to hold at once:
//!
//! 1. The 3D geometry renders.
//! 2. The 2D overlay renders, in the place 2D coordinates say it should.
//! 3. Neither camera has disturbed the other.
//!
//! Skipped, not failed, without an adapter — same rule as the other GPU
//! suites.

use glam::{DVec3, Vec2, Vec3};
use void_engine::renderer::batch::{Batch, Material, Surface, Vertex};
use void_engine::renderer::camera::{Camera2D, Camera3D, CameraUniform};
use void_engine::renderer::depth::{main_pipeline_state, DEPTH_FORMAT};
use void_engine::renderer::mesh3d::{Mesh3D, Vertex3D};
use void_engine::renderer::render3d::depth_state;

mod common;
use common::{Gpu, Readback};

const W: u32 = 128;
const H: u32 = 128;

fn gpu() -> Option<Gpu> {
    common::gpu("ui over 3d test device")
}

/// Looking at the origin from along -Y.
fn camera_3d() -> Camera3D {
    let mut c = Camera3D::new(Vec2::new(W as f32, H as f32));
    c.position = DVec3::new(0.0, -6.0, 0.0);
    c.target = DVec3::ZERO;
    c
}

/// A blue box filling the middle of the frame.
fn scene(cam: &Camera3D) -> Mesh3D {
    let mut m = Mesh3D::new();
    m.push_box(
        cam.world_to_camera_offset(DVec3::ZERO),
        Vec3::splat(2.0),
        [0.0, 0.0, 1.0, 1.0],
    );
    m
}

/// A red bar along the bottom of the screen, in 2D pixel coordinates —
/// the shape a health bar or hotbar has.
///
/// Deliberately *not* centred: a 2D element drawn through the 3D camera
/// by mistake would land somewhere else entirely, and a centred one could
/// coincidentally overlap the box and hide the error.
fn overlay() -> Batch {
    let mut b = Batch::new();
    b.set_surface(Surface::new(Material::Solid));
    // Camera2D is centred on the origin with +Y up, so this is the
    // lower-left quadrant.
    b.rect(
        Vec2::new(-40.0, -48.0),
        Vec2::new(48.0, 12.0),
        [1.0, 0.0, 0.0, 1.0],
    );
    b.clear_surface();
    b
}

/// Render 3D geometry and optionally a 2D batch over it, in one pass.
///
/// Mirrors what `frame.rs` does: 3D binds the 3D camera and draws first
/// with depth; the 2D batch binds its own camera, has the no-op depth
/// state, and composites over the finished scene.
fn render(gpu: &Gpu, mesh: &Mesh3D, cam3d: &Camera3D, ui: Option<&Batch>) -> Vec<u8> {
    use wgpu::util::DeviceExt;

    let shader_3d = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shader3d"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../src/renderer/shader3d.wgsl").into(),
        ),
    });
    let shader_2d = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shader"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../src/renderer/shader.wgsl").into(),
        ),
    });

    let camera_bgl = common::uniform_bgl(gpu, wgpu::ShaderStages::VERTEX);

    // The two cameras, in *separate* buffers. This is the whole point:
    // writing both into one would mean the second overwrote the first.
    let make_cam = |u: CameraUniform, label: &str| {
        let buf = gpu
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::bytes_of(&u),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bg = common::uniform_bg(gpu, &camera_bgl, &buf);
        (buf, bg)
    };
    let (_c3_buf, cam_3d_bg) = make_cam(cam3d.build_uniform(), "camera_3d");
    let (_c2_buf, cam_2d_bg) =
        make_cam(Camera2D::new(Vec2::new(W as f32, H as f32)).build_uniform(), "camera_2d");

    // ---- 3D pipeline ----------------------------------------------------
    let instance_bgl = common::uniform_bgl(gpu, wgpu::ShaderStages::VERTEX);
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct Instance {
        model: [[f32; 4]; 4],
        tint: [f32; 4],
    }
    let inst = Instance {
        model: glam::Mat4::IDENTITY.to_cols_array_2d(),
        tint: [1.0; 4],
    };
    let inst_buf = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("instance"),
            contents: bytemuck::bytes_of(&inst),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let inst_bg = common::uniform_bg(gpu, &instance_bgl, &inst_buf);

    let (shadow_bgl, shadow_bg) = neutral_shadow(gpu);
    let (lights_bgl, lights_bg) = neutral_lights(gpu);

    let format = Readback::FORMAT;
    let pipeline_3d = gpu
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("scene"),
            layout: Some(&gpu.device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&camera_bgl, &instance_bgl, &shadow_bgl, &lights_bgl],
                    push_constant_ranges: &[],
                },
            )),
            vertex: wgpu::VertexState {
                module: &shader_3d,
                entry_point: "vs_main",
                buffers: &[Vertex3D::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_3d,
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

    // ---- 2D pipeline ----------------------------------------------------
    let (tex_bgl, tex_bg) = white_texture(gpu);
    let pipeline_2d = gpu
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ui"),
            layout: Some(&gpu.device.create_pipeline_layout(
                &wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&camera_bgl, &tex_bgl],
                    push_constant_ranges: &[],
                },
            )),
            vertex: wgpu::VertexState {
                module: &shader_2d,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader_2d,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            // The identity depth state the 2D path uses: present so it can
            // share the pass, but testing and writing nothing.
            depth_stencil: main_pipeline_state(),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

    // ---- targets and buffers --------------------------------------------
    let readback = Readback::new(gpu, W, H);
    let depth_view = common::depth_texture(gpu, W, H);

    let vb3 = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
    let ib3 = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

    let ui_buffers = ui.map(|b| {
        (
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&b.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            gpu.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&b.indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
            b.indices.len() as u32,
        )
    });

    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut rp = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("main_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &readback.view,
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

        // 3D first, through the 3D camera.
        rp.set_pipeline(&pipeline_3d);
        rp.set_bind_group(0, &cam_3d_bg, &[]);
        rp.set_bind_group(1, &inst_bg, &[]);
        rp.set_bind_group(2, &shadow_bg, &[]);
        rp.set_bind_group(3, &lights_bg, &[]);
        rp.set_vertex_buffer(0, vb3.slice(..));
        rp.set_index_buffer(ib3.slice(..), wgpu::IndexFormat::Uint32);
        rp.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);

        // Then 2D over it, through the 2D camera, in the same pass.
        if let Some((vb, ib, n)) = &ui_buffers {
            rp.set_pipeline(&pipeline_2d);
            rp.set_bind_group(0, &cam_2d_bg, &[]);
            rp.set_bind_group(1, &tex_bg, &[]);
            rp.set_vertex_buffer(0, vb.slice(..));
            rp.set_index_buffer(ib.slice(..), wgpu::IndexFormat::Uint32);
            rp.draw_indexed(0..*n, 0, 0..1);
        }
    }

    readback.copy_from_texture(&mut enc);
    gpu.queue.submit([enc.finish()]);
    readback.pixels(gpu)
}

/// An empty shadow map and an overhead sun: nothing occludes anything.
fn neutral_shadow(gpu: &Gpu) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    use wgpu::util::DeviceExt;
    #[repr(C)]
    #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
    struct ShadowUniform {
        light_view_proj: [[f32; 4]; 4],
        light_dir_ambient: [f32; 4],
    }
    let u = ShadowUniform {
        light_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
        light_dir_ambient: [0.32, 0.43, 0.84, 0.25],
    };
    let buf = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::bytes_of(&u),
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    {
        let mut e = gpu.device.create_command_encoder(&Default::default());
        e.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &tex.create_view(&Default::default()),
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        gpu.queue.submit([e.finish()]);
    }
    let view = tex.create_view(&Default::default());
    let sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
        compare: Some(wgpu::CompareFunction::LessEqual),
        ..Default::default()
    });
    let bgl = gpu
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
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
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
    (bgl, bg)
}

fn neutral_lights(gpu: &Gpu) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    use wgpu::util::DeviceExt;
    let zeros = vec![0u8; 16 + 16 * 32];
    let buf = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &zeros,
            usage: wgpu::BufferUsages::UNIFORM,
        });
    let bgl = common::uniform_bgl(gpu, wgpu::ShaderStages::FRAGMENT);
    let bg = common::uniform_bg(gpu, &bgl, &buf);
    (bgl, bg)
}

/// The 1x1 white texel the 2D solid-colour path samples.
fn white_texture(gpu: &Gpu) -> (wgpu::BindGroupLayout, wgpu::BindGroup) {
    use wgpu::util::DeviceExt;
    let tex = gpu.device.create_texture_with_data(
        &gpu.queue,
        &wgpu::TextureDescriptor {
            label: Some("white"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &[255, 255, 255, 255],
    );
    let view = tex.create_view(&Default::default());
    let sampler = gpu.device.create_sampler(&Default::default());
    let bgl = gpu
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
    let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });
    (bgl, bg)
}

fn pixel(px: &[u8], x: u32, y: u32) -> [u8; 4] {
    let i = ((y * W + x) * 4) as usize;
    [px[i], px[i + 1], px[i + 2], px[i + 3]]
}

/// Where the 2D bar should land, in framebuffer coordinates.
///
/// `Camera2D` is centred with +Y up; the framebuffer has +Y down. The bar
/// is at UI (-40, -48), so it is left of centre and below it — which is
/// *down*-frame.
fn bar_probe() -> (u32, u32) {
    (W / 2 - 40, H / 2 + 48)
}

/// The end-to-end claim: both draw, in one frame, in the right places.
#[test]
fn a_2d_overlay_draws_on_top_of_a_3d_scene() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let cam = camera_3d();
    let mesh = scene(&cam);

    let px = render(&gpu, &mesh, &cam, Some(&overlay()));

    // The box is blue and covers the centre.
    let centre = pixel(&px, W / 2, H / 2);
    assert!(
        centre[2] > centre[0],
        "the 3D box should still be visible in the centre, got {centre:?}",
    );

    // The bar is red and sits where 2D coordinates put it.
    let (bx, by) = bar_probe();
    let bar = pixel(&px, bx, by);
    assert!(
        bar[0] > bar[2],
        "the 2D overlay should be red at ({bx}, {by}), got {bar:?} — if it \
         is the clear colour the overlay did not draw; if it is blue the \
         2D geometry went through the 3D camera and landed elsewhere",
    );
}

/// The 3D camera must not disturb the 2D one *across frames*: rendering
/// the overlay with and without a scene present must put it in exactly
/// the same pixels.
///
/// Note what this does and does not cover. It catches a 3D camera whose
/// matrix leaks into the 2D one depending on what the frame contains —
/// but not the 2D path simply binding the wrong camera, since then both
/// renders here are equally wrong and still match. Verified: binding the
/// 3D camera for the 2D draw leaves this passing while the other two
/// tests in this file fail. They are the ones that pin *which* camera the
/// overlay goes through; this one pins that the choice is stable.
#[test]
fn the_3d_camera_does_not_move_the_2d_overlay() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let cam = camera_3d();

    // Overlay over a scene, and overlay over an empty scene.
    let with_scene = render(&gpu, &scene(&cam), &cam, Some(&overlay()));
    let without = render(&gpu, &Mesh3D::new(), &cam, Some(&overlay()));

    let (bx, by) = bar_probe();
    assert_eq!(
        pixel(&with_scene, bx, by),
        pixel(&without, bx, by),
        "the overlay landed on a different pixel depending on whether 3D \
         geometry was present — the two cameras are sharing state",
    );
}

/// The 2D path is alpha-blended and depth-neutral, so it composites over
/// the scene rather than being occluded by it. A UI element hidden behind
/// world geometry is the bug this rules out.
#[test]
fn the_overlay_is_not_occluded_by_3d_geometry_behind_it() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let cam = camera_3d();

    // A box large enough to cover the whole frame, including where the
    // bar goes.
    let mut big = Mesh3D::new();
    big.push_box(
        cam.world_to_camera_offset(DVec3::ZERO),
        Vec3::new(40.0, 2.0, 40.0),
        [0.0, 0.0, 1.0, 1.0],
    );

    let px = render(&gpu, &big, &cam, Some(&overlay()));
    let (bx, by) = bar_probe();
    let bar = pixel(&px, bx, by);
    assert!(
        bar[0] > bar[2],
        "the overlay should draw over the box covering that pixel, got \
         {bar:?} — it is being depth-tested against the scene",
    );
}
