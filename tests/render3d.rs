//! Draw 3D geometry on a real GPU and check what comes out.
//!
//! Phase 2 of `docs/3d-spec.md`. The unit tests beside `mesh3d.rs` and
//! `render3d.rs` check the vertex layout and the depth states in isolation;
//! none of them draws anything, so none would notice a shader that compiles
//! and renders nothing, a winding that culls every face, or a depth
//! comparison pointing the wrong way.
//!
//! This stands up a headless device and renders through the **real**
//! `shader3d.wgsl`, the real `Vertex3D::desc()` and the real
//! `render3d::depth_state()`, then reads the pixels back.
//!
//! Skipped, not failed, when no adapter is available — same rule as
//! `materials_render.rs` and `depth_buffer.rs`.

use glam::{DVec3, Vec2, Vec3};
use void_engine::renderer::camera::Camera3D;
use void_engine::renderer::depth::DEPTH_FORMAT;
use void_engine::renderer::mesh3d::{Mesh3D, Vertex3D};
use void_engine::renderer::render3d::depth_state;

const W: u32 = 128;
const H: u32 = 128;

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
            label: Some("render3d test device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .ok()?;
    Some(Gpu { device, queue })
}

/// A camera looking at the origin from along -Y, far enough back to see a
/// 2 m box whole.
fn camera() -> Camera3D {
    let mut c = Camera3D::new(Vec2::new(W as f32, H as f32));
    c.position = DVec3::new(0.0, -6.0, 0.0);
    c.target = DVec3::ZERO;
    c
}

/// Render a mesh through the real 3D shader and pipeline state, returning
/// tightly-packed RGBA.
///
/// Vertices are camera-relative, matching `Camera3D`'s contract — the
/// caller subtracts the eye before handing geometry over.
fn render(gpu: &Gpu, mesh: &Mesh3D, cam: &Camera3D) -> Vec<u8> {
    use wgpu::util::DeviceExt;

    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("shader3d"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../src/renderer/shader3d.wgsl").into(),
        ),
    });

    let uniform = cam.build_uniform();
    let camera_buffer = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });
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
    let camera_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &camera_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
    });

    let layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&camera_bgl],
            push_constant_ranges: &[],
        });
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let pipeline = gpu
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex3D::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
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
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: depth_state(),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

    let target = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("target"),
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
        label: Some("depth"),
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_tex.create_view(&Default::default());

    let vbuf = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
    let ibuf = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

    let unpadded = W * 4;
    let padded = unpadded.div_ceil(256) * 256;
    let readback = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
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
                    // 1.0 = far plane, matching `frame.rs` and
                    // `Camera3D`'s `perspective_rh` 0..1 range.
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&pipeline);
        rpass.set_bind_group(0, &camera_bg, &[]);
        rpass.set_vertex_buffer(0, vbuf.slice(..));
        rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
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

fn centre_pixel(px: &[u8]) -> [u8; 4] {
    let i = ((H / 2) * W + (W / 2)) as usize * 4;
    [px[i], px[i + 1], px[i + 2], px[i + 3]]
}

fn lit_fraction(px: &[u8]) -> f32 {
    let lit = px.chunks(4).filter(|p| p[0] > 0 || p[1] > 0 || p[2] > 0).count();
    lit as f32 / (px.chunks(4).count() as f32)
}

/// A box in front of the camera must produce visible pixels.
///
/// This is the end-to-end smoke test the unit tests cannot be: it fails if
/// the shader compiles but draws nothing, if the winding culls every face,
/// if the projection points the wrong way, or if the depth comparison
/// rejects everything against the 1.0 clear.
#[test]
fn a_box_in_front_of_the_camera_is_visible_and_lit() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let cam = camera();

    let mut mesh = Mesh3D::new();
    mesh.push_box(
        // Camera-relative: the box is at the world origin and the eye is
        // 6 m away on -Y, so relative to the eye it sits at +6 on Y.
        cam.world_to_camera_offset(DVec3::ZERO),
        Vec3::splat(2.0),
        [1.0, 1.0, 1.0, 1.0],
    );

    let px = render(&gpu, &mesh, &cam);
    let centre = centre_pixel(&px);
    assert!(
        centre[0] > 0 || centre[1] > 0 || centre[2] > 0,
        "the box should cover the centre pixel, got {centre:?}",
    );

    let frac = lit_fraction(&px);
    assert!(
        frac > 0.05 && frac < 0.95,
        "a 2 m box at 6 m should cover a modest part of the frame, covered {:.1}%",
        frac * 100.0,
    );

    // Lambertian shading against a fixed light: the visible face should be
    // shaded, not full white, or the lighting is not being applied at all.
    assert!(
        centre[0] < 255,
        "the lit face should be shaded below full white, got {centre:?} — \
         the fragment shader's Lambert term may not be applied",
    );
}

/// Geometry behind the camera must not appear.
///
/// Guards the projection's handedness. With a left-handed projection, or a
/// view matrix looking the wrong way, a box *behind* the eye still lands on
/// screen — and the test above would pass regardless.
#[test]
fn a_box_behind_the_camera_is_not_visible() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let cam = camera();

    let mut mesh = Mesh3D::new();
    // The camera is at -Y looking toward +Y, so -Y of the eye is behind it.
    mesh.push_box(
        cam.world_to_camera_offset(DVec3::new(0.0, -20.0, 0.0)),
        Vec3::splat(2.0),
        [1.0, 1.0, 1.0, 1.0],
    );

    let px = render(&gpu, &mesh, &cam);
    assert_eq!(
        lit_fraction(&px),
        0.0,
        "geometry behind the camera was drawn — the projection or view \
         matrix has the wrong handedness",
    );
}

/// The nearer of two overlapping boxes must win.
///
/// This is what the depth buffer is *for*, and it is the one thing no
/// amount of 2D testing could establish. The far box is drawn second, so
/// under painter's algorithm it would overwrite the near one; with
/// `Less` + depth writes it must not.
#[test]
fn the_nearer_box_occludes_the_farther_one_regardless_of_draw_order() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let cam = camera();

    let mut mesh = Mesh3D::new();
    // Near box: red, closer to the eye.
    mesh.push_box(
        cam.world_to_camera_offset(DVec3::new(0.0, -2.0, 0.0)),
        Vec3::splat(2.0),
        [1.0, 0.0, 0.0, 1.0],
    );
    // Far box: blue, drawn *after* and directly behind the red one.
    mesh.push_box(
        cam.world_to_camera_offset(DVec3::new(0.0, 2.0, 0.0)),
        Vec3::splat(2.0),
        [0.0, 0.0, 1.0, 1.0],
    );

    let px = render(&gpu, &mesh, &cam);
    let centre = centre_pixel(&px);
    assert!(
        centre[0] > centre[2],
        "the near (red) box should occlude the far (blue) one drawn after \
         it; centre pixel was {centre:?}. Painter's ordering would show \
         blue — the depth test is not doing its job.",
    );
}
