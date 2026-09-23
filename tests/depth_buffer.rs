//! Prove the depth buffer changed nothing about 2D rendering.
//!
//! Phase 0 of the 3D work (`docs/3d-spec.md`) attaches a depth buffer to the
//! main pass and gives `main_pipeline` depth state. The claim that justifies
//! doing it before anything else is that it is *invisible* in 2D: the state
//! is `depth_compare: Always` with writes off, which is the identity.
//!
//! A claim like that is worth nothing unasserted, and the existing
//! `materials_render.rs` cannot check it — that file builds its own pipeline
//! with `depth_stencil: None`, so it would keep passing however this was
//! wired. This test draws the same geometry twice, once through a pipeline
//! carrying the real `depth::main_pipeline_state()` against a real depth
//! attachment and once through the no-depth path that preceded it, and
//! requires the two framebuffers to be byte-identical.
//!
//! It also covers the failure this change could plausibly introduce:
//! declaring depth state on a pipeline whose pass supplies no attachment is
//! a validation error, so "the depth pipeline builds and draws at all" is
//! itself part of what is being asserted.
//!
//! # What this does *not* catch, and what does
//!
//! Changing the compare to `Less` while writes stay off passes this test,
//! and always has. With nothing written to the depth buffer every
//! fragment tests against the 1.0 clear and passes, so the frame is
//! identical — the compare only bites once something records depth.
//!
//! The guard against that is the `const` assertion at the bottom of this
//! file: flipping [`MAIN_WRITES_DEPTH`] fails the *build*, before any
//! pixel is drawn. The two together cover the state; neither does alone,
//! which is worth knowing before trusting a green run here.
//!
//! Skipped, not failed, when no adapter is available — same rule as
//! `materials_render.rs`.

use void_engine::renderer::batch::{Batch, Material, Surface};
use void_engine::renderer::depth::{main_pipeline_state, DEPTH_FORMAT, MAIN_WRITES_DEPTH};

mod common;
use common::{Gpu, Readback};

const W: u32 = 128;
const H: u32 = 128;

fn gpu() -> Option<Gpu> {
    common::gpu("depth test device")
}

/// Overlapping geometry, drawn back-to-front.
///
/// Overlap is the point: under painter's algorithm the last quad wins, and
/// that is exactly what a wrongly-enabled depth test would change. Flat
/// colours on a solid material keep the comparison about ordering rather
/// than about shader detail, which `materials_render.rs` already covers.
fn overlapping_quads() -> Batch {
    let mut batch = Batch::new();
    batch.set_surface(Surface::new(Material::Solid));
    batch.rect(glam::Vec2::new(-16.0, -16.0), glam::Vec2::new(64.0, 64.0), [1.0, 0.0, 0.0, 1.0]);
    batch.rect(glam::Vec2::new(0.0, 0.0), glam::Vec2::new(64.0, 64.0), [0.0, 1.0, 0.0, 1.0]);
    batch.rect(glam::Vec2::new(16.0, 16.0), glam::Vec2::new(64.0, 64.0), [0.0, 0.0, 1.0, 1.0]);
    batch.clear_surface();
    batch
}

/// Draw `batch` through the real `shader.wgsl`, with or without the depth
/// buffer, and read the framebuffer back as RGBA.
///
/// The two paths differ *only* in the depth state and attachment, so any
/// pixel difference is attributable to this change alone.
fn render(gpu: &Gpu, batch: &Batch, with_depth: bool) -> Vec<u8> {
    use wgpu::util::DeviceExt;

    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("main shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../src/renderer/shader.wgsl").into()),
    });

    let half_w = W as f32 * 0.5;
    let half_h = H as f32 * 0.5;
    let proj = glam::Mat4::orthographic_rh(-half_w, half_w, -half_h, half_h, -1.0, 1.0);
    let camera_buffer = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera"),
            contents: bytemuck::bytes_of(&proj),
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

    let white = gpu.device.create_texture_with_data(
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
    let white_view = white.create_view(&Default::default());
    let sampler = gpu.device.create_sampler(&Default::default());
    let tex_bgl = gpu
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
    let tex_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &tex_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&white_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    let layout = gpu
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&camera_bgl, &tex_bgl],
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
                buffers: &[void_engine::renderer::batch::Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            // The whole point of the test: the engine's real depth state,
            // not a locally invented one.
            depth_stencil: if with_depth { main_pipeline_state() } else { None },
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

    let readback = Readback::new(gpu, W, H);
    let target_view = &readback.view;

    // Mirrors `DepthBuffer::new`. Built here rather than through that type
    // because it is `pub(super)` — the format is what must match, and that
    // is imported from the engine.
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
            contents: bytemuck::cast_slice(&batch.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
    let ibuf = gpu
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&batch.indices),
            usage: wgpu::BufferUsages::INDEX,
        });

    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
        let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: with_depth.then_some(
                wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                },
            ),
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        rpass.set_pipeline(&pipeline);
        rpass.set_bind_group(0, &camera_bg, &[]);
        rpass.set_bind_group(1, &tex_bg, &[]);
        rpass.set_vertex_buffer(0, vbuf.slice(..));
        rpass.set_index_buffer(ibuf.slice(..), wgpu::IndexFormat::Uint32);
        rpass.draw_indexed(0..batch.indices.len() as u32, 0, 0..1);
    }
    readback.copy_from_texture(&mut enc);
    gpu.queue.submit([enc.finish()]);

    readback.pixels(gpu)
}

#[test]
fn the_depth_buffer_does_not_change_a_single_2d_pixel() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };

    let batch = overlapping_quads();
    let without = render(&gpu, &batch, false);
    let with = render(&gpu, &batch, true);

    assert_eq!(without.len(), with.len(), "readback sizes should match");

    // Guard against the comparison being vacuous: if the geometry never
    // drew, two black images would "match" and prove nothing.
    assert!(
        without.chunks(4).any(|p| p[0] > 0 || p[1] > 0 || p[2] > 0),
        "the no-depth render is entirely black — the test geometry never drew, \
         so an equality check between the two paths would prove nothing",
    );

    let diff = without
        .chunks(4)
        .zip(with.chunks(4))
        .enumerate()
        .find(|(_, (a, b))| a != b);
    if let Some((i, (a, b))) = diff {
        panic!(
            "depth buffer changed pixel {} of {}: {:?} without depth, {:?} with. \
             The main pipeline's depth state is supposed to be the identity \
             (`Always`, no writes) — see src/renderer/depth.rs.",
            i,
            without.len() / 4,
            a,
            b,
        );
    }
}

/// Guards the constant the no-op rests on, at compile time.
///
/// If someone flips it to `true` without giving 2D geometry real z values,
/// the main pass starts depth-testing coplanar quads at `z = 0.0` against
/// each other — a coin-flip on GPU tie-breaking rather than an ordering
/// anyone chose (`shader.wgsl` hardcodes `z = 0.0` for every vertex). A 3D
/// path should use its own pipeline with its own depth state instead.
///
/// A `const` assertion rather than a `#[test]`: the value is known at
/// compile time, so `assert!` on it is what clippy's `assertions_on_
/// constants` rightly objects to. This fails the build instead, which is
/// strictly earlier and needs no GPU.
const _: () = assert!(!MAIN_WRITES_DEPTH);
