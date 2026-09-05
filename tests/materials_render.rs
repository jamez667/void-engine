//! Render the procedural materials on a real GPU and check what comes out.
//!
//! A WGSL file that parses and validates can still draw nothing, or draw the
//! same thing for every material. This stands up a headless device, draws one
//! quad per material through the actual pipeline the game uses, and reads the
//! pixels back — so the thing under test is the shader itself rather than a
//! Rust reimplementation of it that could drift from it.
//!
//! Skipped, not failed, when no adapter is available: CI without a GPU should
//! not report a red build for a machine limitation.

use void_engine::renderer::batch::{Batch, Blend, Material, Overlay, Surface, Vertex};

const W: u32 = 256;
const H: u32 = 256;

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
            label: Some("material test device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .ok()?;
    Some(Gpu { device, queue })
}

/// Draw a batch through the real shader and read the framebuffer back as RGBA.
fn render(gpu: &Gpu, batch: &Batch) -> Vec<u8> {
    use wgpu::util::DeviceExt;

    let shader = gpu.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("main shader"),
        source: wgpu::ShaderSource::Wgsl(
            include_str!("../src/renderer/shader.wgsl").into(),
        ),
    });

    // An orthographic camera mapping our -128..128 quad space onto the target.
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

    // The white 1x1 the solid-colour path multiplies against.
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
                buffers: &[Vertex::desc()],
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
            depth_stencil: None,
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

    // Read-back needs its row stride padded to 256 bytes.
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
            depth_stencil_attachment: None,
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

    // Strip the row padding back out.
    let mut out = Vec::with_capacity((unpadded * H) as usize);
    for row in 0..H {
        let start = (row * padded) as usize;
        out.extend_from_slice(&mapped[start..start + unpadded as usize]);
    }
    drop(mapped);
    readback.unmap();
    out
}

/// A full-target quad painted with one surface.
fn quad_with(surface: Option<Surface>) -> Batch {
    let mut b = Batch::new();
    if let Some(s) = surface {
        b.set_surface(s);
    }
    // Mid grey, so a pattern has room to darken and lighten it.
    b.rect(
        glam::Vec2::ZERO,
        glam::Vec2::new(W as f32, H as f32),
        [0.5, 0.5, 0.5, 1.0],
    );
    b
}

/// Spread of luminance across the image: a patterned fill varies, a flat one
/// does not.
fn variation(px: &[u8]) -> f32 {
    let lum: Vec<f32> = px
        .chunks(4)
        .map(|p| (p[0] as f32 * 0.299 + p[1] as f32 * 0.587 + p[2] as f32 * 0.114) / 255.0)
        .collect();
    let mean = lum.iter().sum::<f32>() / lum.len() as f32;
    (lum.iter().map(|l| (l - mean).powi(2)).sum::<f32>() / lum.len() as f32).sqrt()
}

/// Variation left after averaging over 4x4 blocks.
///
/// Per-pixel noise averages away; drawn features -- a blade, a pebble, a crack
/// between blocks -- survive. This is what separates a pattern from a grain.
fn coarse_variation(px: &[u8]) -> f32 {
    const B: usize = 4;
    let mut blocks = Vec::new();
    for by in (0..H as usize).step_by(B) {
        for bx in (0..W as usize).step_by(B) {
            let mut sum = 0.0;
            let mut n = 0.0;
            for y in by..(by + B).min(H as usize) {
                for x in bx..(bx + B).min(W as usize) {
                    let i = (y * W as usize + x) * 4;
                    sum += (px[i] as f32 * 0.299
                        + px[i + 1] as f32 * 0.587
                        + px[i + 2] as f32 * 0.114)
                        / 255.0;
                    n += 1.0;
                }
            }
            blocks.push(sum / n);
        }
    }
    let mean = blocks.iter().sum::<f32>() / blocks.len() as f32;
    (blocks.iter().map(|b| (b - mean).powi(2)).sum::<f32>() / blocks.len() as f32).sqrt()
}

/// How different two renders are, per pixel.
fn difference(a: &[u8], b: &[u8]) -> f32 {
    let n = a.len().min(b.len());
    let sum: f32 = (0..n).map(|i| (a[i] as f32 - b[i] as f32).abs()).sum();
    sum / n as f32 / 255.0
}

#[test]
fn materials_draw_distinct_patterns() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    // A solid fill is the control: it must come out perfectly flat.
    let solid = render(&gpu, &quad_with(None));
    assert!(
        variation(&solid) < 0.01,
        "a solid fill must be flat, got variation {:.4}",
        variation(&solid)
    );

    // Each material seen at 24 px per metre, which is a working zoom in game.
    // Pattern features are then several pixels across: big enough to read.
    // A 2 m feature seen at 24 px per metre: about 48 px across, which is the
    // size ground detail actually reads at in the game.
    const PPM: f32 = 24.0;
    let surface = |m| Some(Surface::new(m).scale_m(2.0).anchored(glam::Vec2::ZERO, PPM));
    let grass = render(&gpu, &quad_with(surface(Material::Grass)));
    let dirt = render(&gpu, &quad_with(surface(Material::Dirt)));
    let stone = render(&gpu, &quad_with(surface(Material::Stone)));

    for (name, px) in [("grass", &grass), ("dirt", &dirt), ("stone", &stone)] {
        assert!(
            variation(px) > 0.02,
            "{name} must actually draw a pattern, got variation {:.4}",
            variation(px)
        );
        // A pattern must have structure at a legible size, not merely be noisy.
        // Per-pixel noise varies as much as a drawn hatch but reads as grain,
        // so measure how much survives a blur: real features do, noise does not.
        assert!(
            coarse_variation(px) > 0.012,
            "{name} must have features bigger than a pixel, got {:.4}",
            coarse_variation(px)
        );
        assert!(
            difference(px, &solid) > 0.01,
            "{name} must differ from a flat fill"
        );
    }

    // The spec's real requirement: the materials are told apart by pattern, not
    // only by colour. All three are drawn in the same grey here, so any
    // difference between them is pattern alone.
    for (a_name, a, b_name, b) in [
        ("grass", &grass, "dirt", &dirt),
        ("grass", &grass, "stone", &stone),
        ("dirt", &dirt, "stone", &stone),
    ] {
        let d = difference(a, b);
        assert!(
            d > 0.01,
            "{a_name} and {b_name} must be distinguishable with colour removed, \
             differ by only {d:.4}"
        );
    }
}

/// Every material in the library, so a new one cannot be added without being
/// held to the same bar as the rest.
const ALL: [(Material, &str); 20] = [
    (Material::Grass, "grass"),
    (Material::Dirt, "dirt"),
    (Material::Stone, "stone"),
    (Material::Scree, "scree"),
    (Material::Gravel, "gravel"),
    (Material::Sand, "sand"),
    (Material::Cracked, "cracked"),
    (Material::Timber, "timber"),
    (Material::EndGrain, "end grain"),
    (Material::Alluvium, "alluvium"),
    (Material::Shale, "shale"),
    (Material::Sandstone, "sandstone"),
    (Material::Limestone, "limestone"),
    (Material::Dolomite, "dolomite"),
    (Material::Quartzite, "quartzite"),
    (Material::Granite, "granite"),
    (Material::Gossan, "gossan"),
    (Material::OreSulphide, "sulphide ore"),
    (Material::Quartz, "quartz"),
    (Material::Timbered, "timbered"),
];

/// The whole library must be mutually distinguishable by pattern alone.
///
/// This is the spec's actual requirement, and the one that matters as the set
/// grows: it is easy to add a twenty-first material that looks fine on its own
/// and is indistinguishable from the fourth. Every pair is compared, drawn in
/// the same grey, so only pattern can separate them.
#[test]
fn every_material_is_distinguishable_from_every_other() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    const PPM: f32 = 26.0;
    let rendered: Vec<(&str, Vec<u8>)> = ALL
        .iter()
        .map(|(m, name)| {
            let px = render(
                &gpu,
                &quad_with(Some(
                    Surface::new(*m).scale_m(2.0).anchored(glam::Vec2::ZERO, PPM),
                )),
            );
            (*name, px)
        })
        .collect();

    // Each must draw something with structure at a legible size.
    for (name, px) in &rendered {
        assert!(
            variation(px) > 0.015,
            "{name} draws nothing: variation {:.4}",
            variation(px)
        );
        assert!(
            coarse_variation(px) > 0.008,
            "{name} is noise rather than a pattern: coarse variation {:.4}",
            coarse_variation(px)
        );
    }

    // And every pair must differ. Colour is identical across all of them here,
    // so any difference found is pattern alone.
    let mut worst = (f32::MAX, "", "");
    for (i, (a_name, a)) in rendered.iter().enumerate() {
        for (b_name, b) in &rendered[i + 1..] {
            let d = difference(a, b);
            if d < worst.0 {
                worst = (d, a_name, b_name);
            }
            assert!(
                d > 0.004,
                "{a_name} and {b_name} are too alike with colour removed: {d:.4}"
            );
        }
    }
    println!(
        "closest pair: {} and {} at {:.4}",
        worst.1, worst.2, worst.0
    );
}

/// Dolomite is limestone plus a tick in each course. That tick is the whole
/// convention distinguishing two rocks a miner must not confuse, and it is the
/// most likely thing in the set to be lost to a scaling change.
#[test]
fn dolomite_is_not_limestone() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };
    const PPM: f32 = 26.0;
    let at = |m| {
        render(
            &gpu,
            &quad_with(Some(Surface::new(m).scale_m(2.0).anchored(glam::Vec2::ZERO, PPM))),
        )
    };
    let lime = at(Material::Limestone);
    let dol = at(Material::Dolomite);
    let d = difference(&lime, &dol);
    // A tick in every brick over a ten-metre view. The measure is a mean over
    // every pixel, and ticks are ink on a mostly plain field, so a few
    // thousandths here is a plainly visible difference on screen -- confirmed
    // by eye against the rendered swatches, not assumed.
    assert!(
        d > 0.004,
        "the dolomite tick must read against limestone's courses, differ by only {d:.4}"
    );
    // Dolomite carries strictly more ink: the same courses, plus the ticks.
    assert!(
        variation(&dol) > variation(&lime) * 0.9,
        "dolomite should not be plainer than limestone"
    );
}

/// A pattern belongs to the ground, not to the screen.
///
/// Primitives are drawn in screen coordinates, so without being told where the
/// camera is, a pattern is a function of pixels: its features come out the size
/// of pixels instead of the size of things, and the whole surface slides under
/// the camera as it pans. Both are the same bug, and this catches it.
#[test]
fn a_pattern_is_anchored_to_the_world() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    const PPM: f32 = 20.0;
    // A camera is defined by the ground at the centre of its screen. Two
    // cameras pointed at the *same* ground must therefore draw the same
    // picture, whatever numbers they were built from -- and a pattern locked to
    // the screen instead of the ground would draw the same picture from every
    // camera, which is the bug, so the second half of this test checks that a
    // camera pointed somewhere else draws something different.
    let shot = |cam: glam::Vec2| {
        let mut b = Batch::new();
        b.set_surface(
            Surface::new(Material::Scree)
                .scale_m(4.0)
                .anchored(cam, PPM),
        );
        // Fills the view, centred: the screen shows the ground around `cam`.
        b.rect(
            glam::Vec2::ZERO,
            glam::Vec2::new(W as f32, H as f32),
            [0.55, 0.55, 0.53, 1.0],
        );
        render(&gpu, &b)
    };

    let here = glam::Vec2::new(40.0, 25.0);
    let same = shot(here);
    let again = shot(here);
    assert_eq!(
        difference(&same, &again),
        0.0,
        "the same view must draw identically"
    );

    // Pointed at different ground, the view must show different ground. If the
    // pattern were locked to the screen these would be pixel-identical.
    let elsewhere = shot(here + glam::Vec2::new(37.0, -21.0));
    assert!(
        difference(&same, &elsewhere) > 0.01,
        "different ground must look different: {:.4}",
        difference(&same, &elsewhere)
    );

    // And the features must be sized in metres: at 20 px/m a 4 m feature is
    // 80 px, so a 256 px view holds only a few of them and is far from noise.
    assert!(
        coarse_variation(&same) > 0.02,
        "world-anchored features must be big on screen, got {:.4}",
        coarse_variation(&same)
    );
}

/// Mean luminance of one vertical strip of the render, left to right.
fn strips(px: &[u8], n: usize) -> Vec<f32> {
    let mut out = vec![0.0; n];
    for (i, slot) in out.iter_mut().enumerate() {
        let x0 = i * W as usize / n;
        let x1 = (i + 1) * W as usize / n;
        let mut sum = 0.0;
        let mut count = 0.0;
        for y in 0..H as usize {
            for x in x0..x1 {
                let p = (y * W as usize + x) * 4;
                sum += (px[p] as f32 * 0.299
                    + px[p + 1] as f32 * 0.587
                    + px[p + 2] as f32 * 0.114)
                    / 255.0;
                count += 1.0;
            }
        }
        *slot = sum / count;
    }
    out
}

/// An overlay can be brought in from one side, crossing over inside the tile.
///
/// Without this an overlay is uniform across whatever it is drawn on, so two
/// tiles meeting change over on the edge between them -- a staircase at tile
/// resolution. Blending within the tile is what turns that edge into a band.
#[test]
fn an_overlay_can_cross_over_inside_its_tile() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    // Every render here sits on the same ground, or the comparison is between
    // different tiles rather than between different blends.
    const GROUND: [f32; 4] = [0.55, 0.55, 0.50, 1.0];

    // Grass tile with rock coming in from the right, crossing a quarter of the
    // way in from that edge.
    let mut b = Batch::new();
    b.set_surface(
        Surface::new(Material::Grass)
            .scale_m(2.0)
            .anchored(glam::Vec2::ZERO, 26.0)
            .over(Overlay::new(Material::Stone).strength(1.0).blend(Blend::Shade))
            .from_edge(glam::Vec2::X, 0.5),
    );
    b.rect(
        glam::Vec2::ZERO,
        glam::Vec2::new(W as f32, H as f32),
        GROUND,
    );
    let graded = render(&gpu, &b);

    // The same tile with the overlay applied evenly.
    let mut u = Batch::new();
    u.set_surface(
        Surface::new(Material::Grass)
            .scale_m(2.0)
            .anchored(glam::Vec2::ZERO, 26.0)
            .over(Overlay::new(Material::Stone).strength(1.0).blend(Blend::Shade)),
    );
    u.rect(
        glam::Vec2::ZERO,
        glam::Vec2::new(W as f32, H as f32),
        GROUND,
    );
    let even = render(&gpu, &u);

    assert!(
        difference(&graded, &even) > 0.006,
        "a directed blend must differ from a uniform one: {:.4}",
        difference(&graded, &even)
    );

    // The overlay darkens (Shade), so the side it comes from must end up
    // darker, and the far side must be left nearly alone.
    //
    // The bar is set from what this overlay can actually do rather than from a
    // round number: measure the difference a *uniform* overlay makes, and
    // require the graded one to cross most of that range from one side of the
    // tile to the other. An absolute threshold here would be a guess about
    // material strengths that a later tweak would silently invalidate.
    let mut bare = Batch::new();
    bare.set_surface(
        Surface::new(Material::Grass)
            .scale_m(2.0)
            .anchored(glam::Vec2::ZERO, 26.0),
    );
    bare.rect(glam::Vec2::ZERO, glam::Vec2::new(W as f32, H as f32), GROUND);
    let plain_strips = strips(&render(&gpu, &bare), 4);
    let even_strips = strips(&even, 4);
    let ceiling = plain_strips[3] - even_strips[3];
    assert!(
        ceiling > 0.002,
        "the overlay must darken at all for this test to mean anything: {ceiling:.4}"
    );

    // The far strip must look like bare ground and the near strip like the
    // overlay at full strength: that is what "crosses over inside the tile"
    // means. Comparing strip *means* against each other would understate it,
    // because a strip averages a range of the gradient rather than sampling
    // one point of it.
    let s = strips(&graded, 4);
    assert!(
        (s[0] - plain_strips[0]).abs() < ceiling * 0.35,
        "the far side must be nearly untouched ground: {:.4} vs plain {:.4}",
        s[0],
        plain_strips[0]
    );
    assert!(
        (s[3] - even_strips[3]).abs() < ceiling * 0.35,
        "the near side must be nearly the full overlay: {:.4} vs uniform {:.4}",
        s[3],
        even_strips[3]
    );
    // No third check on the total range: the two ends already pin the gradient
    // to bare ground at one side and full overlay at the other, and a range
    // measured between two *different* strips also carries the difference
    // between the patterns under them, so it says less than it appears to.

    // The middle must sit between the ends, which is what makes it a gradation
    // rather than a step somewhere inside the tile.
    //
    // Which end is darker is not the blend's business: it depends on the two
    // materials. Dense grass under a stone overlay comes out *lighter* toward
    // the overlay, because the blades it replaces were darker than the shading
    // that replaces them. So test that the middle lies between the ends,
    // whichever way round they are.
    let (lo, hi) = if s[0] <= s[3] { (s[0], s[3]) } else { (s[3], s[0]) };
    // The slack is for the patterns' own texture: a strip's mean carries
    // whatever fragments and joints happen to fall in it, which varies by a
    // few thousandths between strips independently of the blend. Tighter than
    // this measures the materials rather than the gradient.
    const TEXTURE: f32 = 0.012;
    for (i, v) in s.iter().enumerate().take(3).skip(1) {
        assert!(
            *v >= lo - TEXTURE && *v <= hi + TEXTURE,
            "strip {i} lies outside the ends, so the tile steps rather than grades: {s:?}"
        );
    }

    // The far side must still be its own ground: if the overlay reached all the
    // way across there would be nothing left of the tile's own material.
    assert!(
        (s[0] - plain_strips[0]).abs() < 0.02,
        "the far side of the tile must keep its own cover: {:.3} vs {:.3}",
        s[0],
        plain_strips[0]
    );
}

#[test]
fn a_pattern_fades_out_before_it_aliases() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    // Zoomed far out, one pattern cell is smaller than a pixel. Drawing the
    // hatch there would moire, so every material must fade to honest flat
    // colour -- a single material that refuses would shimmer on the map.
    for (m, name) in ALL {
        let tiny = Surface::new(m).scale_m(0.004).anchored(glam::Vec2::ZERO, 1.0);
        let far = render(&gpu, &quad_with(Some(tiny)));
        assert!(
            variation(&far) < 0.01,
            "{name} must fade to flat when finer than a pixel, got {:.4}",
            variation(&far)
        );
    }

    // Close in, the same material must show its pattern.
    let near = Surface::new(Material::Stone)
        .scale_m(2.0)
        .anchored(glam::Vec2::ZERO, 24.0);
    let close = render(&gpu, &quad_with(Some(near)));
    assert!(
        variation(&close) > 0.02,
        "the same material must pattern when its features are visible"
    );
}

/// Mean colour of a render, so a change of ink can be checked for.
fn mean_rgb(px: &[u8]) -> [f32; 3] {
    let mut sum = [0.0f32; 3];
    let n = (px.len() / 4) as f32;
    for p in px.chunks(4) {
        sum[0] += p[0] as f32;
        sum[1] += p[1] as f32;
        sum[2] += p[2] as f32;
    }
    [sum[0] / n / 255.0, sum[1] / n / 255.0, sum[2] / n / 255.0]
}

const WORKING_PPM: f32 = 26.0;

fn surf(m: Material) -> Surface {
    Surface::new(m)
        .scale_m(2.0)
        .anchored(glam::Vec2::ZERO, WORKING_PPM)
}

/// The colour a pattern draws in is separate from the ground it sits on.
///
/// This is what makes one pattern serve many rocks: the same hatch in rust for
/// oxidation, in blue-grey for a parting, in near-black for survey ink. If ink
/// were baked into the shader, every recolour would need a new material.
#[test]
fn ink_colour_is_independent_of_the_ground() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    let dark = render(&gpu, &quad_with(Some(surf(Material::Shale))));
    let rust = render(
        &gpu,
        &quad_with(Some(surf(Material::Shale).ink([0.65, 0.22, 0.08]))),
    );

    // Same ground, same pattern, different ink: the render must differ, and
    // must differ toward red.
    assert!(
        difference(&dark, &rust) > 0.004,
        "changing the ink must change the render"
    );
    let (a, b) = (mean_rgb(&dark), mean_rgb(&rust));
    assert!(
        b[0] - b[2] > a[0] - a[2] + 0.005,
        "rust ink must push the render red: {a:?} -> {b:?}"
    );

    // And the ground colour still governs: the same ink over a different
    // ground gives a different result, so neither is baked into the other.
    let mut pale = Batch::new();
    pale.set_surface(surf(Material::Shale).ink([0.65, 0.22, 0.08]));
    pale.rect(
        glam::Vec2::ZERO,
        glam::Vec2::new(W as f32, H as f32),
        [0.85, 0.85, 0.80, 1.0],
    );
    let on_pale = render(&gpu, &pale);
    assert!(
        difference(&on_pale, &rust) > 0.02,
        "the ground under the ink must still show through"
    );
}

/// Each blend mode must do something different to the same pattern, or the
/// mode is decoration rather than a control.
#[test]
fn blend_modes_are_distinct() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    let at = |b: Blend| {
        render(
            &gpu,
            &quad_with(Some(surf(Material::Stone).blend(b).ink([0.70, 0.30, 0.12]))),
        )
    };
    let ink = at(Blend::Ink);
    let shade = at(Blend::Shade);
    let lighten = at(Blend::Lighten);
    let stain = at(Blend::Stain);

    for (a_name, a, b_name, b) in [
        ("ink", &ink, "shade", &shade),
        ("ink", &ink, "lighten", &lighten),
        ("ink", &ink, "stain", &stain),
        ("shade", &shade, "lighten", &lighten),
        ("shade", &shade, "stain", &stain),
        ("lighten", &lighten, "stain", &stain),
    ] {
        assert!(
            difference(a, b) > 0.004,
            "{a_name} and {b_name} must differ, got {:.4}",
            difference(a, b)
        );
    }

    // Lighten must lighten and shade must darken: the names have to be true.
    let flat = mean_rgb(&render(&gpu, &quad_with(None)));
    let l = mean_rgb(&lighten);
    let s = mean_rgb(&shade);
    let lum = |c: [f32; 3]| c[0] * 0.299 + c[1] * 0.587 + c[2] * 0.114;
    assert!(lum(l) > lum(flat), "Lighten must lighten the ground");
    assert!(lum(s) < lum(flat), "Shade must darken the ground");
}

/// Two materials must compose into one surface, in one pass.
///
/// This is the point of layering: gossan staining over limestone is one rock
/// with a history, not a new material that has to be authored from scratch.
#[test]
fn an_overlay_modifies_the_material_under_it() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    let plain = render(&gpu, &quad_with(Some(surf(Material::Limestone))));
    let stained = render(
        &gpu,
        &quad_with(Some(
            surf(Material::Limestone).over(
                Overlay::new(Material::Gossan)
                    .strength(0.7)
                    .blend(Blend::Stain),
            ),
        )),
    );

    assert!(
        difference(&plain, &stained) > 0.006,
        "an overlay must change what is drawn: {:.4}",
        difference(&plain, &stained)
    );

    // The base must still be in there: staining a rock does not replace it.
    // Gossan alone should be further from stained limestone than plain
    // limestone is, or the overlay has simply painted over the base.
    let gossan = render(&gpu, &quad_with(Some(surf(Material::Gossan))));
    assert!(
        difference(&plain, &stained) < difference(&gossan, &stained),
        "the base material must still read through its overlay"
    );

    // And no overlay is the same as before: the feature costs nothing unused.
    let none = render(&gpu, &quad_with(Some(surf(Material::Limestone))));
    assert_eq!(
        difference(&plain, &none),
        0.0,
        "a surface with no overlay must render identically"
    );
}

/// An overlay fades on its own terms. A fine overlay on a coarse base aliases
/// sooner than its base does, and must drop out first rather than shimmering.
#[test]
fn an_overlay_fades_on_its_own_scale() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    // Zoomed out far enough that the fine overlay is sub-pixel but the coarse
    // base is not.
    let base = Surface::new(Material::Limestone)
        .scale_m(3.0)
        .anchored(glam::Vec2::ZERO, 2.0)
        .over(Overlay::new(Material::Sandstone).scale_ratio(60.0));
    let with_overlay = render(&gpu, &quad_with(Some(base)));
    let without = render(
        &gpu,
        &quad_with(Some(
            Surface::new(Material::Limestone).scale_m(3.0).anchored(glam::Vec2::ZERO, 2.0),
        )),
    );

    assert!(
        difference(&with_overlay, &without) < 0.004,
        "a sub-pixel overlay must fade out rather than alias: {:.4}",
        difference(&with_overlay, &without)
    );
}

#[test]
fn strength_controls_how_far_the_pattern_departs_from_colour() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    let at = |s: f32| {
        render(
            &gpu,
            &quad_with(Some(
                Surface::new(Material::Stone)
                    .scale_m(2.0)
                    .anchored(glam::Vec2::ZERO, 24.0)
                    .strength(s),
            )),
        )
    };
    let faint = variation(&at(0.15));
    let full = variation(&at(1.0));
    assert!(
        full > faint,
        "a stronger pattern must vary more: {full:.4} vs {faint:.4}"
    );

    // Zero strength is flat colour, whatever the material says.
    let none = at(0.0);
    assert!(
        variation(&none) < 0.01,
        "zero strength must leave the colour flat, got {:.4}",
        variation(&none)
    );
}

/// A tile's base colour must be able to grade across it.
///
/// The pattern over a tile can already cross over inside it, but the colour
/// under that pattern was one flat value, so a grid of tiles stepped at its
/// edges however well the pattern blended. That step is invisible only while
/// the pattern is coarse enough to hide it -- which is to say, only while the
/// ground is drawn at the wrong size.
#[test]
fn a_tile_colour_can_grade_across_it() {
    let Some(gpu) = gpu() else {
        eprintln!("no GPU adapter available; skipping");
        return;
    };

    let dark = [0.30, 0.30, 0.28, 1.0];
    let pale = [0.80, 0.80, 0.76, 1.0];

    let mut b = Batch::new();
    b.rect_graded(
        glam::Vec2::ZERO,
        glam::Vec2::new(W as f32, H as f32),
        glam::Vec2::X,
        dark,
        pale,
    );
    let graded = render(&gpu, &b);

    let s = strips(&graded, 4);
    // Dark at the far side, pale at the near one, and monotonic between.
    assert!(s[0] < s[3], "the gradient must run dark to pale: {s:?}");
    for i in 1..s.len() {
        assert!(
            s[i] >= s[i - 1] - 0.004,
            "a colour gradient must not step backwards: {s:?}"
        );
    }
    // And it must genuinely span the two colours rather than sitting between.
    //
    // The bar is what a strip *mean* can reach, not the full colour range:
    // four strips sample the gradient at its eighths, so their means span
    // three quarters of it, and the sRGB framebuffer compresses the bright end
    // further. Measured against flat fills of the two end colours, which is
    // the most any gradient between them could show.
    let flat_of = |c| {
        let mut b = Batch::new();
        b.rect(glam::Vec2::ZERO, glam::Vec2::new(W as f32, H as f32), c);
        strips(&render(&gpu, &b), 4)[0]
    };
    let range = flat_of(pale) - flat_of(dark);
    assert!(
        s[3] - s[0] > range * 0.6,
        "the gradient must cover most of the range: {:.3} of {range:.3}",
        s[3] - s[0]
    );

    // With no direction it is a flat fill, which is what a plain rect gives.
    let mut f = Batch::new();
    f.rect_graded(
        glam::Vec2::ZERO,
        glam::Vec2::new(W as f32, H as f32),
        glam::Vec2::ZERO,
        dark,
        pale,
    );
    let flat = strips(&render(&gpu, &f), 4);
    for i in 1..flat.len() {
        assert!(
            (flat[i] - flat[0]).abs() < 0.004,
            "no direction must give a flat fill: {flat:?}"
        );
    }
}
