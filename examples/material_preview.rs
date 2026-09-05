//! Render the procedural materials to a PNG so they can be looked at.
//!
//! Draws every material as a swatch, each at several zooms, on the real GPU
//! through the real shader — so what you see is what the game draws, not a
//! reimplementation of it.
//!
//! Run with:
//!     cargo run -p void_engine --example material_preview -- out.png

use void_engine::renderer::batch::{Batch, Blend, Material, Overlay, Surface, Vertex};

const W: u32 = 1000;
const H: u32 = 820;

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "materials.png".into());

    let instance = wgpu::Instance::default();
    let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
        eprintln!("no GPU adapter available");
        std::process::exit(1);
    };
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: None,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: Default::default(),
        },
        None,
    ))
    .expect("device");

    // Two boards: the material library, and what colouring and layering do to
    // it. Pass "blend" as a second argument for the latter.
    let show_blend = std::env::args().nth(2).as_deref() == Some("blend");

    const PPM: f32 = 26.0;
    let base = |m| {
        Surface::new(m)
            .scale_m(0.35 * PPM)
            .m_per_px(1.0 / PPM)
    };

    // (surface, ground colour, label)
    let mut swatches: Vec<(Surface, [f32; 4], String)> = Vec::new();

    if show_blend {
        let lime = [0.62, 0.71, 0.72, 1.0];
        let rock = [0.56, 0.56, 0.54, 1.0];
        let rust = [0.62, 0.26, 0.10];
        let pale = [0.92, 0.90, 0.84];

        // One pattern, four inks: the ink is not baked into the material.
        swatches.push((base(Material::Shale), lime, "shale, survey ink".into()));
        swatches.push((base(Material::Shale).ink(rust), lime, "shale, rust ink".into()));
        swatches.push((base(Material::Shale).ink([0.18, 0.31, 0.48]), lime, "shale, blue ink".into()));
        swatches.push((base(Material::Shale).ink(pale), [0.35, 0.33, 0.30, 1.0], "shale, pale ink".into()));

        // One pattern, four blend modes.
        swatches.push((base(Material::Stone).ink(rust).blend(Blend::Ink), rock, "stone, ink".into()));
        swatches.push((base(Material::Stone).ink(rust).blend(Blend::Shade), rock, "stone, shade".into()));
        swatches.push((base(Material::Stone).ink(pale).blend(Blend::Lighten), rock, "stone, lighten".into()));
        swatches.push((base(Material::Stone).ink(rust).blend(Blend::Stain), rock, "stone, stain".into()));

        // Layering: one rock, then the same rock with a history.
        swatches.push((base(Material::Limestone), lime, "limestone".into()));
        swatches.push((
            base(Material::Limestone).ink(rust).over(
                Overlay::new(Material::Gossan).strength(0.75).blend(Blend::Stain),
            ),
            lime,
            "limestone + gossan".into(),
        ));
        swatches.push((
            base(Material::Quartzite).ink(pale).over(
                Overlay::new(Material::Quartz).strength(0.8).blend(Blend::Lighten),
            ),
            [0.70, 0.62, 0.60, 1.0],
            "quartzite + quartz".into(),
        ));
        swatches.push((
            base(Material::Shale).over(
                Overlay::new(Material::OreSulphide)
                    .strength(0.9)
                    .blend(Blend::Lighten)
                    .scale_ratio(0.8),
            ),
            [0.44, 0.46, 0.50, 1.0],
            "shale + sulphide".into(),
        ));
    } else {
        let lib: [(Material, [f32; 4], &str); 20] = [
            (Material::Grass, [0.38, 0.46, 0.28, 1.0], "grass"),
            (Material::Dirt, [0.52, 0.42, 0.30, 1.0], "dirt"),
            (Material::Stone, [0.56, 0.56, 0.54, 1.0], "stone"),
            (Material::Scree, [0.60, 0.57, 0.51, 1.0], "scree"),
            (Material::Gravel, [0.63, 0.60, 0.54, 1.0], "gravel"),
            (Material::Sand, [0.78, 0.70, 0.52, 1.0], "sand"),
            (Material::Cracked, [0.66, 0.58, 0.46, 1.0], "cracked mud"),
            (Material::Timber, [0.42, 0.33, 0.21, 1.0], "timber"),
            (Material::EndGrain, [0.58, 0.46, 0.30, 1.0], "end grain"),
            (Material::Alluvium, [0.85, 0.79, 0.64, 1.0], "alluvium"),
            (Material::Shale, [0.55, 0.58, 0.63, 1.0], "shale"),
            (Material::Sandstone, [0.82, 0.66, 0.36, 1.0], "sandstone"),
            (Material::Limestone, [0.62, 0.71, 0.72, 1.0], "limestone"),
            (Material::Dolomite, [0.66, 0.69, 0.65, 1.0], "dolomite"),
            (Material::Quartzite, [0.77, 0.66, 0.63, 1.0], "quartzite"),
            (Material::Granite, [0.69, 0.54, 0.51, 1.0], "granite"),
            (Material::Gossan, [0.56, 0.35, 0.18, 1.0], "gossan"),
            (Material::OreSulphide, [0.42, 0.44, 0.47, 1.0], "sulphide ore"),
            (Material::Quartz, [0.86, 0.84, 0.80, 1.0], "quartz"),
            (Material::Timbered, [0.47, 0.40, 0.31, 1.0], "timbered"),
        ];
        for (m, c, name) in lib {
            swatches.push((base(m), c, name.into()));
        }
    }

    const COLS: usize = 4;
    let pad = 14.0;
    let cell_w = (W as f32 - pad * (COLS as f32 + 1.0)) / COLS as f32;
    let rows = swatches.len().div_ceil(COLS);
    let cell_h = (H as f32 - pad * (rows as f32 + 1.0)) / rows as f32;

    let mut batch = Batch::new();
    for (i, (surface, colour, _)) in swatches.iter().enumerate() {
        let (col, row) = (i % COLS, i / COLS);
        let x = -(W as f32) * 0.5 + pad + col as f32 * (cell_w + pad) + cell_w * 0.5;
        let y = (H as f32) * 0.5 - pad - row as f32 * (cell_h + pad) - cell_h * 0.5;

        batch.with_surface(*surface, |b| {
            b.rect(glam::Vec2::new(x, y), glam::Vec2::new(cell_w, cell_h), *colour);
        });
        batch.rect(
            glam::Vec2::new(x, y + cell_h * 0.5),
            glam::Vec2::new(cell_w, 1.5),
            [0.17, 0.15, 0.13, 1.0],
        );
    }

    let pixels = render(&device, &queue, &batch);
    write_png(&out, W, H, &pixels);
    println!("wrote {out}");
    for (i, (_, _, name)) in swatches.iter().enumerate() {
        println!("  r{} c{}  {name}", i / COLS, i % COLS);
    }
}

fn render(device: &wgpu::Device, queue: &wgpu::Queue, batch: &Batch) -> Vec<u8> {
    use wgpu::util::DeviceExt;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(include_str!("../src/renderer/shader.wgsl").into()),
    });
    let proj = glam::Mat4::orthographic_rh(
        -(W as f32) * 0.5,
        W as f32 * 0.5,
        -(H as f32) * 0.5,
        H as f32 * 0.5,
        -1.0,
        1.0,
    );
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::bytes_of(&proj),
        usage: wgpu::BufferUsages::UNIFORM,
    });
    let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    let camera_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &camera_bgl,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
    });

    let white = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: None,
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
    let sampler = device.create_sampler(&Default::default());
    let tex_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
    let tex_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
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

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&camera_bgl, &tex_bgl],
        push_constant_ranges: &[],
    });
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview: None,
        cache: None,
    });

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&Default::default());

    let vbuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&batch.vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let ibuf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&batch.indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    let unpadded = W * 4;
    let padded = unpadded.div_ceil(256) * 256;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: (padded * H) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut enc = device.create_command_encoder(&Default::default());
    {
        let mut rpass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.93,
                        g: 0.90,
                        b: 0.84,
                        a: 1.0,
                    }),
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
    queue.submit([enc.finish()]);

    let slice = readback.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
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

/// Minimal PNG writer: the engine takes no image dependencies, so this encodes
/// the one format it needs by hand, stored (uncompressed) deflate blocks.
fn write_png(path: &str, w: u32, h: u32, rgba: &[u8]) {
    fn crc32(data: &[u8]) -> u32 {
        let mut table = [0u32; 256];
        for (i, e) in table.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *e = c;
        }
        let mut c = 0xFFFF_FFFFu32;
        for &b in data {
            c = table[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
        }
        c ^ 0xFFFF_FFFF
    }
    fn adler32(data: &[u8]) -> u32 {
        let (mut a, mut b) = (1u32, 0u32);
        for &byte in data {
            a = (a + byte as u32) % 65521;
            b = (b + a) % 65521;
        }
        (b << 16) | a
    }
    fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
        out.extend_from_slice(&(body.len() as u32).to_be_bytes());
        let mut with_kind = kind.to_vec();
        with_kind.extend_from_slice(body);
        out.extend_from_slice(&with_kind);
        out.extend_from_slice(&crc32(&with_kind).to_be_bytes());
    }

    // Filter byte 0 per scanline.
    let mut raw = Vec::with_capacity((w * h * 4 + h) as usize);
    for y in 0..h {
        raw.push(0);
        let start = (y * w * 4) as usize;
        raw.extend_from_slice(&rgba[start..start + (w * 4) as usize]);
    }

    // zlib with stored deflate blocks: no compression, but no dependency.
    let mut z = vec![0x78, 0x01];
    for (i, block) in raw.chunks(65_535).enumerate() {
        let last = if (i + 1) * 65_535 >= raw.len() { 1 } else { 0 };
        z.push(last);
        z.extend_from_slice(&(block.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(block.len() as u16)).to_le_bytes());
        z.extend_from_slice(block);
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());

    let mut png = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA
    chunk(&mut png, b"IHDR", &ihdr);
    chunk(&mut png, b"IDAT", &z);
    chunk(&mut png, b"IEND", &[]);
    std::fs::write(path, png).expect("write png");
}
