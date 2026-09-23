//! Shared harness for the GPU-backed tests.
//!
//! `depth_buffer`, `render3d` and `shadow3d` each stand up a headless
//! wgpu device, render something small, and read the pixels back. Those
//! three steps were written out in all three files — the adapter request
//! and device descriptor were byte-identical apart from a label, and the
//! 256-byte row-padding arithmetic in the readback is the kind of detail
//! that is wrong in one copy and right in the others.
//!
//! `materials_render` deliberately keeps its own copy: it predates this
//! and tests the 2D path, so rewriting it to prove a point about the 3D
//! work would be unrelated churn on a file that already passes.
//!
//! Each integration test is its own crate, so this is included with
//! `mod common;` rather than imported. A test that uses only part of it
//! will see dead-code warnings, hence the blanket allow — the module is
//! shared infrastructure, not every consumer's whole surface.

#![allow(dead_code)]

/// A headless device and queue.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

/// Request a headless adapter and device.
///
/// Returns `None` when no adapter is available, so a caller skips rather
/// than fails: CI without a GPU should not report a red build for a
/// machine limitation. Every caller is expected to `eprintln!` and return
/// on `None` so the skip is visible in the log rather than silent.
pub fn gpu(label: &'static str) -> Option<Gpu> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::default(),
        compatible_surface: None,
        force_fallback_adapter: false,
    }))?;
    let (device, queue) = pollster::block_on(adapter.request_device(
        &wgpu::DeviceDescriptor {
            label: Some(label),
            required_features: wgpu::Features::empty(),
            // `downlevel_defaults` deliberately: it is the weakest profile
            // wgpu offers, so a test passing here passes on the hardware
            // the engine actually targets rather than only on a dev box.
            required_limits: wgpu::Limits::downlevel_defaults(),
            memory_hints: wgpu::MemoryHints::default(),
        },
        None,
    ))
    .ok()?;
    Some(Gpu { device, queue })
}

/// A colour target plus its readback buffer, sized for `width x height`.
pub struct Readback {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    /// Row stride rounded up to wgpu's 256-byte alignment requirement.
    padded: u32,
}

impl Readback {
    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    pub fn new(gpu: &Gpu, width: u32, height: u32) -> Self {
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("readback_target"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        // `copy_texture_to_buffer` requires each row to start on a
        // 256-byte boundary, so the buffer is wider than the image and
        // `pixels` strips the padding back off.
        let padded = (width * 4).div_ceil(256) * 256;
        let buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback_buffer"),
            size: (padded * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self { texture, view, buffer, width, height, padded }
    }

    /// Queue the copy from the colour target into the readback buffer.
    pub fn copy_from_texture(&self, enc: &mut wgpu::CommandEncoder) {
        enc.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Map the buffer and return tightly-packed RGBA, padding stripped.
    ///
    /// Call after submitting the encoder that ran `copy_from_texture`.
    pub fn pixels(&self, gpu: &Gpu) -> Vec<u8> {
        let slice = self.buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device.poll(wgpu::Maintain::Wait);
        let mapped = slice.get_mapped_range();

        let unpadded = (self.width * 4) as usize;
        let mut out = Vec::with_capacity(unpadded * self.height as usize);
        for row in 0..self.height {
            let start = (row * self.padded) as usize;
            out.extend_from_slice(&mapped[start..start + unpadded]);
        }
        drop(mapped);
        self.buffer.unmap();
        out
    }
}

/// A depth texture matching the engine's depth format.
pub fn depth_texture(gpu: &Gpu, width: u32, height: u32) -> wgpu::TextureView {
    let tex = gpu.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("test_depth"),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: void_engine::renderer::depth::DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    tex.create_view(&Default::default())
}

/// A bind group layout for a single uniform buffer at binding 0.
///
/// The camera, instance and point-light groups all have this shape, so
/// the three test files were each writing it out two or three times.
pub fn uniform_bgl(
    gpu: &Gpu,
    visibility: wgpu::ShaderStages,
) -> wgpu::BindGroupLayout {
    gpu.device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
}

/// A bind group binding `buffer` at binding 0 of `layout`.
pub fn uniform_bg(
    gpu: &Gpu,
    layout: &wgpu::BindGroupLayout,
    buffer: &wgpu::Buffer,
) -> wgpu::BindGroup {
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    })
}
