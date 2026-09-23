//! The 3D render pipeline (feature `render3d`).
//!
//! A pipeline beside `main_pipeline`, not a replacement for it. The two
//! share the main pass, the camera bind group layout and the depth
//! attachment added in Phase 0; they differ in vertex format, shader, and
//! — crucially — depth state.
//!
//! **This is the pipeline that actually uses the depth buffer.** The 2D
//! pipeline carries `depth_compare: Always` with writes off, which is the
//! identity and keeps painter's ordering intact (see `depth.rs`). This one
//! uses `Less` with writes on, because in 3D what occludes what is a
//! property of the geometry rather than of call order. Both are legal in
//! one pass: depth state is per-pipeline.
//!
//! Back-face culling is on here and off everywhere else. The 2D path draws
//! quads whose winding nobody has ever had to think about, so culling them
//! would silently drop geometry; a closed 3D mesh is wound outward by
//! construction ([`crate::renderer::mesh3d::Mesh3D::push_box`]) and
//! culling halves its fragment
//! work.

use bytemuck::{Pod, Zeroable};

use super::depth::DEPTH_FORMAT;
use super::mesh3d::{Mesh3D, Vertex3D};

/// Ceiling on mesh instances submitted in one frame.
///
/// The instance ring is allocated up front at this size, matching how
/// `lights.rs` sizes its per-light ring. At 80 bytes rounded to a 256-byte
/// alignment slot that is 1 MB, which is cheap for the headroom; draws past
/// the cap are dropped with a warning rather than silently overwriting slot
/// zero and drawing every remaining mesh on top of each other.
pub const MAX_MESH_INSTANCES_PER_FRAME: usize = 4096;

/// Per-instance transform and tint. Must match `InstanceUniform` in
/// `shader3d.wgsl`.
///
/// 80 bytes: a 64-byte matrix and a 16-byte tint, both naturally aligned,
/// so unlike `LightUniform` this needs no explicit tail padding to satisfy
/// std140.
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub(super) struct InstanceUniform {
    pub model: [[f32; 4]; 4],
    pub tint: [f32; 4],
}

/// A mesh that lives on the GPU.
///
/// Uploaded once and drawn every frame, unlike `Batch`, which is rebuilt
/// per frame. That difference is the point of the type: static geometry
/// should not pay to be re-uploaded 60 times a second.
pub struct GpuMesh3D {
    pub(super) vertex_buffer: wgpu::Buffer,
    pub(super) index_buffer: wgpu::Buffer,
    pub(super) index_count: u32,
}

impl GpuMesh3D {
    /// Upload a [`Mesh3D`]. Returns `None` for an empty mesh: wgpu rejects
    /// a zero-sized buffer, and a mesh with no triangles has nothing to
    /// draw anyway.
    pub fn upload(device: &wgpu::Device, mesh: &Mesh3D) -> Option<Self> {
        use wgpu::util::DeviceExt;
        if mesh.is_empty() || mesh.vertices.is_empty() {
            return None;
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh3d_vbuf"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mesh3d_ibuf"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Some(Self {
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
        })
    }

    pub fn index_count(&self) -> u32 {
        self.index_count
    }
}

/// The depth state for 3D geometry: test and write, nearest wins.
///
/// The counterpart to `depth::main_pipeline_state`, which is deliberately
/// the identity. Split out so a test can assert the two differ in exactly
/// the way the design intends — if this ever silently became `Always` too,
/// 3D would fall back to draw-order and look subtly wrong rather than
/// fail.
pub fn depth_state() -> Option<wgpu::DepthStencilState> {
    Some(wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: true,
        // `Less`, with the pass clearing to 1.0 and `Camera3D`'s
        // `perspective_rh` mapping near→0 and far→1. Getting any of those
        // three out of step blanks the frame.
        depth_compare: wgpu::CompareFunction::Less,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    })
}

pub(super) struct Render3D {
    pub pipeline: wgpu::RenderPipeline,
    /// Ring of per-instance uniforms, read at a dynamic offset. One bind
    /// group serves every instance in the frame.
    pub instance_buffer: wgpu::Buffer,
    pub instance_bg: wgpu::BindGroup,
    /// Slot size, rounded up to the device's uniform offset alignment.
    pub instance_stride: u64,
}

impl Render3D {
    /// Build the 3D pipeline against the surface format and the *existing*
    /// camera bind group layout — the same one the 2D path uses, which is
    /// what `Camera3D` filling the same
    /// [`CameraUniform`](super::camera::CameraUniform) buys.
    pub fn new(
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("shader3d.wgsl"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader3d.wgsl").into()),
        });
        // Slot size for the instance ring, rounded to the device's
        // alignment — nearly always 256 bytes against an 80-byte struct.
        // Same shape as the per-light ring in `lights.rs`.
        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let raw = std::mem::size_of::<InstanceUniform>() as u64;
        let instance_stride = raw.div_ceil(align) * align;

        let instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("render3d_instance_bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: std::num::NonZeroU64::new(raw),
                },
                count: None,
            }],
        });

        let instance_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("render3d_instance_ring"),
            size: instance_stride * MAX_MESH_INSTANCES_PER_FRAME as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let instance_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render3d_instance_bg"),
            layout: &instance_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &instance_buffer,
                    offset: 0,
                    size: std::num::NonZeroU64::new(raw),
                }),
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("render3d_pl"),
            // Camera at 0, per-instance transform at 1. The 2D path's
            // second group is the glyph atlas, which this shader does not
            // sample: `uv` rides along for a future textured path but
            // nothing binds a texture yet.
            bind_group_layouts: &[camera_bgl, &instance_bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render3d_pipeline"),
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
                    // Opaque. Alpha blending on depth-tested geometry needs
                    // back-to-front sorting to be correct, which nothing
                    // here does yet — better to draw solid than to draw
                    // wrong.
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
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        Self {
            pipeline,
            instance_buffer,
            instance_bg,
            instance_stride,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The two pipelines' depth states must differ in the way the design
    /// intends: 2D is the identity, 3D actually tests and writes. If 3D
    /// silently became `Always`, occlusion would fall back to draw order
    /// and look subtly wrong rather than fail.
    #[test]
    fn the_3d_pipeline_tests_depth_where_the_2d_one_does_not() {
        let three = depth_state().expect("3D always has depth state");
        let two = super::super::depth::main_pipeline_state().expect("2D state exists");

        assert!(three.depth_write_enabled, "3D geometry must write depth");
        assert_eq!(three.depth_compare, wgpu::CompareFunction::Less);

        assert!(!two.depth_write_enabled, "the 2D path must stay a no-op");
        assert_eq!(two.depth_compare, wgpu::CompareFunction::Always);

        assert_eq!(
            three.format, two.format,
            "both pipelines share one depth attachment, so the formats must match",
        );
    }
}
