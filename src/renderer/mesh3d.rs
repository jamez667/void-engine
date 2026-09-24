//! The 3D vertex format and CPU-side mesh (feature `render3d`).
//!
//! # Why a separate vertex from `Vertex`
//!
//! [`crate::renderer::batch::Vertex`] is 84 bytes across 9 attributes, and
//! several of them encode assumptions that only hold on a flat plane:
//! `pattern` is a world-metre coordinate for procedural materials locked to
//! the ground, `local` is a position within a tile for intra-tile blending,
//! and `scale` is the metres-per-pixel used to fade hatching before it
//! aliases. Under perspective there is no single metres-per-pixel per frame,
//! so `scale` stops being a per-vertex quantity at all.
//!
//! Widening that type would also move two buffer caps derived from
//! `size_of::<Vertex>()` (`frame.rs`), which the repo has already been bitten
//! by twice — once producing a guard that passed and then panicked, once
//! letting a frame draw from never-uploaded contents. So: a separate,
//! smaller vertex, and the 2D path untouched.
//!
//! # Why a retained mesh rather than a `Batch`
//!
//! [`crate::renderer::batch::Batch`] is immediate-mode: it accumulates
//! vertices every frame and uploads them once per frame. That is right for
//! 2D, where what is drawn changes constantly and the geometry is cheap to
//! rebuild. It is wrong for static 3D geometry, which is the same every
//! frame and wants to live on the GPU. [`crate::renderer::mesh3d::Mesh3D`]
//! is the CPU-side data;
//! `GpuMesh3D` (in `render3d.rs`) is what it becomes once uploaded.

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3};

/// A single 3D vertex: position, normal, uv, colour.
///
/// 48 bytes, against the 2D `Vertex`'s 84. Position is **camera-relative**,
/// matching [`crate::renderer::camera::Camera3D`] — see that type on why the
/// eye sits at the origin.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct Vertex3D {
    pub pos: [f32; 3],
    /// Surface normal, expected normalised. The shader renormalises after
    /// interpolation anyway, but an unnormalised normal here scales the
    /// lighting rather than only its direction.
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

impl Vertex3D {
    pub fn new(pos: Vec3, normal: Vec3, uv: Vec2, color: [f32; 4]) -> Self {
        Self {
            pos: pos.to_array(),
            normal: normal.to_array(),
            uv: uv.to_array(),
            color,
        }
    }

    /// The vertex buffer layout.
    ///
    /// Offsets are **derived** via `offset_of!`, not written as literals.
    /// The 2D `Vertex::desc` hand-writes nine of them, and `frame.rs`
    /// records two separate incidents caused by a hardcoded stride drifting
    /// from the real one. Deriving them means a field reorder cannot
    /// silently produce a layout that compiles and renders garbage.
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem::offset_of;
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex3D>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: offset_of!(Vertex3D, pos) as wgpu::BufferAddress,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: offset_of!(Vertex3D, normal) as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: offset_of!(Vertex3D, uv) as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: offset_of!(Vertex3D, color) as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

/// CPU-side triangle mesh, ready to upload.
///
/// Indices are `u32` to match the 2D path's index buffer format, so both
/// can share `wgpu::IndexFormat::Uint32` at the draw site.
#[derive(Clone, Debug, Default)]
pub struct Mesh3D {
    pub vertices: Vec<Vertex3D>,
    pub indices: Vec<u32>,
}

impl Mesh3D {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Triangle count. `indices.len() / 3` — the shape every pipeline here
    /// uses is a triangle list.
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Append a triangle, computing its face normal.
    ///
    /// Winding is counter-clockwise when viewed from the side the normal
    /// points toward, matching `FrontFace::Ccw`. A degenerate triangle
    /// (zero-area, so no defined normal) is dropped rather than admitted
    /// with a NaN normal, which would propagate through the shader's
    /// `normalize` and blacken every fragment it touches.
    pub fn push_triangle(&mut self, a: Vec3, b: Vec3, c: Vec3, color: [f32; 4]) {
        let cross = (b - a).cross(c - a);
        if cross.length_squared() <= 0.0 || !cross.is_finite() {
            return;
        }
        let n = cross.normalize();
        let base = self.vertices.len() as u32;
        self.vertices.push(Vertex3D::new(a, n, Vec2::ZERO, color));
        self.vertices.push(Vertex3D::new(b, n, Vec2::ZERO, color));
        self.vertices.push(Vertex3D::new(c, n, Vec2::ZERO, color));
        self.indices.extend([base, base + 1, base + 2]);
    }

    /// Append a quad as two triangles, wound so both share a face normal.
    pub fn push_quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3, color: [f32; 4]) {
        self.push_triangle(a, b, c, color);
        self.push_triangle(a, c, d, color);
    }

    /// An axis-aligned box centred on `center` with the given full extents.
    ///
    /// Here because a 3D path with no geometry at all cannot be seen to
    /// work, and a box is the smallest thing that shows perspective,
    /// depth ordering and per-face lighting at once. Faces are wound
    /// outward, so back-face culling keeps the interior hidden.
    pub fn push_box(&mut self, center: Vec3, size: Vec3, color: [f32; 4]) {
        let h = size * 0.5;
        let (x0, y0, z0) = (center.x - h.x, center.y - h.y, center.z - h.z);
        let (x1, y1, z1) = (center.x + h.x, center.y + h.y, center.z + h.z);

        let p = [
            Vec3::new(x0, y0, z0), // 0
            Vec3::new(x1, y0, z0), // 1
            Vec3::new(x1, y1, z0), // 2
            Vec3::new(x0, y1, z0), // 3
            Vec3::new(x0, y0, z1), // 4
            Vec3::new(x1, y0, z1), // 5
            Vec3::new(x1, y1, z1), // 6
            Vec3::new(x0, y1, z1), // 7
        ];

        self.push_quad(p[4], p[5], p[6], p[7], color); // +Z (up)
        self.push_quad(p[1], p[0], p[3], p[2], color); // -Z (down)
        self.push_quad(p[0], p[1], p[5], p[4], color); // -Y
        self.push_quad(p[2], p[3], p[7], p[6], color); // +Y
        self.push_quad(p[3], p[0], p[4], p[7], color); // -X
        self.push_quad(p[1], p[2], p[6], p[5], color); // +X
    }

    /// A UV sphere centred on `center`, flat-shaded like everything else.
    ///
    /// Sixteen slices by eight stacks: round enough at the distances this
    /// engine's cameras sit at, and small enough that a handful of them
    /// on one body is not the mesh's whole vertex budget.
    ///
    /// Every latitude band is emitted as quads, poles included. At the
    /// poles one edge of each quad has zero length, so its second
    /// triangle is degenerate — and [`Mesh3D::push_triangle`] drops
    /// degenerates rather than admitting a NaN normal. That is what lets
    /// this loop stay uniform instead of special-casing the caps.
    pub fn push_sphere(&mut self, center: Vec3, radius: f32, color: [f32; 4]) {
        const SLICES: usize = 16;
        const STACKS: usize = 8;
        use std::f32::consts::PI;

        let at = |i: usize, j: usize| {
            let phi = PI * i as f32 / STACKS as f32;
            let theta = 2.0 * PI * j as f32 / SLICES as f32;
            center
                + radius
                    * Vec3::new(phi.sin() * theta.cos(), phi.sin() * theta.sin(), phi.cos())
        };

        for i in 0..STACKS {
            for j in 0..SLICES {
                // Wound so the face normal points outward: down the
                // stack first, then round the slice, which is
                // counter-clockwise seen from outside the sphere.
                self.push_quad(at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1), color);
            }
        }
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layout must describe the struct it claims to. A mismatch here
    /// is not a compile error — it renders garbage, or reads past the
    /// buffer.
    #[test]
    fn the_vertex_layout_matches_the_struct() {
        let d = Vertex3D::desc();
        assert_eq!(
            d.array_stride,
            std::mem::size_of::<Vertex3D>() as wgpu::BufferAddress,
        );

        // Every attribute must sit inside the stride, and the formats must
        // sum to no more than it.
        let mut covered = 0u64;
        for a in d.attributes {
            let size = match a.format {
                wgpu::VertexFormat::Float32x2 => 8,
                wgpu::VertexFormat::Float32x3 => 12,
                wgpu::VertexFormat::Float32x4 => 16,
                f => panic!("unexpected format {f:?}"),
            };
            assert!(
                a.offset + size <= d.array_stride,
                "attribute at offset {} ({} bytes) runs past the {}-byte stride",
                a.offset,
                size,
                d.array_stride,
            );
            covered += size;
        }
        assert!(covered <= d.array_stride);
    }

    /// Shader locations must match `shader3d.wgsl`'s `@location` values, in
    /// order. Nothing else checks this: a mismatch binds normals to the uv
    /// slot and draws nonsense.
    #[test]
    fn shader_locations_are_sequential_from_zero() {
        let d = Vertex3D::desc();
        let locs: Vec<u32> = d.attributes.iter().map(|a| a.shader_location).collect();
        assert_eq!(locs, vec![0, 1, 2, 3]);
    }

    #[test]
    fn a_triangle_gets_an_outward_normal() {
        let mut m = Mesh3D::new();
        // CCW seen from +Z, so the normal should point +Z.
        m.push_triangle(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            [1.0; 4],
        );
        assert_eq!(m.triangle_count(), 1);
        for v in &m.vertices {
            assert!(
                (Vec3::from(v.normal) - Vec3::Z).length() < 1e-5,
                "expected +Z normal, got {:?}",
                v.normal,
            );
        }
    }

    /// A zero-area triangle has no normal. Admitting one would put NaN in
    /// the buffer, and the shader's `normalize` turns that into black
    /// fragments rather than an error anyone can trace.
    #[test]
    fn a_degenerate_triangle_is_dropped_rather_than_given_a_nan_normal() {
        let mut m = Mesh3D::new();
        m.push_triangle(Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, [1.0; 4]);
        m.push_triangle(
            Vec3::ZERO,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, 0.0, 0.0), // collinear
            [1.0; 4],
        );
        assert!(m.is_empty(), "degenerate triangles should not be admitted");
        assert!(m.vertices.iter().all(|v| Vec3::from(v.normal).is_finite()));
    }

    #[test]
    fn a_box_has_twelve_triangles_and_finite_normals() {
        let mut m = Mesh3D::new();
        m.push_box(Vec3::ZERO, Vec3::splat(2.0), [1.0; 4]);
        assert_eq!(m.triangle_count(), 12, "6 faces x 2 triangles");
        assert!(m.vertices.iter().all(|v| Vec3::from(v.normal).is_finite()));
    }

    /// Each face should point away from the centre. If a face is wound the
    /// wrong way its normal points inward, and with back-face culling on
    /// it vanishes — a bug that looks like a hole in the geometry.
    #[test]
    fn every_box_face_is_wound_outward() {
        let mut m = Mesh3D::new();
        m.push_box(Vec3::ZERO, Vec3::splat(2.0), [1.0; 4]);
        for v in &m.vertices {
            let pos = Vec3::from(v.pos);
            let n = Vec3::from(v.normal);
            assert!(
                pos.dot(n) > 0.0,
                "face at {pos:?} has inward normal {n:?} — it would be culled",
            );
        }
    }
}
