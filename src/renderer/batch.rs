use bytemuck::{Pod, Zeroable};
use glam::Vec2;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

impl Vertex {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 16,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        }
    }
}

pub struct Batch {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Batch {
    pub fn new() -> Self {
        Self {
            vertices: Vec::with_capacity(8192),
            indices: Vec::with_capacity(32768),
        }
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
    }

    pub fn push_quad(&mut self, corners: [Vec2; 4], uv: [[f32; 2]; 4], color: [f32; 4]) {
        let base = self.vertices.len() as u32;
        for (i, c) in corners.iter().enumerate() {
            self.vertices.push(Vertex {
                pos: [c.x, c.y],
                uv: uv[i],
                color,
            });
        }
        self.indices
            .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    /// Axis-aligned colored rect (no rotation)
    pub fn rect(&mut self, center: Vec2, size: Vec2, color: [f32; 4]) {
        let h = size * 0.5;
        let corners = [
            center + Vec2::new(-h.x, h.y),
            center + Vec2::new(h.x, h.y),
            center + Vec2::new(h.x, -h.y),
            center + Vec2::new(-h.x, -h.y),
        ];
        let uv = [[0.5, 0.5]; 4];
        self.push_quad(corners, uv, color);
    }

    /// Rotated rect (angle in radians)
    pub fn quad(&mut self, center: Vec2, size: Vec2, angle: f32, color: [f32; 4]) {
        let h = size * 0.5;
        let (sin, cos) = angle.sin_cos();
        let rotate = |v: Vec2| Vec2::new(v.x * cos - v.y * sin, v.x * sin + v.y * cos);
        let corners = [
            center + rotate(Vec2::new(-h.x, h.y)),
            center + rotate(Vec2::new(h.x, h.y)),
            center + rotate(Vec2::new(h.x, -h.y)),
            center + rotate(Vec2::new(-h.x, -h.y)),
        ];
        let uv = [[0.5, 0.5]; 4];
        self.push_quad(corners, uv, color);
    }

    /// Line drawn as a thin rotated rect
    pub fn line(&mut self, a: Vec2, b: Vec2, thickness: f32, color: [f32; 4]) {
        let dir = b - a;
        let len = dir.length();
        if len < 0.001 {
            return;
        }
        let center = (a + b) * 0.5;
        let angle = dir.y.atan2(dir.x);
        self.quad(center, Vec2::new(len, thickness), angle, color);
    }

    /// Circle approximated as triangle fan
    pub fn circle(&mut self, center: Vec2, radius: f32, color: [f32; 4], segments: u32) {
        let n = segments.max(6);
        for i in 0..n {
            let a0 = (i as f32 / n as f32) * std::f32::consts::TAU;
            let a1 = ((i + 1) as f32 / n as f32) * std::f32::consts::TAU;
            let p0 = center + Vec2::new(a0.cos(), a0.sin()) * radius;
            let p1 = center + Vec2::new(a1.cos(), a1.sin()) * radius;
            let base = self.vertices.len() as u32;
            self.vertices.push(Vertex {
                pos: [center.x, center.y],
                uv: [0.5, 0.5],
                color,
            });
            self.vertices.push(Vertex {
                pos: [p0.x, p0.y],
                uv: [0.5, 0.5],
                color,
            });
            self.vertices.push(Vertex {
                pos: [p1.x, p1.y],
                uv: [0.5, 0.5],
                color,
            });
            self.indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
    }

    /// Filled triangle.
    pub fn triangle(&mut self, p0: Vec2, p1: Vec2, p2: Vec2, color: [f32; 4]) {
        let base = self.vertices.len() as u32;
        for p in [p0, p1, p2] {
            self.vertices.push(Vertex { pos: [p.x, p.y], uv: [0.5, 0.5], color });
        }
        self.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }

    /// Irregular polygon: `radii[i]` is the radius of vertex i, `rot` rotates the whole shape.
    pub fn polygon(&mut self, center: Vec2, radii: &[f32], rot: f32, color: [f32; 4]) {
        let n = radii.len();
        if n < 3 { return; }
        for i in 0..n {
            let a0 = rot + (i as f32 / n as f32) * std::f32::consts::TAU;
            let a1 = rot + ((i + 1) as f32 / n as f32) * std::f32::consts::TAU;
            let p0 = center + Vec2::new(a0.cos(), a0.sin()) * radii[i];
            let p1 = center + Vec2::new(a1.cos(), a1.sin()) * radii[(i + 1) % n];
            let base = self.vertices.len() as u32;
            self.vertices.push(Vertex { pos: [center.x, center.y], uv: [0.5, 0.5], color });
            self.vertices.push(Vertex { pos: [p0.x, p0.y], uv: [0.5, 0.5], color });
            self.vertices.push(Vertex { pos: [p1.x, p1.y], uv: [0.5, 0.5], color });
            self.indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
    }

    /// Outline of an irregular polygon (line segments between adjacent vertices).
    pub fn polygon_outline(&mut self, center: Vec2, radii: &[f32], rot: f32, thickness: f32, color: [f32; 4]) {
        let n = radii.len();
        if n < 3 { return; }
        for i in 0..n {
            let a0 = rot + (i as f32 / n as f32) * std::f32::consts::TAU;
            let a1 = rot + ((i + 1) as f32 / n as f32) * std::f32::consts::TAU;
            let p0 = center + Vec2::new(a0.cos(), a0.sin()) * radii[i];
            let p1 = center + Vec2::new(a1.cos(), a1.sin()) * radii[(i + 1) % n];
            self.line(p0, p1, thickness, color);
        }
    }

    /// Circle outline
    pub fn circle_outline(
        &mut self,
        center: Vec2,
        radius: f32,
        thickness: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        self.arc(center, radius, 0.0, std::f32::consts::TAU, thickness, color, segments);
    }

    /// Partial circle outline: a `sweep`-radian arc starting at
    /// `start_angle`, walked as `segments` line pieces. `circle_outline`
    /// is this with a full-turn sweep. Useful for gauge arcs and for
    /// specular highlight sweeps on curved surfaces.
    #[allow(clippy::too_many_arguments)]
    pub fn arc(
        &mut self,
        center: Vec2,
        radius: f32,
        start_angle: f32,
        sweep: f32,
        thickness: f32,
        color: [f32; 4],
        segments: u32,
    ) {
        let n = segments.max(1);
        for i in 0..n {
            let a0 = start_angle + sweep * (i as f32 / n as f32);
            let a1 = start_angle + sweep * ((i + 1) as f32 / n as f32);
            let p0 = center + Vec2::new(a0.cos(), a0.sin()) * radius;
            let p1 = center + Vec2::new(a1.cos(), a1.sin()) * radius;
            self.line(p0, p1, thickness, color);
        }
    }
}

impl Default for Batch {
    fn default() -> Self {
        Self::new()
    }
}
