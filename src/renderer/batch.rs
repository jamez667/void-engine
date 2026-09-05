use bytemuck::{Pod, Zeroable};
use glam::Vec2;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct Vertex {
    pub pos: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    /// Position in *pattern space*, which for world geometry is world metres.
    ///
    /// Procedural materials are a function of this rather than of screen or UV
    /// space, so a hatch stays locked to the ground: it does not swim when the
    /// camera pans, and its scale is stable under zoom.
    pub pattern: [f32; 2],
    /// Material id in the low 16 bits, flags in the high 16. Zero is the plain
    /// solid-colour material every existing primitive already uses, so a vertex
    /// built without thinking about materials behaves exactly as before.
    pub material: u32,
    /// Metres per pixel at the time the vertex was built.
    ///
    /// The fragment shader needs this to know how dense the hatch would come
    /// out on screen, so it can fade to flat colour before the pattern aliases
    /// into moire rather than after.
    pub scale: f32,
}

impl Default for Vertex {
    fn default() -> Self {
        Vertex {
            pos: [0.0, 0.0],
            uv: [0.5, 0.5],
            color: [1.0, 1.0, 1.0, 1.0],
            pattern: [0.0, 0.0],
            material: 0,
            scale: 1.0,
        }
    }
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
                wgpu::VertexAttribute {
                    offset: 32,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 40,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Uint32,
                },
                wgpu::VertexAttribute {
                    offset: 44,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

/// A procedural surface pattern, generated in the fragment shader.
///
/// These are not textures: nothing is sampled and no image exists. Each is a
/// function of position, evaluated per fragment, so it costs no memory, never
/// tiles visibly, and stays sharp at any zoom.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
#[repr(u32)]
pub enum Material {
    /// Flat colour. What every primitive draws unless told otherwise.
    #[default]
    Solid = 0,

    // --- Ground cover ---------------------------------------------------
    /// Fine short strokes at varied angles, denser in clumps: turf.
    Grass = 1,
    /// Irregular fine stipple with occasional coarser grains: bare earth.
    Dirt = 2,
    /// Angular cracks and mottled patches: broken rock.
    Stone = 3,
    /// Loose angular fragments of every size: talus, and a mine's waste dump.
    Scree = 4,
    /// Rounded, sorted, water-worn: a creek bed or a placer wash.
    Gravel = 5,
    /// Wind-drifted ripples, no fragments at all.
    Sand = 6,
    /// Cracked polygons of dried mud: a playa or a dry wash bottom.
    Cracked = 7,
    /// Rough bark and radiating grain: a stump or a cut end of timber.
    Timber = 8,
    /// Sawn end grain: concentric rings, for a beam seen end-on.
    EndGrain = 9,

    // --- Lithology ------------------------------------------------------
    // The eight the UI spec names, each with the pattern a survey drawing
    // uses for it. Colour never distinguishes a rock on its own.
    /// Scattered stipple dots of irregular size.
    Alluvium = 16,
    /// Dense horizontal fine lines, close spaced: bedded shale.
    Shale = 17,
    /// Uniform fine stipple, even across the field.
    Sandstone = 18,
    /// Brick courses: horizontal lines with offset verticals.
    Limestone = 19,
    /// Brick courses with a 45 degree tick in each: the dolomite convention.
    Dolomite = 20,
    /// Fine cross-hatch at 45 and 135 degrees.
    Quartzite = 21,
    /// Randomised crosses and plus marks at low density.
    Granite = 22,
    /// Irregular blotches: the iron-oxide cap over a sulphide body.
    Gossan = 23,

    // --- Mineralisation and workings ------------------------------------
    /// Bright cubic glints: galena and the like in a face.
    OreSulphide = 32,
    /// Massive, glassy, faintly banded: a quartz vein.
    Quartz = 33,
    /// Timbered ground: regular set marks, for a driven and supported drift.
    Timbered = 34,
}

/// How a material is applied over its region.
#[derive(Copy, Clone, Debug)]
pub struct Surface {
    pub material: Material,
    /// Size of the pattern's features, in pattern-space units (world metres).
    /// Larger means a coarser, more open pattern.
    pub scale_m: f32,
    /// How strongly the pattern departs from the flat colour, 0..1.
    pub strength: f32,
    /// Metres per pixel on screen, used to fade the pattern out before it
    /// aliases. Set this from the camera zoom.
    pub m_per_px: f32,
}

impl Surface {
    pub fn new(material: Material) -> Self {
        Surface { material, scale_m: 1.0, strength: 1.0, m_per_px: 1.0 }
    }

    pub fn scale_m(mut self, m: f32) -> Self {
        self.scale_m = m.max(0.0001);
        self
    }

    pub fn strength(mut self, s: f32) -> Self {
        self.strength = s.clamp(0.0, 1.0);
        self
    }

    pub fn m_per_px(mut self, m: f32) -> Self {
        self.m_per_px = m.max(0.0);
        self
    }

    /// Pack the material id and its strength into the vertex material word.
    fn packed(&self) -> u32 {
        let strength = (self.strength.clamp(0.0, 1.0) * 255.0).round() as u32;
        (self.material as u32 & 0xFFFF) | (strength << 16)
    }
}

pub struct Batch {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// The surface applied to geometry pushed from now on. `None` means flat
    /// colour, which is what every existing caller gets without asking.
    surface: Option<Surface>,
}

impl Batch {
    pub fn new() -> Self {
        Self {
            vertices: Vec::with_capacity(8192),
            indices: Vec::with_capacity(32768),
            surface: None,
        }
    }

    pub fn clear(&mut self) {
        self.vertices.clear();
        self.indices.clear();
        self.surface = None;
    }

    /// Draw following geometry with a procedural surface.
    ///
    /// This is deliberately modal rather than a parameter on every primitive:
    /// a caller filling a band of rock sets the surface once and then draws
    /// with the same `rect`/`polygon` calls as everything else, and every
    /// existing call site keeps its signature and its flat colour.
    pub fn set_surface(&mut self, surface: Surface) {
        self.surface = Some(surface);
    }

    /// Go back to flat colour.
    pub fn clear_surface(&mut self) {
        self.surface = None;
    }

    /// Run `f` with a surface applied, restoring whatever was set before.
    pub fn with_surface<R>(&mut self, surface: Surface, f: impl FnOnce(&mut Self) -> R) -> R {
        let prev = self.surface;
        self.surface = Some(surface);
        let out = f(self);
        self.surface = prev;
        out
    }

    /// Stamp the current surface onto a vertex at a world position.
    ///
    /// Every primitive funnels through here, so a material added later is
    /// picked up by all of them without touching any of their code.
    #[inline]
    fn vertex(&self, pos: Vec2, uv: [f32; 2], color: [f32; 4]) -> Vertex {
        match self.surface {
            None => Vertex { pos: [pos.x, pos.y], uv, color, ..Default::default() },
            Some(s) => Vertex {
                pos: [pos.x, pos.y],
                uv,
                color,
                // Pattern space is world space, divided by the feature size so
                // the shader always works in units of one pattern cell.
                pattern: [pos.x / s.scale_m, pos.y / s.scale_m],
                material: s.packed(),
                // Carried in the same units, so the shader can compare feature
                // size against pixel size and fade before it aliases.
                scale: s.m_per_px / s.scale_m,
            },
        }
    }

    pub fn push_quad(&mut self, corners: [Vec2; 4], uv: [[f32; 2]; 4], color: [f32; 4]) {
        let base = self.vertices.len() as u32;
        for (i, c) in corners.iter().enumerate() {
            self.vertices.push(self.vertex(*c, uv[i], color));
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
            self.vertices.push(self.vertex(center, [0.5, 0.5], color));
            self.vertices.push(self.vertex(p0, [0.5, 0.5], color));
            self.vertices.push(self.vertex(p1, [0.5, 0.5], color));
            self.indices.extend_from_slice(&[base, base + 1, base + 2]);
        }
    }

    /// Filled triangle.
    pub fn triangle(&mut self, p0: Vec2, p1: Vec2, p2: Vec2, color: [f32; 4]) {
        let base = self.vertices.len() as u32;
        for p in [p0, p1, p2] {
            self.vertices.push(self.vertex(p, [0.5, 0.5], color));
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
            self.vertices.push(self.vertex(center, [0.5, 0.5], color));
            self.vertices.push(self.vertex(p0, [0.5, 0.5], color));
            self.vertices.push(self.vertex(p1, [0.5, 0.5], color));
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
