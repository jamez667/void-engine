//! Depth buffer for the main pass.
//!
//! Until this existed, ordering in the main pass was painter's-algorithm
//! only: every pipeline set `depth_stencil: None` and every pass set
//! `depth_stencil_attachment: None`, so what a caller drew last won. That
//! is workable for 2D, where `Batch` emits primitives in the order the
//! game asks for them, and it is what the split-index machinery in
//! `frame.rs` exists to preserve across composite boundaries.
//!
//! It is not workable for 3D, where what occludes what is a property of
//! the geometry rather than of call order. This module owns the depth
//! texture that makes the difference.
//!
//! **Why only the main pass.** Three pipelines take `Vertex::desc()` and
//! therefore draw geometry: `main_pipeline`, `offscreen_pipeline` and
//! `shadow_mask_pipeline`. The other eleven are fullscreen triangles, for
//! which depth is meaningless. Of the three, only the main pass draws the
//! scene 3D geometry will join: the offscreen pass renders a 2D sub-scene
//! that is then blurred, and the wall mask is a flat binary occluder map
//! feeding the screen-space effects. Giving either of those a depth buffer
//! would mean a second and third depth texture (the mask is 2x surface
//! size) to no purpose, so they keep `depth_stencil: None`.
//!
//! **Why the composites stay depth-free.** The main pass interleaves
//! fullscreen composite draws between ranges of the main batch. Those
//! pipelines keep `depth_stencil: None` while `main_pipeline` gains depth
//! state. Within one render pass that is legal and deliberate: a
//! composite is a full-screen overlay that must not be clipped by scene
//! geometry drawn before it.
//!
//! **Why 2D is unaffected.** [`crate::renderer::depth::MAIN_WRITES_DEPTH`]
//! is false and the compare
//! is `Always`, so the main pipeline tests and writes nothing: the state is
//! the identity. `tests/depth_buffer.rs` asserts that by rendering the same
//! overlapping geometry down both paths and requiring byte-identical
//! framebuffers.

/// Depth format. `Depth32Float` rather than `Depth24PlusStencil8` because
/// nothing here wants a stencil buffer, and 32-bit float depth is
/// supported as a render target on every backend wgpu targets without a
/// feature check.
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Whether the main pipeline writes and tests depth.
///
/// **False, deliberately, and this is what keeps 2D pixel-identical.**
/// The attachment exists and is cleared every frame, but with
/// `depth_compare: Always` and `depth_write_enabled: false` the main
/// pipeline behaves exactly as it did with no depth buffer at all: every
/// fragment passes, nothing is recorded, and draw order alone decides the
/// result. All 2D geometry sits at `z = 0.0` (`shader.wgsl` hardcodes it),
/// so enabling the test today would be a coin-flip on equal depths rather
/// than a correctness win.
///
/// A 3D path flips this to true *for its own pipeline*, which is a
/// separate pipeline with a separate shader — it does not need this
/// constant to change. This is here to document that the 2D pipeline's
/// depth state is a considered no-op rather than an oversight.
pub const MAIN_WRITES_DEPTH: bool = false;

/// The depth-stencil state the main pipeline uses.
///
/// Returned as an `Option` mirroring the field it fills so the call site
/// reads as a swap for the `None` that was there before. Public so a 3D
/// pipeline sharing the main pass can match the format.
pub fn main_pipeline_state() -> Option<wgpu::DepthStencilState> {
    Some(wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: MAIN_WRITES_DEPTH,
        // `Always` with writes off is the identity: it reproduces
        // no-depth-buffer behaviour exactly. A 3D pipeline wants `Less`.
        depth_compare: wgpu::CompareFunction::Always,
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    })
}

pub(super) struct DepthBuffer {
    pub view: wgpu::TextureView,
    // Held so the view stays valid; the texture is never touched again.
    // Unlike the effect passes this carries no width/height, because
    // nothing here reallocates conditionally — `resize` rebuilds it
    // unconditionally alongside the colour targets.
    _tex: wgpu::Texture,
}

impl DepthBuffer {
    /// Allocate the depth texture at surface size.
    ///
    /// Returns `None` on a zero-size surface, matching the other offscreen
    /// passes: a minimised window must not panic.
    ///
    /// **The `None` case cannot reach the main pass, and that matters.**
    /// Unlike the effect passes, which simply skip when absent, a missing
    /// depth attachment here would be a validation error rather than a
    /// downgrade: `main_pipeline` declares depth state, and wgpu requires
    /// a pass binding it to supply an attachment. That is safe only
    /// because the two `None` routes are both unreachable from a live main
    /// pass — a zero-size surface yields no swapchain texture, so
    /// `end_frame` returns before opening the pass, and `create_texture`
    /// panics rather than returning on allocation failure.
    ///
    /// If this ever gains a route that returns `None` on a renderable
    /// surface, `main_pipeline_state` must become conditional too, or the
    /// main pass will fail validation instead of losing depth.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Option<Self> {
        if width == 0 || height == 0 {
            return None;
        }

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("depth_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            // No TEXTURE_BINDING: nothing samples depth yet. A 3D shadow
            // pass would add it here.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor::default());

        Some(Self {
            view,
            _tex: tex,
        })
    }

    /// The attachment for the main pass. Clears to 1.0 (the far plane)
    /// each frame and stores, so a later pass could sample it.
    pub fn attachment(&self) -> wgpu::RenderPassDepthStencilAttachment<'_> {
        wgpu::RenderPassDepthStencilAttachment {
            view: &self.view,
            depth_ops: Some(wgpu::Operations {
                load: wgpu::LoadOp::Clear(1.0),
                store: wgpu::StoreOp::Store,
            }),
            stencil_ops: None,
        }
    }
}
