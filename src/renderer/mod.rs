//! wgpu renderer. This module owns the [`Renderer`] struct definition,
//! `resize`, and the small queue/accessor surface; the two bulky halves
//! live in sibling child modules:
//!
//! - [`init`] — `Renderer::new`: GPU bring-up, pipelines, bind-group
//!   layouts, offscreen pass allocation.
//! - [`frame`] — `begin_frame` / `end_frame`: the per-frame encoder pass.
//!
//! Both are children of this module, so they reach `Renderer`'s private
//! fields without any visibility widening.

pub mod context;
pub mod camera;
pub mod batch;
mod postprocess;
mod shadow;
mod lights;
mod sun;
mod godray;
mod init;
mod frame;

use context::GpuContext;
use camera::Camera2D;
use batch::{Batch, Vertex};
use postprocess::PostProcess;
use shadow::ShadowPass;
use lights::{LightPass, LightUniform, MAX_LIGHTS_PER_FRAME, AMBIENT_CLEAR};
use sun::SunPass;
use godray::GodrayPass;
use glam::Vec2;
use bytemuck;
use wgpu::util::DeviceExt;
use std::sync::Arc;
use winit::window::Window;

pub struct ScreenshotData {
    pub pixels: Vec<u8>,   // RGBA8, no row padding
    pub width:  u32,
    pub height: u32,
}

pub struct Renderer {
    pub gpu: GpuContext,
    pub camera: Camera2D,
    pub batch: Batch,
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    white_texture_bind_group: wgpu::BindGroup,
    vertex_capacity: usize,
    index_capacity: usize,
    pub shake_trauma: f32,
    shake_offset: Vec2,
    pub screenshot_pending: bool,
    pub screenshot_data:    Option<ScreenshotData>,
    /// Last-second rollup of frame timings + vertex count, published by the
    /// engine's `PerfStats::record` after each rollup tick (~1 Hz). Defaults
    /// to zero on the first frame. Surfaced to `App::render` so the F3
    /// debug overlay can show the same numbers as the `[perf]` log line
    /// without duplicating the rolling-average bookkeeping.
    pub last_perf: crate::app::PerfSnapshot,
    window: Arc<Window>,

    // Post-process pipeline. `postprocess` is `None` on hardware where
    // texture allocation failed or the surface has zero-size — in that
    // case `begin_offscreen_batch` returns a no-op batch and
    // `blur_and_composite` skips, so the main pass still renders.
    postprocess: Option<PostProcess>,
    offscreen_batch: Batch,
    offscreen_vbuf: wgpu::Buffer,
    offscreen_ibuf: wgpu::Buffer,
    offscreen_vcap: usize,
    offscreen_icap: usize,
    // Set by `blur_and_composite`, consumed and cleared by `end_frame`.
    // Radius stored so multi-pass structure lives inside the encoder step
    // instead of leaking through the client API.
    pending_blur_radius: Option<f32>,
    // Main-batch index count captured at the moment `blur_and_composite`
    // was called. The main draw is split into pre-composite and post-
    // composite ranges around this point so the blurred layer paints on
    // top of the backdrop but under the active-floor geometry.
    composite_split_index: Option<u32>,
    // Cached bind group layouts so `resize` can rebuild `postprocess`
    // against the current surface format without reconstructing the whole
    // renderer.
    camera_bgl: wgpu::BindGroupLayout,
    texture_bgl: wgpu::BindGroupLayout,

    // Shadow pipeline. Mirrors the blur pipeline: `None` when texture
    // allocation fails or the surface has zero size — `begin_wall_mask_batch`
    // still returns a scratch batch and `raycast_and_composite_shadows`
    // becomes a no-op so the main frame keeps rendering.
    shadow: Option<ShadowPass>,
    mask_batch: Batch,
    mask_vbuf:  wgpu::Buffer,
    mask_ibuf:  wgpu::Buffer,
    mask_vcap:  usize,
    mask_icap:  usize,
    // Set by `raycast_and_composite_shadows`; consumed + cleared by
    // `end_frame`. Split index records where in the main batch the shadow
    // composite should slot in — mirrors `composite_split_index` for blur.
    pending_shadow: Option<(Vec2, f32)>,
    shadow_split_index: Option<u32>,

    // Per-light additive light-map pipeline (on-foot lighting). Mirrors
    // `shadow` — `None` when texture allocation fails or the surface has
    // zero size; the client API becomes a no-op so the frame still renders.
    // Reuses the `mask_batch` / `wall_mask` texture from `shadow` as its
    // occlusion source (same tile+character silhouettes, different consumer).
    lights_pass: Option<LightPass>,
    /// Emitters queued for this frame via `push_light`. Drained + uploaded
    /// into the LightPass ring buffer inside `end_frame`.
    pending_lights: Vec<LightUniform>,
    /// Split index into the main batch where `render_lights_and_composite`
    /// should slot the light-map multiply. Mirrors `shadow_split_index`.
    lights_split_index: Option<u32>,
    /// Set by `render_lights_and_composite` to opt the frame into the
    /// light pass. Even with zero pushed lights we still clear + composite
    /// the ambient light-map so the scene darkens correctly.
    lights_pending: bool,

    // Global directional-sun pass. Shares the wall_mask texture with the
    // shadow + lights pipelines — no additional mask geometry needed.
    // `None` when texture allocation fails or the surface has zero size
    // (matches the other passes); `queue_sun_pass` becomes a no-op.
    sun_pass: Option<SunPass>,
    /// Set by `queue_sun_pass`. Tuple: (sun_dir_screen, intensity, color).
    /// Consumed + cleared by `end_frame`.
    pending_sun: Option<(Vec2, f32, [f32; 3], f32, f32)>,
    /// Split index into the main batch where the sun composite slots in.
    sun_split_index: Option<u32>,

    // Offscreen god-ray pass. Streams soft volumetric shafts from lit
    // windows into interior spaces. `None` on the same failure modes as
    // the other pipelines.
    godray_pass: Option<GodrayPass>,
    /// Set by `queue_godray_pass`. Same tuple as `pending_sun`.
    pending_godray: Option<(Vec2, f32, [f32; 3], f32, f32)>,
    /// Split index into the main batch where the godray composite slots in.
    godray_split_index: Option<u32>,
}

impl Renderer {

    pub fn resize(&mut self, width: u32, height: u32) {
        self.gpu.resize(width, height);
        self.camera.viewport_size = Vec2::new(width as f32, height as f32);
        // Postprocess textures are viewport-sized; recreate them so the blur
        // shader continues sampling at 1:1. Skip when width/height is zero
        // (minimised) — surface reconfigure would have skipped too.
        if width > 0 && height > 0 {
            self.postprocess = PostProcess::new(
                &self.gpu.device,
                &self.camera_bgl,
                &self.texture_bgl,
                self.gpu.format,
                width,
                height,
            );
            if self.postprocess.is_none() {
                log::warn!("[renderer] postprocess pipeline unavailable after resize");
            }
            self.shadow = ShadowPass::new(
                &self.gpu.device,
                &self.camera_bgl,
                &self.texture_bgl,
                self.gpu.format,
                width,
                height,
            );
            if self.shadow.is_none() {
                log::warn!("[renderer] shadow pipeline unavailable after resize");
            }
            // Rebuild the light pass against the new wall_mask view (bind
            // group holds the old view otherwise, which points at a texture
            // that's been dropped).
            self.lights_pass = self.shadow.as_ref().and_then(|sh| LightPass::new(
                &self.gpu.device,
                &sh.wall_mask_view,
                self.gpu.format,
                width,
                height,
            ));
            if self.lights_pass.is_none() {
                log::warn!("[renderer] light pipeline unavailable after resize");
            }
            // Rebuild the sun pass against the new wall_mask view too.
            self.sun_pass = self.shadow.as_ref().and_then(|sh| SunPass::new(
                &self.gpu.device,
                &sh.wall_mask_view,
                self.gpu.format,
                width,
                height,
            ));
            if self.sun_pass.is_none() {
                log::warn!("[renderer] sun pipeline unavailable after resize");
            }
            self.godray_pass = self.shadow.as_ref().and_then(|sh| GodrayPass::new(
                &self.gpu.device,
                &sh.wall_mask_view,
                self.gpu.format,
                width,
                height,
            ));
            if self.godray_pass.is_none() {
                log::warn!("[renderer] godray pipeline unavailable after resize");
            }
        }
    }

    /// Returns a batch the caller can populate with geometry that should be
    /// rendered into the offscreen blur target instead of the main pass.
    /// Cleared at the start of every frame (`begin_frame`); the recorded
    /// commands are consumed by `blur_and_composite` inside `end_frame`.
    pub fn begin_offscreen_batch(&mut self) -> &mut Batch {
        &mut self.offscreen_batch
    }

    /// Marks the current offscreen batch for blur + composite into the main
    /// pass. Runs H+V Gaussian at the given pixel radius. Cheap no-op when
    /// the postprocess pipeline isn't available (see `postprocess` field).
    /// Returns a batch the caller populates with white rects at every
    /// shadow caster (solid tile centres + character positions) for the
    /// current frame. Cleared each `begin_frame`; consumed by
    /// `raycast_and_composite_shadows`.
    pub fn begin_wall_mask_batch(&mut self) -> &mut Batch {
        &mut self.mask_batch
    }

    /// Records intent to run the shadow raycast pass. Splits the main
    /// batch at the current index — geometry written before this call
    /// gets multiply-darkened by the shadow; geometry written after
    /// (character sprites already drawn should typically go before,
    /// but this is the caller's choice) is drawn on top of the shadow.
    /// `sun_dir` is the screen-space direction shadows fall; the shader
    /// marches the opposite direction to find occluders.
    pub fn raycast_and_composite_shadows(&mut self, sun_dir: Vec2, shadow_length_px: f32) {
        if self.shadow.is_none() { return; }
        self.pending_shadow = Some((sun_dir, shadow_length_px.max(0.0)));
        self.shadow_split_index = Some(self.batch.indices.len() as u32);
    }

    /// Queue a point light emitter for this frame's light-map. `pos_px` is
    /// the light centre in surface pixel space with the top-left origin
    /// (matches wall_mask uv). Silently drops lights past
    /// `MAX_LIGHTS_PER_FRAME` — the on-foot lattice + windows fits well
    /// inside this budget in normal play. `color` is linear RGB; `radius_px`
    /// is the falloff cutoff; `intensity` scales the peak contribution.
    pub fn push_light(
        &mut self,
        pos_px: Vec2,
        color: [f32; 3],
        radius_px: f32,
        intensity: f32,
    ) {
        let Some(lp) = self.lights_pass.as_ref() else { return; };
        if self.pending_lights.len() >= MAX_LIGHTS_PER_FRAME { return; }
        self.pending_lights.push(lp.make_uniform(
            pos_px.into(), color, radius_px.max(0.0), intensity.max(0.0),
        ));
    }

    /// Queue a rectangle (directional beam) light emitter. `pos_px` is the
    /// beam origin (top-left origin, same as `push_light`); `dir` is the
    /// facing direction the beam shines in (need not be unit). `length_px`
    /// is the along-axis reach where the linear falloff hits zero;
    /// `half_width_px` is the perpendicular half-width (hard edges).
    /// Windows use this to stream light in a fixed direction rather than
    /// pooling radially.
    pub fn push_rect_light(
        &mut self,
        pos_px: Vec2,
        dir: Vec2,
        length_px: f32,
        half_width_px: f32,
        color: [f32; 3],
        intensity: f32,
    ) {
        let Some(lp) = self.lights_pass.as_ref() else { return; };
        if self.pending_lights.len() >= MAX_LIGHTS_PER_FRAME { return; }
        self.pending_lights.push(lp.make_rect_uniform(
            pos_px.into(),
            dir.into(),
            length_px.max(0.0),
            half_width_px.max(0.0),
            color,
            intensity.max(0.0),
        ));
    }

    /// Split the main batch here and slot the light-map composite in at
    /// this point in `end_frame`. Everything drawn to the main batch
    /// *before* this call gets multiplied by the light-map (dark base +
    /// per-light additive contribution); anything drawn *after* (HUD, UI)
    /// paints on top of the lit scene at full brightness.
    ///
    /// Cheap no-op when the light pipeline isn't available (see
    /// `lights_pass` field).
    pub fn render_lights_and_composite(&mut self) {
        if self.lights_pass.is_none() { return; }
        self.lights_pending = true;
        self.lights_split_index = Some(self.batch.indices.len() as u32);
    }

    /// Queue the global directional-sun pass for this frame. `sun_dir_screen`
    /// is the direction TOWARD the sun in screen pixel space (y grows
    /// downward to match wall_mask uv). `intensity` scales the sun colour;
    /// `color` is a linear-RGB tint (typical daylight ~[1.0, 0.95, 0.85]).
    ///
    /// Splits the main batch here — geometry drawn *before* this call is
    /// additively lit by the sun; geometry drawn *after* (HUD, UI) paints
    /// on top of the sunlit scene at full brightness.
    ///
    /// Cheap no-op when the sun pipeline isn't available (see `sun_pass`).
    pub fn queue_sun_pass(&mut self, sun_dir_screen: Vec2, intensity: f32, color: [f32; 3], tile_px: f32, time: f32) {
        if self.sun_pass.is_none() { return; }
        self.pending_sun = Some((sun_dir_screen, intensity.max(0.0), color, tile_px.max(1.0), time));
        self.sun_split_index = Some(self.batch.indices.len() as u32);
    }

    /// Queue the offscreen god-ray pass. Mirrors `queue_sun_pass` — same
    /// inputs; the pass reads the same wall_mask, seeds at lit windows,
    /// marches away from the sun into interior space, and composites the
    /// resulting soft shafts additively on top of the main frame.
    ///
    /// Runs INDEPENDENTLY of the sun pass at its own split index, so the
    /// caller can order it after `queue_sun_pass` for beams to paint on
    /// top of the direct sun contribution.
    ///
    /// Cheap no-op when the godray pipeline isn't available.
    pub fn queue_godray_pass(&mut self, sun_dir_screen: Vec2, intensity: f32, color: [f32; 3], tile_px: f32, time: f32) {
        if self.godray_pass.is_none() { return; }
        self.pending_godray = Some((sun_dir_screen, intensity.max(0.0), color, tile_px.max(1.0), time));
        self.godray_split_index = Some(self.batch.indices.len() as u32);
    }

    pub fn blur_and_composite(&mut self, radius: f32) {
        if self.postprocess.is_none() { return; }
        self.pending_blur_radius = Some(radius.max(0.0));
        // Anything already written to the main batch (backdrop, starfield…)
        // paints under the composite; anything written after paints on top.
        self.composite_split_index = Some(self.batch.indices.len() as u32);
    }

    pub fn set_vsync(&mut self, enabled: bool) {
        self.gpu.set_vsync(enabled);
    }

    pub fn window(&self) -> &std::sync::Arc<Window> {
        &self.window
    }


    pub fn screen_to_world(&self, screen_pos: Vec2) -> Vec2 {
        self.camera.screen_to_world(screen_pos)
    }

    pub fn viewport_size(&self) -> Vec2 {
        self.camera.viewport_size
    }
}
