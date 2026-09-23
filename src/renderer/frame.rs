//! Per-frame recording: `begin_frame` (batch reset + camera shake decay)
//! and `end_frame` (the encoder pass that uploads batches, runs the
//! shadow / lights / sun / godray / blur composites in split-index order,
//! and presents).
//!
//! Split out of `renderer/mod.rs` (was 1351 lines). `end_frame` alone is
//! ~590 lines of pass sequencing; isolating it keeps `mod.rs` to the struct
//! definition and the small `queue_*` / accessor surface that callers
//! actually read.
//!
//! This is a child module of `renderer`, so it reaches `Renderer`'s private
//! fields directly — no visibility widening was needed for the split.

use super::*;

/// Grow the given vertex/index buffers to fit `batch.vertices` /
/// `batch.indices` (doubling on overflow), then upload the batch's
/// contents. Shared by the main + offscreen batch upload paths so a bug
/// fix or perf tweak lands in one place.
// Takes the vertex and index buffer/capacity/label triples as separate
// `&mut` arguments precisely so one function can serve the main, offscreen
// and mask paths. Grouping them would need a struct holding three `&mut`
// borrows into `Renderer`, which the borrow checker rejects at the call
// sites this exists to share.
#[allow(clippy::too_many_arguments)]
fn upload_batch(
    device: &wgpu::Device,
    queue:  &wgpu::Queue,
    batch:  &Batch,
    vbuf: &mut wgpu::Buffer, vcap: &mut usize, vlabel: &str,
    ibuf: &mut wgpu::Buffer, icap: &mut usize, ilabel: &str,
) {
    // Hard upper bound on batch size, set against the wgpu default
    // `max_buffer_size` limit of 256MB. Above it, buffer creation fails
    // and the whole client panics with "wgpu errors as fatal" (observed
    // under heavy scene stress when a burst of vertex-heavy geometry
    // briefly hit 3M+ verts and doubled to a ~6M-vert allocation).
    //
    // `Vertex` is 84 bytes (9 attributes — see `batch.rs`), NOT the 32
    // this comment claimed until the numbers were actually checked. So
    // the real ceiling is 256MB / 84B ≈ 3.19M verts, and the 8M cap
    // below sat *above* the limit it was written to respect: a batch
    // between 3.2M and 8M verts passed the guard and then panicked in
    // buffer creation, which is the exact failure the cap exists to
    // prevent. `MAX_VERTS` is now derived from the stride instead of
    // hardcoded, so it cannot drift out of step with `Vertex` again.
    //
    // First cut of this fix set the cap at 1M verts, which was
    // *too aggressive* — truncating verts while leaving indices
    // intact meant indices pointed past the sliced-off tail and
    // draws produced garbage (visible symptom: geometry vanished
    // entirely on any overflow). Truncating indices in step with
    // verts is only safe when we also find the matched vertex
    // boundary; simpler + correct: keep the cap generous enough
    // that real scenes never trip it. 8M verts covers thousands of
    // dynamic entities with their full geometry + labels.
    // 256MB is wgpu's default `max_buffer_size`. The growth step below is
    // clamped to these caps too, so the cap can be the true limit rather
    // than a fraction of it.
    const MAX_BUFFER_BYTES: usize = 256 * 1024 * 1024;
    const MAX_VERTS: usize = MAX_BUFFER_BYTES / std::mem::size_of::<Vertex>();
    // Indices are u32, and the 3:1 ratio matches the triangle-list
    // topology every primitive in `batch.rs` emits (6 indices per 4-vert
    // quad, 3 per triangle) — keep them in step so a truncating batch
    // never leaves indices pointing past the sliced-off vertex tail.
    const MAX_INDICES: usize = MAX_VERTS * 3;
    let vlen_raw = batch.vertices.len();
    let ilen_raw = batch.indices.len();
    let vlen = vlen_raw.min(MAX_VERTS);
    let ilen = ilen_raw.min(MAX_INDICES);
    if vlen_raw > MAX_VERTS || ilen_raw > MAX_INDICES {
        // Warn once per frame — deliberately not rate-limited by tick
        // here (this is the renderer, no tick counter); env_logger
        // dedupe + downstream logs handle spam.
        log::warn!(
            "[renderer] batch overflow — capping verts {vlen_raw} → {MAX_VERTS}, indices {ilen_raw} → {MAX_INDICES} (labels: {vlabel}/{ilabel})"
        );
    }
    if vlen > *vcap {
        // Clamp the doubling, not just the batch: `vlen * 2` on a batch near
        // MAX_VERTS is what actually reaches `create_buffer`, so leaving it
        // unclamped reintroduces the same over-limit allocation the cap
        // exists to prevent.
        *vcap = (vlen * 2).min(MAX_VERTS);
        *vbuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(vlabel),
            size: (*vcap * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }
    if ilen > *icap {
        *icap = (ilen * 2).min(MAX_INDICES);
        *ibuf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(ilabel),
            size: (*icap * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    }
    if vlen > 0 {
        queue.write_buffer(vbuf, 0, bytemuck::cast_slice(&batch.vertices[..vlen]));
        queue.write_buffer(ibuf, 0, bytemuck::cast_slice(&batch.indices[..ilen]));
    }
}

impl Renderer {
    pub fn begin_frame(&mut self) {
        self.batch.clear();
        self.offscreen_batch.clear();
        self.mask_batch.clear();
        self.pending_blur_radius = None;
        self.composite_split_index = None;
        self.pending_shadow = None;
        self.shadow_split_index = None;
        self.pending_lights.clear();
        self.lights_split_index = None;
        self.lights_pending = false;
        self.pending_sun = None;
        self.sun_split_index = None;
        self.pending_godray = None;
        self.godray_split_index = None;
        // The draw list is per-frame like every other pending list above.
        // Retained meshes themselves are NOT cleared — that is the whole
        // point of the store. Transients are, which is what keeps
        // `draw_mesh_transient`'s upload-per-call from leaking.
        #[cfg(feature = "render3d")]
        {
            self.pending_draws.clear();
            self.point_lights_3d.clear();
            for h in std::mem::take(&mut self.transient_handles) {
                self.meshes.remove(h);
            }
        }
        self.shake_trauma = (self.shake_trauma - 0.02).max(0.0);
        let shake_amount = self.shake_trauma * self.shake_trauma * 8.0;
        self.shake_offset = Vec2::new(
            (self.shake_trauma * 13.7).sin() * shake_amount,
            (self.shake_trauma * 17.3).cos() * shake_amount,
        );
    }

    pub fn end_frame(&mut self) {
        // `Lost`/`Outdated` are recoverable and routine — a device reset, a
        // driver update, or alt-tabbing out of exclusive fullscreen all
        // produce one. They mean "reconfigure and ask again", so do exactly
        // that once. Swallowing them (the old `Err(_) => return`) left the
        // window black permanently and silently, because nothing else in the
        // engine ever reconfigures the surface outside `resize`.
        let surface_tex = match self.gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let (w, h) = (self.gpu.surface_config.width, self.gpu.surface_config.height);
                log::warn!("surface lost/outdated; reconfiguring at {w}x{h}");
                self.gpu.surface.configure(&self.gpu.device, &self.gpu.surface_config);
                match self.gpu.surface.get_current_texture() {
                    Ok(t) => t,
                    // Still bad after a reconfigure: skip this frame rather
                    // than spin. The next frame retries from the top.
                    Err(e) => {
                        log::error!("surface still unavailable after reconfigure: {e}");
                        return;
                    }
                }
            }
            // OOM is unrecoverable; Timeout resolves on its own next frame.
            // Both are worth a line in the log — the old code produced none.
            Err(e) => {
                log::error!("surface acquire failed: {e}");
                return;
            }
        };
        let view = surface_tex.texture.create_view(&Default::default());

        let cam_with_shake = Camera2D {
            position: self.camera.position + self.shake_offset,
            zoom: self.camera.zoom,
            viewport_size: self.camera.viewport_size,
        };
        let uniform = cam_with_shake.build_uniform();
        // The 2D camera always owns this buffer. It used to be shared,
        // and a 3D camera overwrote it for the whole frame — which is
        // what made a frame 2D *or* 3D. The 3D camera now has its own
        // below, so a HUD can sit over a scene.
        self.gpu
            .queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));

        // The 3D camera, into its own buffer. Written even when unset so
        // the buffer never holds a matrix from an earlier frame: a game
        // that clears its 3D camera mid-session would otherwise draw the
        // next 3D geometry through a stale view.
        #[cfg(feature = "render3d")]
        {
            let u3d = match self.camera_3d.as_ref() {
                Some(c) => c.build_uniform(),
                None => uniform,
            };
            self.gpu
                .queue
                .write_buffer(&self.camera_3d_buffer, 0, bytemuck::bytes_of(&u3d));
        }

        // Fill the per-instance ring, grouped by mesh, and record the
        // order so the draw loop can walk the same slots. The grouping
        // happens exactly once: `HashMap` iteration order is not stable
        // across two traversals, so regrouping at draw time would pair
        // instances with the wrong transforms.
        #[cfg(feature = "render3d")]
        let mesh_order: Vec<(super::mesh_store::MeshHandle, u32)> = {
            let mut order = Vec::new();
            if let Some(r3d) = self.render3d.as_ref() {
                let grouped = super::mesh_store::group_by_mesh(&self.pending_draws);
                let mut slot = 0u64;
                for (handle, instances) in grouped {
                    order.push((handle, instances.len() as u32));
                    for d in instances {
                        let u = super::render3d::InstanceUniform {
                            model: d.model.to_cols_array_2d(),
                            tint: d.color,
                        };
                        self.gpu.queue.write_buffer(
                            &r3d.instance_buffer,
                            slot * r3d.instance_stride,
                            bytemuck::bytes_of(&u),
                        );
                        slot += 1;
                    }
                }
            }
            order
        };

        // Upload the 3D lighting: the sun's matrix (shared by the shadow
        // pass and the main pass's lookup) and the point-light block.
        #[cfg(feature = "render3d")]
        {
            if let (Some(r3d), Some(sh)) = (self.render3d.as_ref(), self.shadow3d.as_ref()) {
                // The shadow frustum is centred on the camera, in the
                // camera-relative frame everything else uses — so the
                // origin is where the eye is.
                let (dir, radius) = self.sun_3d.unwrap_or((glam::Vec3::Z, 1.0));
                let m = super::shadow3d::light_view_proj(dir, glam::Vec3::ZERO, radius);

                let cam = super::camera::CameraUniform {
                    view_proj: m.to_cols_array_2d(),
                };
                self.gpu.queue.write_buffer(
                    &sh.light_camera_buffer,
                    0,
                    bytemuck::bytes_of(&cam),
                );

                let d = dir.normalize_or_zero();
                let u = super::shadow3d::ShadowUniform {
                    light_view_proj: m.to_cols_array_2d(),
                    light_dir_ambient: [d.x, d.y, d.z, self.ambient_3d],
                };
                self.gpu
                    .queue
                    .write_buffer(&sh.uniform_buffer, 0, bytemuck::bytes_of(&u));

                let mut lights = super::render3d::PointLightsUniform::default();
                lights.count[0] = self.point_lights_3d.len() as u32;
                for (i, l) in self.point_lights_3d.iter().enumerate() {
                    lights.lights[i] = super::render3d::PointLightGpu {
                        pos_radius: [l.position.x, l.position.y, l.position.z, l.radius],
                        color_intensity: [l.color[0], l.color[1], l.color[2], l.intensity],
                    };
                }
                self.gpu.queue.write_buffer(
                    &r3d.lights_buffer,
                    0,
                    bytemuck::bytes_of(&lights),
                );

            }
        }

        // Grow + upload the main batch's vertex/index buffers. Same pattern
        // used for the offscreen batch below.
        upload_batch(
            &self.gpu.device, &self.gpu.queue, &self.batch,
            &mut self.vertex_buffer, &mut self.vertex_capacity, "vbuf",
            &mut self.index_buffer,  &mut self.index_capacity,  "ibuf",
        );

        // Offscreen batch (for the blur pipeline). Only runs when the
        // caller actually queued a blur pass this frame.
        let run_blur = self.pending_blur_radius.is_some()
            && !self.offscreen_batch.indices.is_empty()
            && self.postprocess.is_some();
        if run_blur {
            upload_batch(
                &self.gpu.device, &self.gpu.queue, &self.offscreen_batch,
                &mut self.offscreen_vbuf, &mut self.offscreen_vcap, "offscreen_vbuf",
                &mut self.offscreen_ibuf, &mut self.offscreen_icap, "offscreen_ibuf",
            );
        }

        // Shadow batch. Runs when the caller queued a shadow pass and
        // populated the mask batch. Empty mask → no walls → nothing to
        // shadow, so skip. Also skipped when the shadow pipeline itself
        // failed to allocate (see `shadow` field).
        let run_shadow = self.pending_shadow.is_some()
            && !self.mask_batch.indices.is_empty()
            && self.shadow.is_some();
        // Light pass runs whenever the caller opted in via
        // `render_lights_and_composite`, even when no lights were pushed —
        // the ambient clear alone squashes the scene toward dark, which is
        // the whole point of the "everything dark by default" model.
        let run_lights = self.lights_pending && self.lights_pass.is_some();
        let run_sun = self.pending_sun.is_some() && self.sun_pass.is_some();
        let run_godray = self.pending_godray.is_some() && self.godray_pass.is_some();
        // The wall_mask texture is shared with the shadow / light / sun /
        // godray pipelines. If the shadow pass isn't running we still need
        // to render the mask so downstream consumers can occlude against it.
        // Even with no fresh stamps, we still need to CLEAR the mask each
        // frame — otherwise last frame's walls persist and downstream
        // passes read stale occluders. Drop the `mask_batch.is_empty`
        // guard so we always run the clear.
        let need_mask_only_for_lights =
            (run_lights || run_sun || run_godray) && !run_shadow
                && self.shadow.is_some();
        if run_shadow || need_mask_only_for_lights {
            upload_batch(
                &self.gpu.device, &self.gpu.queue, &self.mask_batch,
                &mut self.mask_vbuf, &mut self.mask_vcap, "mask_vbuf",
                &mut self.mask_ibuf, &mut self.mask_icap, "mask_ibuf",
            );
        }
        // Upload the queued light uniforms into their ring slots. Empty
        // vec is fine — the ambient clear still runs so the scene darkens.
        if run_lights {
            let lp = self.lights_pass.as_ref().unwrap();
            for (i, u) in self.pending_lights.iter().enumerate() {
                let off = i as u64 * lp.slot_stride as u64;
                self.gpu.queue.write_buffer(&lp.light_uniforms, off, bytemuck::bytes_of(u));
            }
        }

        // Staging buffer for screenshot (allocated before encoder so it outlives the copy cmd)
        let w = self.gpu.surface_config.width;
        let h = self.gpu.surface_config.height;
        let screenshot_staging = self.screenshot_pending.then(|| {
            let bpr = (w * 4).div_ceil(256) * 256;  // align to 256 bytes
            let buf = self.gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (bpr * h) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            (buf, bpr)
        });

        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&Default::default());

        // Offscreen sub-scene → H blur → V blur. Structured as separate
        // encoder passes (rather than folded into the main pass) so the
        // fragment shader can sample the previous target — a render pass
        // can't read and write the same attachment. Same shape lets future
        // effects (bloom, fog-of-war) piggy-back on the ping-pong pair.
        if run_blur {
            let pp = self.postprocess.as_ref().unwrap();
            let radius = self.pending_blur_radius.unwrap_or(0.0);
            pp.write_uniforms(&self.gpu.queue, radius);

            // 1. Offscreen batch → blur_a. Clear transparent so tiles the
            //    caller didn't paint stay 0 alpha and don't leak into the
            //    composite as dark halos.
            {
                let mut off = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("offscreen_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &pp.blur_a_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                off.set_pipeline(&pp.offscreen_pipeline);
                off.set_bind_group(0, &self.camera_bind_group, &[]);
                off.set_bind_group(1, &self.white_texture_bind_group, &[]);
                off.set_vertex_buffer(0, self.offscreen_vbuf.slice(..));
                off.set_index_buffer(self.offscreen_ibuf.slice(..), wgpu::IndexFormat::Uint32);
                off.draw_indexed(0..self.offscreen_batch.indices.len() as u32, 0, 0..1);
            }

            // 2. Horizontal blur: sample blur_a, write blur_b.
            {
                let mut hp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("blur_h_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &pp.blur_b_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                hp.set_pipeline(&pp.blur_pipeline);
                hp.set_bind_group(0, &pp.bg_sample_a_h, &[]);
                hp.draw(0..3, 0..1);
            }

            // 3. Vertical blur: sample blur_b, write back to blur_a. The
            //    final blurred image lives in blur_a for the composite step.
            {
                let mut vp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("blur_v_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &pp.blur_a_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                vp.set_pipeline(&pp.blur_pipeline);
                vp.set_bind_group(0, &pp.bg_sample_b_v, &[]);
                vp.draw(0..3, 0..1);
            }
        }

        // Wall mask population — needed by shadow, lights, or both. Cleared
        // to black = "no wall" so pixels outside authored geometry never
        // occlude a light.
        if run_shadow || need_mask_only_for_lights {
            let sh = self.shadow.as_ref().unwrap();
            // Mask pass uses its own camera uniform — enlarged viewport
            // that shrinks the on-screen viewport to the central portion
            // of the wall_mask texture, so off-screen walls still land
            // inside the mask. See `ShadowPass::write_mask_camera` +
            // `MASK_SCALE`.
            sh.write_mask_camera(
                &self.gpu.queue,
                self.camera.position + self.shake_offset,
                self.camera.zoom,
                self.camera.viewport_size,
            );
            let mut mp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("wall_mask_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &sh.wall_mask_view,
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
            mp.set_pipeline(&sh.mask_pipeline);
            mp.set_bind_group(0, &sh.mask_camera_bind_group, &[]);
            mp.set_bind_group(1, &self.white_texture_bind_group, &[]);
            mp.set_vertex_buffer(0, self.mask_vbuf.slice(..));
            mp.set_index_buffer(self.mask_ibuf.slice(..), wgpu::IndexFormat::Uint32);
            mp.draw_indexed(0..self.mask_batch.indices.len() as u32, 0, 0..1);
        }

        // Shadow raycast pipeline: wall_mask → shadow_map. Encoded before
        // the main pass so `shadow_map` is ready when the composite step
        // multiplies it onto the frame. (Currently unused by the on-foot
        // view — the light pipeline supersedes it — but kept live for
        // future callers.)
        if run_shadow {
            let sh = self.shadow.as_ref().unwrap();
            let (sun_dir, len_px) = self.pending_shadow.unwrap();
            sh.write_uniforms(&self.gpu.queue, sun_dir, len_px);
            let mut sp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow_raycast_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &sh.shadow_map_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            sp.set_pipeline(&sh.raycast_pipeline);
            sp.set_bind_group(0, &sh.bg_raycast, &[]);
            sp.draw(0..3, 0..1);
        }

        // Light pipeline: clear light_map to ambient, then run one
        // additive fullscreen pass per queued light. Each pass reads
        // wall_mask for occlusion and blends its wall-occluded radial
        // contribution into the accumulator.
        if run_lights {
            let lp = self.lights_pass.as_ref().unwrap();
            let lights = self.pending_lights.len();
            // First pass clears; subsequent passes load. If there are zero
            // lights we still emit a clear-only pass so the composite has
            // a valid (ambient-only) texture to sample.
            {
                let mut lp0 = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("light_map_clear"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &lp.light_map_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(self.ambient),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                // Drawing the first light in the clear pass avoids a
                // separate load-op pass with the same bind group.
                if lights > 0 {
                    lp0.set_pipeline(&lp.light_pipeline);
                    lp0.set_bind_group(0, &lp.bg_light, &[0]);
                    lp0.draw(0..3, 0..1);
                }
            }
            for i in 1..lights {
                let off = i as u32 * lp.slot_stride;
                let mut lpi = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("light_add_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &lp.light_map_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                lpi.set_pipeline(&lp.light_pipeline);
                lpi.set_bind_group(0, &lp.bg_light, &[off]);
                lpi.draw(0..3, 0..1);
            }
        }

        // Sun pipeline: wall_mask → sun_map. Fullscreen march toward the
        // sun, writes per-pixel sun contribution. Composite additively
        // blends into the main pass at the sun split index.
        if run_sun {
            let sp = self.sun_pass.as_ref().unwrap();
            let (sun_dir_screen, intensity, color, tile_px, time) = self.pending_sun.unwrap();
            sp.write_uniforms(&self.gpu.queue, sun_dir_screen, intensity, color, tile_px, time);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("sun_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &sp.sun_map_view,
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
            pass.set_pipeline(&sp.sun_pipeline);
            pass.set_bind_group(0, &sp.bg_sun, &[]);
            pass.draw(0..3, 0..1);
        }

        // Godray pipeline: seed lit windows → radial march away from sun →
        // beam_map. Composited additively into the main pass at the godray
        // split index. Two sub-passes because a render pass can't sample
        // an attachment it writes.
        if run_godray {
            let gp = self.godray_pass.as_ref().unwrap();
            let (sun_dir_screen, intensity, color, tile_px, time) = self.pending_godray.unwrap();
            gp.write_uniforms(&self.gpu.queue, sun_dir_screen, intensity, color, tile_px, time);
            // 1. Seed: wall_mask → seed_map. Clear black so non-window
            //    pixels contribute nothing to the march.
            {
                let mut sp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("godray_seed_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &gp.seed_view,
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
                sp.set_pipeline(&gp.seed_pipeline);
                sp.set_bind_group(0, &gp.bg_seed, &[]);
                sp.draw(0..3, 0..1);
            }
            // 2. March: seed_map + wall_mask → beam_map.
            {
                let mut mp = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("godray_march_pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &gp.beam_view,
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
                mp.set_pipeline(&gp.march_pipeline);
                mp.set_bind_group(0, &gp.bg_march, &[]);
                mp.draw(0..3, 0..1);
            }
        }

        // Shadow pass: the same geometry from the light's point of view,
        // depth only. Must precede the main pass, which samples what this
        // writes. Skipped entirely when no sun is set or nothing is
        // queued — an empty shadow map reads as "everything lit", which
        // is the right answer for a scene with no caster.
        #[cfg(feature = "render3d")]
        if let (Some(r3d), Some(sh)) = (self.render3d.as_ref(), self.shadow3d.as_ref()) {
            if self.sun_3d.is_some() && !self.pending_draws.is_empty() {
                let mut spass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("shadow3d_pass"),
                    // No colour target: the pipeline has no fragment
                    // stage, so there is nothing to write but depth.
                    color_attachments: &[],
                    depth_stencil_attachment: Some(sh.attachment()),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                spass.set_pipeline(&sh.pipeline);
                // The light's matrix stands in for the camera's, through
                // the same bind group layout.
                spass.set_bind_group(0, &sh.light_camera_bg, &[]);

                let mut slot = 0u32;
                for (handle, count) in &mesh_order {
                    let Some(mesh) = self.meshes.get(*handle) else {
                        slot += *count;
                        continue;
                    };
                    spass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                    spass.set_index_buffer(
                        mesh.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    for _ in 0..*count {
                        let offset = slot as u64 * r3d.instance_stride;
                        spass.set_bind_group(
                            1,
                            &r3d.instance_bg,
                            &[offset as wgpu::DynamicOffset],
                        );
                        spass.draw_indexed(0..mesh.index_count, 0, 0..1);
                        slot += 1;
                    }
                }
            }
        }

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("main_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.02,
                            g: 0.02,
                            b: 0.05,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                // Present whenever the depth texture allocated. The 2D
                // pipeline's depth state is the identity, so this changes
                // no 2D pixel; under `render3d` it is what the 3D pipeline
                // (`Less`, writes on) actually tests against. `None` on a
                // zero-size surface, matching the pipeline's tolerance for
                // a missing buffer.
                depth_stencil_attachment: self.depth.as_ref().map(|d| d.attachment()),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            // 3D geometry first: it is opaque and depth-tested, so laying
            // it down before the 2D batch lets the alpha-blended 2D
            // primitives composite over a finished scene. Drawing it after
            // would blend 3D over 2D in whatever order the meshes happened
            // to be queued.
            #[cfg(feature = "render3d")]
            if let Some(r3d) = self.render3d.as_ref() {
                if !self.pending_draws.is_empty() {
                    rpass.set_pipeline(&r3d.pipeline);
                    // The *3D* camera, not the shared one. The 2D batch
                    // drawn later in this same pass binds its own, which
                    // is what lets a HUD composite over the scene.
                    rpass.set_bind_group(0, &self.camera_3d_bind_group, &[]);
                    // Shadow map at 2, point lights at 3. Both are
                    // frame-constant, so they bind once outside the loop.
                    if let Some(sh) = self.shadow3d.as_ref() {
                        rpass.set_bind_group(2, &sh.sample_bg, &[]);
                    }
                    rpass.set_bind_group(3, &r3d.lights_bg, &[]);

                    // `mesh_order` was produced by the same grouping that
                    // filled the instance ring above, so walking it here
                    // keeps slot N holding instance N. Regrouping instead
                    // of reusing it would be a bug: `HashMap` iteration
                    // order is not stable between two traversals.
                    let mut slot = 0u32;
                    for (handle, count) in &mesh_order {
                        let Some(mesh) = self.meshes.get(*handle) else {
                            // A stale handle draws nothing. Still advance
                            // the ring, or every later instance would read
                            // the wrong slot.
                            slot += *count;
                            continue;
                        };
                        rpass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                        rpass.set_index_buffer(
                            mesh.index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        for _ in 0..*count {
                            let offset = slot as u64 * r3d.instance_stride;
                            rpass.set_bind_group(
                                1,
                                &r3d.instance_bg,
                                &[offset as wgpu::DynamicOffset],
                            );
                            rpass.draw_indexed(0..mesh.index_count, 0, 0..1);
                            slot += 1;
                        }
                    }
                }
            }

            // The main batch is drawn in ranges separated by composite
            // steps. Each split names an index in the main batch AND a
            // composite pipeline to run at that boundary. Splits are
            // sorted so out-of-order caller invocations still render in
            // batch-index order.
            //
            // `Composite::Blur`   → postprocess blur_a composited over.
            // `Composite::Shadow` → shadow_map multiply-blended over.
            #[derive(Copy, Clone)]
            enum Composite { Blur, Shadow, Lights, Sun, Godray }
            let mut splits: Vec<(u32, Composite)> = Vec::with_capacity(5);
            if run_blur {
                if let Some(i) = self.composite_split_index {
                    splits.push((i, Composite::Blur));
                }
            }
            if run_shadow {
                if let Some(i) = self.shadow_split_index {
                    splits.push((i, Composite::Shadow));
                }
            }
            if run_lights {
                if let Some(i) = self.lights_split_index {
                    splits.push((i, Composite::Lights));
                }
            }
            if run_sun {
                if let Some(i) = self.sun_split_index {
                    splits.push((i, Composite::Sun));
                }
            }
            if run_godray {
                if let Some(i) = self.godray_split_index {
                    splits.push((i, Composite::Godray));
                }
            }
            splits.sort_by_key(|(i, _)| *i);

            // Cap the draw range to the same cap `upload_batch` applies,
            // so a draw can never name indices past what was actually
            // written to the GPU buffer.
            //
            // **Derived, not a literal.** This read `24_000_000` until
            // 2026-09-11, while `upload_batch` truncates at
            // `MAX_VERTS * 3` — which with an 84-byte `Vertex` is
            // 9,586,980. The two disagreed by 14.4M indices, so a frame
            // in that window would draw from buffer contents that were
            // never uploaded: exactly the "indices point past the
            // sliced-off tail, geometry turns to garbage" failure the
            // comment at the top of `upload_batch` records as already
            // having happened once.
            //
            // That earlier fix made `MAX_VERTS` stride-derived
            // specifically so it "cannot drift out of step with `Vertex`
            // again", and missed this copy. Third time the stride has
            // been assumed rather than asked for — so this asks.
            const MAX_BUFFER_BYTES: usize = 256 * 1024 * 1024;
            const MAX_INDICES_DRAW: usize =
                (MAX_BUFFER_BYTES / std::mem::size_of::<Vertex>()) * 3;
            let total = (self.batch.indices.len().min(MAX_INDICES_DRAW)) as u32;
            let has_main = !self.batch.indices.is_empty();

            if has_main {
                rpass.set_pipeline(&self.pipeline);
                rpass.set_bind_group(0, &self.camera_bind_group, &[]);
                rpass.set_bind_group(1, &self.white_texture_bind_group, &[]);
                rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                rpass.set_index_buffer(
                    self.index_buffer.slice(..),
                    wgpu::IndexFormat::Uint32,
                );
                let mut cursor: u32 = 0;
                for (mid_raw, kind) in &splits {
                    let mid = (*mid_raw).min(total);
                    if mid > cursor {
                        rpass.draw_indexed(cursor..mid, 0, 0..1);
                    }
                    match kind {
                        Composite::Blur => {
                            let pp = self.postprocess.as_ref().unwrap();
                            rpass.set_pipeline(&pp.composite_pipeline);
                            rpass.set_bind_group(0, &pp.bg_composite, &[]);
                            rpass.draw(0..3, 0..1);
                        }
                        Composite::Shadow => {
                            let sh = self.shadow.as_ref().unwrap();
                            rpass.set_pipeline(&sh.composite_pipeline);
                            rpass.set_bind_group(0, &sh.bg_composite, &[]);
                            rpass.draw(0..3, 0..1);
                        }
                        Composite::Lights => {
                            let lp = self.lights_pass.as_ref().unwrap();
                            rpass.set_pipeline(&lp.composite_pipeline);
                            // Composite bg still binds a dynamic-offset
                            // uniform — pass 0 (contents ignored by shader).
                            rpass.set_bind_group(0, &lp.bg_composite, &[0]);
                            rpass.draw(0..3, 0..1);
                        }
                        Composite::Sun => {
                            let sp = self.sun_pass.as_ref().unwrap();
                            rpass.set_pipeline(&sp.composite_pipeline);
                            rpass.set_bind_group(0, &sp.bg_composite, &[]);
                            rpass.draw(0..3, 0..1);
                        }
                        Composite::Godray => {
                            let gp = self.godray_pass.as_ref().unwrap();
                            rpass.set_pipeline(&gp.composite_pipeline);
                            rpass.set_bind_group(0, &gp.bg_composite, &[]);
                            rpass.draw(0..3, 0..1);
                        }
                    }
                    // Rebind the main pipeline for whatever comes next.
                    rpass.set_pipeline(&self.pipeline);
                    rpass.set_bind_group(0, &self.camera_bind_group, &[]);
                    rpass.set_bind_group(1, &self.white_texture_bind_group, &[]);
                    rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
                    rpass.set_index_buffer(
                        self.index_buffer.slice(..),
                        wgpu::IndexFormat::Uint32,
                    );
                    cursor = mid;
                }
                if cursor < total {
                    rpass.draw_indexed(cursor..total, 0, 0..1);
                }
            } else {
                // No main geometry — composites still fire so a full-
                // viewport blur/shadow layer at least paints something.
                for (_, kind) in &splits {
                    match kind {
                        Composite::Blur => {
                            let pp = self.postprocess.as_ref().unwrap();
                            rpass.set_pipeline(&pp.composite_pipeline);
                            rpass.set_bind_group(0, &pp.bg_composite, &[]);
                            rpass.draw(0..3, 0..1);
                        }
                        Composite::Shadow => {
                            let sh = self.shadow.as_ref().unwrap();
                            rpass.set_pipeline(&sh.composite_pipeline);
                            rpass.set_bind_group(0, &sh.bg_composite, &[]);
                            rpass.draw(0..3, 0..1);
                        }
                        Composite::Lights => {
                            let lp = self.lights_pass.as_ref().unwrap();
                            rpass.set_pipeline(&lp.composite_pipeline);
                            rpass.set_bind_group(0, &lp.bg_composite, &[0]);
                            rpass.draw(0..3, 0..1);
                        }
                        Composite::Sun => {
                            let sp = self.sun_pass.as_ref().unwrap();
                            rpass.set_pipeline(&sp.composite_pipeline);
                            rpass.set_bind_group(0, &sp.bg_composite, &[]);
                            rpass.draw(0..3, 0..1);
                        }
                        Composite::Godray => {
                            let gp = self.godray_pass.as_ref().unwrap();
                            rpass.set_pipeline(&gp.composite_pipeline);
                            rpass.set_bind_group(0, &gp.bg_composite, &[]);
                            rpass.draw(0..3, 0..1);
                        }
                    }
                }
            }
        }

        // Copy rendered frame to staging buffer for screenshot
        if let Some((ref staging, bpr)) = screenshot_staging {
            encoder.copy_texture_to_buffer(
                surface_tex.texture.as_image_copy(),
                wgpu::ImageCopyBuffer {
                    buffer: staging,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(bpr),
                        rows_per_image: None,
                    },
                },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
        }

        self.gpu.queue.submit(std::iter::once(encoder.finish()));
        surface_tex.present();

        // Read back screenshot data (blocks until GPU done — acceptable for a one-off capture)
        if let Some((staging, bpr)) = screenshot_staging {
            self.screenshot_pending = false;
            staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            self.gpu.device.poll(wgpu::Maintain::Wait);
            let is_bgra = matches!(self.gpu.format,
                wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb);
            let view = staging.slice(..).get_mapped_range();
            let raw: Vec<u8> = view.to_vec();
            drop(view);
            staging.unmap();
            // Strip row padding and convert BGRA→RGBA if needed
            let mut pixels = Vec::with_capacity((w * h * 4) as usize);
            for row in 0..h {
                let row_start = (row * bpr) as usize;
                let row_src = &raw[row_start..row_start + (w * 4) as usize];
                if is_bgra {
                    for px in row_src.as_chunks::<4>().0 {
                        pixels.extend_from_slice(&[px[2], px[1], px[0], px[3]]);
                    }
                } else {
                    pixels.extend_from_slice(row_src);
                }
            }
            self.screenshot_data = Some(ScreenshotData { pixels, width: w, height: h });
        }
    }
}
