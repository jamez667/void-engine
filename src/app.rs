//! The game-facing entry points.
//!
//! [`App`] is the simulation half and exists in every build. [`ClientApp`],
//! [`EngineCtx`] and [`run`] are the windowed half and need the `client`
//! feature; a dedicated server drives [`App`] with
//! [`crate::app_headless::run_headless`] instead.

#[cfg(feature = "client")]
use winit::application::ApplicationHandler;
#[cfg(feature = "client")]
use winit::event::{WindowEvent, DeviceEvent, DeviceId, MouseScrollDelta};
#[cfg(feature = "client")]
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
#[cfg(feature = "client")]
use winit::window::{Window, WindowId, WindowAttributes};
#[cfg(feature = "client")]
use winit::keyboard::PhysicalKey;
#[cfg(feature = "client")]
use glam::Vec2;
#[cfg(feature = "client")]
use std::sync::Arc;
#[cfg(feature = "client")]
use std::sync::mpsc;
#[cfg(feature = "client")]
use std::time::Instant;

#[cfg(feature = "client")]
use crate::renderer::Renderer;
use crate::input::InputState;
use crate::ecs::World;
#[cfg(feature = "client")]
use crate::time::Timestep;

// `PerfSnapshot` moved to `crate::perf` so `renderer` can store one without
// depending on `app`. Re-exported here because that is where consumers have
// always found it.
pub use crate::perf::PerfSnapshot;

/// What a fixed step gets. No renderer: this is the context a dedicated
/// server has, and therefore the context simulation code must be written
/// against if it is to run on one.
///
/// `dt` is this loop's step duration, read from its [`Timestep`] rather
/// than a global constant — a client passes 1/60 and a server 1/30, and
/// the same `fixed_update` body is correct under both.
pub struct SimCtx<'a> {
    pub world: &'a mut World,
    pub input: &'a InputState,
    pub dt: f32,
}

/// The simulation half of a game. Everything here runs identically on a
/// client and on a headless server, and none of it can touch the GPU.
///
/// A dedicated server implements only this and never links `wgpu`/`winit`
/// (see the `client` feature). A game that also draws implements
/// [`ClientApp`] on the same type, so the sim is written exactly once.
pub trait App: 'static {
    fn init(&mut self, ctx: &mut SimCtx);
    fn fixed_update(&mut self, ctx: &mut SimCtx);
    /// Checked before every fixed step. Return `false` to defer the tick
    /// instead of running it.
    ///
    /// Exists for deterministic lockstep, where a tick may not run until
    /// both peers' inputs for it are in hand — the game cannot simply
    /// advance with input it does not have. Returning `false` stops the
    /// catch-up loop and refunds the un-run steps to the timestep
    /// accumulator, so they are deferred to a later frame rather than lost.
    ///
    /// Defaulted to `true`, so single-player games never implement it and
    /// behave exactly as before.
    fn can_advance(&self) -> bool { true }
}

/// What a client fixed step gets: a [`SimCtx`] plus the renderer.
///
/// Exists so a client can reach the renderer from `init` (uploading static
/// geometry, sizing buffers) without that possibility leaking into [`App`],
/// which must stay compilable with no GPU. Deref to the sim context so the
/// familiar `ctx.world` / `ctx.input` / `ctx.dt` still work.
#[cfg(feature = "client")]
pub struct EngineCtx<'a, 'b> {
    pub sim: SimCtx<'a>,
    pub renderer: &'b mut Renderer,
}

#[cfg(feature = "client")]
impl<'a> std::ops::Deref for EngineCtx<'a, '_> {
    type Target = SimCtx<'a>;
    fn deref(&self) -> &Self::Target { &self.sim }
}

#[cfg(feature = "client")]
impl std::ops::DerefMut for EngineCtx<'_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.sim }
}

/// The drawing half of a game: everything that needs a window and a GPU.
///
/// Split from [`App`] rather than defaulted on it so that a server build
/// cannot accidentally depend on rendering, and so `render`'s signature can
/// name `Renderer` without that type having to exist in a headless build.
#[cfg(feature = "client")]
pub trait ClientApp: App {
    fn render(
        &mut self,
        renderer: &mut Renderer,
        world: &World,
        input: &InputState,
        alpha: f32,
    );
    fn on_resize(&mut self, _width: u32, _height: u32) {}
    /// Window title. Override to name your window; default keeps the
    /// engine generic. Called once at `resumed`, so a static string is
    /// enough — no need to react to state changes here.
    fn window_title(&self) -> &'static str { "void_engine app" }
}

#[cfg(feature = "client")]
struct PerfLogger {
    tx: mpsc::SyncSender<String>,
    _thread: std::thread::JoinHandle<()>,
}

#[cfg(feature = "client")]
impl PerfLogger {
    fn new() -> Self {
        let (tx, rx) = mpsc::sync_channel::<String>(4);
        let thread = std::thread::spawn(move || {
            use std::io::Write;
            let mut out = std::io::BufWriter::new(std::io::stderr());
            for msg in rx {
                let _ = writeln!(out, "{msg}");
                let _ = out.flush();
            }
        });
        Self { tx, _thread: thread }
    }

    fn send(&self, msg: String) {
        // non-blocking: drop the message if the channel is full rather than stall the game
        let _ = self.tx.try_send(msg);
    }
}

#[cfg(feature = "client")]
struct PerfStats {
    frame_times: Vec<f64>,
    accum_update_ms: f64,
    accum_batch_ms: f64,
    accum_present_ms: f64,
    last_report: Instant,
    vertex_count: usize,
    logger: PerfLogger,
}

#[cfg(feature = "client")]
impl PerfStats {
    fn new() -> Self {
        Self {
            frame_times: Vec::with_capacity(120),
            accum_update_ms: 0.0,
            accum_batch_ms: 0.0,
            accum_present_ms: 0.0,
            last_report: Instant::now(),
            vertex_count: 0,
            logger: PerfLogger::new(),
        }
    }

    fn record(
        &mut self,
        renderer: &mut Renderer,
        frame_ms: f64,
        update_ms: f64,
        batch_ms: f64,
        present_ms: f64,
        verts: usize,
    ) {
        self.frame_times.push(frame_ms);
        self.accum_update_ms += update_ms;
        self.accum_batch_ms += batch_ms;
        self.accum_present_ms += present_ms;
        self.vertex_count = verts;

        let elapsed = self.last_report.elapsed().as_secs_f64();
        if elapsed >= 1.0 {
            let n = self.frame_times.len() as f64;
            let fps = n / elapsed;
            let avg_frame   = self.frame_times.iter().sum::<f64>() / n;
            let avg_update  = self.accum_update_ms  / n;
            let avg_batch   = self.accum_batch_ms   / n;
            let avg_present = self.accum_present_ms / n;

            self.frame_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50 = self.frame_times[(n * 0.50) as usize];
            let p95 = self.frame_times[(n * 0.95) as usize];
            let p99 = self.frame_times[((n * 0.99) as usize).min(self.frame_times.len() - 1)];
            let worst = self.frame_times.last().copied().unwrap_or(0.0);

            let msg = format!(
                "[perf] fps={fps:.1}  avg={avg_frame:.2}ms  p50={p50:.2}ms  p95={p95:.2}ms  p99={p99:.2}ms  worst={worst:.2}ms  |  update={avg_update:.2}ms  batch={avg_batch:.2}ms  present={avg_present:.2}ms  verts={}",
                self.vertex_count
            );
            self.logger.send(msg);

            // Publish the same numbers to the renderer so App::render can
            // surface them in the F3 debug overlay. Single source of truth
            // for "what the [perf] log line says".
            renderer.last_perf = PerfSnapshot {
                fps: fps as f32,
                avg_frame_ms:   avg_frame   as f32,
                p50_frame_ms:   p50         as f32,
                p95_frame_ms:   p95         as f32,
                p99_frame_ms:   p99         as f32,
                worst_frame_ms: worst       as f32,
                avg_update_ms:  avg_update  as f32,
                avg_batch_ms:   avg_batch   as f32,
                avg_present_ms: avg_present as f32,
                vertex_count:   self.vertex_count as u32,
            };

            self.frame_times.clear();
            self.accum_update_ms = 0.0;
            self.accum_batch_ms = 0.0;
            self.accum_present_ms = 0.0;
            self.last_report = Instant::now();
        }
    }
}

#[cfg(feature = "client")]
struct Handler<A: ClientApp> {
    app: A,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    world: World,
    input: InputState,
    timestep: Timestep,
    last_frame: Instant,
    perf: PerfStats,
}

#[cfg(feature = "client")]
impl<A: ClientApp> Handler<A> {
    fn new(app: A) -> Self {
        Self {
            app,
            window: None,
            renderer: None,
            world: World::new(),
            input: InputState::default(),
            timestep: Timestep::new(),
            last_frame: Instant::now(),
            perf: PerfStats::new(),
        }
    }
}

#[cfg(feature = "client")]
impl<A: ClientApp> ApplicationHandler for Handler<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = WindowAttributes::default()
            .with_title(self.app.window_title())
            .with_inner_size(winit::dpi::LogicalSize::new(1920u32, 1080u32));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let renderer = Renderer::new(window.clone());
        self.renderer = Some(renderer);
        self.window = Some(window);

        let dt = self.timestep.dt();
        let mut ctx = SimCtx {
            world: &mut self.world,
            input: &self.input,
            dt,
        };
        self.app.init(&mut ctx);
        self.last_frame = Instant::now();
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = self.renderer.as_mut() {
                    r.resize(size.width, size.height);
                    self.app.on_resize(size.width, size.height);
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    if event.state == winit::event::ElementState::Pressed {
                        self.input.on_key_down(code);
                    } else {
                        self.input.on_key_up(code);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if state == winit::event::ElementState::Pressed {
                    self.input.on_mouse_down(button);
                } else {
                    self.input.on_mouse_up(button);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input.on_mouse_move(
                    Vec2::new(position.x as f32, position.y as f32),
                    Vec2::ZERO,
                );
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let y = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
                };
                self.input.on_scroll(y);
            }
            WindowEvent::RedrawRequested => {
                // OS-requested repaint (window uncovered etc.) — just render, no timing
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.begin_frame();
                    self.app.render(renderer, &self.world, &self.input, 1.0);
                    renderer.end_frame();
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _: &ActiveEventLoop,
        _: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.input.on_mouse_move(
                self.input.mouse_pos,
                Vec2::new(delta.0 as f32, delta.1 as f32),
            );
        }
    }

    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if self.renderer.is_none() { return; }

        // Frame cap: sleep most of the budget, spin only the last 1ms for precision
        const TARGET_FRAME_S: f64 = 1.0 / 62.0;
        const SPIN_THRESHOLD_S: f64 = 0.001;
        let sleep_until = TARGET_FRAME_S - SPIN_THRESHOLD_S;
        let elapsed = self.last_frame.elapsed().as_secs_f64();
        if elapsed < sleep_until {
            std::thread::sleep(std::time::Duration::from_secs_f64(sleep_until - elapsed));
        }
        while self.last_frame.elapsed().as_secs_f64() < TARGET_FRAME_S {
            std::hint::spin_loop();
        }

        let frame_start = Instant::now();
        let frame_dt = frame_start.duration_since(self.last_frame).as_secs_f32();
        self.last_frame = frame_start;

        let (steps, alpha) = self.timestep.advance(frame_dt);
        let dt = self.timestep.dt();

        let update_start = Instant::now();
        for step in 0..steps {
            // Lockstep games stall here when the peer's input for the next
            // tick has not arrived. Refund the steps we are not going to run
            // so they come back on a later frame — dropping them would let
            // this peer fall permanently behind the other one.
            //
            // Checked BEFORE the step, so a stall on step 0 leaves the input
            // edge flags untouched: they are cleared only after a step has
            // actually consumed them (see below), and a deferred tick has
            // consumed nothing. Clearing them here would swallow the
            // keypress entirely, which is the bug the `step == 0` guard was
            // added to fix in the first place.
            if !self.app.can_advance() {
                self.timestep.refund(steps - step);
                break;
            }
            let mut ctx = SimCtx {
                world: &mut self.world,
                input: &self.input,
                dt,
            };
            self.app.fixed_update(&mut ctx);
            // Clear the pressed/released edge flags after the FIRST step,
            // not after the whole catch-up loop.
            //
            // `key_pressed` means "went down this instant". When the
            // renderer falls behind, `advance` returns several steps to
            // catch up, and every one of them saw the same edge flag —
            // so one keypress typed N characters into the login field
            // (observed as five at ~25 fps against a 60 Hz fixed step).
            //
            // Clearing here keeps the earlier fix intact: the flags still
            // survive a frame where NO step ran (the render loop runs at
            // ~62 Hz against a 1/60 step, so that is common, and clearing
            // unconditionally dropped keypresses instead). They are now
            // consumed exactly once, by exactly one step.
            if step == 0 {
                self.input.begin_frame();
            }
        }
        let update_ms = update_start.elapsed().as_secs_f64() * 1000.0;

        let renderer = self.renderer.as_mut().unwrap();
        renderer.begin_frame();
        let batch_start = Instant::now();
        self.app.render(renderer, &self.world, &self.input, alpha);
        let batch_ms = batch_start.elapsed().as_secs_f64() * 1000.0;
        let verts = renderer.batch.vertices.len();
        let present_start = Instant::now();
        renderer.end_frame();
        let present_ms = present_start.elapsed().as_secs_f64() * 1000.0;

        let frame_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
        self.perf.record(renderer, frame_ms, update_ms, batch_ms, present_ms, verts);
    }
}

/// Open a window and run the game loop at the client rate (60 Hz fixed
/// step, ~62 Hz render). Blocks until the window closes.
#[cfg(feature = "client")]
pub fn run<A: ClientApp>(app: A) {
    let _ = env_logger::try_init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut handler = Handler::new(app);
    event_loop.run_app(&mut handler).unwrap();
}
