use winit::application::ApplicationHandler;
use winit::event::{WindowEvent, DeviceEvent, DeviceId, MouseScrollDelta};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId, WindowAttributes};
use winit::keyboard::PhysicalKey;
use glam::Vec2;
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Instant;

use crate::renderer::Renderer;
use crate::input::InputState;
use crate::ecs::World;
use crate::time::{Timestep, FIXED_DT};

/// Per-second rollup of render-loop timings + vertex count. Published by
/// the engine on `Renderer::last_perf` after each rollup tick (~1 Hz) so
/// `App::render` can surface the same numbers the `[perf]` log line shows.
/// Defaults to zero on the first frame.
#[derive(Default, Clone, Copy, Debug)]
pub struct PerfSnapshot {
    pub fps:           f32,
    pub avg_frame_ms:  f32,
    pub p50_frame_ms:  f32,
    pub p95_frame_ms:  f32,
    pub p99_frame_ms:  f32,
    pub worst_frame_ms: f32,
    pub avg_update_ms:  f32,
    pub avg_batch_ms:   f32,
    pub avg_present_ms: f32,
    pub vertex_count:   u32,
}

pub struct EngineCtx<'a> {
    pub world: &'a mut World,
    pub input: &'a InputState,
    pub renderer: &'a mut Renderer,
    pub dt: f32,
}

pub trait App: 'static {
    fn init(&mut self, ctx: &mut EngineCtx);
    fn fixed_update(&mut self, ctx: &mut EngineCtx);
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

struct PerfLogger {
    tx: mpsc::SyncSender<String>,
    _thread: std::thread::JoinHandle<()>,
}

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

struct PerfStats {
    frame_times: Vec<f64>,
    accum_update_ms: f64,
    accum_batch_ms: f64,
    accum_present_ms: f64,
    last_report: Instant,
    vertex_count: usize,
    logger: PerfLogger,
}

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

struct Handler<A: App> {
    app: A,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    world: World,
    input: InputState,
    timestep: Timestep,
    last_frame: Instant,
    perf: PerfStats,
}

impl<A: App> Handler<A> {
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

impl<A: App> ApplicationHandler for Handler<A> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = WindowAttributes::default()
            .with_title(self.app.window_title())
            .with_inner_size(winit::dpi::LogicalSize::new(1920u32, 1080u32));
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        let renderer = Renderer::new(window.clone());
        self.renderer = Some(renderer);
        self.window = Some(window);

        let renderer = self.renderer.as_mut().unwrap();
        let mut ctx = EngineCtx {
            world: &mut self.world,
            input: &self.input,
            renderer,
            dt: FIXED_DT,
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

        let update_start = Instant::now();
        for step in 0..steps {
            let renderer = self.renderer.as_mut().unwrap();
            let mut ctx = EngineCtx {
                world: &mut self.world,
                input: &self.input,
                renderer,
                dt: FIXED_DT,
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

pub fn run<A: App>(app: A) {
    let _ = env_logger::try_init();
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut handler = Handler::new(app);
    event_loop.run_app(&mut handler).unwrap();
}
