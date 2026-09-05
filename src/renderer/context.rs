use wgpu::SurfaceTarget;
use winit::window::Window;

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub format: wgpu::TextureFormat,
}

impl GpuContext {
    pub fn new(window: std::sync::Arc<Window>) -> Self {
        pollster::block_on(Self::new_async(window))
    }

    async fn new_async(window: std::sync::Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(SurfaceTarget::Window(Box::new(window.clone())))
            .unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no GPU adapter found");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .expect("device request failed");
        // Custom uncaptured-error handler — wgpu's default just panics
        // with "Handling wgpu errors as fatal by default" and swallows
        // the actual validation / OOM message. Log the full error text
        // via the crate logger before letting the default kick in, so
        // logs/client.log captures what actually died.
        device.on_uncaptured_error(Box::new(|err| {
            log::error!("[wgpu] uncaptured error: {err}");
            // Fall through to the default (panic) — we still want the
            // process to stop rather than continue rendering garbage.
            panic!("wgpu fatal: {err}");
        }));

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);

        // Prefer Mailbox (triple-buffer): get_current_texture() never blocks,
        // GPU handles vsync internally. Fifo (vsync) as fallback.
        let present_mode = if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::Fifo
        };

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 1,
        };
        surface.configure(&device, &surface_config);

        Self {
            device,
            queue,
            surface,
            surface_config,
            format,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
    }

    /// Swap the present mode (Fifo = vsync on, Mailbox/Immediate = vsync off)
    /// and reconfigure the surface. Falls back to Fifo if the requested mode
    /// isn't supported by this adapter.
    pub fn set_vsync(&mut self, enabled: bool) {
        let target = if enabled {
            wgpu::PresentMode::Fifo
        } else {
            // Prefer Mailbox over Immediate so we still get triple buffering
            // when the driver supports it.
            wgpu::PresentMode::Mailbox
        };
        if self.surface_config.present_mode == target { return; }
        self.surface_config.present_mode = target;
        self.surface.configure(&self.device, &self.surface_config);
    }
}
