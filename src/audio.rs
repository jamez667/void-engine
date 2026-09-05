//! Generic audio playback built on `rodio`.
//!
//! Pieces:
//! - [`Mixer`] — owns the output stream + master/sfx/music gain atomics. Play
//!   one-shot samples or hold looping streams; both mix through the same output.
//! - [`MusicPlayer`] — plays background music from files with a crossfade on
//!   swap. Generation-guarded so stale worker threads discard their decoded
//!   source instead of appending to a new track. Single-flight decoder so the
//!   game's fixed-update tick can call `tick()` every frame without stacking
//!   worker threads.
//! - [`LoopBuf`] — a `Source` that repeats an f32 mono buffer forever. Useful
//!   for synthesized loops (thrust rumble, etc.) that the game generates in
//!   memory.
//!
//! The game side keeps the flavor: the SFX table, waveform synthesis, and the
//! music-mode enum all live in the client crate.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink};
use rodio::buffer::SamplesBuffer;
use rodio::source::Source;

/// A looping mono buffer at a fixed sample rate. Cycles forever — pair with
/// `Sink::set_volume(0.0)` to "mute" while keeping the stream alive.
pub struct LoopBuf {
    data: Arc<Vec<f32>>,
    sample_rate: u32,
    pos:  usize,
}

impl LoopBuf {
    pub fn new(data: Vec<f32>, sample_rate: u32) -> Self {
        Self { data: Arc::new(data), sample_rate, pos: 0 }
    }
}

impl Iterator for LoopBuf {
    type Item = f32;
    fn next(&mut self) -> Option<f32> {
        if self.data.is_empty() { return None; }
        let s = self.data[self.pos];
        self.pos = (self.pos + 1) % self.data.len();
        Some(s)
    }
}

impl Source for LoopBuf {
    fn current_frame_len(&self) -> Option<usize> { None }
    fn channels(&self) -> u16 { 1 }
    fn sample_rate(&self) -> u32 { self.sample_rate }
    fn total_duration(&self) -> Option<Duration> { None }
}

/// Owns the rodio output stream + shared master/sfx/music gain atomics.
///
/// Cheap to clone the `Arc<Mixer>` and hand out to whatever wants to play.
///
/// NOTE: `rodio::OutputStream` is `!Send + !Sync`, so `Arc<Mixer>` is not
/// actually shareable across threads — clippy's `arc_with_non_send_sync`
/// fires here and is correct about the type. It is sound today only because
/// every holder (`void_claim::audio::AudioSystem`) lives on the main thread
/// and never sends it anywhere; the `Arc` buys refcounting, not thread
/// safety. Left as `Arc` deliberately: switching to `Rc` would force
/// `AudioSystem` (handed around as `Arc<Self>`) to change shape too, which
/// is a design change rather than a lint fix. If audio ever needs to move
/// off the main thread, the stream must be owned by a dedicated audio
/// thread and driven by messages instead.
pub struct Mixer {
    _stream: OutputStream,
    handle:  OutputStreamHandle,
    // f32-bits so audio thread and UI can both read/write without locking.
    master_vol: AtomicU32,
    music_vol:  AtomicU32,
    sfx_vol:    AtomicU32,
}

impl Mixer {
    /// Try to open the default output device. Returns `None` if no device is
    /// available (headless build server, muted default, etc.).
    pub fn new() -> Option<Arc<Self>> {
        let (stream, handle) = OutputStream::try_default().ok()?;
        // See the type-level note: !Send + !Sync by way of OutputStream.
        #[allow(clippy::arc_with_non_send_sync)]
        Some(Arc::new(Self {
            _stream: stream,
            handle,
            master_vol: AtomicU32::new(1.0_f32.to_bits()),
            music_vol:  AtomicU32::new(0.45_f32.to_bits()),
            sfx_vol:    AtomicU32::new(0.7_f32.to_bits()),
        }))
    }

    /// The underlying `OutputStreamHandle`. Use it to build a `Sink` when you
    /// need to hold on to one (e.g. a looping source).
    pub fn handle(&self) -> &OutputStreamHandle { &self.handle }

    /// Play a one-shot f32 mono buffer at `sample_rate` with `channels`.
    /// `gain` is multiplied by `master * sfx` before hitting the sink.
    pub fn play_sample(&self, buf: Vec<f32>, sample_rate: u32, channels: u16, gain: f32) {
        if let Ok(sink) = Sink::try_new(&self.handle) {
            sink.append(SamplesBuffer::new(channels, sample_rate, buf));
            let master = self.master();
            let sfx    = self.sfx();
            sink.set_volume((master * sfx * gain).clamp(0.0, 1.0));
            sink.detach();
        }
    }

    /// Build a new empty sink on this mixer's output. Caller owns lifetime;
    /// dropping the sink drops the sound.
    pub fn new_sink(&self) -> Option<Sink> { Sink::try_new(&self.handle).ok() }

    pub fn set_master(&self, v: f32) {
        self.master_vol.store(v.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }
    pub fn set_music(&self, v: f32) {
        self.music_vol.store(v.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }
    pub fn set_sfx(&self, v: f32) {
        self.sfx_vol.store(v.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
    }

    pub fn master(&self) -> f32 { f32::from_bits(self.master_vol.load(Ordering::Relaxed)) }
    pub fn music(&self)  -> f32 { f32::from_bits(self.music_vol.load(Ordering::Relaxed)) }
    pub fn sfx(&self)    -> f32 { f32::from_bits(self.sfx_vol.load(Ordering::Relaxed)) }
}

// ── MusicPlayer ──────────────────────────────────────────────────────────────

/// How many `tick()` calls a crossfade takes to fully hand off.
/// Roughly 3 s at 30 Hz.
const FADE_TICKS: i32 = 90;

/// File-based background music with crossfade. Not aware of "modes" — the
/// game picks the next track and calls `play_track()`; the player handles
/// the actual crossfade + decode + generation guard.
pub struct MusicPlayer {
    handle:        OutputStreamHandle,
    music_sink:    Arc<Mutex<Sink>>,
    fade_sink:     Arc<Mutex<Sink>>,
    fade_count:    Arc<AtomicI32>,
    gen:           Arc<AtomicU64>,     // bumped so stale loads abort
    loading:       Arc<AtomicBool>,    // a worker is decoding
    paused:        AtomicBool,
    current_gain:  AtomicU32,          // per-track gain, set on play_track
    current_track: Mutex<Option<PathBuf>>,
}

impl MusicPlayer {
    /// Build a music player attached to the given mixer's output. Fails if the
    /// initial sinks can't be constructed.
    pub fn new(mixer: &Mixer) -> Option<Arc<Self>> {
        let music_sink = Sink::try_new(mixer.handle()).ok()?;
        music_sink.set_volume(0.45);
        let fade_sink = Sink::try_new(mixer.handle()).ok()?;
        fade_sink.set_volume(0.0);
        Some(Arc::new(Self {
            handle:       mixer.handle().clone(),
            music_sink:   Arc::new(Mutex::new(music_sink)),
            fade_sink:    Arc::new(Mutex::new(fade_sink)),
            fade_count:   Arc::new(AtomicI32::new(0)),
            gen:          Arc::new(AtomicU64::new(0)),
            loading:      Arc::new(AtomicBool::new(false)),
            paused:       AtomicBool::new(false),
            current_gain: AtomicU32::new(0.45_f32.to_bits()),
            current_track: Mutex::new(None),
        }))
    }

    /// Ramp down the crossfade counter and mute the fade sink when it hits zero.
    /// Call this every fixed-update tick.
    pub fn tick(&self) {
        let count = self.fade_count.load(Ordering::Relaxed);
        if count > 0 {
            let remaining = count - 1;
            let fade_vol = if remaining < 60 {
                remaining as f32 / 60.0 * 0.45
            } else {
                0.45
            };
            if let Ok(fade) = self.fade_sink.lock() { fade.set_volume(fade_vol); }
            self.fade_count.store(remaining, Ordering::Relaxed);
            if remaining <= 0 {
                if let Ok(fade) = self.fade_sink.lock() { fade.set_volume(0.0); }
            }
        }
    }

    /// True when the current music sink is empty (playback finished / never
    /// started) and no decoder worker is in flight. Game code polls this to
    /// decide when to queue the next track.
    pub fn is_drained(&self) -> bool {
        if self.loading.load(Ordering::Relaxed) { return false; }
        if self.paused.load(Ordering::Relaxed)  { return false; }
        let sink = match self.music_sink.lock() { Ok(s) => s, Err(_) => return false };
        sink.empty()
    }

    /// True while a decoder worker is decoding the queued track.
    pub fn is_loading(&self) -> bool { self.loading.load(Ordering::Relaxed) }

    pub fn is_paused(&self) -> bool { self.paused.load(Ordering::Relaxed) }

    pub fn current_track(&self) -> Option<PathBuf> {
        self.current_track.lock().ok().and_then(|g| g.clone())
    }

    /// Apply mixer gain to the music sink: `master * music`. Called by the
    /// game side whenever volume sliders change so the currently-playing
    /// track updates without waiting for the next crossfade.
    pub fn apply_gain(&self, master_times_music: f32) {
        let gain = master_times_music.clamp(0.0, 1.0);
        self.current_gain.store(gain.to_bits(), Ordering::Relaxed);
        if let Ok(sink) = self.music_sink.lock() { sink.set_volume(gain); }
    }

    /// Toggle pause on the underlying sink. Returns the new paused state.
    pub fn toggle_pause(&self) -> bool {
        let now_paused = !self.paused.load(Ordering::Relaxed);
        self.paused.store(now_paused, Ordering::Relaxed);
        if let Ok(sink) = self.music_sink.lock() {
            if now_paused { sink.pause(); } else { sink.play(); }
        }
        now_paused
    }

    /// Force the paused flag without touching the sink (used when the game
    /// switches modes and wants playback to resume implicitly).
    pub fn set_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::Relaxed);
    }

    /// Crossfade from the currently-playing track into a new one.
    ///
    /// `gain` is the target volume for the new sink (before fade-in).
    /// Kicks off a background decoder that will `append` on the music sink
    /// when finished — unless the generation moved in the meantime.
    pub fn play_track(&self, path: PathBuf, gain: f32) {
        self.fade_count.store(FADE_TICKS, Ordering::Relaxed);

        // Bump generation BEFORE swapping so any in-flight decoder discards
        // its source rather than appending to the fresh sink.
        self.gen.fetch_add(1, Ordering::Relaxed);
        self.current_gain.store(gain.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);

        if let Ok(new_sink) = Sink::try_new(&self.handle) {
            new_sink.set_volume(gain.clamp(0.0, 1.0));
            if let Ok(mut sink) = self.music_sink.lock() {
                sink.stop(); // some rodio versions leave source playing on Drop alone
                *sink = new_sink;
            }
        }
        if let Ok(mut cur) = self.current_track.lock() {
            *cur = Some(path.clone());
        }

        self.queue_decode(path);
    }

    /// Stop the current track (with the same crossfade behavior) without
    /// queueing a replacement. `apply_gain` on future `play_track` calls
    /// restores volume.
    pub fn stop(&self) {
        self.fade_count.store(FADE_TICKS, Ordering::Relaxed);
        self.gen.fetch_add(1, Ordering::Relaxed);
        if let Ok(new_sink) = Sink::try_new(&self.handle) {
            new_sink.set_volume(0.0);
            if let Ok(mut sink) = self.music_sink.lock() {
                sink.stop();
                *sink = new_sink;
            }
        }
        if let Ok(mut cur) = self.current_track.lock() { *cur = None; }
    }

    /// Replace the music sink with an empty one at the given gain, discarding
    /// any in-flight decoder. Use when you want an immediate cut (skip track)
    /// without the fade — game code then follows up with `play_track`.
    pub fn skip(&self, gain: f32) {
        self.gen.fetch_add(1, Ordering::Relaxed);
        self.current_gain.store(gain.clamp(0.0, 1.0).to_bits(), Ordering::Relaxed);
        if let Ok(new_sink) = Sink::try_new(&self.handle) {
            new_sink.set_volume(gain.clamp(0.0, 1.0));
            if let Ok(mut sink) = self.music_sink.lock() {
                sink.stop();
                *sink = new_sink;
            }
        }
    }

    fn queue_decode(&self, path: PathBuf) {
        // Single-flight: refuse to dispatch a new worker while one is
        // decoding. Without this guard, callers that tick every frame would
        // saturate the CPU with redundant decoders.
        if self.loading.swap(true, Ordering::AcqRel) { return; }
        let sink = Arc::clone(&self.music_sink);
        let gen  = Arc::clone(&self.gen);
        let loading = Arc::clone(&self.loading);
        let gen_at_dispatch = gen.load(Ordering::Relaxed);
        std::thread::spawn(move || {
            struct Guard(Arc<AtomicBool>);
            impl Drop for Guard {
                fn drop(&mut self) { self.0.store(false, Ordering::Release); }
            }
            let _guard = Guard(loading);

            let file = match std::fs::File::open(&path) {
                Ok(f)  => f,
                Err(e) => { log::warn!("[audio] open {:?} failed: {}", path, e); return; }
            };
            let buf = std::io::BufReader::new(file);
            // Pick decoder explicitly by extension. MP3 auto-probe slurps the
            // stream and interferes with format-specific decoders.
            let ext = path.extension().and_then(|s| s.to_str()).map(|s| s.to_lowercase());
            macro_rules! play_with {
                ($call:expr) => {{
                    match $call {
                        Ok(d) => {
                            if gen.load(Ordering::Relaxed) != gen_at_dispatch {
                                log::info!("[audio] track {:?} aborted — mode changed", path.file_name());
                                return;
                            }
                            log::info!("[audio] play track {:?}", path.file_name());
                            let sink = sink.lock().unwrap();
                            sink.append(d);
                            return;
                        }
                        Err(e) => { log::warn!("[audio] decode {:?} failed: {}", path, e); return; }
                    }
                }};
            }
            match ext.as_deref() {
                Some("mp3") => play_with!(Decoder::new_mp3(buf)),
                _           => play_with!(Decoder::new(buf)),
            }
        });
    }
}

// ── file-pool helpers ────────────────────────────────────────────────────────

/// List every `.mp3` file directly under `dir` (non-recursive). Returns an
/// empty vec if the directory is unreadable.
pub fn list_mp3_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(rd) = std::fs::read_dir(dir) {
        for entry in rd.flatten() {
            let path = entry.path();
            if path.is_file() {
                if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
                    if ext.eq_ignore_ascii_case("mp3") {
                        out.push(path);
                    }
                }
            }
        }
    }
    out
}
