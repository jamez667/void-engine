//! Rotating file logger + in-process bounded relay ring + optional
//! boot-time Loki push. Every record hits `logs/<log_name>.log`
//! (rotated at 10 MB × 5) and is pushed onto a bounded ring the game's
//! net layer drains once per tick to forward through its transport.
//!
//! `LogEvent` is the engine's serialisation-agnostic log record; games
//! typically wrap it in their own transport packet (e.g. `LogPacket`
//! with bitcode derives) via `From<LogEvent>`.
//!
//! Boot pusher: strictly ERROR-level, opt-in via a caller-supplied
//! `boot_loki_url` — used only to capture pre-connect panics /
//! start-up failures that the normal server-relay path would miss.
//! Off by default.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use log::{Metadata, Record};

const MAX_LOGS: usize = 5;
const MAX_LOG_SIZE: u64 = 1024 * 1024 * 10; // 10 MB
const RELAY_CAPACITY: usize = 2000;

/// Serialisation-agnostic log record. Games map this into their
/// transport packet via `From<LogEvent>`.
#[derive(Clone, Debug)]
pub struct LogEvent {
    pub level:  u8,
    pub target: String,
    pub msg:    String,
    pub ts_ms:  u64,
}

// ── relay channel (logger → net layer) ──────────────────────────────────────

struct LogRelay {
    buf: std::collections::VecDeque<LogEvent>,
    dropped: u64,
}

static RELAY: OnceLock<Mutex<LogRelay>> = OnceLock::new();

fn relay() -> &'static Mutex<LogRelay> {
    RELAY.get_or_init(|| Mutex::new(LogRelay {
        buf: std::collections::VecDeque::with_capacity(RELAY_CAPACITY),
        dropped: 0,
    }))
}

fn push_relay(evt: LogEvent) {
    if let Ok(mut g) = relay().lock() {
        if g.buf.len() >= RELAY_CAPACITY {
            g.buf.pop_front();
            g.dropped = g.dropped.saturating_add(1);
        }
        g.buf.push_back(evt);
    }
}

/// Drain all queued log events. Called by the game's net layer once per
/// tick (or however often it batches log messages onto the wire).
/// Callers map the returned `LogEvent`s into their transport packet.
pub fn drain_log_events() -> Vec<LogEvent> {
    match relay().lock() {
        Ok(mut g) => g.buf.drain(..).collect(),
        Err(_)    => Vec::new(),
    }
}

// ── boot-time direct Loki path (errors only) ────────────────────────────────

struct BootEvent {
    ts_ns: u128,
    level: &'static str,
    line:  String,
}

static BOOT_TX: OnceLock<Option<std::sync::mpsc::SyncSender<BootEvent>>> = OnceLock::new();

/// Config for a `RotatingLogger` instance.
#[derive(Clone, Debug)]
pub struct LogConfig {
    /// Directory holding the rotated file set (created if missing).
    pub log_dir: PathBuf,
    /// Base file name (e.g. `"client.log"`).
    pub log_name: String,
    /// Substrings whose full message body is dropped before writing —
    /// tag known-noisy driver spam here so it doesn't bloat the log or
    /// the server relay.
    pub noise: Vec<&'static str>,
    /// Log targets (crate / module prefixes) dropped at the `enabled`
    /// gate, before the record is even formatted. For chatty
    /// dependencies that log per-frame — put whole crates here, and use
    /// `noise` only for one-off messages you cannot address by target.
    pub quiet_targets: Vec<&'static str>,
    /// Optional boot-Loki URL for the direct ERROR-only path. `None`
    /// disables the boot pusher entirely.
    pub boot_loki_url: Option<String>,
    /// Loki `service` label for events shipped through the boot pusher.
    pub boot_service: &'static str,
}

pub struct RotatingLogger {
    cfg: LogConfig,
    /// The open log file, kept across calls.
    ///
    /// This used to `create_dir_all` + `OpenOptions::open` + close on
    /// EVERY line, synchronously on whichever thread logged — which for
    /// a client is the render thread. Three syscalls per line at tens of
    /// thousands of lines a session is a measurable stutter, and it is
    /// pure waste: the path never changes.
    file: Mutex<Option<fs::File>>,
}

impl RotatingLogger {
    pub fn new(cfg: LogConfig) -> Self {
        let _ = fs::create_dir_all(&cfg.log_dir);
        let file = fs::OpenOptions::new()
            .create(true).append(true)
            .open(cfg.log_dir.join(&cfg.log_name))
            .ok();
        Self { cfg, file: Mutex::new(file) }
    }
}

impl log::Log for RotatingLogger {
    /// Reject noisy dependencies **before** the record is formatted.
    ///
    /// `log`'s macros check this first, so a rejected target costs a
    /// string compare rather than a `format!` plus a file write plus a
    /// relay push. wgpu logs `Device::maintain` at INFO once per frame:
    /// one session produced 43,942 of 45,144 lines (97%) from that alone,
    /// each one formatted, written to disk, and shipped to Loki.
    fn enabled(&self, metadata: &Metadata) -> bool {
        // Quiet targets lose INFO/DEBUG/TRACE only. WARN and ERROR still
        // come through — surface loss, device lost and shader compile
        // failures are exactly the things you need when the graphics
        // stack misbehaves, and muting a whole crate would hide them.
        if metadata.level() <= log::Level::Warn { return true; }
        !self.cfg.quiet_targets.iter().any(|t| metadata.target().starts_with(t))
    }

    fn log(&self, record: &Record) {
        // `log_enabled!` short-circuits most calls, but a direct
        // `Log::log` (or a macro that skipped the check) still lands
        // here, so re-test rather than trust the caller.
        if !self.enabled(record.metadata()) { return; }
        let level = record.level();
        let body  = format!("{}", record.args());
        if self.cfg.noise.iter().any(|n| body.contains(n)) { return; }

        let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S");
        if let Ok(mut guard) = self.file.lock() {
            if let Some(file) = guard.as_mut() {
                let _ = writeln!(file, "[{}] [{}] {}", timestamp, level, body);
            }
        }

        let ts_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        push_relay(LogEvent {
            level:  level as u8,
            target: record.target().to_string(),
            msg:    body.clone(),
            ts_ms,
        });

        if level == log::Level::Error {
            if let Some(tx) = BOOT_TX.get().and_then(|o| o.as_ref()) {
                let ts_ns = (ts_ms as u128) * 1_000_000;
                let _ = tx.try_send(BootEvent { ts_ns, level: level_str(level), line: body });
            }
        }

        check_rotate(&self.cfg);
    }

    fn flush(&self) {}
}

fn level_str(l: log::Level) -> &'static str {
    match l {
        log::Level::Error => "ERROR",
        log::Level::Warn  => "WARN",
        log::Level::Info  => "INFO",
        log::Level::Debug => "DEBUG",
        log::Level::Trace => "TRACE",
    }
}

fn check_rotate(cfg: &LogConfig) {
    let log_path = cfg.log_dir.join(&cfg.log_name);
    let metadata = match fs::metadata(&log_path) { Ok(m) => m, Err(_) => return };
    if metadata.len() > MAX_LOG_SIZE { rotate_logs(cfg); }
}

fn rotate_logs(cfg: &LogConfig) {
    let log_dir = cfg.log_dir.as_path();
    for i in (1..MAX_LOGS).rev() {
        let src = log_dir.join(format!("{}.{}", cfg.log_name, i));
        let dst = log_dir.join(format!("{}.{}", cfg.log_name, i + 1));
        let _ = fs::rename(&src, &dst);
    }
    let src = log_dir.join(&cfg.log_name);
    let dst = log_dir.join(format!("{}.1", cfg.log_name));
    let _ = fs::rename(&src, &dst);
}

/// Install `RotatingLogger` as the global `log` crate sink at
/// `LevelFilter::Info`. Starts the boot pusher if `cfg.boot_loki_url` is
/// `Some`. Safe to call once; subsequent calls silently no-op (matching
/// `log`'s `set_boxed_logger` behaviour).
pub fn init(cfg: LogConfig) {
    if let Some(url) = cfg.boot_loki_url.clone() {
        let service = cfg.boot_service;
        BOOT_TX.get_or_init(|| Some(start_boot_pusher(url, service)));
    } else {
        BOOT_TX.get_or_init(|| None);
    }

    log::set_boxed_logger(Box::new(RotatingLogger::new(cfg)))
        .map(|_| log::set_max_level(log::LevelFilter::Info))
        .ok();
}

// ── boot pusher (ERROR-only, direct to Loki) ────────────────────────────────

const BOOT_FLUSH_INTERVAL: Duration = Duration::from_secs(2);
const BOOT_BATCH_MAX:      usize    = 32;

fn start_boot_pusher(loki_url: String, service: &'static str) -> std::sync::mpsc::SyncSender<BootEvent> {
    let (tx, rx) = std::sync::mpsc::sync_channel::<BootEvent>(128);
    let endpoint = format!("{}/loki/api/v1/push", loki_url.trim_end_matches('/'));
    let host = hostname();

    std::thread::Builder::new()
        .name("loki-boot-push".into())
        .spawn(move || {
            let agent = ureq::AgentBuilder::new().timeout(Duration::from_secs(5)).build();
            let mut buf: Vec<BootEvent> = Vec::with_capacity(BOOT_BATCH_MAX);
            let mut last_flush = Instant::now();

            loop {
                let timeout = BOOT_FLUSH_INTERVAL.saturating_sub(last_flush.elapsed());
                match rx.recv_timeout(timeout) {
                    Ok(evt) => {
                        buf.push(evt);
                        if buf.len() >= BOOT_BATCH_MAX {
                            flush_boot(&agent, &endpoint, &host, service, &mut buf);
                            last_flush = Instant::now();
                        }
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                        if !buf.is_empty() { flush_boot(&agent, &endpoint, &host, service, &mut buf); }
                        last_flush = Instant::now();
                    }
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                        if !buf.is_empty() { flush_boot(&agent, &endpoint, &host, service, &mut buf); }
                        break;
                    }
                }
            }
        })
        .ok();

    tx
}

fn hostname() -> String {
    std::env::var("COMPUTERNAME")
        .or_else(|_| std::env::var("HOSTNAME"))
        .unwrap_or_else(|_| "unknown".into())
}

fn flush_boot(agent: &ureq::Agent, endpoint: &str, host: &str, service: &str, buf: &mut Vec<BootEvent>) {
    let body = build_boot_payload(host, service, buf);
    buf.clear();
    let _ = agent.post(endpoint).set("Content-Type", "application/json").send_string(&body);
}

fn build_boot_payload(host: &str, service: &str, events: &[BootEvent]) -> String {
    use std::collections::BTreeMap;
    let mut by_level: BTreeMap<&'static str, Vec<&BootEvent>> = BTreeMap::new();
    for e in events { by_level.entry(e.level).or_default().push(e); }

    let mut s = String::with_capacity(events.len() * 80);
    s.push_str("{\"streams\":[");
    let mut first_stream = true;
    for (level, group) in &by_level {
        if !first_stream { s.push(','); }
        first_stream = false;
        s.push_str("{\"stream\":{\"service\":\"");
        s.push_str(service);
        s.push_str("\",\"host\":\"");
        push_json_escaped(&mut s, host);
        s.push_str("\",\"level\":\"");
        s.push_str(level);
        s.push_str("\"},\"values\":[");
        let mut first_val = true;
        for e in group {
            if !first_val { s.push(','); }
            first_val = false;
            s.push_str("[\"");
            s.push_str(&e.ts_ns.to_string());
            s.push_str("\",\"");
            push_json_escaped(&mut s, &e.line);
            s.push_str("\"]");
        }
        s.push_str("]}");
    }
    s.push_str("]}");
    s
}

fn push_json_escaped(out: &mut String, src: &str) {
    for c in src.chars() {
        match c {
            '"'  => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => { out.push_str(&format!("\\u{:04x}", c as u32)); }
            c => out.push(c),
        }
    }
}

// Path is only used through PathBuf/Path already imported.
#[allow(dead_code)]
fn _path_marker(_: &Path) {}

#[cfg(test)]
mod filter_tests {
    use super::*;
    use log::{Level, Log};

    fn logger(quiet: &[&'static str]) -> RotatingLogger {
        RotatingLogger::new(LogConfig {
            log_dir:  std::env::temp_dir().join("void_log_filter_test"),
            log_name: "t.log".into(),
            noise:    Vec::new(),
            quiet_targets: quiet.to_vec(),
            boot_loki_url: None,
            boot_service:  "test",
        })
    }

    fn meta(target: &'static str, level: Level) -> Metadata<'static> {
        Metadata::builder().target(target).level(level).build()
    }

    /// The reason this exists: wgpu logs `Device::maintain` at INFO once
    /// per frame. One session produced 43,942 of 45,144 lines from it,
    /// each formatted, written to disk and queued for the relay on the
    /// render thread.
    #[test]
    fn quiet_targets_drop_their_info_chatter() {
        let l = logger(&["wgpu_core"]);
        assert!(!l.enabled(&meta("wgpu_core::device", Level::Info)));
        assert!(!l.enabled(&meta("wgpu_core", Level::Debug)));
        assert!(!l.enabled(&meta("wgpu_core", Level::Trace)));
    }

    /// ...but a quiet target is NOT muted. Surface loss, device lost and
    /// shader compile failures all arrive as WARN/ERROR from exactly
    /// these crates, and are the whole reason to read graphics logs.
    #[test]
    fn quiet_targets_still_report_problems() {
        let l = logger(&["wgpu_core"]);
        assert!(l.enabled(&meta("wgpu_core::device", Level::Warn)));
        assert!(l.enabled(&meta("wgpu_core::device", Level::Error)));
    }

    /// Prefix matching, so one entry covers a crate's submodules — and
    /// does not accidentally cover an unrelated crate that merely shares
    /// a prefix boundary.
    #[test]
    fn quiet_matching_is_by_target_prefix() {
        let l = logger(&["wgpu_core"]);
        assert!(!l.enabled(&meta("wgpu_core::device::resource", Level::Info)));
        assert!(l.enabled(&meta("void_claim::net", Level::Info)),
            "game targets must keep their INFO");
        assert!(l.enabled(&meta("void_sim", Level::Info)));
    }

    /// With no quiet list configured nothing is filtered — the setting
    /// is opt-in, so existing consumers keep their behaviour.
    #[test]
    fn an_empty_quiet_list_filters_nothing() {
        let l = logger(&[]);
        assert!(l.enabled(&meta("wgpu_core", Level::Info)));
        assert!(l.enabled(&meta("anything", Level::Trace)));
    }
}
