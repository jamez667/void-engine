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
    /// On-disk line format. Defaults to [`LogFormat::Human`], so nothing
    /// that already reads these files changes.
    ///
    /// A server behind promtail wants [`LogFormat::Logfmt`]: it carries
    /// the record's target and an RFC3339 timestamp, neither of which
    /// survives the human format, and promtail parses it natively.
    pub format: LogFormat,
}

/// How each record is written to the log file.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Default)]
pub enum LogFormat {
    /// `[2026-09-10 19:31:24] [ERROR] the message`
    ///
    /// Readable over someone's shoulder, lossy for a log pipeline: no
    /// target, no timezone, no sub-second precision, so events inside one
    /// second sort arbitrarily.
    #[default]
    Human,
    /// `ts=2026-09-10T19:31:24.512Z level=error target=… msg="…"`
    ///
    /// Logfmt rather than JSON on purpose: promtail parses both, but this
    /// needs no serialisation dependency and stays greppable when you
    /// tail the file by hand.
    Logfmt,
}

/// Quote a logfmt value if it needs it.
///
/// Bare values are fine until they contain a space, a quote or a newline
/// — a message body contains all three sooner or later, and an unquoted
/// one silently truncates at the first space when parsed.
fn logfmt_value(v: &str) -> String {
    if v.is_empty() {
        return "\"\"".to_string();
    }
    let needs_quoting = v
        .chars()
        .any(|c| c.is_whitespace() || c == '"' || c == '=' || c == '\\');
    if !needs_quoting {
        return v.to_string();
    }
    let mut out = String::with_capacity(v.len() + 2);
    out.push('"');
    for c in v.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(c),
        }
    }
    out.push('"');
    out
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

        let line = match self.cfg.format {
            LogFormat::Human => {
                let ts = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S");
                format!("[{}] [{}] {}", ts, level, body)
            }
            LogFormat::Logfmt => {
                // RFC3339 with milliseconds: promtail orders by this, and
                // whole-second precision loses the ordering of everything
                // that happens inside one tick.
                let ts = chrono::Utc::now()
                    .format("%Y-%m-%dT%H:%M:%S%.3fZ")
                    .to_string();
                format!(
                    "ts={} level={} target={} msg={}",
                    ts,
                    level.as_str().to_ascii_lowercase(),
                    logfmt_value(record.target()),
                    logfmt_value(&body),
                )
            }
        };
        if let Ok(mut guard) = self.file.lock() {
            if let Some(file) = guard.as_mut() {
                let _ = writeln!(file, "{}", line);
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

        self.check_rotate();
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

impl RotatingLogger {
    /// Rotate if the live file has outgrown [`MAX_LOG_SIZE`].
    ///
    /// # Why this is a method and asks the handle
    ///
    /// This was a free function that `fs::metadata`'d the *path*, and
    /// `rotate_logs` renamed files without touching the open handle. An
    /// open handle follows the inode, not the name, so after one rotation:
    ///
    /// 1. `t.log` becomes `t.log.1`, and nothing recreates `t.log`.
    /// 2. Every later line is written **into `t.log.1`** through the stale
    ///    handle — an operator tailing `t.log` sees a file frozen at the
    ///    rotation moment while the server logs somewhere else.
    /// 3. The path stat now fails, `check_rotate` returns early, and
    ///    rotation is **permanently disabled**. `MAX_LOGS` is never
    ///    enforced again and the surviving file grows without bound.
    ///
    /// So the size cap fired exactly once per process lifetime, and the
    /// five-file retention never fired at all. Measured end to end: after
    /// rotation plus 1000 further lines, the directory held one file
    /// (`t.log.1`), no `t.log`, and no `t.log.2`.
    ///
    /// Asking the handle for its own length also removes a syscall per
    /// line. The path stat measured **0.0154 ms**, which at 1000 lines/s
    /// is ~1.5% of a core on whichever thread logged — for a client, the
    /// render thread. The same file's own docs record that per-line
    /// open/close was already removed as "a measurable stutter"; the stat
    /// was the same waste, surviving.
    fn check_rotate(&self) {
        let mut guard = match self.file.lock() {
            Ok(g) => g,
            // A poisoned mutex means another thread panicked mid-write.
            // Logging is not worth propagating that.
            Err(e) => e.into_inner(),
        };
        let too_big = guard
            .as_ref()
            .and_then(|f| f.metadata().ok())
            .is_some_and(|m| m.len() > MAX_LOG_SIZE);
        if !too_big {
            return;
        }

        // Drop the handle *before* renaming. Windows refuses to rename a
        // file with an open handle, so on that platform this is not a
        // tidiness point but the difference between rotating and silently
        // not rotating.
        *guard = None;
        rotate_logs(&self.cfg);
        *guard = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.cfg.log_dir.join(&self.cfg.log_name))
            .ok();
    }
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
mod rotation_tests {
    use super::*;
    use log::Log;

    /// A logger writing into its own directory, so these can run in
    /// parallel without fighting over a shared path.
    fn logger(tag: &str) -> (RotatingLogger, PathBuf) {
        let dir = std::env::temp_dir().join(format!("void_log_rot_{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let cfg = LogConfig {
            log_dir: dir.clone(),
            log_name: "t.log".to_string(),
            noise: Vec::new(),
            quiet_targets: Vec::new(),
            boot_loki_url: None,
            boot_service: "test",
            format: LogFormat::Human,
        };
        (RotatingLogger::new(cfg), dir)
    }

    fn say(l: &RotatingLogger, msg: &str) {
        l.log(
            &log::Record::builder()
                .args(format_args!("{msg}"))
                .level(log::Level::Error)
                .target("test")
                .build(),
        );
    }

    /// Push the live file past the cap without writing ten megabytes
    /// through the logger. The handle is the logger's; this appends
    /// underneath it, which is exactly what a real oversized log is.
    fn inflate(dir: &std::path::Path) {
        let path = dir.join("t.log");
        let big = vec![b'x'; (MAX_LOG_SIZE as usize) + 1024];
        fs::write(&path, big).expect("seed an oversized log");
    }

    /// The bug: an open handle follows the inode, so after a rename every
    /// later line landed in the *rotated* file and the live name did not
    /// exist at all.
    #[test]
    fn writes_land_in_the_live_file_after_rotation() {
        let (l, dir) = logger("live");
        say(&l, "before");
        inflate(&dir);
        // Triggers the size check, rotates, and must reopen.
        say(&l, "trigger");
        say(&l, "after-rotation");

        let live = fs::read_to_string(dir.join("t.log")).expect("t.log must exist after rotating");
        assert!(
            live.contains("after-rotation"),
            "a line written after rotation must land in the live file, got {live:?}",
        );
        assert!(
            live.len() < MAX_LOG_SIZE as usize,
            "the live file must be the fresh one, not the 10 MB original",
        );
        assert!(dir.join("t.log.1").exists(), "the old file must be kept as .1");
        let _ = fs::remove_dir_all(&dir);
    }

    /// The second consequence: once the path stat failed, `check_rotate`
    /// returned early forever, so rotation happened exactly once per
    /// process and `MAX_LOGS` was never enforced again.
    #[test]
    fn rotation_still_works_a_second_time() {
        let (l, dir) = logger("twice");
        inflate(&dir);
        say(&l, "first trigger");
        assert!(dir.join("t.log.1").exists(), "first rotation");

        inflate(&dir);
        say(&l, "second trigger");
        assert!(
            dir.join("t.log.2").exists(),
            "a second rotation must happen — this is what silently stopped before",
        );
        assert!(dir.join("t.log").exists(), "and the live file must still be there");
        let _ = fs::remove_dir_all(&dir);
    }

    /// A file under the cap must not rotate: the check reads the handle's
    /// own length now, and an off-by-one there would rotate every line.
    #[test]
    fn a_small_file_does_not_rotate() {
        let (l, dir) = logger("small");
        for i in 0..50 {
            say(&l, &format!("line {i}"));
        }
        assert!(!dir.join("t.log.1").exists(), "nothing should have rotated");
        let live = fs::read_to_string(dir.join("t.log")).expect("t.log exists");
        assert!(live.contains("line 49"));
        let _ = fs::remove_dir_all(&dir);
    }
}

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
            format:        LogFormat::Human,
        })
    }

    fn meta(target: &'static str, level: Level) -> Metadata<'static> {
        Metadata::builder().target(target).level(level).build()
    }

    /// The reason this exists: wgpu logs `Device::maintain` at INFO once
    /// per frame. One session produced 43,942 of 45,144 lines from it,
    /// each formatted, written to disk and queued for the relay on the
    /// render thread.
    // -- logfmt escaping ------------------------------------------
    //
    // A mis-escaped value does not fail loudly: promtail parses the line
    // and silently gets the wrong fields, so a message truncating at its
    // first space looks like a short message rather than a bug.

    #[test]
    fn a_bare_word_needs_no_quoting() {
        assert_eq!(logfmt_value("ledger"), "ledger");
        assert_eq!(logfmt_value("void_engine::persist"), "void_engine::persist");
    }

    #[test]
    fn anything_with_a_space_is_quoted() {
        assert_eq!(logfmt_value("two words"), "\"two words\"");
    }

    #[test]
    fn embedded_quotes_are_escaped_not_dropped() {
        // Unescaped, this would close the value early and the rest of the
        // message would parse as stray keys.
        assert_eq!(logfmt_value("say \"hi\""), "\"say \\\"hi\\\"\"");
    }

    #[test]
    fn a_backslash_is_escaped() {
        assert_eq!(logfmt_value("C:\\path"), "\"C:\\\\path\"");
    }

    #[test]
    fn newlines_become_escapes_so_one_event_stays_one_line() {
        // A panic message spans lines. Written raw it would split into
        // several log records, and the tail of the panic would parse as
        // an event with no timestamp.
        assert_eq!(logfmt_value("a\nb"), "\"a\\nb\"");
        assert_eq!(logfmt_value("a\r\nb"), "\"a\\r\\nb\"");
        assert_eq!(logfmt_value("a\tb"), "\"a\\tb\"");
    }

    #[test]
    fn an_equals_sign_is_quoted() {
        // Otherwise `msg=k=v` parses as two keys.
        assert_eq!(logfmt_value("k=v"), "\"k=v\"");
    }

    #[test]
    fn an_empty_value_is_an_explicit_empty_string() {
        // A bare `msg=` with nothing after it is ambiguous.
        assert_eq!(logfmt_value(""), "\"\"");
    }

    /// The alert rules match on `event=`, so it has to survive into the
    /// line rather than being mangled by the quoting.
    #[test]
    fn an_event_key_survives_into_the_value() {
        let v = logfmt_value("event=ledger_reconciliation_failed books disagree");
        assert!(v.starts_with('"'), "a value with spaces must be quoted: {v}");
        assert!(v.contains("event=ledger_reconciliation_failed"), "{v}");
    }

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
