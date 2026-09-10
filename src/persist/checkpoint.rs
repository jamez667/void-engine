//! Putting a [`Snapshot`] on disk, and getting it back after a crash.
//!
//! # The guarantee
//!
//! A checkpoint file is either wholly the old one or wholly the new one.
//! There is no state in which a reader sees half of each, however the
//! process dies. That comes from writing to a temp file, flushing it to
//! the device, and only then renaming it over the target — rename being
//! the one filesystem operation that is atomic on both POSIX and Windows.
//!
//! # What crashes cost
//!
//! Killing the process mid-write loses *the write*, never the previous
//! checkpoint. Killing it between checkpoints loses the ticks since the
//! last one. That is the whole trade: cadence buys recency, and the
//! ceiling on loss is one interval.
//!
//! # Platform notes
//!
//! `fs::rename` replaces an existing file on both platforms — verified,
//! not assumed. The POSIX belt-and-braces step of fsyncing the *parent
//! directory* after a rename (so the directory entry itself is durable)
//! fails with `EACCES` on Windows, so it is attempted and tolerated
//! rather than treated as an error.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::snapshot::{from_bytes, to_bytes, Snapshot, SnapshotError};

/// How many historical checkpoints to keep beside the current one.
///
/// More than zero because the newest file is the one most likely to be
/// bad: it is the one that was being written when something went wrong.
/// A predecessor to fall back to turns "the save is corrupt" into "we
/// lost one interval".
pub const DEFAULT_KEEP: usize = 3;

/// Where checkpoints live and how many are kept.
#[derive(Clone, Debug)]
pub struct CheckpointConfig {
    /// Directory holding the checkpoint set. Created if missing.
    pub dir: PathBuf,
    /// Base file name; rotated predecessors get `.1`, `.2`, … suffixes,
    /// matching the convention `log.rs` already uses in this crate.
    pub name: String,
    /// Historical copies to retain. `0` keeps only the current file and
    /// gives up the fallback described on [`DEFAULT_KEEP`].
    pub keep: usize,
}

impl CheckpointConfig {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into(), name: "world.ckpt".to_string(), keep: DEFAULT_KEEP }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn with_keep(mut self, keep: usize) -> Self {
        self.keep = keep;
        self
    }

    /// The current checkpoint's path.
    pub fn current(&self) -> PathBuf {
        self.dir.join(&self.name)
    }

    /// Path of the `n`th historical copy (`1` is the most recent).
    pub fn nth(&self, n: usize) -> PathBuf {
        self.dir.join(format!("{}.{n}", self.name))
    }
}

/// Why a checkpoint could not be written or read.
#[derive(Debug)]
pub enum CheckpointError {
    /// The snapshot could not be encoded or decoded.
    Snapshot(SnapshotError),
    /// A filesystem operation failed. Carries the path, because "No such
    /// file or directory" without one is a useless thing to find in a log
    /// at 3am.
    Io { path: PathBuf, source: std::io::Error },
    /// Every candidate file was unreadable or corrupt.
    NoUsableCheckpoint { tried: Vec<PathBuf> },
}

impl std::fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckpointError::Snapshot(e) => write!(f, "{e}"),
            CheckpointError::Io { path, source } => {
                write!(f, "{}: {source}", path.display())
            }
            CheckpointError::NoUsableCheckpoint { tried } => {
                write!(f, "no usable checkpoint among {} candidate(s): ", tried.len())?;
                for (i, p) in tried.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", p.display())?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for CheckpointError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CheckpointError::Snapshot(e) => Some(e),
            CheckpointError::Io { source, .. } => Some(source),
            CheckpointError::NoUsableCheckpoint { .. } => None,
        }
    }
}

impl From<SnapshotError> for CheckpointError {
    fn from(e: SnapshotError) -> Self {
        CheckpointError::Snapshot(e)
    }
}

fn io_err(path: &Path, source: std::io::Error) -> CheckpointError {
    CheckpointError::Io { path: path.to_path_buf(), source }
}

/// Write `snapshot` as the current checkpoint, atomically.
///
/// The sequence is: encode, write a temp file, `sync_all` it so the bytes
/// are on the device rather than merely in the page cache, rotate the
/// existing checkpoints, then rename the temp over the target. A crash at
/// any point leaves either the previous checkpoint or the new one intact
/// — never a partial file.
///
/// The temp name carries the process id, so two processes pointed at the
/// same directory cannot scribble over each other's in-progress write.
pub fn save(cfg: &CheckpointConfig, snapshot: &Snapshot) -> Result<(), CheckpointError> {
    fs::create_dir_all(&cfg.dir).map_err(|e| io_err(&cfg.dir, e))?;

    let bytes = to_bytes(snapshot)?;

    let tmp = cfg.dir.join(format!(
        "{}.{}.tmp",
        cfg.name,
        std::process::id(),
    ));

    {
        let mut f = fs::File::create(&tmp).map_err(|e| io_err(&tmp, e))?;
        f.write_all(&bytes).map_err(|e| io_err(&tmp, e))?;
        // Without this the rename can land while the contents are still
        // only in the page cache, which after a power loss yields an
        // atomically-renamed file full of zeroes.
        f.sync_all().map_err(|e| io_err(&tmp, e))?;
    }

    rotate(cfg);

    fs::rename(&tmp, cfg.current()).map_err(|e| io_err(&tmp, e))?;

    // Make the directory entry itself durable. POSIX only: this fails
    // with EACCES on Windows, where a directory is not openable as a
    // file, so a failure here is not an error.
    if let Ok(d) = fs::File::open(&cfg.dir) {
        let _ = d.sync_all();
    }

    Ok(())
}

/// Shuffle existing checkpoints down one slot, dropping the oldest.
///
/// Best-effort by design, exactly as `log.rs` rotates its files: a
/// missing predecessor is the normal case on the first few saves, and a
/// rotation that cannot happen must not prevent the new checkpoint from
/// being written.
fn rotate(cfg: &CheckpointConfig) {
    if cfg.keep == 0 {
        return;
    }
    for i in (1..cfg.keep).rev() {
        let _ = fs::rename(cfg.nth(i), cfg.nth(i + 1));
    }
    let _ = fs::rename(cfg.current(), cfg.nth(1));
}

/// Load the newest usable checkpoint.
///
/// Tries the current file, then each historical copy in turn. A corrupt
/// or truncated file is skipped rather than fatal — the newest file is
/// precisely the one a crash was most likely to damage, and falling back
/// costs one interval instead of the whole world.
///
/// Returns `Ok(None)` when the directory holds no checkpoint at all,
/// which is the ordinary first-boot case and not an error.
pub fn load(cfg: &CheckpointConfig) -> Result<Option<Snapshot>, CheckpointError> {
    let mut tried = Vec::new();
    let mut found_any = false;

    for path in candidates(cfg) {
        match fs::read(&path) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => {
                log::warn!("checkpoint {}: {e}", path.display());
                found_any = true;
                tried.push(path);
            }
            Ok(bytes) => {
                found_any = true;
                match from_bytes(&bytes) {
                    Ok(s) => return Ok(Some(s)),
                    Err(e) => {
                        // Worth a loud line: falling back is correct, but
                        // silently doing so hides a systematic problem.
                        log::warn!(
                            "checkpoint {} unusable ({e}); trying an older one",
                            path.display(),
                        );
                        tried.push(path);
                    }
                }
            }
        }
    }

    if found_any {
        Err(CheckpointError::NoUsableCheckpoint { tried })
    } else {
        Ok(None)
    }
}

/// Current file first, then historical copies newest-first.
fn candidates(cfg: &CheckpointConfig) -> Vec<PathBuf> {
    std::iter::once(cfg.current())
        .chain((1..=cfg.keep).map(|i| cfg.nth(i)))
        .collect()
}

/// Delete every checkpoint file in the set.
///
/// For tests and for a deliberate "start fresh" path. Best-effort: a file
/// that is already gone is success.
pub fn clear(cfg: &CheckpointConfig) {
    for path in candidates(cfg) {
        let _ = fs::remove_file(path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist::{capture, register_engine_components, Registry, RngStreams};
    use crate::World;

    /// Each test gets its own directory, so a failure in one cannot
    /// cascade into another through leftover files.
    fn temp_dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir()
            .join("void_engine_ckpt_tests")
            .join(format!("{tag}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    fn snapshot_with(tick: u64) -> Snapshot {
        let mut reg = Registry::new();
        register_engine_components(&mut reg).unwrap();
        let mut w = World::new();
        let e = w.spawn();
        w.insert(e, crate::components::Transform2D {
            pos: glam::DVec2::new(tick as f64, 0.0),
            rot: 0.0,
        });
        capture(&w, &reg, tick, &RngStreams::new()).unwrap()
    }

    #[test]
    fn save_then_load_round_trips() {
        let cfg = CheckpointConfig::new(temp_dir("round_trip"));
        save(&cfg, &snapshot_with(7)).unwrap();
        let loaded = load(&cfg).unwrap().expect("a checkpoint was just written");
        assert_eq!(loaded.tick, 7);
        let _ = fs::remove_dir_all(&cfg.dir);
    }

    /// First boot is not an error condition.
    #[test]
    fn loading_an_empty_directory_yields_none() {
        let cfg = CheckpointConfig::new(temp_dir("empty"));
        assert!(load(&cfg).unwrap().is_none());
    }

    #[test]
    fn saving_creates_the_directory() {
        let cfg = CheckpointConfig::new(temp_dir("mkdir").join("nested").join("deeper"));
        save(&cfg, &snapshot_with(1)).unwrap();
        assert!(cfg.current().exists());
        let _ = fs::remove_dir_all(&cfg.dir);
    }

    /// The newest save wins, and predecessors are kept beside it.
    #[test]
    fn successive_saves_rotate() {
        let cfg = CheckpointConfig::new(temp_dir("rotate")).with_keep(3);
        for tick in 1..=4 {
            save(&cfg, &snapshot_with(tick)).unwrap();
        }
        assert_eq!(load(&cfg).unwrap().unwrap().tick, 4, "current must be newest");

        // .1/.2/.3 hold 3, 2, 1 respectively.
        for (slot, want) in [(1, 3), (2, 2), (3, 1)] {
            let bytes = fs::read(cfg.nth(slot)).expect("rotated copy missing");
            assert_eq!(from_bytes(&bytes).unwrap().tick, want, "slot .{slot}");
        }
        let _ = fs::remove_dir_all(&cfg.dir);
    }

    #[test]
    fn retention_drops_the_oldest() {
        let cfg = CheckpointConfig::new(temp_dir("retain")).with_keep(2);
        for tick in 1..=5 {
            save(&cfg, &snapshot_with(tick)).unwrap();
        }
        assert!(!cfg.nth(3).exists(), "keep=2 must not retain a third copy");
        let _ = fs::remove_dir_all(&cfg.dir);
    }

    /// The property the whole module exists for: a damaged newest file
    /// costs one interval, not the world.
    #[test]
    fn a_corrupt_current_falls_back_to_the_predecessor() {
        let cfg = CheckpointConfig::new(temp_dir("fallback"));
        save(&cfg, &snapshot_with(1)).unwrap();
        save(&cfg, &snapshot_with(2)).unwrap();

        // Simulate a torn write: truncate the current checkpoint.
        fs::write(cfg.current(), b"\x00\x01\x02 not a snapshot").unwrap();

        let loaded = load(&cfg).unwrap().expect("must fall back");
        assert_eq!(loaded.tick, 1, "should have loaded the predecessor");
        let _ = fs::remove_dir_all(&cfg.dir);
    }

    /// An empty file is a plausible crash artefact (created, never
    /// written) and must be skipped like any other corruption.
    #[test]
    fn an_empty_current_falls_back() {
        let cfg = CheckpointConfig::new(temp_dir("empty_file"));
        save(&cfg, &snapshot_with(9)).unwrap();
        save(&cfg, &snapshot_with(10)).unwrap();
        fs::write(cfg.current(), b"").unwrap();

        assert_eq!(load(&cfg).unwrap().unwrap().tick, 9);
        let _ = fs::remove_dir_all(&cfg.dir);
    }

    /// If nothing is readable, say so with the paths tried rather than
    /// returning `None`, which would look like a fresh install and
    /// silently start a new world over the top of a broken one.
    #[test]
    fn all_corrupt_is_an_error_not_a_fresh_start() {
        let cfg = CheckpointConfig::new(temp_dir("all_bad")).with_keep(1);
        save(&cfg, &snapshot_with(1)).unwrap();
        save(&cfg, &snapshot_with(2)).unwrap();
        fs::write(cfg.current(), b"junk").unwrap();
        fs::write(cfg.nth(1), b"junk").unwrap();

        // Deliberately not formatting the Ok value: `Snapshot` holds the
        // encoded component columns, so a `Debug` of it in panic output
        // would be pages of bytes rather than a diagnosis.
        match load(&cfg) {
            Err(CheckpointError::NoUsableCheckpoint { tried }) => {
                assert_eq!(tried.len(), 2, "both candidates should be reported");
            }
            Err(e) => panic!("expected NoUsableCheckpoint, got error: {e}"),
            Ok(Some(s)) => panic!("expected failure, but a snapshot at tick {} loaded", s.tick),
            Ok(None) => panic!("expected failure, but the set looked empty"),
        }
        let _ = fs::remove_dir_all(&cfg.dir);
    }

    /// A crash mid-write leaves a `.tmp` behind. It must not be mistaken
    /// for a checkpoint, and the previous one must still load.
    #[test]
    fn a_stray_temp_file_is_ignored() {
        let cfg = CheckpointConfig::new(temp_dir("stray_tmp"));
        save(&cfg, &snapshot_with(3)).unwrap();
        fs::write(cfg.dir.join(format!("{}.99999.tmp", cfg.name)), b"partial").unwrap();

        assert_eq!(load(&cfg).unwrap().unwrap().tick, 3);
        let _ = fs::remove_dir_all(&cfg.dir);
    }

    #[test]
    fn clear_removes_the_whole_set() {
        let cfg = CheckpointConfig::new(temp_dir("clear"));
        for tick in 1..=3 {
            save(&cfg, &snapshot_with(tick)).unwrap();
        }
        clear(&cfg);
        assert!(load(&cfg).unwrap().is_none());
        let _ = fs::remove_dir_all(&cfg.dir);
    }

    /// `keep = 0` is legal: only the current file, no fallback.
    #[test]
    fn keep_zero_retains_only_the_current_file() {
        let cfg = CheckpointConfig::new(temp_dir("keep0")).with_keep(0);
        save(&cfg, &snapshot_with(1)).unwrap();
        save(&cfg, &snapshot_with(2)).unwrap();
        assert_eq!(load(&cfg).unwrap().unwrap().tick, 2);
        assert!(!cfg.nth(1).exists());
        let _ = fs::remove_dir_all(&cfg.dir);
    }

    /// An IO error names the file it happened to. A bare "access denied"
    /// with no path is the log line everyone hates.
    #[test]
    fn io_errors_carry_their_path() {
        let e = io_err(Path::new("some/where.ckpt"),
            std::io::Error::new(std::io::ErrorKind::PermissionDenied, "nope"));
        assert!(e.to_string().contains("some/where.ckpt"), "got {e}");
    }
}
