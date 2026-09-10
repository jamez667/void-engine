# void_engine — MMO readiness backlog

Sorted by severity. Derived from the full-repo audit of `a1dbc30`; every
number below was measured in release on a dev box, not estimated.

**Status (2026-09-09):** every S1-S3 item is done. Tests 146 -> 167, clippy
clean on all three feature axes, CI and hot-path budget guards in place. The
R-items below are untouched and remain multi-week projects.

Severity key:
- **S1** — correctness bug or missing safety net. Silent failure or no
  coverage on load-bearing code.
- **S2** — wrong or misleading in a way that will bite later.
- **S3** — infrastructure debt. Nothing breaks today; everything breaks
  quietly later.
- **R** — roadmap. Multi-week projects, not tasks. Design decisions
  attached.

---

## S1 — correctness

- [x] **Surface errors blank the window permanently.** `renderer/frame.rs:108`
      maps every `get_current_texture` failure to `Err(_) => return`.
      `SurfaceError::Lost` and `Outdated` are recoverable by reconfiguring;
      discarding them means a device reset, driver update, or alt-tab from
      exclusive fullscreen renders black forever with no log line.
- [x] **`ecs/world.rs` has zero tests.** Generation-based id reuse and
      "despawn wipes every component slot" are the correctness foundation the
      whole engine sits on. A stale-slot bug here silently corrupts any game
      built on it. 146 tests in the repo, none on this file.

## S2 — wrong in a way that will bite

- [x] **Overflow guard documents the wrong stride.** `renderer/frame.rs:27`
      sizes its cap as "8M verts × 32B = 256MB". `size_of::<Vertex>()` is
      **84 bytes**, so 8M verts is 672MB — the guard does not protect the
      limit it claims to.
- [x] **`query_pairs` is order-nondeterministic.** `collision.rs` dedups
      through a `HashSet<u64>` with per-process `RandomState` seeding, so pair
      order varies run to run. Breaks replay determinism before lockstep is
      even on the table. Fix: sort before returning, or use a fixed-seed
      hasher.
- [x] **Clippy exits non-zero.** `input/mod.rs:174` has `for step in 0..0`
      inside an otherwise-good regression test; `reversed_empty_ranges` is
      deny-by-default. Rewrite the test without the empty range.

## S3 — infrastructure

- [x] **No CI.** 146 tests exist and pass. Nothing enforces that they keep
      passing. Added `.github/workflows/ci.yml`: build, test and clippy across
      all three feature axes (default / `net` / headless), plus the hot-path
      budgets as a separate job.

      No `cargo fmt --check` gate: rustfmt would rewrite ~586 sites and undo
      the hand-aligned style this codebase uses throughout. Adopting it is a
      reasonable call but belongs in its own commit, not a CI gate that turns
      `main` red immediately.
- [x] **No benchmarks.** The audit's numbers have no regression protection.
      Lock in the two that matter: ECS query overhead and collision
      rebuild+query cost.

---

## R — roadmap (not tonight)

Each of these is a project with design decisions attached. Ordered by what
unblocks the most downstream work.

- [ ] **R1 — Split the client out of the loop.** Feature-gate `winit`/`wgpu`
      behind a `client` feature; add a `SimCtx` without a renderer and a
      `run_headless` driver over the existing `Timestep`.

      *Blocked on nothing; breaks the `App`/`EngineCtx`/`run` API.*

      **Decision made (2026-09-09): tick rate becomes a parameter,
      server default 30Hz.** This resolves the contradiction already in the
      tree — `time.rs` hardcodes `FIXED_DT = 1.0/60.0` while
      `net/interp.rs` defaults `tick_hz: 30.0`. Client render stays free.

      Note: `cargo tree --no-default-features` currently still pulls
      `wgpu v22.1.0` and `winit v0.30.13` — the README's headless claim is
      not accurate until this lands.

- [ ] **R2 — Make entities serializable, then persistent.** Component registry
      with stable ids, versioned schema, world snapshot/restore, then a store.
      Today exactly one `Serialize` derive exists in 13.5k lines, for
      keybinds; no component derives it.

      *Open decision: is the authority a database of record, or a
      periodically-checkpointed sim? This changes the whole design.*

- [ ] **R3 — Replication with interest management.** Per-client AoI query →
      relevancy set → baseline+delta snapshot → quantized, bit-packed encode →
      the existing `net/chunk.rs` `send_chunked`.

      *`SpatialGrid` is already the right structure but is referenced only
      inside `collision.rs`. Without AoI, per-client bandwidth is O(world
      entities) — the hard wall between a session game and an MMO.*

- [ ] **R4 — Chunk the world; incremental broadphase.** Replace the flat
      `TileGrid` (`Vec<T>` of `w*h`) with a chunk table keyed by `Sector2D`,
      generated on demand. Give `SpatialGrid` `update`/`remove` so it stops
      reallocating every tick (50k colliders = 26.4ms rebuild).

      *The terrain noise is already pure `f(position, seed)` and streams
      perfectly. It is held back by the container, not its own design.*

- [ ] **R5 — Rebuild the client draw path.** Instancing (every draw is
      currently `0..1`), texture atlas or `D2Array` (one 1×1 white pixel
      exists today), SDF glyph atlas (text is one quad per lit font pixel —
      72KB per 10-char nameplate), depth buffer or explicit layer sort, and a
      single clustered light pass (`frame.rs:382` opens a fullscreen pass
      *per light*, `MAX_LIGHTS_PER_FRAME = 384`).

      *A rewrite of the draw path, not an optimization pass. Independent of
      the server work and safely deferrable.*

---

## Measured baselines

Release build, dev box. These are what the benchmarks should defend. The
"At audit" column is the original `a1dbc30` measurement, kept so a
regression is recognisable as one.

**ECS query cost (done, 2026-09-09).** Not an R-item, but flagged in the
audit as structural and cheapest to fix before R3 builds per-client
snapshot queries on the same iterator. `iter`/`iter2`/`iter_mut` each
collected into a heap-allocated `Vec` of raw pointers per call — the
pointers existed only to escape a borrow-checker conflict, and profiling
put ~83% of query time in the collect versus ~7% in the actual iteration.
Making them lazy (splitting the `alive`/`generations` borrow from the
storage borrow) removed both the allocation and the pointers: **9.7×
faster, no call-site changes, no storage rewrite, `unsafe` count in the
ECS drops to zero.**

| Measurement | Value | At audit |
| --- | --- | --- |
| ECS `iter2`, 50k entities × 20 systems | **1.17 ms/tick** | 11.32 ms |
| Same workload over contiguous arrays | 0.39 ms/tick | 0.39 ms |
| ECS query overhead factor | **2.8×** | 28.7× |
| ECS `iter2`, 250k entities, 1 system | **0.43 ms/tick** | 2.73 ms |
| Collision rebuild+query, 10k colliders | 5.1 ms/tick | 4.4 ms |
| Collision rebuild+query, 50k colliders | 26.4 ms/tick |
| A*, 256×256 open grid, corner-to-corner | 14.5 ms |
| `size_of::<Vertex>()` | 84 bytes |
| Text: 10-char nameplate | 864 verts / 72 KB |
