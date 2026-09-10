# void_engine — MMO readiness backlog

Sorted by severity. Derived from the full-repo audit of `a1dbc30`; every
number below was measured in release on a dev box, not estimated.

**Status (2026-09-10):** every S1-S3 item is done, plus the ECS query-cost
fix and **R1 (headless split)**. Tests 146 -> 187 across three feature axes,
clippy clean on all of them, CI and hot-path budget guards in place.
R2-R5 remain multi-week projects.

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

- [x] **R1 — Split the client out of the loop.** Done 2026-09-10.

      `client` feature (default on) gates `wgpu` and every module that needs
      it: `renderer`, `ui`, `fx`, `text`, plus `ClientApp`/`EngineCtx`/`run`.
      A headless build links **no wgpu at all** — verified with
      `cargo tree -e normal --no-default-features`, which now reports zero
      wgpu/naga edges. The README's headless claim is finally true.

      `winit` stays unconditional by choice: `input`/`keybinds` use
      `KeyCode`/`MouseButton` as plain data (a field-less enum used as a
      bitset index), and a replay harness or keybind loader on a server still
      needs to name keys. It is pure Rust and opens no OS windowing libraries
      unless a window is created.

      **Trait split.** `App` = `init`/`fixed_update`/`can_advance` over
      `SimCtx` (world + input + dt, no renderer) and exists in every build.
      `ClientApp: App` adds `render`/`on_resize`/`window_title`. One game type
      implements both; a server implements only `App`. Simulation is written
      once and runs in both.

      **Tick rate is now per-`Timestep`**, server default 30Hz
      (`SERVER_HZ`), resolving the contradiction where `FIXED_DT` hardcoded
      60 while `net/interp.rs` assumed 30. Read `dt` from `SimCtx`, not the
      constant. `FIXED_DT` remains as the 60Hz client value.

      **Migration** (clean break, as agreed):
      - `fn fixed_update(&mut self, ctx: &mut EngineCtx)` →
        `ctx: &mut SimCtx`; same for `init`.
      - Move `render`/`on_resize`/`window_title` into
        `impl ClientApp for Game`.
      - Replace `FIXED_DT` in game logic with `ctx.dt`.
      - Servers: `default-features = false`, drive with
        `run_headless(app, || keep_going)`.

      `EngineCtx` still exists for clients (now `SimCtx` + renderer, with
      `Deref`, so `ctx.world`/`ctx.dt` keep working), but `fixed_update` no
      longer receives one — reach the renderer from `render`.

      Six integration tests in `tests/headless_server.rs` prove a server
      ticks with no window, `init` populates the world, `can_advance` defers,
      `should_run` stops the loop, and `dt` tracks the configured rate.
      `HeadlessConfig::uncapped` runs an exact tick count with a synthetic
      clock so replays and CI are reproducible.

- [ ] **R2 - Make entities serializable, then persistent.** Designed
      2026-09-10, not yet built. Full design published as an artifact; the
      decisions it fixes are recorded here so they outlive the link.

      **Decision: Postgres is the source of truth; checkpoints are a
      disposable cache.** Every durable fact lives in the DB. Checkpoints are
      a periodic binary dump of the live `World` that makes restart fast.
      Deleting every checkpoint costs startup time and nothing else. That
      invariant is what stops the two mechanisms becoming two half-truths.

      **Decision: three persistence classes**, declared per component.
      Durable (DB + checkpoint) - inventory, currency, health. Volatile
      (checkpoint only) - position, velocity, AI state. Transient (neither) -
      particles, client markers. Keeps DB write volume proportional to
      player-meaningful change rather than to world size.

      **Four constraints verified against the tree, not assumed:**
      - `TypeId` cannot key a save file - opaque 128-bit hash, no cross-build
        stability guarantee. Every component needs a hand-assigned stable id.
        Engine reserves ids below 1000.
      - RNG position is not capturable: `Pcg32 { state, inc }` are private
        and `Lcg` is a bare tuple struct. A seed-only checkpoint replays the
        stream from the beginning and diverges. Needs accessors.
      - `World` has no restore path - all four fields private, and `spawn()`
        assigns the next id rather than a requested one, so restoring through
        the public API would renumber every entity. `EntityId`'s fields are
        public, so exact reconstruction is possible with a deliberate ctor.
      - The engine creates no async runtime (zero `Runtime::new`/`block_on`),
        and `fixed_update` is sync. Making it async would infect every game's
        sim code and defeat R1's headless split.

      **Decision: the tick never awaits.** `fixed_update` appends durable
      changes to an in-memory journal; a writer thread owns the tokio runtime
      and the Postgres pool and commits them. A checkpoint may only claim
      tick N once every durable change up to N has committed, so DB and
      checkpoint can be stale relative to each other but never disagree.
      Journal needs backpressure - unbounded is a memory leak with extra
      steps.

      **Decision: binary serde format** (postcard or bincode) shared with
      R3. `net/chunk.rs` is explicitly format-agnostic, so choosing here
      settles it for both rather than growing two encodings.

      **Decision: JSONB component payloads** in Postgres, not a table per
      component - the component set is defined by the game, so a fixed
      relational schema would force an engine migration per game component.
      `shard_id` in the schema from day one so sharding is not a migration
      later, though cross-shard movement is explicitly out of scope.

      **Build order:** (1) registry + snapshot/restore, no DB; (2) on-disk
      checkpoints, atomic write, restore-on-boot; (3) storage trait + journal
      + writer thread against an in-memory store; (4) Postgres behind the
      trait, feature-gated; (5) crash matrix at every kill boundary.

      *Open risk to review before phase 1 ships: component ids are frozen by
      the first save file ever written.*

      *Sizing: `Transform2D + Velocity + Collider` is 60 B/entity, so 100k
      entities is ~5.7 MB and 1M is ~57 MB of raw component data - fine for a
      local checkpoint every few seconds, not fine through Postgres at tick
      rate.*

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
