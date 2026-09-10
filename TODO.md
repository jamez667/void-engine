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

- [ ] **R2 - Persistence and ledger.** Designed 2026-09-10 (rev 3), not
      built. Full design published as an artifact; decisions recorded here so
      they outlive the link.

      **Rev 2 supersedes rev 1.** Rev 1 stored balances as mutable values
      written transactionally and logged changes beside them. That cannot
      detect duping: recording *that* a value changed says nothing about why
      or from what. Anything of value is now derived from an append-only
      ledger - a balance is SUM(entries), never a float anything can write.

      **Evidence this is the right correction, from void-claim (shipped):**
      - Its `credit_ledger` write is fire-and-forget - `log::warn!` on
        failure, bounded `mpsc::channel(10_000)`. A minute of Scylla downtime
        means those credit movements never existed.
      - `Wallet { credits: f64 }` is mutated directly at ~8 sites, several
        emitting no event at all: deaths.rs:153 (respawn fee), deaths.rs:388
        (NPC reward), missions.rs:61 (payout), on_foot_actions.rs:43,
        ingress.rs:111.
      - The code documents its own gap: "On-foot purchases skip the
        CreditEvent pipeline", plus three `let mut throwaway:
        Vec<CreditEvent>` sites that exist to discard events.
      - `PlayerRecord` carries a comment that SurrealDB rejected
        `#[serde(flatten)]` "which silently broke every save" - inventory not
        persisting, no error. Exactly the failure class to design out.
      - Worth keeping from it: the `reason` taxonomy (buy_kind, sell_kind,
        debt_garnish, station_repair, weapon_upgrade_cannon ...), per-player
        time-bucketed partitioning, and a worker thread for genuinely
        non-critical streams. Positions and chat *should* be fire-and-forget.

      **Decision: three tiers, by Cargo feature** (structural, not runtime,
      following the existing audio/client/net pattern). Default = no
      persistence, as today. `persist` = registry + snapshot/restore +
      on-disk checkpoints, no DB. `ledger` = the above plus append-only value
      tracking, double-entry, idempotency, reconciliation, Postgres; implies
      `persist`. Most games never need tier 3.

      **Decision: classification is Ledgered / Volatile / Transient.** Rev
      1's "Durable" class - a mutable value written transactionally - is
      deleted; it was the duping surface. Ledgered = currency, items,
      structures, claims. Volatile = position, velocity, health (checkpoint
      only). Transient = particles, client markers (never serialised).

      **Decision: every entry has a counterparty.** Value moves, never
      appears - from a player, a shop, a loot table, or an explicit
      mint/burn account. A transfer is two rows in one transaction summing to
      zero. Duping becomes *detectable*: if the ledger does not sum to zero
      per asset, something created value outside the API.

      **Decision: idempotency key is UNIQUE in the DB**, derived from
      (session, client_seq, action). A retried packet - the classic dupe
      vector - inserts nothing the second time, enforced by the database
      rather than by game logic.

      **Decision: fixed-point, not f64.** void-claim uses `credits: f64`;
      float rounding makes exact reconciliation impossible. Currency is an
      integer of minor units.

      **Decision: the API is unbypassable.** No `pub credits` field exists.
      `ledger.transfer(..) -> Result<Receipt, LedgerError>` is the only way
      to move value; `Wallet` keeps a private cached balance with no setter,
      updated only by applying a committed receipt. Contrast void-claim's
      `try_spend -> bool`, which reads then mutates in two steps - a
      time-of-check/time-of-use shape.

      **Decision: the tick never awaits, and pending debits are reserved.**
      `fixed_update` is sync and stays so. A transfer returns a pending
      receipt; the reservation prevents spending the same credits twice
      while a commit is in flight. A checkpoint may never claim a tick ahead
      of the ledger's committed watermark - restoring a checkpoint showing a
      purchase the ledger never recorded *is* a dupe.

      **Decision: entity-level JSONB, not per-component rows.** Rev 1 had one
      row per (entity, component), so a player with 40 inventory slots was 40
      rows on login. One row per entity matches how it is read.
      `ledger_entries` is append-only, enforced by grants, not convention -
      corrections are compensating entries.

      **Reconciliation is the part that catches cheating:** zero-sum per
      asset; cached balance vs SUM(entries) per account; rate/shape anomalies
      by reason; item conservation (a unique item has exactly one holder).
      An audit trail nobody checks is a log file.

      **Decision (rev 3): component identity is an explicit string, not a
      number and not the Rust type path.** `registry.register::<Transform2D>
      ("transform2d", Persist::Volatile)`. Rev 2 said numeric ids were frozen
      by the first save file ever written; that framing was wrong. What a
      save file commits to is a *name*, and a name you assign is data you
      control rather than a fact about your source tree - so renaming a Rust
      type or moving a module must never change what a save file says.

      Not `TypeId`: opaque 128-bit hash, no cross-build guarantee, changes
      silently on recompile. Not `type_name`: `core::any` says its output "is
      not specified", is "intended for diagnostic use", and "may change
      between versions of the compiler" - and it embeds the module path, so
      moving a component into a submodule silently changes every key, the
      same shape as void-claim's serde-flatten bug. Not integers: they need a
      taken-numbers registry, two branches both picking 1047 collide silently
      at merge, and a save file of bare numbers is hostile to debug; a
      duplicate *name* fails loudly at startup.

      Names intern to a `u16` at startup, so checkpoints and R3's per-entity
      wire encoding pay two bytes, not a string - the name appears once per
      component type in a storage header, never per entity.
      `registry.rename("old", "new")` is applied by the load path, so a bad
      name is recoverable, which a wrong integer id is not. Precedent:
      void-claim's `persist.rs` keys station records by a stable id, with a
      test asserting "records must leave keyed by the stable id, never the
      entity index" - the same lesson one layer up.

      **Three constraints still hold from rev 1, verified against the tree:**
      RNG position is not capturable (`Pcg32 { state, inc }` private, `Lcg` a
      bare tuple struct - seed-only checkpoints replay from the start and
      diverge); `World` has no restore path (all fields private, `spawn()`
      cannot be told which id to assign); the engine creates no async runtime
      and `fixed_update` is sync.

      **Build order:** (1) registry + snapshot/restore, no DB; (2) on-disk
      checkpoints - `persist` tier ships, useful alone; (3) ledger core
      in-memory, semantics proven without IO; (4) Postgres behind the ledger
      trait, feature-gated; (5) reconciliation + crash matrix, exit criterion
      being that an injected synthetic dupe is caught by reconciliation
      rather than by a player noticing.

      *Open risks: a component's registry name is fixed once written to a
      save file (downgraded from rev 2's "frozen forever" - it lives in the
      registry, not the type, so refactoring is free and `rename` recovers a
      bad choice); ledger volume needs partitioning and an archive-not-delete
      retention policy; a reconciliation mismatch must page a human, not
      append to a log nobody reads.*

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
