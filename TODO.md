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
      it: `renderer`, `ui`, `fx`, `text`, plus `ClientApp`/`run`.
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

      `EngineCtx` was introduced here as `SimCtx` + renderer with a `Deref`,
      but **it was never constructed** — `resumed` built a plain `SimCtx` and
      the renderer was unreachable from `init`, so the type was dead API that
      the docs described as live. A review caught it and it has since been
      removed. Reach the renderer from `render`.

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

      **Build order:** (1) registry + snapshot/restore, no DB - **DONE
      2026-09-10**; (2) on-disk checkpoints - **DONE 2026-09-10**, the
      `persist` tier now ships and is useful alone; (3) ledger core in-memory, semantics proven without IO -
      **DONE 2026-09-10**; (4)
      Postgres behind the ledger trait, feature-gated - **DONE 2026-09-10**; (5) reconciliation +
      crash matrix, exit criterion being that an injected synthetic dupe is
      caught by reconciliation rather than by a player noticing.

      **Phase 1 shipped (2026-09-10).** `persist` feature, off by default.
      34 new tests; four CI axes now (default / net / headless / persist).

      - `persist::registry` - names, not integers or `TypeId`. Interned to
        `NameId(u16)`. `register` carries the serde bounds;
        `register_transient` deliberately does not, so a particle needs no
        serde to be classified. Duplicate names and one-type-two-names fail
        at startup. `rename` aliases make a bad name recoverable. `audit`
        catches a component present in the world but never classified.
      - `persist::snapshot` - `capture`/`restore`, `to_bytes`/`from_bytes`.
        Walks the registry, not the world's storages, so an unregistered
        component is not silently saved. Columns keep their holes so slot
        indices survive; the allocator (`generations`/`alive`/`free_list`)
        is captured verbatim rather than replayed through `spawn()`, which
        would renumber every entity and dangle saved cross-references.
      - `Pcg32::state()`/`from_state()` - checkpoints capture stream
        *position*, not seed. `Lcg` needed nothing: its state is a public
        field, pinned by a test so a future refactor cannot quietly break it.
      - Codec design forced by bincode: `serialize` is generic over
        `T: Serialize` and cannot cross a `&dyn` boundary, so `Entry` holds
        monomorphised `encode`/`decode`/`make_storage` fn pointers captured
        at `register` time.
      - Rejections are errors, never silent: unknown component, schema
        mismatch, future format version, inconsistent allocator arrays.
      - `World` gained a doc-hidden persistence seam - `allocator_state`,
        `restore_allocator`, `column_for`, `install_column`,
        `present_component_types`. Engine seam, not game API.
      - serde derives on engine components are `cfg_attr`-gated on
        `persist`, and `persist` enables `glam/serde`, so a non-saving game
        compiles none of it.

      **Phase 2 shipped (2026-09-10).** `persist::checkpoint` - 16 more
      tests, 237 total across four axes.

      - Atomic write: encode, write `<name>.<pid>.tmp`, `sync_all` so the
        bytes reach the device rather than the page cache, rotate, then
        rename over the target. A checkpoint file is wholly the old one or
        wholly the new one, however the process dies.
      - Two platform facts verified rather than assumed: `fs::rename` does
        replace an existing file on Windows, and fsync-ing the parent
        directory (the POSIX durability step for the directory entry)
        fails `EACCES` on Windows - so it is attempted and tolerated, never
        fatal.
      - Retention keeps `DEFAULT_KEEP = 3` predecessors as `.1`/`.2`/`.3`,
        matching `log.rs`'s existing rotation convention. The newest file
        is the one a crash was most likely to damage, so a fallback turns
        "the save is corrupt" into "we lost one interval".
      - `load` tries current, then each predecessor; a corrupt or empty
        file is skipped with a warning. If *nothing* is usable that is an
        error carrying every path tried - never `Ok(None)`, which would
        look like a fresh install and start a new world over a broken one.
      - `Ok(None)` is reserved for a genuinely empty directory: first boot
        must not be an error.
      - Temp files carry the pid, so two processes sharing a directory
        cannot scribble over each other's in-progress write; a stray
        `.tmp` left by a crash is never mistaken for a checkpoint.
      - `tests/crash_recovery.rs` is the exit criterion, driven through
        the public API: a world survives a restart, a headless server
        resumes at tick 30 rather than restarting at 10, a torn write
        costs one interval, and a first boot with no checkpoint starts
        fresh. CI asserts both this and the phase 1 count so a broken cfg
        gate cannot turn either into a silent no-op.

      *Deferred: no cadence policy in the engine - a game decides when to
      call `save`, because how much loss is acceptable is a game question,
      not an engine one. `tests/crash_recovery.rs` shows the every-N-ticks
      shape.*

      **Phase 3 shipped (2026-09-10).** `persist::ledger` behind a new
      `ledger` feature (tier 3, implies `persist`). 27 more tests, 264
      total across five axes.

      - A balance is *derived*, never stored: `Ledger::balance` sums the
        log, and there is no setter, so no path changes value without
        leaving a record. This is the structural answer to void-claim's
        advisory `CreditEvent` pipeline that eight sites simply skipped.
      - Every entry has a counterparty. `Account::Mint`/`Burn` are where
        value enters and leaves, so loot and repair costs still balance.
        `audit_zero_sum` is the dupe detector: a non-zero sum per asset
        means value was created outside the API.
      - `IdemKey` from (session, client_seq, action). A retried packet
        returns the *original* receipt and moves nothing - and succeeds
        even if the player has since spent the money, because the original
        transfer already happened. Checked before balances for that reason.
      - `Amount` is `i64` minor units. Verified the ceiling: 10M accounts
        at 100bn display units each fits with orders of magnitude spare,
        and audits accumulate in `i128` so reconciliation cannot itself
        overflow. Confirmed f64 is exact at 100bn but *not* at 2^53+1 -
        precisely why void-claim's `credits: f64` cannot reconcile exactly.
      - Direction lives in `from`/`to`, never the sign, so a negative
        amount cannot quietly reverse a transfer. Zero and negative are
        refused outright.
      - Reservations hold funds against in-flight commits, so the same
        credits cannot be spent twice while a phase-4 write is in the air.
        `available` = balance - reserved, and that is what a spend checks.
      - Items reuse the same machinery: a unique item is an asset with a
        total supply of one, and `total_minted` proves it.
      - Both audits are proven non-vacuous: one test forges an entry the
        way an exploit would and asserts it is caught; another drifts the
        balance cache and asserts the same.
      - `tests/ledger_properties.rs` is the exit criterion - random
        sequences of transfers, retries and reservations over a seeded LCG
        (no proptest dependency; a failure prints its seed to replay),
        asserting after *every* step that the books balance and the cache
        matches. Plus 3000-step depth, heavy retry pressure, and a unique
        item traded 500 times that is never duplicated or lost.

      *Deferred to phase 4: the ledger is in memory. Durability, the
      writer thread, the acked-tick watermark and Postgres come next; the
      invariants proven here are what that has to preserve.*

      **Phase 4 shipped (2026-09-10).** `persist::ledger_pg` behind a
      `ledger-pg` feature (implies `ledger`). 8 more tests, 272 total
      across six axes.

      - `LedgerStore` is the contract both backends satisfy, and it is
        deliberately *sync*: `fixed_update` is sync and making it async
        would infect every game's simulation code. The durable backend
        therefore validates against the in-memory core, applies to
        balances, returns a **pending** receipt, and journals the write.
      - Postgres is durability, not a second implementation of the rules.
        Two implementations of "can this player afford it" would
        eventually disagree, and the disagreement would be a dupe.
      - Schema proven against real Postgres 17.9 before any Rust depended
        on it: `UNIQUE (idem_key, side)` rejected a replayed insert
        (SQLSTATE 23505), the `side` CHECK rejected a bad value, and
        rollback left zero rows. The store detects a retry by SQLSTATE
        code, never by matching message text.
      - Replay on open rebuilds the core from `ledger_entries` through the
        normal `transfer` path, so the balance cache is rebuilt by exactly
        the code that maintains it. A separate rebuild path would be a
        second implementation that can disagree.
      - **Bug found and fixed by the tests:** `flush` originally waited
        for the *journal* to empty, but the writer drains work into a
        local batch before committing, so there was a window where the
        queue was empty and nothing was durable. `flush` returned success
        during it and every restart test reopened onto zero rows. Now an
        `accepted`/`committed` counter pair makes that window
        unobservable. This is precisely the class of bug the phase existed
        to surface.
      - Backpressure: past `max_journal` transfers are refused with
        `WriterBehind` rather than growing an unbounded queue. A dead
        writer sets `WriterFailed` and every subsequent transfer is
        refused - continuing to accept value movements with no way to
        persist them is silent data loss wearing the costume of a working
        ledger.
      - `tests/ledger_pg.rs` runs against a real database, each test in
        its own schema via `search_path`. It skips (not fails) without
        `VOID_ENGINE_PG_URL`, so a machine with no database still passes
        `cargo test`; CI runs it against a Postgres service container and
        asserts the count so a misconfigured env cannot look like a pass.

      **Phase 5, first slice (2026-09-10): surviving a database outage.**
      3 more tests, 281 total across six axes.

      This started as a question about policy and turned out not to be
      one. If the durable store is unreachable, every value movement the
      sim accepts is unbacked, so "degrade to read-only" and "queue and
      hope" are not real options - the only correct behaviour is to stop
      accepting value movements. Presenting that as a product decision was
      a mistake.

      The real defect was the opposite of laxity: the writer died on the
      *first* error with no retry, so a three-second failover permanently
      bricked the ledger until a restart. Now:

      - `is_transient` classifies by SQLSTATE. A constraint or schema
        fault (unique, check, undefined table/column, datatype) is fatal -
        retrying `UNDEFINED_TABLE` forever hides a real problem behind an
        infinite loop. Anything with no `DbError` at all means the server
        never answered: a closed socket, a timeout. Retryable.
      - Exponential backoff, capped, bounded by `max_retries`, with a
        reconnect when the connection is closed.
      - A **degraded** state distinct from **failed**. Both refuse writes,
        because value the store cannot accept must not be accepted; but
        degraded clears itself once the connection returns, so a failover
        no longer needs an operator. `WriterHealth` exposes the three
        states for a health check.
      - **Reconciliation before trusting the log again.** On recovery the
        writer sums every delta per asset in the database and requires
        exactly zero before clearing the flag. An outage is precisely when
        things diverge, and a divergence is the dupe signal this tier
        exists to catch. A failed reconciliation is terminal, not
        retryable.
      - Reads stay available throughout - refusing them would break the
        exact tooling an operator needs mid-outage.

      `tests/ledger_outage.rs` stops and starts a real container
      mid-flight: writes refused while down, the ledger healing itself
      without a restart, and reads working throughout. CI runs them in a
      job that owns its own docker, since a `services:` container cannot
      be stopped from inside the job.

      **Phase 5 complete (2026-09-10).** 4 more tests, 293 total across
      six axes. R2 is done.

      Nine boundaries were already covered by earlier phases: process
      death between checkpoints, a torn checkpoint write, first boot, the
      database vanishing mid-flight, recovery, reads during an outage, a
      rolled-back transaction, a full journal, and a retry across a
      restart. `tests/crash_matrix.rs` closes the four that were not:

      - **The exit criterion.** A row forged directly into Postgres -
        value with no counterparty, the way a bad migration, a direct
        edit or a compromised service would create it - is caught by
        `reconcile_now`, naming the offending asset. A dupe found by a job
        rather than by a player noticing their balance is wrong.
      - Half a double-entry pair reaching disk is caught the same way. A
        transaction makes this unreachable through the writer, so it is
        forged directly; the point is that the audit does not depend on
        the writer having behaved.
      - A checkpoint may not claim a tick above the watermark. A transfer
        accepted but not flushed does not advance `acked_tick`, so a
        checkpoint taken mid-flight records the durable value, not the
        sim's current tick. Restoring one that showed an uncommitted
        purchase would be a dupe: the player keeps the goods, the payment
        never happened.
      - A fatal fault terminates rather than retrying forever. Dropping
        the table gives `UNDEFINED_TABLE`, which the classifier calls
        fatal; with `max_retries` set to 10,000 the writer still stops,
        proving the *classifier* stopped it and not the retry budget. And
        it stays stopped - no self-healing from an unrecoverable fault.

      `reconcile_now` is public because an operator wants "check the
      books" as a command, not only as something the writer does after an
      outage. It opens its own short-lived connection so it works while
      the writer is busy or degraded.

      Reconciliation failures now shout through the log pipeline
      (`event=ledger_reconciliation_failed`), so the earlier risk about
      having nowhere to page is closed - see the promtail config in
      `docs/`.

      **Reservation lifecycle (2026-09-10).** 9 more tests, 302 total. The
      last open item from the review, and a real funds-lock bug rather
      than a doc mismatch.

      `transfer` never consulted a reservation, so a hold taken and not
      explicitly released depressed `available` for the life of the
      process: the balance moved, the hold stayed. The doc claimed release
      happened "when the transfer commits or is abandoned", which was
      never true.

      - `TransferRequest.spends: Option<ReservationId>` names the hold a
        transfer consumes. Explicit rather than matched by amount: two
        holds on one account for the same sum are indistinguishable, and
        picking the wrong one is a silent error on money. A mismatched
        account/asset, a too-small hold, or an unknown id are all refused.
      - Released on success **only**. A refused transfer leaves the hold
        standing so the caller can retry against it; releasing on entry
        would turn one failed attempt into lost protection.
      - `reserve` takes `expires_after_tick`. Ticks not wall-clock: a
        replay must reproduce exactly, and a stalled server should not
        have holds lapse early because real time kept moving. This is the
        backstop for a caller that panics or disconnects - the one case
        spend-and-release cannot cover.
      - **Expiry bites on read, not on a sweep.** `available_at` ignores
        elapsed holds immediately, so a server that never calls
        `expire_reservations` leaks a little memory but never a player's
        money. The sweep only reclaims space.
      - A hold guarantees its own spend: the reserved transfer checks
        against availability *plus* its own hold, so other traffic cannot
        consume the funds underneath it.
      - The property test now staggers deadlines so some holds lapse
        naturally and some are released explicitly, with the zero-sum
        invariant asserted after every step either way.

      *Mutation-tested: removing the release line makes
      `a_spent_reservation_is_released` fail with the hold still counted.*

      *Open risks: a component's registry name is fixed once written to a
      save file (downgraded from rev 2's "frozen forever" - it lives in the
      registry, not the type, so refactoring is free and `rename` recovers a
      bad choice); ledger volume needs partitioning and an archive-not-delete
      retention policy; a reconciliation mismatch must page a human, not
      append to a log nobody reads.*

- [x] **R3 — Replication with interest management.** Per-client AoI query →
      relevancy set → baseline+delta snapshot → quantized, bit-packed encode →
      the existing `net/chunk.rs` `send_chunked`. **Complete**, and proven
      against a real QUIC connection rather than only a fake sink.

      *Without AoI, per-client bandwidth is O(world entities) — the hard wall
      between a session game and an MMO.*

      **Done: the AoI primitive, the wire format, and the packet.**
      `SpatialGrid::query_circle_into` + `AoiScratch` (guarded in
      `benches/hot_paths.rs`), `net::bitpack`, `net::replication`
      (`Relevancy`, `ClientLink`, `Ack`), `net::snapshot::SnapshotPacket`,
      and `Replicate` on the registry. Behind `replication = ["net",
      "persist"]` with its own CI axis and four count assertions.

      The pipeline is proven to compose: `tests/replication_e2e.rs` drives
      relevancy → diff → encode → chunk → decode across several ticks with
      entities entering and leaving, asserting the client's reconstructed
      view equals the server's at every tick, including a chunked keyframe
      and a recycled entity index.

      `ChunkHint` also lives here now. Halving alone packed 28 items where
      93 fit — 66 datagrams against an ideal 20, 33% utilisation — so a
      connection remembers capacity across sends. That recovers half the
      waste (66 → 34 datagrams, 63% utilisation); the rest needs a second
      remembered size for continuation chunks, which carry no bulk header
      and so fit more than the one measured size allows.

      `KeyframeBudget` closes a thundering herd. Keyframes were decided
      per client with no shared limit: one costs 0.388 ms to encode, so 64
      clients taking one on the same tick is 25.9 ms of a 33.3 ms tick and
      1000 clients is **413 ms** — twelve times the budget, reachable on
      any restart or shard migration. A per-tick allowance (13 ≈ 5 ms)
      turns that into a queue draining in 2.6 s, and `plan` returns
      `Plan::{Delta, Keyframe, Deferred}` so a deferred client cannot be
      mistaken for one that may take a delta — that mistake would apply
      changes against a baseline the server knows it lacks.

      Encoding is guarded too: 1000 clients × 40 items is 5.46 ms, so
      relevancy plus encode is ~12.5 ms of the tick.

      And proven against a socket: `examples/replication_server.rs` runs
      both ends over loopback for 120 ticks, one keyframe and 119 deltas
      in 602 datagrams, after which the client's 280 reconstructed
      entities equal the server's 280 visible ones. Acknowledgements make
      the full round trip; nothing is undeliverable.

      The threading is the part worth reading before writing a real
      server. `fixed_update` is synchronous and stays that way, so the
      simulation owns a thread and runs `run_headless_with` on it. The
      network side owns a second thread with a current-thread runtime,
      because accepting a connection and reading a stream are async and
      this axis resolves `tokio/rt` but not `rt-multi-thread`. What makes
      the split cheap is that **`send_datagram` needs no runtime** —
      measured, it reports a 1162-byte path MTU and sends from a plain
      thread — so snapshots go out on the simulation thread through
      `QuinnSink` with no hop, and only the ack path is driven on the
      runtime. Note `tokio/macros` is absent too, so `select!` does not
      exist here; `tokio::time::timeout` is the substitute.

      *`quic.rs` still has `server_endpoint` but no client equivalent, so
      a client calls `quinn::Endpoint::client` directly. An asymmetry in
      the engine's surface, not a blocker.*

      *Known gap: CI never runs `cargo doc`, so a broken intra-doc link
      goes unnoticed until someone runs it by hand. Two were found that
      way and both are fixed — one in `net/mod.rs` from `7ac5124`, and
      `renderer/mod.rs:257` from `a1dbc30`, where an inserted function had
      also split a doc comment away from the one it described. Every axis
      measures zero warnings today, so a `cargo doc` step could be added
      without first cleaning anything up.*

      The brief assumed `SpatialGrid` could be used as-is. It is the right
      structure, but the naive path measured **42.7 ms** for 100k colliders
      × 1000 clients against a **33.3 ms** tick at 30 Hz — over budget on
      relevancy alone, before encoding anything. The cost was not the
      algorithm: allocation was 28 ms and the `HashSet` dedupe 12 ms, i.e.
      99.3%. A caller-owned scratch with generation-stamped dedupe gives
      ~12.4 ms (~7.3 ms on the bench's denser lattice), leaving ~20 ms of
      tick for the rest of replication.

      Measured and rejected along the way: sharing one widened query across
      a bucket of nearby clients (39.9 ms — the wider radius cancels the
      fewer queries exactly), and staggering AoI across ticks (works, 14.9 ms
      at 1/4, but 132 ms to refresh is visible pop-in; keep it in reserve for
      the far ring only). **R4 is not a prerequisite** — grid rebuild is
      4.7 ms against ~50 ms of queries, so incremental broadphase does not
      unblock this.

      Settled for the remaining work: components go on the wire as
      `NameId(u16)`, but ids are registration-order and *not* stable across
      runs, so each connection needs a name→id header once on the reliable
      stream (`framing.rs`) before any datagram uses them. `EntityId` is
      8 bytes raw — 440 MB/s of pure identity at 1834 entities × 1000
      clients × 30 Hz — so index is varint'd and generation moves into
      spawn/despawn events rather than every delta. No quantization or
      bit-packing exists in the crate yet; that layer is new work, not
      wiring. The module wants `replication = ["net", "persist"]` (following
      `ledger = ["persist"]`) at `src/net/replication.rs`, plus a seventh CI
      axis: `--features net` builds with persist *off* today, so a `net`-only
      gate referencing `persist::registry` would fail that existing job.

- [ ] **R4 — Chunk the world; incremental broadphase.** Replace the flat
      `TileGrid` (`Vec<T>` of `w*h`) with a chunk table keyed by `Sector2D`,
      generated on demand. Give `SpatialGrid` `update`/`remove` so it stops
      reallocating every tick.

      **`TileGrid` is load-bearing in void-claim, which has no in-tree
      callers to warn you.** `void_sim::module::Module` and
      `void_sim::station_interior::Floor` each hold one as a
      `#[serde(skip)]` field and layer their own overlays on top, and
      `tilemap_editor` calls `rotate_tiles` so authoring rotation cannot
      drift from the runtime. A rewrite must preserve: `empty`,
      `is_empty`, `width`, `height`, `tile_at`, `tile_at_glyphs`,
      `rebuild_from_glyphs`, `as_slice`, `set`, and the free function
      `rotate_tiles`. They use it as a value type through that surface
      rather than reaching into it, so a chunk table behind the same
      methods is viable; changing the methods is not. mini-miner-2 does
      not use it at all.

      **The premise was wrong, and the real bottleneck is now fixed — but
      neither half of this entry's stated scope has been done.** No chunk
      table, no `update`/`remove`; the checkbox stays open.

      What was wrong: the rebuild this entry is built around was never the
      bottleneck. The 26.4 ms it cites for 50k colliders measures 1.69 ms.
      `query_pairs` was the cost, for a reason the entry does not mention —
      it returned every pair sharing a cell with no distance test, and on a
      mixed-radius world three quarters of those had bounding squares that
      never touched. Rejecting them before the dedupe set (`fd56689`)
      halved it: 67.6 ms to 33.0, 42.3 to 20.1, both now inside a 30 Hz
      tick that neither previously fit.

      So the *performance* case for an incremental broadphase is much
      weaker than written, and whoever picks this up should re-justify it
      on streaming and world size rather than on rebuild cost. The
      `Relevancy` warning below still applies in full.

      *A dense world still costs ~180 ms at 100k, and that one is not a
      broadphase problem. It carries 9.6 overlaps per entity against a
      narrow phase costing 1.45 ms — collision resolution has already
      failed at that density, because no body resolves ten simultaneous
      penetrations per tick. Realistic densities measure 1.3 and 0.3
      overlaps per entity.*

      *Measured and rejected on the way: sweep-and-prune (3–4x faster on
      uniform radii, returns the **wrong pair set** once radii vary, and
      swings 387x on collider orientation); size tiering (wins only at
      the densest row, flat expensive cost elsewhere); per-partition or
      per-cluster grids (84 ms against a 33.3 ms tick, so a caller cannot
      work around this today).*

      **This breaks `net::replication::Relevancy`, by construction.** That
      type maps a grid index back to an `EntityId` by recording entities in
      insertion order, which is correct *only* because the grid is rebuilt
      from scratch every tick and indices are therefore assigned fresh.
      With `update`/`remove`, an index outlives the tick that created it and
      the parallel vector has to be maintained rather than rebuilt — or the
      grid has to hand back a stable id instead of a dense index. Decide
      which before writing the incremental path, not after: the failure mode
      is a client being told about the wrong entity, which no type checks.

      *Measured while building R3: the rebuild itself is not the bottleneck
      it looks like. At 100k colliders a full rebuild is 4.7 ms against
      ~50 ms of per-client queries, so incremental update is worth doing for
      the tile grid and for larger worlds, but it does not unblock
      replication and should not be justified on that basis.*

      *The terrain noise is already pure `f(position, seed)` and streams
      perfectly. It is held back by the container, not its own design.*

- [ ] **R5 — Rebuild the client draw path.** Instancing (every draw is
      currently `0..1`), texture atlas or `D2Array` (one 1×1 white pixel
      exists today), SDF glyph atlas, depth buffer or explicit layer sort,
      and a single clustered light pass (`MAX_LIGHTS_PER_FRAME = 384`).

      *A rewrite of the draw path, not an optimization pass. Independent of
      the server work and safely deferrable.*

      **Measured, 2026-09-11, RTX 3080 Ti at 1080p.** All five claims hold;
      one was cited at the wrong line. Treat every number as a best case —
      pass submission is driver-bound and a weaker GPU punishes it harder.

      **Text is the biggest win, and the entry under-sold it.** "72 KB per
      nameplate" is a memory figure; the real cost is CPU. `draw_text`
      emits one quad per *lit font pixel* (mean 20.8 set bits per glyph,
      max 37), so 200 nameplates costs **9.8 ms** — 6.8 ms building the
      batch, 3.0 ms uploading 16.7 MB — before a single draw call. 1000
      costs 39.7 ms and 83.6 MB, more than two frames on its own.
      One quad per glyph instead measures **23.7× fewer vertices and
      22–76× faster batch building**: 200 nameplates drop to 0.150 ms and
      180 KB.

      The rewrite is unusually safe. `Vertex` already carries `uv`
      (offset 8, location 1) and `shader.wgsl:505` already does
      `textureSample(t_diffuse, s_diffuse, in.uv) * in.color`, with the
      white 1×1 bound at group 1 in four places — so an atlas glyph is the
      existing path with different UVs. No shader change, no vertex-layout
      change. `text.rs` has no atlas or cache today and exposes five
      functions; downstream calls only `draw_text` (177 + 22) and
      `draw_text_centered` (142 + 6), and *never* the metric functions. The
      contract to preserve is the geometry: 8 px glyph + 1 px spacing
      (`(chars * 9 - 1) * scale`), `pos.y` as the row-0 baseline with
      glyphs spanning `[pos.y - 7s, pos.y + s]`, and the `+3s` centring
      offset. Change any of those and 347 call sites shift silently.

      **The light cap is reachable and catastrophic at the cap.** The real
      site is `frame.rs:411`/`435`, not 382 — 382 is `shadow_raycast_pass`.
      `for i in 1..lights` opens a fresh render pass per light, each a
      fullscreen triangle running up to `LIGHT_TAPS = 12` raycast samples
      per covered pixel, early-outing past `radius_px`:

      | lights | radius | ms | frame at 60 Hz |
      | --- | --- | --- | --- |
      | 384 | 1200 px | **25.1 ms** | **152%** |
      | 384 | 400 px | 11.9 ms | 72% |
      | 384 | 120 px | 7.6 ms | 46% |
      | 64 | 1200 px | 4.0 ms | 24% |

      Cost is area-proportional, so the 25 ms case needs 384 large
      overlapping lights. A plausible interior at 64 lights stays under
      4 ms, which matches the note at `mod.rs:275` that the on-foot
      lattice fits comfortably. *Measured first with a constant-colour
      fragment, which gave 5.3 ms at 384 and made the claim look weak —
      that measured submission overhead alone. The march is the cost.*

      **Instancing and the missing depth buffer are confirmed but
      unquantified.** All 14 draw calls are `0..1`; every
      `depth_stencil_attachment` is `None` and no pipeline sets
      `depth_stencil: Some`, so ordering is painter's-algorithm only. The
      main pass already splits `draw_indexed` into ranges around composite
      points (`frame.rs:588`), which is the structure instancing has to
      preserve.

      *Downstream surface, the same constraint `TileGrid` has: roughly
      2,100 `Batch` primitive call sites across void-claim and
      mini-miner-2 (638 `rect`, 581 `line`, 351 `draw_text` in void-claim
      alone) and **zero** `push_quad` in either. Both games go entirely
      through the high-level primitives, so replacing `Batch`'s internals
      behind the same methods moves no call site — and changing those
      methods breaks two shipped games.*

---

## Downstream breakage owed

The engine is kept clean in preference to backward compatibility, so a
breaking change lands here and the consuming game is fixed after. This
records what is currently owed, because nothing else does — neither repo
is in this workspace and `cargo check` here will never notice.

**mini-miner-2 is broken right now.** It depends on this repo by
`path = "../../../void-engine"` (`crates/miner/Cargo.toml`, replacing a
commented-out `rev = "2f42af6"`), so it breaks the moment a signature
changes rather than on a deliberate bump. `eae5e61` gave `send_chunked`
a fifth parameter and two call sites in `crates/miner/src/net/host.rs`
still pass four:

- **`host.rs:545`**, production. The call sits in `send_snapshot_datagrams`,
  a free function invoked from a per-connection async loop (`host.rs:517`)
  that owns `conn` until it closes. That loop is where a `ChunkHint`
  should live — declared before it, threaded in — because a fresh hint
  per snapshot compiles and reproduces the old behaviour while throwing
  away the whole point of remembering capacity across sends.
- **`host.rs:863`**, a test helper. A fresh `ChunkHint::new()` inline is
  right here; the test wants the cold-start policy.

No import change: `host.rs:30` already has
`use void_engine::net::chunk::{self, QuinnSink}`.

*Its tree had uncommitted `Cargo.toml` and `Cargo.lock` changes when this
was written — the `path =` switch — so check what is in flight before
editing.*

**void-claim is not affected.** Worth stating because a substring search
suggests otherwise: it has five files mentioning `send_chunked`, but that
is its own `pub(crate) fn send_chunked(conn, msg)` in
`crates/server/src/net.rs:21`, unrelated to the engine's. It uses no
`ClientLink`, `KeyframeBudget` or `net::replication`; its `.plan(` hits
are `drive_flight_plan`. Its only couplings to this engine are `TileGrid`
(see R4) and one `query_circle` call in `npc.rs:97`, which `fd56689` left
untouched — that commit changed `query_pairs` alone.

It also tracks `branch = "main"` across four crates with a lockfile at
`192232c`, 46 commits behind, so engine changes reach it on a
`cargo update` rather than immediately.

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
| Collision rebuild+query, 10k colliders | **3.68 ms/tick** | 5.1 ms |
| Collision rebuild+query, 50k colliders | **9.33 ms/tick** | 26.4 ms claimed |
| Collision, 10k on the *old* spacing-30 lattice | **1.76 ms/tick** | 5.07 ms |
| `query_pairs`, 100k mixed radii over km | **33.0 ms/tick** | 67.6 ms |
| A*, 256×256 open grid, corner-to-corner | 14.5 ms |
| `size_of::<Vertex>()` | 84 bytes |
| Text: 10-char nameplate | 864 verts / 72 KB |
| Text: 200 nameplates, build + upload | **9.8 ms / 16.7 MB** | — |
| Text: same, one quad per glyph | **0.15 ms / 180 KB** | 23.7× fewer verts |
| Lights: 384 × radius 1200 px | **25.1 ms/frame** | 152% of 16.6 ms |
| Lights: 64 × radius 1200 px | 4.0 ms/frame | 24% |
