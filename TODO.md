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

- [x] **R2 - Persistence and ledger.** Done 2026-09-10, all five phases.
      Three tiers by Cargo feature (`persist` / `ledger` / `ledger-pg`),
      an append-only double-entry ledger whose balances are derived rather
      than stored, and a crash matrix whose exit criterion is met: a row
      forged straight into Postgres with no counterparty is caught by
      `reconcile_now`, not by a player noticing
      (`tests/crash_matrix.rs:99`). Design published as an artifact;
      decisions recorded below so they outlive the link.

      *This header read "Designed 2026-09-10 (rev 3), not built" while the
      body below recorded five shipped phases and said "R2 is done". The
      phase notes were appended as each landed and the header was never
      revisited — the same defect found in R4 and R5, where a summary
      written once outlived the work it described.*

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

      **Phase 5 complete (2026-09-10).** 4 more tests. R2 is done.

      *Every "N total across six axes" figure in these phase notes — 237,
      264, 272, 281, 293 — was measured with `--features X` layered on the
      default `client` feature, which is **not** the build CI gates. CI
      runs each non-default axis as `--no-default-features --features X`.
      Re-measured in CI shape on 2026-09-11: 154 headless, 192 net, 203
      persist, 241 ledger, 256 ledger-pg, 293 replication, 189 default.
      The phase figures are not wrong about what they counted; they are
      incomparable to these. Note the 293 agreeing with replication's
      current count is coincidence, not confirmation.*

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

- [x] **R4 — Incremental broadphase.** Closed. `SpatialGrid` stops
      reallocating every tick — via `clear`, which retains the buckets,
      not via `update`/`remove` as this entry assumed. Those exist too,
      tested and unused. ~~Chunk the world~~ — the `TileGrid` half is
      dropped; it holds authored rooms of a few thousand cells, not a
      world. Both halves are measured below.

      **`TileGrid` is load-bearing in void-claim, which has no in-tree
      callers to warn you.** *Kept as a standing hazard note, not as
      pending work: no `TileGrid` rewrite is planned, and this is what
      would constrain one if a later change reaches for it.*
      `void_sim::module::Module` and
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

      **The two halves of this entry have opposite fates. Measured.**
      The chunk table should not be built. The broadphase half is **done**.

      **What landed.** `SpatialGrid` gained `clear`, `remove`, `update`,
      `slot_count`, `slot_alive`, `resolve`, `id_of`, `insert_tracked`, and
      a generational `ColliderId`. `remove` leaves a hole rather than
      compacting — compaction would renumber a *live* collider, and callers
      hold indices within a tick, which is the "told about the wrong
      entity" failure this was supposed to prevent, not cause. Slots carry
      a generation bumped on every remove, mirroring `EntityId` in the ECS,
      so a stale handle fails closed via `resolve` instead of naming its
      slot's new occupant. Ten tests in `collision::incremental_tests`
      cover reuse, hole-skipping in `query_pairs`, multi-cell unhook,
      bucket membership across `update`, and `clear` semantics.

      **The shipped win is the allocation fix, and it is larger than the
      entry predicted.** Both drivers now call `grid.clear()` instead of
      assigning a fresh `SpatialGrid::new(CELL)`. Rebuild alone, insert
      only:

      | world                  | fresh    | cleared  |
      | ---------------------- | -------- | -------- |
      | 10k lattice, cell 40   |  0.84 ms |  0.32 ms |
      | 50k lattice, cell 40   |  4.09 ms |  1.72 ms |
      | 100k sparse, cell 400  |  3.41 ms |  2.07 ms |
      | 100k sparse, cell 40   | 10.95 ms |  3.00 ms |

      The gain tracks cell count, not collider count — many small buckets
      is where per-tick allocation dominates, so a 40-unit cell gains 3.6x
      where a 400-unit one gains 1.6x. Pair sets verified identical between
      arms at every size, and `clear_leaves_a_grid_equivalent_to_a_fresh_one`
      pins that in the suite.

      *`remove`/`update` are available and tested but **not in use**: both
      drivers still rebuild wholesale, because `clear` gets the measured
      win without touching the relevancy invariant. Do not read the table
      above as their benefit — they are unmeasured, and their case is
      streaming, not allocation.*

      **The `Relevancy` hazard did not bite, for a reason worth keeping.**
      `clear` restarts slot numbering from zero exactly as a fresh grid
      would, so insertion-order `push` stays correct and `Relevancy` needed
      no change at all. The hazard is real only for a caller that removes
      or updates *incrementally*, where an index outlives the tick that
      made it. That warning now lives on the `remove` doc comment, where
      such a caller will actually meet it, rather than only here.

      **The chunk table has no world to chunk.** `TileGrid` never holds a
      world — it holds hand-authored room interiors, one JSON file each.
      void-claim's largest is the lobby at 55 rows × 40 cols = **2,200
      cells**; `apartment_with_window` is 96 and `elevator` is 16. A
      `Vec<T>` of 2,200 is a few kilobytes. Keying that by `Sector2D` and
      generating it on demand is more machinery than the data it holds,
      and the content already streams by being separate files that load
      per module. mini-miner-2 does not use `TileGrid` at all.

      *The "generated on demand" framing assumed a world-sized grid. The
      terrain note below is the tell — the thing that genuinely wants
      streaming is the noise field, which is not a `TileGrid` and is
      already pure `f(position, seed)`. Reopen only if a game authors a
      single grid large enough that `w*h` allocation shows in a profile;
      at 2,200 cells it cannot.*

      *What the case was, before it was acted on: `SpatialGrid` had no
      `clear()`, so "rebuild every tick" meant a fresh allocation every
      tick, and both in-tree drivers assigned
      `self.grid = SpatialGrid::new(CELL)` per tick. It looked modest —
      4.7 ms of rebuild against ~50 ms of queries at 100k. Both drivers now
      call `clear()`, and the measured gain was larger than that framing
      suggested; see the table above.*

      What was wrong: the rebuild this entry is built around was never the
      bottleneck — which is not contradicted by the table above. A cost
      that was never dominant still got 3.6x cheaper, and both facts hold:
      the entry was wrong about *why* the work mattered, and the work was
      worth doing anyway for a smaller, differently-shaped reason.
      The 26.4 ms it cites for 50k colliders measures 1.69 ms.
      `query_pairs` was the cost, for a reason the entry does not mention —
      it returned every pair sharing a cell with no distance test, and on a
      mixed-radius world three quarters of those had bounding squares that
      never touched. Rejecting them before the dedupe set (`fd56689`)
      halved it: 67.6 ms to 33.0, 42.3 to 20.1, both now inside a 30 Hz
      tick that neither previously fit.

      So the *performance* case for an incremental broadphase was much
      weaker than written. An earlier revision of this line said to
      re-justify it "on streaming and world size rather than on rebuild
      cost" — both of those are measured dead: the grids are authored
      rooms, and the thing that wants streaming is the noise field, which
      is not a `TileGrid`. What survived was narrower and was acted on: the
      per-tick allocation at two live call sites, fixed by `clear`. The
      `Relevancy` warning below is what an *incremental* caller must still
      read — see the note above on why `clear` did not trip it.

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

      **An incremental caller breaks `net::replication::Relevancy`, by
      construction — and this is the decision that was made about it.**
      That type maps a grid index back to an `EntityId` by recording
      entities in insertion order, which is correct *only* because indices
      are assigned fresh from zero each tick. `clear` preserves that
      exactly, which is why the allocation fix needed no change here. But
      with `update`/`remove` an index outlives the tick that created it,
      and then insertion order means nothing.

      The resolution taken: **the grid hands back a stable id**
      (`ColliderId`, index + generation, `resolve` to check it) *and* keeps
      the dense index as the in-tick currency. `query_pairs` and
      `AoiScratch::hits` still yield bare `u32` slots, because nothing is
      removed mid-query and resolving a generation per hit would cost more
      than it buys. So `Relevancy` is untouched and stays correct for
      rebuild-style callers; an incremental one holds `ColliderId`s and
      maintains its own mapping. The rule is on the `remove` doc comment:
      finish consuming a query's output before removing anything it named,
      because a bare index will happily address the slot's next occupant.

      *The failure mode this guards is a client being told about the wrong
      entity, which no type checks — hence a generation that fails closed
      rather than a convention that must be remembered.*

      *Measured while building R3: the rebuild is not the bottleneck it
      looks like — the 4.7 ms against ~50 ms of queries cited above. It
      does not unblock replication and must not be justified on that
      basis. An earlier revision added "worth doing for the tile grid and
      for larger worlds"; the tile-grid half is dropped, and no larger
      world exists to point at.*

      *The terrain noise is already pure `f(position, seed)` and streams
      perfectly. It is held back by the container, not its own design.*

- [x] **R5 — Rebuild the client draw path.** Closed without the rewrite.
      ~~Glyph atlas~~ done; ~~clustered lights~~ measured and dropped;
      ~~instancing~~ inspected and dropped (no timing taken — the
      draw-site count settles it); ~~sprite atlas~~ deferred, its premise
      does not hold yet; depth buffer deferred on a 14-pipeline blast
      radius. Each is recorded below with the trigger that would reopen it.

      *Was framed as a rewrite of the draw path. Checking the five claims
      one at a time left nothing to rewrite: one was real and is done, two
      were true and not worth acting on, two are real but premature. The
      entry below is the record of which is which.*

      **Measured, 2026-09-11, RTX 3080 Ti at 1080p.** All five claims are
      *true as stated*, but only two are worth acting on. The glyph atlas
      was real and is now done; the sprite atlas is real and remains open.
      Clustered lights and instancing are true and not worth doing — the
      first only bites at a cap neither game approaches, the second
      describes work `Batch` already performs. The depth buffer is real
      and deferred on blast radius, not on value.

      *An earlier revision of this line read "all five claims hold", which
      was accurate about the claims and misleading about the work. A claim
      being true is not the same as a claim being worth acting on.*

      Treat every number as a best case — pass submission is driver-bound
      and a weaker GPU punishes it harder.

      **Text was the biggest win, and it is done.** The entry under-sold
      it: "72 KB per nameplate" is a memory figure, and the real cost was
      CPU. `draw_text` emitted one quad per *lit font pixel* (mean 20.8
      set bits per glyph, max 37), so 200 nameplates cost **9.8 ms** —
      6.8 building the batch, 3.0 uploading 16.7 MB — before a single draw
      call, and 1000 cost 39.7 ms, more than two frames on their own.

      Each glyph is now one quad UV-mapped into a 128×128 atlas: 200
      nameplates build in **0.153 ms** and 1000 in **1.34 ms**, with
      vertex volume down from 208,800 to 8,800 at 200 plates. Measured
      against the real implementation, which came in fractionally ahead of
      the prototype that justified it.

      It needed no shader or vertex-layout change. `Vertex` already
      carried `uv` (offset 8, location 1) and `shader.wgsl:505` already
      did `textureSample(t_diffuse, s_diffuse, in.uv) * in.color`, so a
      glyph is that path with real UVs instead of all-`0.5`.

      The atlas carries the white texel too, at its centre, which is why
      nothing downstream moved. Every `Batch` primitive writes
      `uv = [0.5, 0.5]`, the renderer binds one texture for the whole main
      pass, and under `FilterMode::Nearest` a sample at 0.5 selects texel
      `floor(0.5 × size)` — measured the same at every size tried, even
      and odd alike, which killed an earlier plan to pad the atlas to an
      odd dimension against a boundary ambiguity that does not exist.
      Ninety-six glyphs fit exactly six rows of sixteen, filling `y < 48`
      and leaving (64, 64) clear.

      The metrics contract is preserved to the float and now has tests:
      8 px glyph + 1 px spacing (`(chars * 9 - 1) * scale`), `pos.y` as
      the row-0 baseline with glyphs spanning `[pos.y - 7s, pos.y + s]`,
      and the `+3s` centring offset. Downstream calls only `draw_text`
      (177 + 22) and `draw_text_centered` (142 + 6) and *never* the metric
      functions, so all 347 sites are untouched.

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
      overlapping lights. *Measured first with a constant-colour fragment,
      which gave 5.3 ms at 384 and made the claim look weak — that
      measured submission overhead alone. The march is the cost.*

      **But do not cluster the lights.** The cap is not where either game
      lives, and the curve at radii they actually use never approaches a
      budget:

      | radius | 8 | 16 | 32 | 64 | 128 lights |
      | --- | --- | --- | --- | --- | --- |
      | 80 px | 0.20 | 0.35 | 0.65 | 1.26 | 2.46 ms |
      | 160 px | 0.21 | 0.38 | 0.71 | 1.39 | 2.73 ms |
      | 320 px | 0.31 | 0.53 | 0.99 | 1.94 | 3.36 ms |
      | 640 px | 0.38 | 0.72 | 1.40 | 2.77 | 5.53 ms |

      void-claim lights ceilings at 4.5 tiles and floods at 12
      (`lights.rs:34`, `:42`), mini-miner-2 candles at 3.2 m, halos at 7
      and portal beams reaching 9 (`lamp.rs`) — all inside that bracket.
      Neither caps its count: void-claim pushes one light per visible tile
      of a kind, mini-miner-2 two per crew member plus one per portal, so
      counts scale with the scene. Even so, reaching 8 ms needs 128 lights
      *and* a 640 px radius, and 25 ms needs 384 at 1200.

      *The same shape as R4: a dramatic figure at the ceiling, and a
      measurement showing nobody stands near it. If this is revisited, the
      trigger is a game that logs light counts above ~128 at large radii —
      void-claim already instruments exactly that
      (`[lights] ceiling_candidates=N`), so the number can be read rather
      than guessed.*

      **Instancing has little to save, and the entry's framing misleads.**
      "Every draw is `0..1`" is literally true and practically empty.
      There are **22** draw sites, not the 14 this entry claimed, and all
      22 are in `frame.rs`. Eighteen are fullscreen triangles
      (`draw(0..3, 0..1)`) — ten of them composites, the rest effect and
      light passes — and a single fullscreen triangle cannot be
      instanced. Only **four** are `draw_indexed`: the offscreen batch
      (`:287`), the wall mask (`:369`), and the main batch split into
      ranges (`:611`, `:659`). `Batch` already merges every primitive into
      one vertex buffer, which is the thing instancing exists to achieve.

      Geometry draws are what instancing could touch, and there are four
      of them regardless of how many sprites the scene holds. The other
      eighteen sites are passes, not batched geometry, and they break down
      as: ten composites inside the main-pass range loop (`:618`–`:644`
      and `:670`–`:694`, five after each `draw_indexed`); two in the
      per-light loop (`:430`/`:451`, submitted once per light, which is
      what makes the light table above the real cost); and six standalone
      fullscreen passes, named here by the pipeline each binds — two blur
      taps (`:308`, `:330`, both `blur_pipeline`), shadow raycast
      (`:397`, `raycast_pipeline` — this is the pass the audit reached for
      when it cited 382 for the light loop; 382 is this pass's `label:`
      line), sun (`:478`), and the godray pair `seed`/`march`
      (`:508`, `:528`).

      *The light-loop citations above and these reconcile once the
      distinction is pass-open versus draw-call: 411/435 open the light
      passes, 430/451 are the draws inside them. Both are right; they name
      different things.*

      So a busy frame submits four indexed draws *plus* up to ten range
      composites, plus the six standalone passes that are active, plus one
      triangle per light. Four is the geometry floor, not the frame's draw
      count — and none of the eighteen is something instancing addresses.

      *If instancing is revisited, it needs a workload where per-draw
      state actually changes — many distinct textures, say — which the
      single-atlas design has just made less likely rather than more.*

      **The sprite atlas has no content to atlas.** The entry's parochial
      detail — "one 1×1 white pixel exists today" — was the whole truth
      about the engine's texture story, and it is no longer even that: the
      white pixel is now the glyph atlas. But an atlas solves *binding
      churn*, and there is none to solve. The main pass binds exactly one
      texture bind group, `white_texture_bind_group`, at all four
      geometry-draw sites (`frame.rs:284`, `:366`, `:601`, `:650`). Every
      other texture the engine creates is a render target for an offscreen
      pass (`godray.rs:379`, `lights.rs:429`, `postprocess.rs:291`,
      `shadow.rs:395`, `sun.rs:285`), not sprite content.

      No public API takes a texture, so a game *cannot* supply sprite
      content today. Neither game tries: every `Batch` primitive writes
      `uv = [0.5, 0.5]` and samples the white texel. mini-miner-2's
      `textured` flag (`map.rs:133`) is not sampling — it is a zoom
      threshold (`px_per_m >= COVER_PPM`, 12.0) choosing between elevation
      tint and a procedural `blend_at`/`blended_surface` colour computed
      on the CPU and written into vertex colours. The name means "shows
      what it is made of", not "reads from an image".

      *So the work is an atlas for sprites that do not exist, reached
      through an API that does not exist. The trigger to reopen is the
      API, not the atlas: when a game needs to draw from an image, the
      texture-binding entry point comes first and the atlas is the obvious
      shape for it — the glyph atlas already proves the pattern, including
      the white texel that keeps untextured primitives working unchanged.*

      **The depth buffer is confirmed and still unmeasured.** Every
      `depth_stencil_attachment` is `None` and no pipeline sets
      `depth_stencil: Some`, so ordering is painter's-algorithm only.
      Adding one touches **14 pipelines across 6 files**
      (`godray`, `init`, `lights`, `postprocess`, `shadow`, `sun`), which
      is a far wider blast radius than any other R5 item — worth keeping
      separate from the rest rather than folded in.

      *Downstream surface, the same constraint `TileGrid` has: roughly
      2,100 `Batch` primitive call sites across void-claim and
      mini-miner-2 (638 `rect`, 581 `line`, 351 `draw_text` in void-claim
      alone) and **zero** `push_quad` in either. Both games go entirely
      through the high-level primitives, so replacing `Batch`'s internals
      behind the same methods moves no call site — and changing those
      methods breaks two shipped games.*

---

## Second audit (2026-09-11): what the first one never looked at

The first audit's five items are closed. Three agents then swept the
territory it never covered — sim/ECS/loop, netcode/robustness,
persistence/ops/coverage — on the standing rule that a finding needs a
file:line, a measured workload where it bites, and a stated workload
where it does not.

- [x] **N1 — `Updated` items forked every respawned entity on the client.**
      Fixed. `encode_into` writes a generation only for `Entered`/`Left`
      (`snapshot.rs:219-221`); `decode` filled the gap with `0`
      (`:288-292`). Both reference clients keyed their map on the full
      `EntityId`, so once any generation went non-zero an update was filed
      under a key no `Entered` had ever created: the client held **two**
      entries for one entity — the real one frozen at its keyframe
      position, and a ghost that moved. `Left` carries the true
      generation, so departure removed only the real one and the ghost
      outlived the connection.

      *Not hostile-input dependent. Any despawn/respawn in ordinary play
      triggered it.* The encoder's premise is sound — a live entity's
      generation cannot change, and sending it per-update would cost
      1-2 bytes × 1834 entities × 500 clients × 30 Hz — so the fix is on
      the receiving side: `EntityItem::key()` (the index) is what a client
      keys by, and `generation_is_authoritative()` says when the full id is
      real.

      **The fix exposed a second defect underneath it.** Collapsing to an
      index key made emission order load-bearing, and both drivers emitted
      `Entered` before `Left` — so a recycled index had its arrival undone
      by the old tenant's departure in the same tick. Two changes, because
      one is not enough: the drivers now emit departures first, and a
      client must check the generation before honouring a `Left` (which is
      what actually makes it safe, since a hostile peer is not bound by
      the ordering rule). Both rules are recorded on `EntityItem::key`.

      *The existing test `a_recycled_index_does_not_confuse_the_client`
      passed throughout, because it stopped at tick 2 — where the entity
      arrives as `Entered`, which carries a generation. Tick 3 is the first
      `Updated`. The test now runs that tick, and was **verified
      load-bearing**: reverting the keying makes it fail at tick 3 with
      `generation: 0` against the server's `1`, while the other five e2e
      tests stay green. A test that would not have caught the bug is worth
      nothing, so this check is the point.*

- [x] **N2 — one forged ack permanently disabled a client's stall
      detection.** Fixed. `Ack::decode` accepts any four bytes as a `u32`
      (`replication.rs:124-127`), `record_ack` checked only monotonicity,
      and `plan` computes `tick.saturating_sub(reference)` (`:342`). A
      single `Ack { tick: u32::MAX }` pinned that difference at zero
      forever: the stall check could never fire, the client was never
      re-keyframed however far it drifted, and the server deltad against a
      baseline it knew the client had not confirmed — holding that
      baseline's memory for the life of the connection.

      Measured before the fix: **0 keyframes over 10,000 ticks** against
      109 for an honest silent client.

      `record_ack` now takes `now: u32` and drops acks from the future.
      Dropped, not clamped — clamping to `now` would let a peer pin the
      watermark at the present tick every tick and never fall behind by
      construction, the same exploit in better manners. The bound lives in
      the method rather than at call sites because the example's net thread
      already re-implemented a monotonic check and *still* would not have
      caught this.

      *Signature change: `record_ack(ack)` → `record_ack(ack, now)`. Nine
      call sites updated. The unit tests pass a real tick rather than
      `u32::MAX`, so the bound stays live in every test instead of being
      bypassed by the tests that exist to check it.*

- [x] **N8 — the ledger's safety mechanisms had no caller and no seam.**
      Done, and the endpoint is read-only permanently rather than
      pending an auth story. The audit found `audit_zero_sum`,
      `audit_balances_match_entries`, `health`, `journal_depth`,
      `reconcile_now` and `drain_log_events` each correct, tested, and
      unreachable from a running server — `SimCtx` carries `world`,
      `input`, `dt`, so a game holding a ledger had to invent its own
      health endpoint to see any of it.

      **`admin` feature (implies `ledger`), off by default.** `GET /`
      serves a self-contained HTML page; `GET /api/status` serves the same
      data as JSON. Both render from one `Status` captured at one instant,
      so the page and the endpoint cannot tell an operator two different
      stories mid-incident — a test pins that they agree on health.

      What it surfaces: zero-sum discrepancies, balance-cache drift,
      resident and lapsed reservation counts (N7's stall predictor),
      writer state and journal depth, acked tick, and N3's `TickHealth` —
      achieved vs target Hz, dropped sim seconds, mean and worst tick.

      *No new dependencies.* The HTTP/1.1 subset (`admin::http`) and the
      JSON writer (`admin::json`) are ~250 lines together, because adding
      a web framework to serve two routes would be the one dependency in
      this crate nobody could justify in a sentence. Bounded deliberately:
      8 KiB of headers, 5 s IO timeout, 16 concurrent connections shed
      rather than queued, bodies never read.

      **It will never mutate anything — decided, not deferred.** An
      endpoint that can act needs authentication, and the cheap answer (a
      shared secret in an env var) is the kind of half-measure that invites
      binding to `0.0.0.0` and calling it secured. A page that cannot act
      needs no such story: the worst a reachable attacker gets is a read of
      what the operator already sees. `mutating_methods_are_refused` pins
      it. `serve` takes the bind address as a required argument rather than
      defaulting it; `loopback()` is the documented choice, and a test
      asserts the bound address is loopback.

      **So maintenance lives on the game's tick**, authenticated by being
      in-process, and the module now carries the worked example:
      `expire_reservations` every tick, `audit_zero_sum` (and
      `reconcile_now` on Postgres) on a cadence. That is the *only* route —
      the page tells you the backlog is growing, `fixed_update` clears it.
      Which means those two still have no caller **in the engine**, and
      correctly so: the engine cannot know a game's cadence, and inventing
      one would be the wrong default in both directions.

      *Two escapers, not one: HTML and JSON escape different character
      sets for different grammars, and merging them is how a value safe in
      one context becomes an injection in the other. Both have hostile-input
      tests — a `<script>` asset name, a database error containing quotes.*

      *`Ledger` does not implement `LedgerStore` — only `PgLedger` does —
      which I assumed the other way round and had to correct after writing
      400 lines against it. Hence `Status::from_ledger` beside
      `from_store`: the in-memory tier is the one with no database to
      query, so if these audits are not on this page they are nowhere.*

## Third audit (2026-09-11): the client, and everything unswept

Three agents over territory the first two never entered: the render path
under load, terrain/pathfind/sector/rng with determinism as the headline,
and input/walk/tilegrid/world/log. Same rule as before — file:line, a
number where it bites, a statement of where it does not.

- [x] **T1 — the draw cap was a stale literal, 2.5× the upload cap.**
      Fixed. `upload_batch` truncates at `MAX_VERTS * 3`, which with an
      84-byte `Vertex` is **9,586,980** indices. The main pass capped its
      draw range at a hardcoded **24,000,000** (`frame.rs`), and its own
      comment claimed the two matched. They disagreed by 14.4M.

      A frame in that window would issue `draw_indexed` over indices past
      what `queue.write_buffer` uploaded — reading buffer contents that
      were never written. That is precisely the failure recorded in
      `upload_batch`'s own header comment as having already happened once:
      *"indices pointed past the sliced-off tail and draws produced
      garbage (visible symptom: geometry vanished entirely on any
      overflow)"*.

      **Third instance of the same mistake.** TODO.md:37-38 records the
      first: a cap sized "8M verts × 32B = 256MB" when the stride was
      really 84 bytes. That fix made `MAX_VERTS` stride-derived
      specifically so it "cannot drift out of step with `Vertex` again" —
      and missed this copy. The draw cap is now derived from the same
      expression, so there is no literal left to drift.

      *Latent, not observed: it needs >1,597,830 quads in one frame, and
      the vertex cap trips first for quad geometry. The reachable shape is
      index-heavy fans (`circle`/`polygon`), which hit neither cap first.
      Every real workload is four orders of magnitude below it — 1000
      nameplates is 44,000 verts.*

- [x] **T2 — `RedrawRequested` was a second, unaccounted render path.**
      Fixed. `app.rs` handled OS repaints by calling
      `begin_frame`/`render`/`end_frame` inline with `alpha = 1.0`, no
      `PerfStats::record`, and no timestep advance — its comment said
      "just render, no timing".

      Two consequences. Frames drawn that way were invisible to `[perf]`
      and to the `PerfSnapshot` overlay, so GPU work under-reported
      exactly while a window was being dragged or uncovered. And it was a
      real extra frame: the 62 Hz cap lives in `about_to_wait`, so a
      repaint storm rendered on both paths at up to double rate.

      The handler is now empty — `about_to_wait` runs unconditionally and
      owns the only render path, so the repaint is honoured within ~16 ms.

      *My first version of this called `window.request_redraw()` from
      inside the `RedrawRequested` handler, which would have fed the
      handler its own next event with nothing else driving it. It compiled
      and all 195 tests passed, because no test drives a window event
      loop. Caught by grepping for other `request_redraw` calls and
      finding mine was the only one in the tree.*

- [x] **T4 — A* truncated its path coordinates to `u16`.** Fixed.
      `reconstruct` cast `i32` search coordinates with `as u16`, which
      wraps silently. On a grid wider than 65,536 the search found the
      *correct* route and then corrupted every coordinate on the way out:
      measured on a 70,000×1 grid, a path from column 65,530 to 65,540
      came back as `…(65534,0), (65535,0), (0,0), (1,0)…`, reporting its
      final cell as column 4. No `None`, no error — a caller following it
      walks backwards across the world.

      Both public entry points now return `Vec<(i32, i32)>`, matching the
      type the search already used internally. Two bytes per waypoint on
      an already-heap-allocated path.

      *Latent in this repo — no in-tree caller builds a grid that wide, and
      void-claim's interiors are far below it. But `astar_bool_grid` is
      re-exported by void-claim's own `pathfind` module and used by four
      call sites there, so the type change will surface downstream at
      compile time. Per the standing policy, the engine stays clean.*

- [x] **T5 — `wrap_pos` was an unbounded loop that hung on infinity.**
      Fixed. Four `while` loops subtracting one sector per iteration:
      correct for the one-crossing case every caller has, unbounded for
      anything else. Measured at sector size 1000, `pos.x = 1e9` cost
      **0.368 ms** and `1e12` cost **381 ms** — one entity stalling a
      60 Hz tick for twenty-three frames. An infinite coordinate never
      terminated at all, since `inf - size` is still `inf`; a NaN exited
      immediately instead and propagated into the sector address.

      *Now three-case arithmetic, and it took **four attempts**. The two
      loops run in sequence, so they are not symmetric halves of one
      operation — a value below `-half` skips the first loop entirely, and
      both `+half` and `-half` are fixed points, making the resting window
      closed at both ends. Every single-expression form I tried
      (`div_euclid` on a reversed shift, `ceil() - 1`, plain `floor`) was
      off by one at a boundary or on the negative side, and each passed
      some of the two existing tests. Six new tests now pin the exact loop
      behaviour: both boundaries as fixed points, the positive/negative
      asymmetry, a 1e9 wrap in one step, non-finite refusal, and a sweep
      asserting every result lands in the window with the sector bump
      accounting for the exact distance moved.*

- [x] **T6 — `dist_to_nearest` skipped the rejection its sibling used.**
      Fixed. It mapped `dist_to_center` over every river unconditionally,
      while `signed_edge_dist` twenty lines above did the same work behind
      a `bbox_dist2` early-out. Measured at 40 rivers × 300 vertices:
      **10.9 µs** per call against 2.4 µs — 716 ms to ask "how far to
      water?" across one 256×256 chunk, on the streaming path, per chunk.

      *The early-out tracks a running minimum rather than a fixed radius,
      so the threshold tightens as it goes and later rivers are rejected
      against the best distance found so far.*

- [x] **T7 — log rotation wrote into the rotated file, then disabled
      itself permanently.** Fixed, and the worst finding of the three
      sweeps. The logger holds one `File` opened at construction;
      `rotate_logs` renamed paths and never touched it. An open handle
      follows the inode, not the name.

      Measured end to end: after `t.log` crossed 10 MB and was renamed to
      `t.log.1`, every subsequent line went **into `t.log.1`** — the file
      contained both the pre- and post-rotation lines, and `t.log` did not
      exist. `check_rotate` then stat'd a missing path and returned early
      on every line thereafter, so after 1000 more lines the directory held
      exactly one file, no `t.log`, and no `t.log.2`.

      Three consequences, all silent: the 10 MB cap fires **once per
      process lifetime**, `MAX_LOGS = 5` is **never** enforced past the
      first rotation, and an operator tailing `client.log` watches a file
      frozen at the rotation moment while the server logs into `.1`.

      `check_rotate` is now a method that asks the *handle* for its length,
      drops it before renaming (Windows refuses to rename a file with an
      open handle — so there this was the difference between rotating and
      silently not rotating), and reopens under the same lock.

      *That also removes a syscall per line, which the input auditor
      measured separately at **0.0154 ms** — ~1.5% of a core at 1000
      lines/s, on the render thread for a client. The file's own docs
      record per-line open/close having been removed as "a measurable
      stutter"; the per-line stat was the same waste, surviving.*

      *Two audits missed this because `log.rs`'s 123-line test module
      covers only logfmt escaping and target filtering — **zero rotation
      coverage**.*

### Closed from the third audit

- [x] **T3 — `Batch` capacity decays instead of ratcheting.** Fixed. A
      windowed peak plus a lazy shrink: `clear` tracks the largest frame
      within a `DECAY_FRAMES` window, and at the window boundary shrinks
      to twice that peak, floored at the initial reserve. A scene in
      constant or oscillating use keeps its buffer; a spike's memory comes
      back.

      *Recovery takes **two** windows, ~4 s at 60 fps, not one — the
      spike's own frame lands inside the first window, so that window's
      peak is the spike and shrinking to twice it is a no-op. Deliberate:
      acting on the first window would shrink a scene that spiked on its
      opening frame before it had shown what it steadily needs. Pinned by
      `recovery_completes_within_two_windows`.*

      *My first version made the peak an all-time high with no reset, so
      after a 200k frame it stayed 200k forever and the slack test
      compared `200000 > 400000` — permanently false. A high-water mark
      that never falls is precisely the ratchet the fix exists to remove,
      and I had rebuilt it inside the fix. Caught by the test failing, not
      by reading.*

      *Six tests, and they are the only ones in `batch.rs` — ~770 lines
      holding the renderer's core geometry type, every primitive and the
      vertex layout, previously covered only indirectly through
      `materials_render`. The gap is narrowed, not closed.*

      *The GPU-side caps in `upload_batch` still ratchet: the same spike
      pins ~137 MB of buffers with no shrink path. Not fixed here —
      recreating a GPU buffer on a decay schedule is a separate change
      with its own failure mode, and the CPU half is where the 92 MB sat.*

- [x] **T8 — `TileGrid` multiplied dimensions in `u32`.** Fixed.
      `new_filled`, `rebuild_from_rows`, `tile_at` and `set` all widen to
      `usize` before multiplying, and the two allocating paths use
      `checked_mul` so an impossible grid fails loudly rather than
      returning one that lies about itself.

      *Four tests. The 65,536² case cannot be allocated in a test, so that
      one asserts the arithmetic directly — `wrapping_mul` is 0, the
      honest product is 4,294,967,296 — which is exactly the condition the
      old code got wrong.*

- [x] **T9 — a walker tunnelled through walls at one tile per tick.**
      Fixed. `integrate_walker` now splits the move into sub-steps of at
      most `MAX_COLLIDE_STEP` (0.5 world units) and resolves collision
      after each, because an overlap-push resolver cannot see a wall the
      walker jumped clean over.

      *`collide` changed from `FnOnce` to `FnMut`, which is what made
      substepping expressible at all. All four in-tree callers pass
      non-capturing closures and are unaffected; void-claim's
      `character::integrate` passes one calling its own tile collision,
      which satisfies `FnMut` unless it moves a captured value.*

      *Wobble is split along with forward motion rather than applied once
      at the end — otherwise a substepped walker would take its whole
      sideways lurch after the final collision check. A stationary walker
      still resolves once, so a wall that moved onto it still pushes out.*

      *Five tests. The load-bearing one drives 1.2 m/tick — 12 m/s
      sprinting at the 30 Hz headless default — straight at a 1 m wall and
      asserts the walker stays on its own side; against the old code it
      ended up through it. One test's tolerance needed loosening from 1e-9
      to 1e-6: ten summed `0.9/√2` steps differ from one `9.0/√2` step by
      1.7e-7 in f64, which is accumulation order, not divergence.*

- [x] **T10 / T11 — key releases are readable, and `key_pressed` says
      what it means.** `key_released` and `mouse_released` added beside
      the existing accessors. `keys_released` had been written and cleared
      on the same one-step schedule as presses since edges existed, with
      no way to read it — so hold-to-charge/release-to-fire could not be
      expressed through this API at all, while the mouse half was readable
      only by being a public field.

      *`key_pressed`'s docs now record that a press may already be over: a
      tap inside one frame sets the press edge and leaves `key_down`
      false, so `if key_pressed { start_hold() }` + `while key_down` never
      starts. Four tests, including that sub-frame tap.*

### Still open from the third audit `new_filled` does `vec![val; (w * h) as usize]` — the multiply
      happens in `u32` before the widening, so it wraps in release. Two
      distinct failures, both measured:

      - `new_filled(70_000, 70_000)` reports `dims() == (70000, 70000)` but
        allocates 605,032,704 cells against a true product of 4.9e9.
        `in_bounds(0, 60_000)` returns **true**, then `tile_at(0, 60_000)`
        **panics** on the index.
      - `new_filled(65_536, 65_536)` wraps to exactly **0**. `dims()` still
        reports `(65536, 65536)`, `is_empty()` is true, every `tile_at`
        returns `T::default()`, and `set` silently no-ops. A completely
        inert grid with no panic and no error.

      `tile_at` computes `r * self.w + c` in `u32` too, so the index wraps
      independently of the allocation. `rebuild_from_rows` has the same
      `(w * h) as usize`.

      *Latent: no in-tree caller passes large dims, and void-claim's use is
      authored rooms of a few thousand cells (R4). This is a trap for a
      future procedural or streaming caller. Fixing it widens to `usize`
      before multiplying and asserts; none of R4's ten preserved method
      signatures change.*

- [ ] **T9 — a walker tunnels through a wall at one tile per tick.**
      MEASURED, not fixed. `tile_collide` resolves by overlap push, not by
      sweeping the movement segment, and `integrate_walker` applies the
      full step before handing the final position over. A step that lands
      past the wall's far face overlaps nothing.

      Threshold measured by bisection at exactly **displacement ≥ tile
      size**. Through `SPRINT_MULT = 3.0`:

      | tick rate | sprinting cutoff | walking cutoff |
      | --- | --- | --- |
      | 60 Hz (`run`) | 20.0 m/s | 60.0 m/s |
      | 30 Hz (`run_headless`) | **10.0 m/s** | 30.0 m/s |

      End-to-end through the real path: 12 m/s sprinting at 30 Hz went
      **through** a solid wall; 5 m/s was blocked.

      *A human-scale walker at 3-5 m/s has a wide margin at either rate. A
      vehicle, dash, or knockback reusing this path at 30 Hz crosses it at
      a plausible 10 m/s — and the engine's own headless default being
      30 Hz halves the margin versus the windowed loop. No in-tree caller
      constructs `WalkParams` at all. The smaller fix is substepping in
      `walk.rs` when `step > tile_size`; the alternative needs
      `tile_collide` to take the pre-move position, which is an API
      change.*

- [ ] **T10 — key release edges are recorded but unreadable.** `InputState`
      maintains `keys_released`, writes it, and clears it on the same
      one-step schedule as `keys_pressed` — but exposes **no accessor**.
      `mouse_buttons_released` is at least a `pub` field; the keyboard half
      is write-only dead state.

      *A game wanting hold-to-charge/release-to-fire cannot read a key
      release from this API. Three lines to add `key_released` beside
      `key_pressed`. Relevant to the replay-driver case `Cargo.toml`
      explicitly keeps `winit` unconditional for.*

- [ ] **T11 — a sub-frame tap reports `key_pressed` but never
      `key_down`.** Measured: press and release within one frame yields
      `key_pressed == true`, `key_down == false`. This is *correct* — the
      press is not lost — but logic shaped as
      `if key_pressed(K) { start_hold() }` followed by `while key_down(K)`
      silently never starts. At 62 Hz polling, a fast tap or a replayed
      input pair can land both events in one frame.

      *Documentation, not a code change: a note on `key_pressed` saying the
      press may already be over.*

- [ ] **T3 — `Batch` capacity ratchets to its high-water mark forever.**
      MEASURED, not fixed. `Batch::clear` clears length but never
      capacity, and nothing shrinks the three `Batch` instances the
      renderer owns (`batch`, `offscreen_batch`, `mask_batch`).

      | workload | CPU `Vec` retained |
      | --- | --- |
      | fresh `Batch::new()` | 0.7 MB |
      | after one 200k-rect frame | **92.0 MB** |
      | after 600 quiet frames (50 rects each) | 92.0 MB |
      | one spike into all three | **264 MB**, retained |

      The GPU side compounds it: `upload_batch` grows
      `*vcap = (vlen*2).min(MAX_VERTS)` with no shrink path, so the same
      spike pins ~137 MB of GPU buffers.

      **Spike-shaped, not steady-state.** A game with a stable per-frame
      vertex count settles and the retention is the intended amortisation.
      It bites a session with a *transient* peak — a zoomed-out map, a
      particle storm, a debug overlay — where the cost is paid once and
      never returned.

      *The fix is a decay policy, not an unconditional shrink: shrinking
      every frame reintroduces exactly the per-frame allocation the
      capacity exists to avoid. Track a rolling high-water mark and
      `shrink_to_fit` when the peak has not been approached for N frames.
      `clear()` is right for the common path; only the ratchet is wrong.*

### Still open from the second audit

- [x] **N3 — silent slow-motion under sustained tick overrun.** Fixed.
      `app_headless.rs` had no instrumentation at all: no timing around
      `fixed_update`, no count of steps requested versus run, no overrun
      signal. When a tick overruns, the loop skips its sleep, feeds true
      elapsed time to `advance`, and `MAX_ACCUM_S` silently discards the
      excess.

      **The counter lives in `Timestep`, not in the loop.** The clamp at
      `time.rs:advance` is the exact moment sim time is destroyed, and it
      is the only place that knows how much: by the time `advance`
      returns, the excess is gone and a caller could only guess at it.
      So `Timestep` now accumulates `dropped_seconds()` and `stepped()`,
      and the loop reads them.

      `HeadlessConfig` gains `on_health: Option<Box<dyn FnMut(TickHealth)>>`
      and `health_every`. A callback rather than a returned handle, matching
      how `should_run` is already passed — the engine takes no view on how
      a server shares this with a health endpoint or a log, and a game
      closes over an `Arc<Mutex<_>>` to read it from another thread.
      `TickHealth` carries ticks, elapsed, achieved vs target hz, dropped
      sim seconds, and mean/worst `fixed_update` time, plus
      `realtime_ratio()` and `keeping_up()`.

      *Measurement is opt-in and costs nothing when off: with no sink the
      loop skips the `Instant::now()` calls entirely rather than timing and
      discarding. `health_reporting_is_off_by_default` pins that.*

      *Six tests: four on the accounting (a healthy loop drops exactly
      zero — an alert that fires in the healthy case gets muted; the clamp
      boundary drops nothing; loss accumulates rather than reporting only
      the last frame; sustained overrun shows a growing deficit), two on
      the loop (a measured run reports, an unmeasured one does not).*

      *Adding two fields to `HeadlessConfig` broke eight struct literals,
      two of them in `tests/crash_recovery.rs` which only compiles on the
      `persist` axis and which the IDE never flagged. Found by enumerating
      every `HeadlessConfig` in the tree at once instead of chasing
      diagnostics one line number at a time — the analyzer shows one axis's
      view and I took it for the whole tree. All eight now use
      `..Default::default()`, so the next field added breaks none of them.*

      | `fixed_update` cost | sim/wall | sim time lost per 3 s |
      | --- | --- | --- |
      | 20 ms | 98.5% | 0.04 s |
      | 35 ms | 94.2% | 0.18 s |
      | 50 ms | 66.0% | 1.07 s |
      | 100 ms | 33.0% | 2.30 s |

      Degradation is smooth, so there is no threshold anyone notices
      crossing. The asymmetry is the point: the *windowed* loop measures
      `update_ms`, rolls it into `PerfStats`, prints `[perf]` every second
      and stores a `PerfSnapshot` for an overlay (`app.rs:393`, `:406`,
      `:182`). The server — the one loop with no human watching it —
      measures nothing. An operator's first evidence of overload is player
      complaints. Fixing it touches `app_headless.rs` only.

- [x] **N4 — `!entered.contains(id)` was a linear scan per visible
      entity.** Fixed, and the largest win of the second audit. Both
      drivers built the update list this way. Steady state was already
      free, but a mass-arrival tick was catastrophic. `diff` returns
      `entered` sorted, so the filter is now a `binary_search_by_key`:

      | 500 clients, 1834 visible | scan | binary search |
      | --- | --- | --- |
      | 0 entered (steady state) | 0.35 ms | 0.00 ms |
      | 280 entered | 71.95 ms | 10.45 ms |
      | 1000 entered | 192.70 ms | 11.75 ms |
      | 1834 entered (full arrival) | 253.95 ms | 13.90 ms |

      *Both filters verified to select identical updates before timing.
      That the worst case is the post-keyframe tick matters: `KeyframeBudget`
      staggers keyframes deliberately, so large-`entered` ticks recur
      rather than passing once.*

      *A sorted-merge would have been faster still, but it requires
      `visible` sorted by `EntityId` — true in both reference drivers only
      because they insert into the grid in id order, and not something
      `diff` can require of an arbitrary caller. The binary search needs no
      such assumption. Nearly designing on that unguaranteed property is
      the same error class as the generation-0 assumption in N1.*

- [x] **N5 — `diff` allocated a fresh `HashSet` per client per tick.**
      Fixed with a caller-owned `DiffScratch`, mirroring `AoiScratch` —
      **but the win is ~5-6% at 1834 visible entities and nothing at 280,
      not the 27.5 ms the audit claimed.**

      That 27.5 ms was the cost of *building the membership set at all*,
      which the fix still pays; what reuse saves is the allocator traffic
      around it. Measured at 500 clients against an otherwise identical
      body: −0.33% at 280 visible, +1.06% at 1000, +5.56% at 1834, +5.87%
      at 1834 with heavy churn.

      *Three measurements disagreed before one was trustworthy, and the
      sequence is worth keeping.* Timing a bare `HashSet::collect` against
      the whole of `diff` made the fix look 33% slower — comparing an
      allocation against a function that also does baseline lookups and two
      sorts. Timing it against a **local copy** of the pre-fix body showed
      the same 20-39% gap, stable across runs, which looked structural;
      the explanation offered for it — that `collect` sizes its table
      better than `clear`+`extend` — was then disproved outright: fresh
      `collect`, cleared `extend`, `+reserve` and `with_capacity` all
      measure within noise (6.56/6.49/6.45/6.63 ms at 1834), and `clear`
      does retain capacity (3584 slots before and after).

      The real cause was that the local copy optimises differently from a
      cross-crate call. A three-arm harness — local+alloc, local+reuse,
      real `diff` — showed the shipped code **fastest of the three** in
      every row (26.40 ms against 34.42 ms at 1834), and the honest
      isolation of the set change is local-vs-local.

      *Only an A/B where every arm is identical except the one variable
      gives a number worth quoting. Four contaminated comparisons in the
      first audit, three here.*

- [x] **N7 — the reservation scan was unobservable, and worse than
      documented.** Accessors added; the scan itself is still open.

      `Ledger::available_at` filters *every* resident reservation on every
      call, and every non-mint transfer calls it. `reservations` was a
      private map with no accessor, so the build-up that drives that cost
      could not be seen at all — the most valuable number for a status
      page was the one number the type would not surface.
      `reservation_count()` and `lapsed_reservations(now_tick)` now do,
      the latter sharing a predicate with `expire_reservations` so it
      cannot promise a sweep that will not happen.

      **Re-measured rather than taken on report**, and the numbers moved:

      | resident holds | one `available_at` |
      | --- | --- |
      | 500 | 3.2 µs |
      | 5,000 | 30.4 µs |
      | 50,000 | 334.9 µs |
      | 180,000 | **1957 µs** |
      | after a sweep | 0.1 µs |

      At 500 players taking a hold every ten seconds, an eight-hour shard
      reaches ~180k holds unswept — where one spend check costs 2 ms and
      sixteen of them exhaust a 33.3 ms tick on affordability tests alone.

      *The audit reported 1.30 / 377.7 / 1.9 µs. The shape reproduced and
      the magnitudes did not, which is why this was re-derived: every other
      number committed today I measured myself, and of the two figures I
      took on trust, one (N4's 271 ms) held and one (N5's 27.5 ms) was
      measuring something other than what it claimed.*

      **The correction that matters: cost tracks *resident* holds, not
      lapsed ones.** 500 live and 500 lapsed measure identically, because
      the filter walks them either way — the audit framed this as a
      sweep-hygiene problem, and it is not. Sweeping helps a shard whose
      holds have lapsed; a shard whose holds are genuinely all live gets no
      relief and needs the scan fixed (index by account, or a per-account
      running total). That remains open.

      *The docs on `Reservation` and `expire_reservations` said forgetting
      the sweep "leaks a little memory but never a player's money". The
      money half is true and verified. The memory half was wrong in kind,
      not degree: it is a time leak on the spend path. Both corrected.*

- [x] **N6 — no timeout on the ack read path.** Fixed at the caller, and
      documented as a property of `read_msg` rather than a bug in it.

      `read_msg` is generic over any `AsyncRead` and cannot hold a clock —
      a timer there would drag a tokio timer feature onto every caller,
      including the tests that drive it over an in-memory buffer with no
      runtime. A control channel idle for an hour is also not an error, so
      the deadline is a per-protocol fact exactly like `max_len`. Its docs
      now say so explicitly, with the `tokio::time::timeout` shape.

      The reference driver imposes the bound: five seconds for four bytes
      (`replication_server.rs`). Without it a peer that opens a uni stream
      and says nothing parks the sequential accept loop, starving every
      later ack on that connection.

      *`net` gains `tokio/time`. The base pin was `io-util` only, so the
      one thing a netcode consumer needed to defend this path was
      available solely to a build that also pulled `ledger-pg` — unrelated
      and absurd. One timer, and the driver can show the right shape.*

- [x] **N9 — `ledger_pg.rs` had no test module at all.** 776 lines of
      writer thread, retry policy, reconnect and replay, reachable only
      through Postgres-gated integration tests that skip without
      `VOID_ENGINE_PG_URL`. On a developer machine `cargo test` reported
      green while exercising none of it, and the `ledger-pg` CI axis —
      which runs without a database — proved only that it compiles.

      `is_transient` is the piece worth rescuing: it decides whether a
      fault retries or kills the ledger, and it is wrong in both
      directions expensively — a fatal fault retried hangs with writes
      refused, a transient one called fatal turns a three-second failover
      into an outage needing a restart. It took a
      `tokio_postgres::Error`, which has no public constructor, so the
      decision was untestable. Split into `is_fatal_code(&SqlState)`,
      which is a pure function over a constructible type, with four tests:
      the fatal set, the transient set, the unlisted-code default, and
      that the two sets do not overlap.

      *Unlisted codes default to **retryable**, deliberately: the fatal
      set is small and well understood, and an unknown SQLSTATE is far
      likelier to be an availability blip than a permanent schema fault.*

      *Three constants I used did not exist — I wrote `SERIALIZATION_FAILURE`
      and `DEADLOCK_DETECTED` from the shape of the API. The crate prefixes
      class-40 codes `T_R_`. Verified all sixteen names against
      `sqlstate.rs` in the registry rather than trusting the compiler's
      "similar name" suggestion, which is the same shortcut that produced
      the bad names.*

      *`replay_into` stays untested: it takes `&[tokio_postgres::Row]`,
      equally unconstructible, and extracting a testable core would mean
      inventing an intermediate row type whose only purpose is the test.
      Recorded rather than done.*

- [x] **N10 — `available_at` scanned every resident hold.** Indexed.
      Two `HashMap`s beside `reservations`: a total per `(account, asset)`,
      and holds grouped by deadline in a `BTreeMap` so the lapsed portion
      is a range query rather than a filter. Maintained at all four
      mutation sites — reserve, spend, release, sweep.

      | holds | old scan | indexed |
      | --- | --- | --- |
      | 500, shared deadline | 3.2 µs | 0.2 µs |
      | 180,000, shared deadline | 1957 µs | 0.2 µs |
      | 50,000, distinct deadlines | ~335 µs | 48 µs |
      | 180,000, distinct deadlines | 1957 µs | **220 µs** |

      **The honest row is the last one.** A real server takes holds on
      different ticks, so they expire on different ticks, and the range
      walks one entry per distinct deadline below `now` — 9x better, not
      constant time. Sixteen spend checks still cost 3.5 ms at 180k.
      `expire_reservations` on the tick is what actually bounds this; the
      index buys headroom, not immunity.

      *Which corrects N7's note that the scan needed "index by account, or
      keep a per-account running total". A per-account total alone is
      exactly wrong: lapsed holds must stop counting the instant their
      deadline passes, and a precomputed total cannot know that. The
      deadline grouping is the part that makes the total usable.*

      *Four tests, because a parallel total that drifts from the holds it
      summarises is a dupe vector in both directions — too low lets a
      player spend reserved funds twice, too high locks money that was
      released. One walks a sequence of reserve/spend/release/sweep
      operations asserting the index equals a full scan after every step.*

### Measured and deliberately NOT fixed

*Recorded so nobody re-derives them. Each is real and each needs a
workload nobody reaches.*

- **Never-shrinking component columns.** `ComponentStorage::insert`
  resizes to `index + 1` and nothing shrinks (`ecs/world.rs:47-52`,
  `:70-74`). The watermark tracks **peak concurrency, not cumulative
  spawns**: 600k spawns over 3,000 ticks left `next_index` at **700** —
  the free list recycles perfectly. Only overlapping lifetimes raise it.
  Needs millions of genuinely concurrent entities to matter; at 500
  players it is a few hundred KiB.
- **Sparse-component queries scan the whole index space.** `iter2` walks
  A's entire column (`ecs/world.rs:416`). Measured 75x slower for 500
  players spawned *after* 200k transients than before them — but the
  absolute cost is 0.0376 ms. Twenty such systems need **~5M concurrent
  entities** to threaten a tick. The ratio is alarming and the number is
  not; reporting it as a defect at 500 players would be exactly the error
  that killed four of the first audit's five items.
- **`despawn` walks every registered storage** (`ecs/world.rs:150-152`):
  13.6 ns at 2 component kinds, 20.9 ns at 10. Linear in kinds, trivial
  constant. 10 µs per tick at 500 despawns.
- **`physics::integrate` clones every mover into a fresh `Vec`**
  (`physics.rs:34-36`): 0.0129 ms at 500 movers, 1.66 ms at 50k. Worth
  revisiting only above ~50k movers.

### Checked and found sound

Ack replay and reordering; malformed-ack rejection; hostile length
prefixes (`max_names`/`max_items` are checked before `with_capacity`, so a
10-byte packet claiming 8192 items costs 170 ns and errors); `bits_remaining`
underflow (unreachable — `read_bit` refuses before advancing); the name
table leaking server-only components (filtered on `on_the_wire()`);
`KeyframeBudget` (1000 clients drain in 77 ticks); chunking and `ChunkHint`
termination; `SpatialGrid::remove`/`update` tripping the documented
bare-index hazard (neither driver calls them); the spiral-of-death guard
(`MAX_ACCUM_S` behaves as documented); entity-id generational safety and
free-list recycling; the narrow-phase SAT helpers (allocation-free, no
hidden quadratic); `tile_collide` (fixed 5×5 neighbourhood, bounded).

---

## Downstream breakage owed

The engine is kept clean in preference to backward compatibility, so a
breaking change lands here and the consuming game is fixed after. This
records what is currently owed, because nothing else does — neither repo
is in this workspace and `cargo check` here will never notice.

*Checked after `77f7c4c`: the glyph atlas replaced the renderer's texture
and broke nothing downstream. mini-miner-2 fails on exactly the two
`send_chunked` call sites below and nothing else; void-claim makes 355
`draw_text` calls but zero `push_quad`, zero raw `uv` writes and zero
`Vertex` constructions, so it only ever touches the high-level API the
atlas kept intact. That was the point of reserving the white texel inside
the atlas rather than switching bind groups mid-pass — worth knowing
before the next draw-path change, because the same reasoning applies to
instancing.*

*Checked after `91da883`: `SpatialGrid` gained `clear`/`remove`/`update`
and **`len()` changed meaning** — it now counts live colliders, where it
used to return `bounds.len()`, which is the index space. With holes those
differ. Nothing downstream notices, and the reason is worth recording so
it is not re-derived: void-claim's nine `self.grid.is_empty()` calls
(`module.rs`, `station_interior/floor.rs`) are all on `TileGrid<TileKind>`
fields, a different type whose `is_empty` was untouched. Its only
`SpatialGrid` field is `npc.rs:78`, which calls `query_circle` and never
`len`/`is_empty`; its other two grids (`collision.rs:142`,
`projectile.rs:144`) are fresh locals. mini-miner-2 does not use
`SpatialGrid` at all. Anything sized per-slot wants the new
`slot_count()`, not `len()`.*

*Those three void-claim grids are all built fresh per call, so `clear()`
buys them nothing without first holding the grid across ticks — an
improvement to offer, not breakage owed.*

**mini-miner-2 was broken and is now fixed** (2026-09-11), though the fix
is *uncommitted in that repo* — its tree carries someone else's in-flight
`Cargo.toml`/`Cargo.lock` work (the `path =` switch documented below), and
staging `host.rs` would have swept that into the same commit. It builds
and its 341 tests pass; committing is the owner's call.

*The two errors turned out to be one break. `ChunkHint` went in as
parameter 4, so the existing `&format_args!(...)` label slid into its slot
— hence `expected ChunkHint, found Arguments` at `:553` alongside the
missing-argument error at `:545`. One call site, two diagnostics.*

*`ChunkHint` now lives in the send loop (`host.rs`, before the `loop` that
owns `conn`) and is threaded into `send_snapshot_datagrams`, which is what
the note below prescribed: a fresh hint per snapshot compiles and behaves
correctly while discarding the capacity memory that is the whole point.
The test helper takes `&mut ChunkHint::new()` inline, because those tests
pin cold-start policy and a shared hint would let one test's discovery
change another's behaviour.*

**What the original breakage was.** It depends on this repo by
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
| Text: 200 nameplates, one quad per glyph | **0.153 ms / 8,800 verts** | 9.8 ms / 208,800 |
| Text: 1000 nameplates | **1.34 ms / 44,000 verts** | 39.7 ms / 1,044,000 |
| Lights: 384 × radius 1200 px | **25.1 ms/frame** | 152% of 16.6 ms |
| Lights: 128 × radius 640 px | 5.53 ms/frame | — |
| Lights: 64 × radius 320 px | 1.94 ms/frame | — |
| Geometry draws per frame | **4 `draw_indexed`** | — |
| Draw sites in `frame.rs` | 22 (18 fullscreen) | 14 claimed |
