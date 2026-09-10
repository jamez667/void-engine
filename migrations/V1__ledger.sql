-- The ledger schema.
--
-- Append-only by intent and, where the database can enforce it, by
-- permission. An entry is never updated or deleted: a correction is a new
-- compensating entry, which is itself auditable. That is the difference
-- between a log you can trust in an investigation and one you cannot.

CREATE TABLE IF NOT EXISTS ledger_entries (
    seq             BIGSERIAL PRIMARY KEY,

    -- Deduplication key, from (session, client_seq, action). UNIQUE is the
    -- whole anti-dupe mechanism: a client that resends a trade after a
    -- timeout — the classic dupe vector — is rejected by the database
    -- rather than by game logic remembering to check. Both entries of one
    -- transfer share the key, so the constraint is on (idem_key, side).
    idem_key        TEXT        NOT NULL,
    -- 'd' debit, 'c' credit. Together with idem_key this is what makes a
    -- transfer exactly two rows and no more.
    side            CHAR(1)     NOT NULL CHECK (side IN ('d', 'c')),

    -- Simulation tick, for correlating an entry with a checkpoint.
    tick            BIGINT      NOT NULL,

    -- Account identity is (kind, id): kind distinguishes player, system,
    -- mint and burn; id is empty for the two singletons.
    account_kind    TEXT        NOT NULL,
    account_id      TEXT        NOT NULL DEFAULT '',
    counterparty_kind TEXT      NOT NULL,
    counterparty_id TEXT        NOT NULL DEFAULT '',

    asset           TEXT        NOT NULL,

    -- Minor units, signed. BIGINT not NUMERIC: the engine's Amount is i64
    -- and exactness matters more than range. Floating point is excluded on
    -- purpose — it is not exact above 2^53, so two paths to the same
    -- balance can differ and reconciliation stops being possible.
    delta           BIGINT      NOT NULL,

    -- Why, from the game's own taxonomy ("buy_kind", "quest_reward").
    -- This is what turns a suspicious number into a diagnosis.
    reason          TEXT        NOT NULL,
    -- Who acted: a player, a GM, or the system. Present from the start so
    -- GM actions are auditable without a later schema change.
    actor           TEXT        NOT NULL,

    committed_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The constraint the anti-dupe guarantee rests on.
CREATE UNIQUE INDEX IF NOT EXISTS ledger_entries_idem
    ON ledger_entries (idem_key, side);

-- Balance derivation: sum every delta for one account and asset.
CREATE INDEX IF NOT EXISTS ledger_entries_account
    ON ledger_entries (account_kind, account_id, asset, seq);

-- The zero-sum audit sums per asset across everything.
CREATE INDEX IF NOT EXISTS ledger_entries_asset
    ON ledger_entries (asset);

-- Correlating durability with checkpoints.
CREATE INDEX IF NOT EXISTS ledger_entries_tick
    ON ledger_entries (tick);


-- Materialised balances. Purely a cache: it is rebuildable from
-- ledger_entries alone, and `audit_balances_match_entries` proves it has
-- not drifted. If it can drift undetected it is a dupe vector, which is
-- why that audit exists rather than trusting the cache.
CREATE TABLE IF NOT EXISTS ledger_balances (
    account_kind    TEXT   NOT NULL,
    account_id      TEXT   NOT NULL DEFAULT '',
    asset           TEXT   NOT NULL,
    amount          BIGINT NOT NULL,
    last_seq        BIGINT NOT NULL,
    PRIMARY KEY (account_kind, account_id, asset)
);


-- Durability watermark, one row. A checkpoint may never claim a tick above
-- acked_tick: restoring a checkpoint that shows a purchase the ledger never
-- recorded is precisely a dupe.
CREATE TABLE IF NOT EXISTS ledger_watermark (
    id          BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (id),
    acked_tick  BIGINT  NOT NULL DEFAULT 0
);

INSERT INTO ledger_watermark (id, acked_tick)
VALUES (TRUE, 0)
ON CONFLICT (id) DO NOTHING;
