-- 02_create_weather_documents.sql
-- Raw document store for narrative weather text harvested from the
-- National Weather Service API.
--
-- One row = one embeddable narrative: either an active alert (headline +
-- description + protective-action instruction) or a single forecast period
-- (its detailedForecast paragraph).

CREATE TABLE IF NOT EXISTS weather_documents (
    -- Stable dedup key. Alerts use "alert:<NWS alert id>"; forecast periods
    -- use "forecast:<office>:<gridX,gridY>:<period start>:<period name>", so
    -- a re-issued forecast for the same period updates in place.
    id              TEXT PRIMARY KEY,

    -- Where this narrative applies.
    location        TEXT NOT NULL,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    grid_office     TEXT,
    grid_x          INTEGER,
    grid_y          INTEGER,

    -- 'alert' or 'forecast'. Lets retrieval filter to safety-critical
    -- products only, and keeps a multi-source pipeline honest.
    source_type     TEXT NOT NULL,

    -- Human-facing labels.
    event           TEXT,
    headline        TEXT,

    -- Alert-only fields; NULL for forecast rows.
    severity        TEXT,
    urgency         TEXT,
    certainty       TEXT,
    area_desc       TEXT,

    -- The free text that actually gets chunked and embedded.
    narrative_text  TEXT NOT NULL,

    issued_at       TIMESTAMPTZ,
    effective_at    TIMESTAMPTZ,
    expires_at      TIMESTAMPTZ,

    -- SHA-256 of narrative_text. The embedding job compares this against the
    -- hash stored on each vector, so re-running a sync only re-embeds text
    -- that genuinely changed.
    content_hash    TEXT NOT NULL,

    -- Full raw API response, kept for provenance and for reprocessing
    -- without re-hitting the API.
    payload         JSONB NOT NULL,

    synced_at       TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT weather_documents_source_type_check
        CHECK (source_type IN ('alert', 'forecast'))
);

CREATE INDEX IF NOT EXISTS idx_weather_documents_location
    ON weather_documents (location);

CREATE INDEX IF NOT EXISTS idx_weather_documents_source_type
    ON weather_documents (source_type);

CREATE INDEX IF NOT EXISTS idx_weather_documents_effective_at
    ON weather_documents (effective_at DESC);

-- Verify the shape of the table.
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'weather_documents'
ORDER BY ordinal_position;
