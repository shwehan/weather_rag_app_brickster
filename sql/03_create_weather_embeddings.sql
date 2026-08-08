-- 03_create_weather_embeddings.sql
-- Chunk-level vector store for weather narratives.
--
-- VECTOR(384) matches sentence-transformers/all-MiniLM-L6-v2, the model used
-- by the ingestion notebook and by the search endpoint's query encoder. If you
-- change EMBEDDING_MODEL in config.py, change the width here to match:
--   all-MiniLM-L6-v2 / all-MiniLM-L12-v2 / bge-small-en-v1.5 -> 384
--   all-mpnet-base-v2 / bge-base-en-v1.5                     -> 768
--   bge-large-en-v1.5                                        -> 1024

CREATE TABLE IF NOT EXISTS weather_embeddings (
    -- "<document_id>#<chunk_index>", so re-embedding a document overwrites
    -- its own chunks instead of duplicating them.
    id            TEXT PRIMARY KEY,

    document_id   TEXT NOT NULL
                  REFERENCES weather_documents (id) ON DELETE CASCADE,

    chunk_index   INTEGER NOT NULL,
    chunk_text    TEXT NOT NULL,

    embedding     VECTOR(384) NOT NULL,

    model_name    TEXT NOT NULL,

    -- Copied from weather_documents.content_hash at embed time. A mismatch
    -- means the source narrative was updated and this vector is stale.
    content_hash  TEXT,

    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),

    CONSTRAINT weather_embeddings_document_chunk_key
        UNIQUE (document_id, chunk_index)
);

-- Join key back to the document store, used by every search query.
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_document_id
    ON weather_embeddings (document_id);

-- Lets the embedding job find stale vectors cheaply.
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_model_hash
    ON weather_embeddings (model_name, content_hash);

-- Approximate nearest-neighbour index for cosine distance. vector_cosine_ops
-- must match the operator used at query time (<=>), or Postgres will fall
-- back to a sequential scan.
--
-- m = 16 and ef_construction = 64 are pgvector's defaults: a good balance of
-- build time and recall at this corpus size. Raise ef_construction for better
-- recall on a much larger corpus, at the cost of a slower build.
CREATE INDEX IF NOT EXISTS idx_weather_embeddings_hnsw
    ON weather_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- On pgvector older than 0.5.0, drop the HNSW index above and use IVFFlat
-- instead. Note that IVFFlat must be built *after* the table holds data,
-- because it clusters existing rows:
--
--   CREATE INDEX idx_weather_embeddings_ivfflat
--       ON weather_embeddings
--       USING ivfflat (embedding vector_cosine_ops)
--       WITH (lists = 100);

-- Verify the vector column really is VECTOR and not an array of floats.
SELECT column_name, data_type, udt_name
FROM information_schema.columns
WHERE table_name = 'weather_embeddings'
ORDER BY ordinal_position;
