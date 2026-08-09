-- 04_verify_schema.sql
-- Run after the notebook has synced and embedded, to inspect Lakebase by hand.

-- 1. Both tables exist, and the vector column has the expected width.
SELECT 'weather_documents'  AS table_name, to_regclass('weather_documents')  IS NOT NULL AS present
UNION ALL
SELECT 'weather_embeddings' AS table_name, to_regclass('weather_embeddings') IS NOT NULL AS present;

SELECT a.attname AS column_name,
       format_type(a.atttypid, a.atttypmod) AS column_type
FROM pg_attribute a
JOIN pg_class c ON c.oid = a.attrelid
WHERE c.relname = 'weather_embeddings'
  AND a.attname = 'embedding';
-- Expect: vector(1024)

-- 2. Row counts and coverage.
SELECT
    (SELECT count(*) FROM weather_documents)                     AS documents,
    (SELECT count(*) FROM weather_documents WHERE source_type = 'alert')    AS alerts,
    (SELECT count(*) FROM weather_documents WHERE source_type = 'forecast') AS forecasts,
    (SELECT count(DISTINCT location) FROM weather_documents)     AS locations,
    (SELECT count(*) FROM weather_embeddings)                    AS chunks,
    (SELECT count(DISTINCT document_id) FROM weather_embeddings) AS embedded_documents;

-- 3. Anything synced but not yet embedded (should be empty after a full run).
SELECT d.id, d.location, d.source_type, d.headline
FROM weather_documents d
WHERE NOT EXISTS (
    SELECT 1 FROM weather_embeddings e
    WHERE e.document_id = d.id
      AND e.content_hash = d.content_hash
)
LIMIT 20;

-- 4. Sample the newest documents.
SELECT id, location, source_type, event, severity,
       left(narrative_text, 160) AS narrative_preview,
       effective_at, synced_at
FROM weather_documents
ORDER BY synced_at DESC
LIMIT 10;

-- 5. Sample chunks, with how many chunks each document produced.
SELECT e.document_id,
       count(*) AS chunk_count,
       min(length(e.chunk_text)) AS shortest_chunk,
       max(length(e.chunk_text)) AS longest_chunk,
       max(e.model_name) AS model_name
FROM weather_embeddings e
GROUP BY e.document_id
ORDER BY chunk_count DESC
LIMIT 10;

-- 6. Confirm the HNSW index exists and is being used. The EXPLAIN below
--    should show an "Index Scan using idx_weather_embeddings_hnsw".
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename = 'weather_embeddings';

-- Replace the literal below with any 1024-dimension vector, e.g. copy one
-- straight out of the table:
--   SELECT embedding FROM weather_embeddings LIMIT 1;
--
-- EXPLAIN ANALYZE
-- SELECT d.headline, 1 - (e.embedding <=> '[...]'::vector) AS similarity
-- FROM weather_embeddings e
-- JOIN weather_documents d ON d.id = e.document_id
-- ORDER BY e.embedding <=> '[...]'::vector
-- LIMIT 5;
