-- 1. Raw documents by source and location.
SELECT location, source_type, COUNT(*) AS document_count
FROM weather_documents
GROUP BY location, source_type
ORDER BY location, source_type;

-- 2. Confirm every vector belongs to a valid raw document.
SELECT COUNT(*) AS orphan_embedding_count
FROM weather_embeddings AS e
LEFT JOIN weather_documents AS d ON d.id = e.document_id
WHERE d.id IS NULL;

-- Expected: 0.

-- 3. Embedding coverage. A gap can be expected immediately after a new sync.
SELECT
    COUNT(*) AS total_documents,
    COUNT(*) FILTER (WHERE embedded.document_id IS NOT NULL) AS embedded_documents,
    COUNT(*) FILTER (WHERE embedded.document_id IS NULL) AS awaiting_embeddings
FROM weather_documents AS d
LEFT JOIN (
    SELECT DISTINCT document_id FROM weather_embeddings
) AS embedded ON embedded.document_id = d.id;

-- 4. Confirm the vector dimension is 384.
SELECT vector_dims(embedding) AS dimensions, COUNT(*) AS embedding_count
FROM weather_embeddings
GROUP BY vector_dims(embedding);

-- 5. Confirm pgvector indexes and constraints.
SELECT indexname, indexdef
FROM pg_indexes
WHERE tablename IN ('weather_documents', 'weather_embeddings')
ORDER BY tablename, indexname;

SELECT conname, pg_get_constraintdef(oid) AS definition
FROM pg_constraint
WHERE conrelid IN (
    'weather_documents'::regclass,
    'weather_embeddings'::regclass
)
ORDER BY conrelid::regclass::text, conname;

