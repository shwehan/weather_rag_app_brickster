"""
The vectorization and retrieval core of the Weather Intelligence pipeline.

This module is deliberately free of Flask and of ``dbutils`` so that the exact
same code path runs in three places: the Databricks App, the interactive
notebook, and the scheduled Job. Whatever you validate in the notebook is what
the web UI executes.

Write path
    documents -> chunks -> vectors -> ``weather_embeddings`` (psycopg2 +
    ``execute_values``, cast to pgvector's ``VECTOR`` type with ``%s::vector``)

Read path
    query -> vector -> cosine distance with pgvector's ``<=>`` operator,
    joined back to ``weather_documents`` for display fields.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Iterable, Sequence

from psycopg2.extras import execute_values

import config
import lakebase

logger = logging.getLogger(__name__)

_model_lock = threading.Lock()
_model_cache: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Model handling
# ---------------------------------------------------------------------------


def get_model(model_name: str = config.EMBEDDING_MODEL_NAME):
    """Load a sentence-transformers model once per process and reuse it.

    Loading MiniLM takes a few seconds and ~90 MB of RAM, so the web app must
    never do it inside a request handler. The first caller pays the cost, and
    every later query -- including every ``POST /weather/search`` -- reuses the
    warm instance behind a lock so two simultaneous first-requests cannot race.
    """
    cached = _model_cache.get(model_name)
    if cached is not None:
        return cached

    with _model_lock:
        cached = _model_cache.get(model_name)
        if cached is not None:
            return cached

        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s", model_name)
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
        return model


def embed_texts(
    texts: Sequence[str],
    model_name: str = config.EMBEDDING_MODEL_NAME,
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed a list of strings and return plain Python float lists."""
    if not texts:
        return []
    model = get_model(model_name)
    vectors = model.encode(
        list(texts), batch_size=batch_size, show_progress_bar=False
    )
    return [[float(value) for value in vector] for vector in vectors]


def to_vector_literal(vector: Sequence[float]) -> str:
    """Render a vector in the text form pgvector accepts: ``[1,2,3]``."""
    return "[" + ",".join(f"{float(value):.7f}" for value in vector) + "]"


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_text(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> list[str]:
    """Split text into overlapping windows.

    Most NWS narratives fit in a single chunk; this matters mainly for long
    alert bodies where the description and the protective-action instruction
    together run past the window. The overlap keeps a sentence that straddles
    a boundary intact in at least one chunk.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    if len(cleaned) <= chunk_size:
        return [cleaned]

    step = max(1, chunk_size - chunk_overlap)
    chunks: list[str] = []
    for start in range(0, len(cleaned), step):
        window = cleaned[start : start + chunk_size].strip()
        if window:
            chunks.append(window)
        if start + chunk_size >= len(cleaned):
            break
    return chunks


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

_DOCUMENT_COLUMNS = (
    "id",
    "location",
    "latitude",
    "longitude",
    "grid_office",
    "grid_x",
    "grid_y",
    "source_type",
    "event",
    "headline",
    "severity",
    "urgency",
    "certainty",
    "area_desc",
    "narrative_text",
    "issued_at",
    "effective_at",
    "expires_at",
    "content_hash",
    "payload",
)


def upsert_documents(documents: Iterable[dict], page_size: int = 100) -> int:
    """Insert or update weather documents, keyed on the stable ``id``.

    Re-running a sync is safe: an alert that has not changed simply overwrites
    itself with identical values, and a re-issued forecast for the same period
    replaces the stale narrative rather than adding a duplicate row.
    """
    rows = []
    for doc in documents:
        rows.append(
            tuple(
                json.dumps(doc.get(column))
                if column == "payload"
                else doc.get(column)
                for column in _DOCUMENT_COLUMNS
            )
        )

    if not rows:
        return 0

    column_list = ", ".join(_DOCUMENT_COLUMNS)
    updates = ", ".join(
        f"{column} = EXCLUDED.{column}"
        for column in _DOCUMENT_COLUMNS
        if column != "id"
    )
    sql = f"""
        INSERT INTO {config.WEATHER_DOCUMENTS_TABLE} ({column_list}, synced_at)
        VALUES %s
        ON CONFLICT (id) DO UPDATE
            SET {updates},
                synced_at = now()
    """
    template = "(" + ", ".join(["%s"] * len(_DOCUMENT_COLUMNS)) + ", now())"

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, rows, template=template, page_size=page_size)
            conn.commit()
    return len(rows)


def fetch_pending_documents(
    limit: int | None = None,
    model_name: str = config.EMBEDDING_MODEL_NAME,
) -> list[dict]:
    """Return documents that have no current embedding.

    "Current" means an embedding produced by this model *from this exact text*.
    Comparing ``content_hash`` rather than just checking for the presence of a
    row means an updated alert gets re-embedded, while an unchanged one is
    skipped -- so the job is cheap to run on a schedule.
    """
    sql = f"""
        SELECT d.id, d.location, d.source_type, d.headline, d.narrative_text,
               d.content_hash
        FROM {config.WEATHER_DOCUMENTS_TABLE} d
        WHERE d.narrative_text IS NOT NULL
          AND length(trim(d.narrative_text)) > 0
          AND NOT EXISTS (
              SELECT 1
              FROM {config.WEATHER_EMBEDDINGS_TABLE} e
              WHERE e.document_id = d.id
                AND e.model_name = %s
                AND e.content_hash = d.content_hash
          )
        ORDER BY d.synced_at DESC
    """
    params: list[Any] = [model_name]
    if limit is not None:
        sql += " LIMIT %s"
        params.append(int(limit))
    return lakebase.run_query(sql, tuple(params))


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def write_embeddings(rows: Sequence[dict], page_size: int = 100) -> int:
    """Batch-write chunk embeddings into Lakebase.

    Each row is a dict with ``document_id``, ``chunk_index``, ``chunk_text``,
    ``embedding``, ``model_name`` and ``content_hash``. The embedding is passed
    as pgvector's text form and cast in SQL with ``%s::vector`` -- no Spark
    JDBC, and no post-hoc ``UPDATE ... ::vector`` pass needed.
    """
    if not rows:
        return 0

    values = [
        (
            f"{row['document_id']}#{row['chunk_index']}",
            row["document_id"],
            int(row["chunk_index"]),
            row["chunk_text"],
            to_vector_literal(row["embedding"]),
            row.get("model_name", config.EMBEDDING_MODEL_NAME),
            row.get("content_hash"),
        )
        for row in rows
    ]

    sql = f"""
        INSERT INTO {config.WEATHER_EMBEDDINGS_TABLE} (
            id, document_id, chunk_index, chunk_text, embedding,
            model_name, content_hash, created_at
        )
        VALUES %s
        ON CONFLICT (id) DO UPDATE
            SET chunk_text = EXCLUDED.chunk_text,
                embedding = EXCLUDED.embedding,
                model_name = EXCLUDED.model_name,
                content_hash = EXCLUDED.content_hash,
                created_at = now()
    """
    template = "(%s, %s, %s, %s, %s::vector, %s, %s, now())"

    with lakebase.get_connection() as conn:
        with conn.cursor() as cur:
            execute_values(cur, sql, values, template=template, page_size=page_size)
            conn.commit()
    return len(values)


def embed_pending_documents(
    limit: int | None = None,
    model_name: str = config.EMBEDDING_MODEL_NAME,
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    batch_size: int = 32,
    progress: bool = False,
) -> dict:
    """Chunk, embed and store every document that needs it.

    This is the whole of Part 2 in one call. The notebook runs it, the
    scheduled Job runs it, and ``POST /weather/embed`` runs it.
    """
    documents = fetch_pending_documents(limit=limit, model_name=model_name)
    if not documents:
        return {
            "documents_processed": 0,
            "chunks_written": 0,
            "model_name": model_name,
        }

    chunk_rows: list[dict] = []
    for doc in documents:
        for index, chunk in enumerate(
            chunk_text(doc["narrative_text"], chunk_size, chunk_overlap)
        ):
            chunk_rows.append(
                {
                    "document_id": doc["id"],
                    "chunk_index": index,
                    "chunk_text": chunk,
                    "content_hash": doc["content_hash"],
                    "model_name": model_name,
                }
            )

    if not chunk_rows:
        return {
            "documents_processed": len(documents),
            "chunks_written": 0,
            "model_name": model_name,
        }

    written = 0
    for start in range(0, len(chunk_rows), batch_size):
        batch = chunk_rows[start : start + batch_size]
        vectors = embed_texts(
            [row["chunk_text"] for row in batch],
            model_name=model_name,
            batch_size=batch_size,
        )
        for row, vector in zip(batch, vectors):
            row["embedding"] = vector
        written += write_embeddings(batch)
        if progress:
            print(f"  embedded {written}/{len(chunk_rows)} chunks")

    return {
        "documents_processed": len(documents),
        "chunks_written": written,
        "model_name": model_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
    }


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def search(
    query: str,
    top_k: int = config.DEFAULT_TOP_K,
    source_type: str | None = None,
    model_name: str = config.EMBEDDING_MODEL_NAME,
) -> list[dict]:
    """Semantic search over the ingested weather documents.

    ``<=>`` is pgvector's cosine *distance*, so ``1 - distance`` gives a
    similarity in ``[0, 1]`` where 1 is an exact match. Ordering by the raw
    distance (rather than by the derived similarity) is what lets the HNSW
    index actually serve the query.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        raise ValueError("Query must be a non-empty string.")

    top_k = max(config.MIN_TOP_K, min(int(top_k), config.MAX_TOP_K))

    vector = to_vector_literal(embed_texts([cleaned], model_name=model_name)[0])

    filter_clause = ""
    params: list[Any] = [vector]
    if source_type:
        filter_clause = "WHERE d.source_type = %s"
        params.append(source_type)
    params.extend([vector, top_k])

    sql = f"""
        SELECT d.id,
               d.location,
               d.source_type,
               d.event,
               d.headline,
               d.severity,
               d.narrative_text,
               d.effective_at,
               d.expires_at,
               e.chunk_index,
               e.chunk_text,
               1 - (e.embedding <=> %s::vector) AS similarity
        FROM {config.WEATHER_EMBEDDINGS_TABLE} e
        JOIN {config.WEATHER_DOCUMENTS_TABLE} d ON d.id = e.document_id
        {filter_clause}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
    """

    rows = lakebase.run_query(sql, tuple(params))
    for row in rows:
        row["similarity"] = round(float(row["similarity"]), 4)
        for key in ("effective_at", "expires_at"):
            if row.get(key) is not None:
                row[key] = row[key].isoformat()
    return rows


def stats() -> dict:
    """Row counts and coverage, used by the health check and the web UI."""
    documents = lakebase.run_query(
        f"""
        SELECT count(*) AS total,
               count(*) FILTER (WHERE source_type = 'alert') AS alerts,
               count(*) FILTER (WHERE source_type = 'forecast') AS forecasts,
               count(DISTINCT location) AS locations,
               max(synced_at) AS last_synced_at
        FROM {config.WEATHER_DOCUMENTS_TABLE}
        """
    )[0]
    embeddings = lakebase.run_query(
        f"""
        SELECT count(*) AS chunks,
               count(DISTINCT document_id) AS embedded_documents,
               max(created_at) AS last_embedded_at
        FROM {config.WEATHER_EMBEDDINGS_TABLE}
        """
    )[0]

    def _iso(value):
        return value.isoformat() if value is not None else None

    return {
        "documents": int(documents["total"] or 0),
        "alerts": int(documents["alerts"] or 0),
        "forecasts": int(documents["forecasts"] or 0),
        "locations": int(documents["locations"] or 0),
        "last_synced_at": _iso(documents["last_synced_at"]),
        "embedded_documents": int(embeddings["embedded_documents"] or 0),
        "chunks": int(embeddings["chunks"] or 0),
        "last_embedded_at": _iso(embeddings["last_embedded_at"]),
        "model_name": config.EMBEDDING_MODEL_NAME,
        "embedding_dim": config.EMBEDDING_DIM,
    }
