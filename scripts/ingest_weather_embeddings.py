#!/usr/bin/env python3
"""Chunk and embed weather documents, then batch-upsert into Lakebase.

Run from the repository root:
    python scripts/ingest_weather_embeddings.py
"""

import argparse
import hashlib
import logging
import sys
from pathlib import Path

from psycopg2.extras import execute_values


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import lakebase  # noqa: E402
from embedding_model import (  # noqa: E402
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    MODEL_NAME,
    chunk_text,
    embed_texts,
    vector_literal,
)


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("weather-embedding-ingestion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP)
    return parser.parse_args()


def fetch_documents_needing_embeddings(limit: int) -> list[dict]:
    return lakebase.run_query(
        """
        SELECT d.id, d.narrative_text, d.content_hash
        FROM weather_documents AS d
        WHERE NOT EXISTS (
            SELECT 1
            FROM weather_embeddings AS e
            WHERE e.document_id = d.id
              AND e.content_hash = d.content_hash
              AND e.model_name = %s
        )
        ORDER BY d.synced_at
        LIMIT %s
        """,
        (MODEL_NAME, limit),
    )


def build_rows(documents: list[dict], chunk_size: int, overlap: int) -> list[tuple]:
    chunk_records = []
    for document in documents:
        chunks = chunk_text(document["narrative_text"], chunk_size, overlap)
        for index, text in enumerate(chunks):
            chunk_records.append(
                {
                    "document_id": document["id"],
                    "chunk_index": index,
                    "chunk_text": text,
                    "content_hash": document["content_hash"],
                }
            )

    vectors = embed_texts(record["chunk_text"] for record in chunk_records)
    rows = []
    for record, vector in zip(chunk_records, vectors, strict=True):
        embedding_id = "embedding:" + hashlib.sha256(
            f"{record['document_id']}|{record['chunk_index']}|{MODEL_NAME}".encode()
        ).hexdigest()
        rows.append(
            (
                embedding_id,
                record["document_id"],
                record["chunk_index"],
                record["chunk_text"],
                vector_literal(vector),
                MODEL_NAME,
                record["content_hash"],
            )
        )
    return rows


def replace_embeddings(rows: list[tuple], document_ids: list[str], batch_size: int) -> int:
    if not rows:
        return 0
    sql = """
        INSERT INTO weather_embeddings (
            id, document_id, chunk_index, chunk_text, embedding,
            model_name, content_hash
        ) VALUES %s
        ON CONFLICT (document_id, chunk_index) DO UPDATE SET
            id = EXCLUDED.id,
            chunk_text = EXCLUDED.chunk_text,
            embedding = EXCLUDED.embedding,
            model_name = EXCLUDED.model_name,
            content_hash = EXCLUDED.content_hash,
            created_at = now()
    """
    with lakebase.get_connection() as conn, conn.cursor() as cur:
        # Remove stale extra chunks when updated content becomes shorter.
        cur.execute(
            "DELETE FROM weather_embeddings WHERE document_id = ANY(%s)",
            (document_ids,),
        )
        execute_values(
            cur,
            sql,
            rows,
            template="(%s, %s, %s, %s, %s::vector, %s, %s)",
            page_size=batch_size,
        )
        conn.commit()
    return len(rows)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    if args.chunk_size < 1 or not 0 <= args.chunk_overlap < args.chunk_size:
        raise SystemExit("Chunk overlap must be smaller than a positive chunk size")

    lakebase.ensure_weather_schema()
    documents = fetch_documents_needing_embeddings(args.limit)
    if not documents:
        logger.info("No new or changed weather documents need embeddings.")
        return

    logger.info("Embedding %s weather documents with %s", len(documents), MODEL_NAME)
    rows = build_rows(documents, args.chunk_size, args.chunk_overlap)
    written = replace_embeddings(
        rows,
        [document["id"] for document in documents],
        args.batch_size,
    )
    logger.info("Wrote %s chunk embeddings to Lakebase.", written)


if __name__ == "__main__":
    main()

