"""Shared Lakebase write operations for normalized weather documents."""

from psycopg2.extras import Json, execute_values

import lakebase


def upsert_weather_documents(documents: list[dict]) -> int:
    """Insert or update normalized documents in one psycopg2 batch."""
    if not documents:
        return 0
    values = [
        (
            document["id"],
            document["location"],
            document["latitude"],
            document["longitude"],
            document["source_type"],
            document["headline"],
            document["narrative_text"],
            document["issued_at"],
            document["effective_at"],
            Json(document["payload"]),
            document["content_hash"],
        )
        for document in documents
    ]
    sql = """
        INSERT INTO weather_documents (
            id, location, latitude, longitude, source_type, headline,
            narrative_text, issued_at, effective_at, payload, content_hash
        ) VALUES %s
        ON CONFLICT (id) DO UPDATE SET
            location = EXCLUDED.location,
            latitude = EXCLUDED.latitude,
            longitude = EXCLUDED.longitude,
            source_type = EXCLUDED.source_type,
            headline = EXCLUDED.headline,
            narrative_text = EXCLUDED.narrative_text,
            issued_at = EXCLUDED.issued_at,
            effective_at = EXCLUDED.effective_at,
            payload = EXCLUDED.payload,
            content_hash = EXCLUDED.content_hash,
            synced_at = now()
    """
    with lakebase.get_connection() as conn, conn.cursor() as cur:
        execute_values(cur, sql, values, page_size=100)
        conn.commit()
    return len(documents)

