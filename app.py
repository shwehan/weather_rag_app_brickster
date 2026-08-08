"""Weather Intelligence Flask app: harvest, store, and semantically retrieve."""

import logging
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
import lakebase
from embedding_model import MODEL_NAME, embed_texts, vector_literal
from weather_client import WeatherClient, WeatherClientError
from weather_store import upsert_weather_documents


load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-intelligence")

app = Flask(__name__)

DEFAULT_LOCATIONS = [
    value.strip()
    for value in os.environ.get("WEATHER_LOCATIONS", "Chicago, IL;Austin, TX").split(";")
    if value.strip()
]


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    logger.exception("Unhandled application error")
    status = getattr(error, "code", 500)
    return jsonify({"error": str(error)}), status if isinstance(status, int) else 500


@app.get("/")
def index():
    return render_template("index.html", default_locations=DEFAULT_LOCATIONS)


@app.get("/healthz")
def healthz():
    return jsonify(
        {
            "status": "ok",
            "model": MODEL_NAME,
            "time": datetime.now(timezone.utc).isoformat(),
        }
    )


@app.get("/weather/stats")
def weather_stats():
    lakebase.ensure_weather_schema()
    rows = lakebase.run_query(
        """
        SELECT
            (SELECT COUNT(*) FROM weather_documents) AS documents,
            (SELECT COUNT(*) FROM weather_embeddings) AS embeddings,
            (SELECT COUNT(DISTINCT location) FROM weather_documents) AS locations,
            (SELECT MAX(synced_at) FROM weather_documents) AS last_synced_at
        """
    )
    return jsonify(rows[0])


@app.post("/weather/sync")
def sync_weather():
    """Fetch NWS alerts and forecasts and upsert normalized documents."""
    body = request.get_json(silent=True) or {}
    locations = body.get("locations") or DEFAULT_LOCATIONS
    if not isinstance(locations, list) or not locations:
        return jsonify({"error": "locations must be a non-empty list"}), 400
    if len(locations) > 20:
        return jsonify({"error": "A sync request supports at most 20 locations"}), 400

    try:
        limit = max(1, min(int(body.get("limit", 50)), 200))
    except (TypeError, ValueError):
        return jsonify({"error": "limit must be an integer between 1 and 200"}), 400

    lakebase.ensure_weather_schema()
    client = WeatherClient()
    total = 0
    results = []
    errors = []

    for requested_location in locations:
        try:
            documents = client.fetch_documents(requested_location, limit=limit)
            synced = upsert_weather_documents(documents)
            total += synced
            results.append(
                {
                    "requested_location": requested_location,
                    "documents_synced": synced,
                }
            )
        except (WeatherClientError, ValueError) as exc:
            errors.append(
                {"requested_location": requested_location, "error": str(exc)}
            )

    status = 200 if results else 502
    return (
        jsonify(
            {
                "synced": total,
                "locations": results,
                "errors": errors,
                "next_step": "Run notebooks/ingest_weather_embeddings.ipynb to embed new or changed documents.",
            }
        ),
        status,
    )


@app.route("/weather/search", methods=["GET", "POST"])
def search_weather():
    """Return weather chunks ranked by pgvector cosine similarity."""
    if request.method == "POST":
        body = request.get_json(silent=True) or {}
        query = body.get("query", "")
        requested_top_k = body.get("top_k", 5)
        source_type = body.get("source_type")
    else:
        query = request.args.get("query", "")
        requested_top_k = request.args.get("top_k", 5)
        source_type = request.args.get("source_type")

    query = query.strip() if isinstance(query, str) else ""
    if not query:
        return jsonify({"error": "query must be a non-empty string"}), 400
    if len(query) > 1000:
        return jsonify({"error": "query must be 1000 characters or fewer"}), 400

    try:
        top_k = max(1, min(int(requested_top_k), 20))
    except (TypeError, ValueError):
        return jsonify({"error": "top_k must be an integer"}), 400

    if source_type not in (None, "", "alert", "forecast"):
        return jsonify({"error": "source_type must be alert or forecast"}), 400

    lakebase.ensure_weather_schema()
    count = lakebase.run_query("SELECT COUNT(*) AS count FROM weather_embeddings")[0][
        "count"
    ]
    if count == 0:
        return (
            jsonify(
                {
                    "error": "No weather embeddings are available yet.",
                    "next_step": "Sync documents, then run notebooks/ingest_weather_embeddings.ipynb",
                }
            ),
            409,
        )

    query_vector = vector_literal(embed_texts([query])[0])
    source_clause = "AND d.source_type = %s" if source_type else ""
    params = [query_vector]
    if source_type:
        params.append(source_type)
    params.extend([query_vector, top_k])

    rows = lakebase.run_query(
        f"""
        SELECT
            d.id,
            d.location,
            d.source_type,
            d.headline,
            d.narrative_text,
            d.issued_at,
            d.effective_at,
            e.chunk_index,
            e.chunk_text,
            1 - (e.embedding <=> %s::vector) AS similarity
        FROM weather_embeddings AS e
        JOIN weather_documents AS d ON d.id = e.document_id
        WHERE TRUE {source_clause}
        ORDER BY e.embedding <=> %s::vector
        LIMIT %s
        """,
        tuple(params),
    )
    matches = []
    for row in rows:
        match = dict(row)
        match["similarity"] = round(float(match["similarity"]), 6)
        matches.append(match)

    return jsonify(
        {
            "query": query,
            "top_k": top_k,
            "source_type": source_type or "all",
            "model": MODEL_NAME,
            "matches": matches,
        }
    )


if __name__ == "__main__":
    lakebase.ensure_weather_schema()
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("DATABRICKS_APP_PORT", "8000")),
    )
