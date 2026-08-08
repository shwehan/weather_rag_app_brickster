"""
Weather Intelligence - Flask API and web UI.

Harvests narrative weather text from the National Weather Service, stores it in
Lakebase (Databricks-managed Postgres), and serves semantic search over the
resulting pgvector embeddings.

Routes
    GET  /                     Web UI
    GET  /healthz              Liveness plus Lakebase reachability
    GET  /weather/stats        Row counts and coverage
    POST /weather/sync         Harvest documents for a list of locations
    POST /weather/embed        Chunk + embed everything not yet vectorized
    POST /weather/search       Semantic search (JSON body)
    GET  /weather/search       Semantic search (query string), optional summary
    GET  /weather/documents    Browse raw synced documents

Run locally:
    python app.py

Deploy as a Databricks App using app.yaml.
"""

from __future__ import annotations

import logging
import os
import threading

import requests
from psycopg2 import errors as pg_errors
from flask import Flask, jsonify, render_template, request

import config
import embedding_pipeline
import lakebase
from weather_client import (
    SOURCE_TYPE_ALERT,
    SOURCE_TYPE_FORECAST,
    LocationResolutionError,
    WeatherClient,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("weather-intelligence")

app = Flask(__name__)

VALID_SOURCE_TYPES = {SOURCE_TYPE_ALERT, SOURCE_TYPE_FORECAST}

_SETUP_HINT = (
    "Run the scripts in sql/ against your Lakebase database to create "
    "weather_documents and weather_embeddings before using this endpoint."
)


def _preload_model() -> None:
    """Warm the sentence-transformers model once, off the request path.

    The model is loaded exactly once per process. Doing it on a background
    thread at startup means the first search is fast without delaying the
    health check while the weights download.
    """
    try:
        embedding_pipeline.get_model()
        logger.info("Embedding model ready: %s", config.EMBEDDING_MODEL_NAME)
    except Exception:
        # A cold model is not fatal -- the first search will retry the load
        # and surface any real problem to the caller.
        logger.exception("Could not preload the embedding model at startup")


if os.environ.get("PRELOAD_EMBEDDING_MODEL", "true").lower() != "false":
    threading.Thread(target=_preload_model, daemon=True).start()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


@app.errorhandler(Exception)
def handle_exception(err):
    """Always answer with JSON so the UI's fetch().json() never sees HTML."""
    status = getattr(err, "code", 500)
    if not isinstance(status, int):
        status = 500

    if isinstance(err, pg_errors.UndefinedTable):
        return jsonify({"error": "Table not found. " + _SETUP_HINT}), 503

    logger.exception("Unhandled error while processing %s", request.path)
    return jsonify({"error": str(err)}), status


def _json_body() -> dict:
    return request.get_json(silent=True) or {}


def _clamp_top_k(value) -> int:
    try:
        top_k = int(value)
    except (TypeError, ValueError):
        top_k = config.DEFAULT_TOP_K
    return max(config.MIN_TOP_K, min(top_k, config.MAX_TOP_K))


def _clean_source_type(value) -> str | None:
    if not value:
        return None
    value = str(value).strip().lower()
    if value in ("", "all", "any"):
        return None
    if value not in VALID_SOURCE_TYPES:
        raise ValueError(
            f"source_type must be one of {sorted(VALID_SOURCE_TYPES)} or omitted."
        )
    return value


# ---------------------------------------------------------------------------
# Pages and health
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/healthz")
def healthz():
    payload = {"status": "ok", "model": config.EMBEDDING_MODEL_NAME}
    try:
        payload["lakebase"] = lakebase.ping()
        payload["tables"] = {
            config.WEATHER_DOCUMENTS_TABLE: lakebase.table_exists(
                config.WEATHER_DOCUMENTS_TABLE
            ),
            config.WEATHER_EMBEDDINGS_TABLE: lakebase.table_exists(
                config.WEATHER_EMBEDDINGS_TABLE
            ),
        }
    except Exception as exc:
        payload["status"] = "degraded"
        payload["error"] = str(exc)
        return jsonify(payload), 503
    return jsonify(payload)


@app.route("/weather/stats")
def weather_stats():
    return jsonify(embedding_pipeline.stats())


# ---------------------------------------------------------------------------
# Part 1 - harvest
# ---------------------------------------------------------------------------


@app.route("/weather/sync", methods=["POST"])
def weather_sync():
    """Harvest alerts and forecasts for a list of locations into Lakebase.

    Body: {"locations": ["Chicago, IL", "Austin, TX"], "limit": 50,
           "include_alerts": true, "include_forecast": true}
    """
    body = _json_body()

    locations = body.get("locations") or config.DEFAULT_LOCATIONS
    if isinstance(locations, str):
        locations = [part for part in locations.split(";") if part.strip()]
    locations = [
        loc.strip() for loc in locations if isinstance(loc, str) and loc.strip()
    ]
    if not locations:
        return jsonify({"error": "Provide at least one location."}), 400
    if len(locations) > 25:
        return jsonify({"error": "Sync at most 25 locations per request."}), 400

    try:
        limit = max(1, min(int(body.get("limit", config.DEFAULT_SYNC_LIMIT)), 200))
    except (TypeError, ValueError):
        return jsonify({"error": "limit must be an integer."}), 400

    include_alerts = bool(body.get("include_alerts", True))
    include_forecast = bool(body.get("include_forecast", True))
    if not (include_alerts or include_forecast):
        return jsonify(
            {"error": "Enable at least one of include_alerts or include_forecast."}
        ), 400

    client = WeatherClient()
    try:
        documents, errors = client.fetch_documents(
            locations,
            limit=limit,
            include_alerts=include_alerts,
            include_forecast=include_forecast,
        )
    except requests.RequestException as exc:
        return jsonify({"error": f"National Weather Service request failed: {exc}"}), 502
    except LocationResolutionError as exc:
        return jsonify({"error": str(exc)}), 400

    synced = embedding_pipeline.upsert_documents(documents)

    return jsonify(
        {
            "synced": synced,
            "alerts": sum(1 for d in documents if d["source_type"] == SOURCE_TYPE_ALERT),
            "forecasts": sum(
                1 for d in documents if d["source_type"] == SOURCE_TYPE_FORECAST
            ),
            "locations": locations,
            "errors": errors,
        }
    )


@app.route("/weather/documents")
def weather_documents():
    """Browse the raw synced documents, newest first."""
    limit = max(1, min(int(request.args.get("limit", 25)), 200))
    try:
        source_type = _clean_source_type(request.args.get("source_type"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    sql = f"""
        SELECT id, location, source_type, event, headline, severity,
               left(narrative_text, 400) AS narrative_preview,
               effective_at, expires_at, synced_at
        FROM {config.WEATHER_DOCUMENTS_TABLE}
    """
    params: list = []
    if source_type:
        sql += " WHERE source_type = %s"
        params.append(source_type)
    sql += " ORDER BY synced_at DESC, effective_at DESC NULLS LAST LIMIT %s"
    params.append(limit)

    rows = lakebase.run_query(sql, tuple(params))
    for row in rows:
        for key in ("effective_at", "expires_at", "synced_at"):
            if row.get(key) is not None:
                row[key] = row[key].isoformat()
    return jsonify({"documents": rows, "count": len(rows)})


# ---------------------------------------------------------------------------
# Part 2 - vectorize
# ---------------------------------------------------------------------------


@app.route("/weather/embed", methods=["POST"])
def weather_embed():
    """Chunk and embed every document that has no current vector.

    The notebook is the primary place to run this, but exposing it here keeps
    the web demo self-contained: sync, embed, then search without leaving the
    page.
    """
    body = _json_body()
    limit = body.get("limit")
    if limit is not None:
        try:
            limit = max(1, min(int(limit), 5000))
        except (TypeError, ValueError):
            return jsonify({"error": "limit must be an integer."}), 400

    result = embedding_pipeline.embed_pending_documents(limit=limit)
    return jsonify(result)


# ---------------------------------------------------------------------------
# Part 3 - retrieve
# ---------------------------------------------------------------------------


def _run_search(query: str, top_k: int, source_type: str | None):
    """Shared search body for the GET and POST variants."""
    if not query or not query.strip():
        return None, (jsonify({"error": "Provide a non-empty 'query'."}), 400)

    if not lakebase.table_exists(config.WEATHER_EMBEDDINGS_TABLE):
        return None, (jsonify({"error": "No embeddings table. " + _SETUP_HINT}), 503)

    results = embedding_pipeline.search(
        query, top_k=top_k, source_type=source_type
    )

    if not results:
        return {
            "query": query,
            "top_k": top_k,
            "source_type": source_type,
            "results": [],
            "message": (
                "No weather documents have been embedded yet. Sync some "
                "locations, then run the embedding step."
            ),
        }, None

    return {
        "query": query,
        "top_k": top_k,
        "source_type": source_type,
        "results": results,
    }, None


@app.route("/weather/search", methods=["POST"])
def weather_search():
    """Semantic search over ingested weather documents.

    Body: {"query": "flash flood risk this weekend", "top_k": 5,
           "source_type": "alert"}
    """
    body = _json_body()
    try:
        source_type = _clean_source_type(body.get("source_type"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    payload, error = _run_search(
        body.get("query", ""), _clamp_top_k(body.get("top_k")), source_type
    )
    if error:
        return error
    return jsonify(payload)


@app.route("/weather/search", methods=["GET"])
def weather_search_get():
    """Query-string variant, with an optional generated summary of the hits."""
    try:
        source_type = _clean_source_type(request.args.get("source_type"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    payload, error = _run_search(
        request.args.get("query", ""),
        _clamp_top_k(request.args.get("top_k")),
        source_type,
    )
    if error:
        return error

    wants_summary = request.args.get("summarize", "").lower() in ("1", "true", "yes")
    if wants_summary and payload["results"]:
        payload["summary"] = _summarize(payload["query"], payload["results"])

    return jsonify(payload)


def _summarize(query: str, results: list[dict]) -> str:
    """Ask a Databricks serving endpoint to summarize the retrieved chunks.

    This is the generation half of retrieval-augmented generation. It is
    optional: without SUMMARY_MODEL_ENDPOINT configured, search still returns
    ranked results and the caller just gets a note instead of a summary.
    """
    if not config.SUMMARY_ENDPOINT:
        return (
            "Set SUMMARY_MODEL_ENDPOINT to a Databricks serving endpoint to get "
            "a written summary alongside these results."
        )

    context = "\n\n".join(
        f"[{i + 1}] {row['location']} - {row['headline']}\n{row['chunk_text']}"
        for i, row in enumerate(results)
    )
    prompt = (
        "You are a weather briefing assistant. Using only the numbered weather "
        f"excerpts below, answer the question: {query}\n\n"
        "Be specific about locations and timing, cite excerpts as [1], [2], and "
        "say so plainly if the excerpts do not cover the question.\n\n"
        f"{context}"
    )

    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

        response = WorkspaceClient().serving_endpoints.query(
            name=config.SUMMARY_ENDPOINT,
            messages=[ChatMessage(role=ChatMessageRole.USER, content=prompt)],
            max_tokens=400,
        )
        return response.choices[0].message.content
    except Exception as exc:
        logger.warning("Summary generation failed: %s", exc)
        return f"Summary unavailable: {exc}"


if __name__ == "__main__":
    host = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_RUN_PORT", "8000"))
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true",
            host=host, port=port)
