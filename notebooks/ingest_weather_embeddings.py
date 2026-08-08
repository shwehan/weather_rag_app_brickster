# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest weather narratives -> pgvector embeddings
# MAGIC
# MAGIC The headless version of `weather_intelligence_pipeline.ipynb`: harvest,
# MAGIC chunk, embed, write, and print a summary. Use the `.ipynb` notebook for
# MAGIC interactive exploration and this script for the scheduled Job (see
# MAGIC `resources/ingest_weather_embeddings_job.yml`).
# MAGIC
# MAGIC It runs unchanged in three places:
# MAGIC
# MAGIC * as a Databricks `notebook_task`, reading Job parameters from widgets
# MAGIC * as a Databricks `spark_python_task`, reading them from CLI flags
# MAGIC * locally with `LAKEBASE_URL` set, for debugging
# MAGIC
# MAGIC Every database write goes through psycopg2 with `execute_values` and an
# MAGIC explicit `%s::vector` cast. No Spark JDBC, no post-hoc array-to-vector
# MAGIC conversion pass.

# COMMAND ----------

# MAGIC %pip install -q --upgrade "databricks-sdk>=0.30.0" psycopg2-binary sentence-transformers requests

# COMMAND ----------

# dbutils.library.restartPython()   # uncomment when running as a notebook task

# COMMAND ----------

"""Harvest NWS weather narratives, embed them, and load them into Lakebase."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------

DEFAULTS = {
    "documents_table": "weather_documents",
    "embeddings_table": "weather_embeddings",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "locations": "Chicago, IL;Austin, TX;Denver, CO;Miami, FL;Seattle, WA",
    "sync_limit": "50",
    "chunk_size": "800",
    "chunk_overlap": "100",
    "nws_user_agent": "weather-intelligence-app (set nws_user_agent)",
    "skip_sync": "false",
}


def _read_parameters() -> dict[str, str]:
    """Read parameters from Databricks widgets, or from CLI flags outside Databricks."""
    try:
        dbutils  # type: ignore[used-before-def]  # noqa: B018
    except NameError:
        parser = argparse.ArgumentParser(description=__doc__)
        for name, default in DEFAULTS.items():
            parser.add_argument(f"--{name.replace('_', '-')}", default=default)
        parsed = vars(parser.parse_args())
        return {name: str(parsed[name]) for name in DEFAULTS}

    values = {}
    for name, default in DEFAULTS.items():
        dbutils.widgets.text(name, default, name.replace("_", " ").title())  # noqa: F821
        values[name] = dbutils.widgets.get(name)  # noqa: F821
    return values


params = _read_parameters()

# config.py reads its defaults from the environment, so set these before it is
# imported anywhere in the process.
os.environ["WEATHER_DOCUMENTS_TABLE"] = params["documents_table"]
os.environ["WEATHER_EMBEDDINGS_TABLE"] = params["embeddings_table"]
os.environ["EMBEDDING_MODEL"] = params["embedding_model"]
os.environ["CHUNK_SIZE"] = params["chunk_size"]
os.environ["CHUNK_OVERLAP"] = params["chunk_overlap"]
os.environ["NWS_USER_AGENT"] = params["nws_user_agent"]

LOCATIONS = [part.strip() for part in params["locations"].split(";") if part.strip()]
SYNC_LIMIT = int(params["sync_limit"])
SKIP_SYNC = params["skip_sync"].strip().lower() in ("1", "true", "yes")

# The project modules live one directory up from notebooks/.
try:
    _notebook_path = (
        dbutils.notebook.entry_point.getDbutils()  # noqa: F821
        .notebook()
        .getContext()
        .notebookPath()
        .get()
    )
    _repo_root = "/Workspace" + os.path.dirname(os.path.dirname(_notebook_path))
except Exception:
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# COMMAND ----------

import config  # noqa: E402
import embedding_pipeline  # noqa: E402
import lakebase  # noqa: E402
from weather_client import WeatherClient  # noqa: E402

print(f"Repo root       : {_repo_root}")
print(f"Documents table : {config.WEATHER_DOCUMENTS_TABLE}")
print(f"Embeddings table: {config.WEATHER_EMBEDDINGS_TABLE}")
print(f"Model           : {config.EMBEDDING_MODEL_NAME} ({config.EMBEDDING_DIM}-dim)")
print(f"Chunking        : size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}")
print(f"Locations       : {LOCATIONS}")

# COMMAND ----------

# Fail fast with a clear message rather than a psycopg2 UndefinedTable trace.
_missing = [
    table
    for table in (config.WEATHER_DOCUMENTS_TABLE, config.WEATHER_EMBEDDINGS_TABLE)
    if not lakebase.table_exists(table)
]
if _missing:
    raise RuntimeError(
        f"Missing table(s): {_missing}. Run the scripts in sql/ against Lakebase first."
    )

print(f"Connected to Lakebase: {lakebase.ping()}")

# COMMAND ----------

# DBTITLE 1,Harvest
sync_summary = {"synced": 0, "alerts": 0, "forecasts": 0, "errors": []}

if SKIP_SYNC:
    print("skip_sync=true — embedding whatever is already in weather_documents.")
else:
    started = time.perf_counter()
    client = WeatherClient()
    documents, errors = client.fetch_documents(LOCATIONS, limit=SYNC_LIMIT)

    sync_summary = {
        "synced": embedding_pipeline.upsert_documents(documents),
        "alerts": sum(1 for d in documents if d["source_type"] == "alert"),
        "forecasts": sum(1 for d in documents if d["source_type"] == "forecast"),
        "errors": errors,
        "seconds": round(time.perf_counter() - started, 2),
    }
    print(json.dumps(sync_summary, indent=2))

# COMMAND ----------

# DBTITLE 1,Chunk, embed and load
started = time.perf_counter()
embed_summary = embedding_pipeline.embed_pending_documents(progress=True)
embed_summary["seconds"] = round(time.perf_counter() - started, 2)

print(json.dumps(embed_summary, indent=2))

# COMMAND ----------

# DBTITLE 1,Summary
final_stats = embedding_pipeline.stats()
print(json.dumps(final_stats, indent=2))

# Surface the run summary as the task output so a downstream Job task, or the
# run page itself, can read what happened without parsing logs.
try:
    dbutils.notebook.exit(  # noqa: F821
        json.dumps({"sync": sync_summary, "embed": embed_summary, "stats": final_stats})
    )
except NameError:
    pass
