# Databricks notebook source
"""Harvest NWS weather narratives, embed them, and load them into Lakebase.

The headless version of ``weather_intelligence_pipeline.ipynb``: harvest,
chunk, embed, write, and print a summary. Use the ``.ipynb`` notebook for
interactive exploration and this script for the scheduled Job (see
``resources/ingest_weather_embeddings_job.yml``).

It runs unchanged in three places:

* as a Databricks ``notebook_task``, reading Job parameters from widgets
* as a Databricks ``spark_python_task``, reading them from CLI flags
* locally with ``LAKEBASE_URL`` set, for debugging

Every database write goes through pg8000 -- a pure-Python Postgres driver,
not psycopg2 -- with ``lakebase.execute_values`` and an explicit
``%s::vector`` cast. No Spark JDBC, no post-hoc array-to-vector conversion
pass, and no compiled C extension that could crash the kernel on serverless
compute. Embeddings come from a Databricks Model Serving endpoint over REST
for the same reason: no local model, no torch.
"""

# `# Databricks notebook source` above must stay the literal first line of the
# file -- it's what makes Databricks render this as a multi-cell notebook
# rather than import it as a plain script. A module docstring and a
# `__future__` import are both still allowed ahead of the rest of the code
# per Python's own rules, so this ordering satisfies both constraints.
from __future__ import annotations

# COMMAND ----------

# Every dependency here is pure Python -- no compiled C extension, no
# torch. That is deliberate: packages like psycopg2 and
# sentence-transformers (which pulls in torch) reliably crash the whole
# kernel with a SIGABRT on Databricks serverless compute, including
# Databricks Free Edition, which is serverless-only. pg8000 is a
# pure-Python Postgres driver, and embeddings come from a Databricks
# Model Serving endpoint called over REST rather than a local model, so
# there is nothing here that can trigger that crash.
%pip install -q --upgrade "databricks-sdk>=0.30.0" "pg8000>=1.31.2" requests

# COMMAND ----------

# The pip install above only takes effect after the Python process restarts.
# dbutils is defined automatically when this runs as a Databricks notebook_task
# (which is how resources/ingest_weather_embeddings_job.yml invokes it) and
# undefined when run as a plain script -- the try/except lets one file cover
# both without a manual edit.
try:
    dbutils.library.restartPython()  # noqa: F821
except NameError:
    pass

# COMMAND ----------

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
    "embedding_model": "databricks-gte-large-en",
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

# Process a limited number of documents at a time to avoid rate limits.
# The pay-per-token databricks-gte-large-en endpoint has strict QPS limits.
embed_summary = embedding_pipeline.embed_pending_documents(limit=5, progress=True)
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