# Weather Intelligence

Semantic search over live weather narratives.

The project harvests free-text weather products from the National Weather
Service, stores them in Lakebase (Databricks-managed Postgres), embeds them into
`pgvector` columns, and serves cosine-similarity retrieval through a Flask REST
API and a small web UI.

The point is that a question like *"roads may be slick this evening"* finds the
right forecast even though no forecast contains that phrase. Matching happens in
vector space, not on keywords.

---

## What's here

```
homework2-weather-intelligence-app/
├── app.py                     Flask REST API + web UI
├── weather_client.py          National Weather Service API client and normalizer
├── embedding_pipeline.py      Chunking, embedding, upserts, vector search
├── lakebase.py                Lakebase connection helper (psycopg2)
├── config.py                  Table names, model, chunking — one source of truth
├── setup_secrets.py           Stores the Lakebase URL in a Databricks secret
├── app.yaml                   Databricks App configuration
├── databricks.yml             Asset Bundle for the scheduled job
├── requirements.txt
├── sql/
│   ├── 01_enable_pgvector.sql
│   ├── 02_create_weather_documents.sql
│   ├── 03_create_weather_embeddings.sql
│   └── 04_verify_schema.sql
├── notebooks/
│   ├── weather_intelligence_pipeline.ipynb   Interactive end-to-end run
│   └── ingest_weather_embeddings.py          Headless version for scheduling
├── templates/
│   └── index.html             Web UI
└── resources/
    └── ingest_weather_embeddings_job.yml     Scheduled Databricks Job
```

Design notes, schema rationale and known limitations are in
[`README_WEATHER.md`](README_WEATHER.md).

---

## Prerequisites

- A Databricks workspace with a **Lakebase** Postgres instance
- A Postgres role on that instance with a static password, and its connection
  URL:
  `postgresql://<role>:<password>@<instance>.database.cloud.databricks.com:5432/databricks_postgres?sslmode=require`
- Permission to create Databricks Apps and secret scopes

There is no weather API key to obtain. The National Weather Service API is open;
it only asks that clients identify themselves with a contact address in the
`User-Agent` header.

---

## Setup

### 1. Add the repository as a Databricks Git folder

In the workspace sidebar: **Workspace → Create → Git folder**, paste the
repository URL, and clone it into your user folder.

### 2. Store the Lakebase connection URL

The connection string never lives in code, `.env`, or `app.yaml` — only the
coordinates of the secret that holds it do. From a notebook cell or a terminal
with the Databricks CLI configured:

```bash
python setup_secrets.py
```

This writes the URL to the secret `database` / `lakebase-url` and grants the
`users` group read access. The deployed App runs as its own service principal,
so grant that too:

```bash
databricks secrets put-acl database <app-service-principal> READ
```

### 3. Create the Lakebase schema

Run the scripts in `sql/`, in order, against your Lakebase database — using the
Databricks SQL editor pointed at the instance, `psql`, or any Postgres client:

1. `01_enable_pgvector.sql` — enables the `vector` extension
2. `02_create_weather_documents.sql` — the raw document store
3. `03_create_weather_embeddings.sql` — `VECTOR(384)` column plus an HNSW
   cosine index

`02` must run before `03`, because `weather_embeddings.document_id` is a
foreign key into `weather_documents`.

### 4. Run the notebook

Open `notebooks/weather_intelligence_pipeline.ipynb` and attach it to a cluster
or serverless compute. It walks the whole pipeline in order and shows you the
data at every stage:

- resolve locations to NWS grid points and harvest alerts and forecasts
- inspect a raw document before it is written
- upsert into `weather_documents`
- see exactly how the chunker splits the longest narrative
- embed the pending chunks and write them as real `VECTOR` values
- run semantic searches and read the ranked results
- confirm the HNSW index exists and check the query plan

Widgets at the top control the locations, table names, model and chunk sizes, so
the same notebook can be pointed at a different set of cities without editing
any code.

### 5. Inspect Lakebase

`sql/04_verify_schema.sql` has the queries worth running by hand: row counts,
the stored vector dimension, anything synced but not embedded, sample chunks,
and the index definitions.

### 6. Deploy the Databricks App

**Compute → Apps → Create app**, then point it at this Git folder. Databricks
reads `app.yaml` for the start command and environment, and `requirements.txt`
for dependencies.

Before deploying, change `NWS_USER_AGENT` in `app.yaml` to a real contact
address, and confirm the app's service principal has read access to the
`database` secret scope.

With the CLI:

```bash
databricks apps deploy weather-intelligence \
    --source-code-path /Workspace/Users/<you>/homework2-weather-intelligence-app
```

### 7. Use the web UI

Open the app URL. The page runs the same three steps the notebook does:

1. **Sync weather data** — enter locations (one per line, as `City, ST` or
   `lat,lon`) and pull their alerts and forecasts
2. **Generate embeddings** — chunk and vectorize everything not yet embedded
3. **Search by meaning** — ask in plain language and read the ranked matches

A status line under the header shows live document, alert, forecast and vector
counts straight from Lakebase, so you can watch the tables fill as you go.

---

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Web UI |
| `GET` | `/healthz` | Liveness, Lakebase reachability, table presence |
| `GET` | `/weather/stats` | Row counts and coverage |
| `POST` | `/weather/sync` | Harvest documents for a list of locations |
| `POST` | `/weather/embed` | Chunk and embed pending documents |
| `POST` | `/weather/search` | Semantic search |
| `GET` | `/weather/search` | Same, via query string, with optional summary |
| `GET` | `/weather/documents` | Browse raw synced documents |

### Sync

```bash
curl -X POST https://<app-url>/weather/sync \
  -H 'Content-Type: application/json' \
  -d '{"locations": ["Chicago, IL", "Austin, TX"], "limit": 50}'
```

```json
{
  "synced": 62,
  "alerts": 3,
  "forecasts": 59,
  "locations": ["Chicago, IL", "Austin, TX"],
  "errors": []
}
```

### Search

```bash
curl -X POST https://<app-url>/weather/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "risk of flooding near rivers", "top_k": 5}'
```

```json
{
  "query": "risk of flooding near rivers",
  "top_k": 5,
  "source_type": null,
  "results": [
    {
      "id": "alert:urn:oid:2.49.0.1.840.0.…",
      "location": "Austin, TX",
      "source_type": "alert",
      "event": "Flood Warning",
      "headline": "Flood Warning issued …",
      "severity": "Severe",
      "chunk_index": 0,
      "chunk_text": "…",
      "similarity": 0.7412,
      "effective_at": "2026-08-07T18:00:00+00:00",
      "expires_at": "2026-08-08T06:00:00+00:00"
    }
  ]
}
```

Add `"source_type": "alert"` or `"forecast"` to restrict retrieval to one kind
of product. `top_k` is clamped to 1–20.

The `GET` variant accepts `?summarize=true`, which asks a Databricks serving
endpoint to write a short briefing over the retrieved chunks. Set
`SUMMARY_MODEL_ENDPOINT` in `app.yaml` to enable it; without it, search still
returns ranked results and notes that summaries are off.

---

## Scheduling

`resources/ingest_weather_embeddings_job.yml` defines a Databricks Job that
re-runs `notebooks/ingest_weather_embeddings.py` every 30 minutes — alerts are
issued continuously, and a warning is not much use to a retrieval system that
refreshes nightly. Unchanged narratives are skipped by content hash, so a quiet
run does almost no work.

```bash
databricks bundle deploy -t dev
databricks bundle run ingest_weather_embeddings_job -t dev
```

The schedule ships paused. Unpause it after a successful manual run.

---

## Running locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # set LAKEBASE_URL and NWS_USER_AGENT
export $(grep -v '^#' .env | xargs)
python app.py             # http://localhost:8000
```

Local runs read `LAKEBASE_URL` from the environment and skip the Databricks
secret lookup entirely, so no workspace authentication is needed as long as the
Lakebase instance accepts connections from your network.
