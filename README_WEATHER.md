# Design notes

Why this data source, why this schema, and what I would change with more time.
Setup and deployment steps are in [`README.md`](README.md).

---

## The data source

**National Weather Service API** (`https://api.weather.gov`), single-sourced.

Two products are harvested:

| Product | Endpoint | Free text used |
| --- | --- | --- |
| Active alerts | `GET /alerts/active?point={lat},{lon}` | `headline` + `description` + `instruction` |
| Multi-day forecast | `GET /gridpoints/{office}/{x},{y}/forecast` | `detailedForecast` per period |

Both are reachable only after resolving a location to a forecast grid cell with
`GET /points/{lat},{lon}`, which also returns the nearest named place — that
becomes the canonical `location` label on every document.

### Why this source

**It is genuinely unstructured.** An alert's `description` is a paragraph a
forecaster wrote — hydrologic reasoning, affected creeks and roads, timing — and
its `instruction` is the protective action in plain English. A `detailedForecast`
is a full sentence-level narrative rather than a temperature field. That is real
prose to embed, not numbers dressed up as text.

**No API key.** Nothing to provision, rotate, or leak, which keeps the project's
attention on harvesting, vectorization and retrieval instead of auth plumbing.
The only secret in the whole project is the Lakebase connection URL.

**It contains genuine paraphrase.** Weather is a domain where people ask
questions in words that never appear in the source: "will my flight get delayed",
"is it safe to drive tonight", "do I need to move the patio furniture". Keyword
search fails those outright, which makes it easy to show that the vector search
is actually doing something.

**Two document types from one source.** Alerts and forecasts have different
register, length and urgency, but normalize into one schema. That gives the
retrieval endpoint a meaningful `source_type` filter — "only tell me about active
warnings" — without the complexity of reconciling two vendors' APIs.

### Why alerts and forecasts, and not hourly forecasts

`/forecast/hourly` returns 156 periods per location, and each `shortForecast` is
a fragment like "Mostly Sunny" — high volume, almost no semantic content. It
would have inflated the index without improving retrieval. The multi-day
narrative forecast carries the actual prose.

### Geocoding

The NWS API only speaks coordinates. `weather_client.py` resolves a location in
three steps: a raw `lat,lon` pair is parsed directly; a `City, ST` string is
looked up in a built-in table of ~70 U.S. metro areas; anything else falls back
to the free U.S. Census geocoder. The built-in table means the common demo path
runs with zero extra dependencies and zero extra network calls.

---

## Schema

### `weather_documents`

One row per embeddable narrative.

| Column | Why it exists |
| --- | --- |
| `id` | Stable dedup key. Alerts use `alert:<NWS alert id>`; forecasts use `forecast:<office>:<x,y>:<period start>:<period name>` |
| `location`, `latitude`, `longitude` | Where the narrative applies; the label is the NWS-resolved place name |
| `grid_office`, `grid_x`, `grid_y` | The forecast grid cell, so a document can be traced back to its exact source URL |
| `source_type` | `'alert'` or `'forecast'`, `CHECK`-constrained. Drives the retrieval filter |
| `event`, `headline` | Display labels — "Flash Flood Warning", "Tuesday Night: Rain Likely" |
| `severity`, `urgency`, `certainty`, `area_desc` | NWS alert metadata; `NULL` on forecast rows |
| `narrative_text` | The free text that gets chunked and embedded |
| `issued_at`, `effective_at`, `expires_at` | Normalized to UTC. `expires_at` is what makes stale-alert cleanup possible |
| `content_hash` | SHA-256 of `narrative_text` |
| `payload` | The raw API response as `JSONB`, for provenance and reprocessing without re-fetching |
| `synced_at` | Last write time |

**The forecast id deserves an explanation.** A forecast has no natural id — the
same period gets re-issued several times a day with revised wording. Keying on
grid cell plus period start means a re-issue updates the row in place, so the
table holds the current forecast for each period rather than an accumulating
pile of near-duplicates.

**Alert description and instruction are stored as one document, not two.** They
are two halves of one message: the description says a creek will crest, the
instruction says do not drive through it. Splitting them would let retrieval
return the hazard without the response, which is the less useful half.

### `weather_embeddings`

One row per chunk.

| Column | Why it exists |
| --- | --- |
| `id` | `<document_id>#<chunk_index>` — re-embedding overwrites a document's own chunks instead of duplicating them |
| `document_id` | FK to `weather_documents(id)` with `ON DELETE CASCADE`, so purging a document takes its vectors with it |
| `chunk_index`, `chunk_text` | Position and the exact text that produced the vector, so results can show what actually matched |
| `embedding` | `VECTOR(1024)` |
| `model_name` | Which model produced it, so two models can coexist during a migration |
| `content_hash` | Copied from the document at embed time; a mismatch marks the vector stale |
| `created_at` | Embed time |

Indexes: HNSW on `embedding` with `vector_cosine_ops`, a btree on `document_id`
for the join, and a btree on `(model_name, content_hash)` so the "what still
needs embedding" query stays cheap.

### The content hash

This is the one piece of the schema that is not obvious, and it is what makes the
job safe to run on a schedule.

A document needs embedding when no vector exists **for this model and this exact
text**. Checking only for the presence of a row would never re-embed an updated
alert; re-embedding everything on each run would waste most of the work, since
most narratives do not change between runs. Comparing hashes gets both: updated
text is re-embedded, unchanged text is skipped, and a quiet run costs one API
call per location and no GPU-equivalent work at all.

---

## Embedding model and chunking

**Model:** `databricks-gte-large-en`, 1024 dimensions, called over REST via a
Databricks Model Serving Foundation Model API endpoint — not a local
`sentence-transformers` model.

That's a deliberate substitution, not a style preference, and it's worth
explaining plainly since the original design used a local MiniLM model. A
local `sentence-transformers` model pulls in `torch`, and `torch`'s compiled
extensions reliably crash the whole Python kernel with a SIGABRT the instant
they're imported on Databricks serverless compute — including Databricks Free
Edition, which is serverless-only. It's a documented pattern on serverless
generally (other C-extension-heavy packages hit the identical symptom there),
not something fixable by editing the importing cell. Calling a hosted
endpoint keeps every native-extension dependency off the process entirely:
the notebook and the app only ever exchange JSON over HTTP. `gte-large-en` is
pre-configured pay-per-token in every workspace's Model Serving, needs no
deployment of your own, and produces embeddings strong enough for
short-passage retrieval — the whole workload here, since weather narratives
are paragraphs rather than documents.

`config.py` maps model names to dimensions and raises a clear error on an
unknown one, so switching models is a two-line change plus a matching
`VECTOR(N)` width in `sql/03_create_weather_embeddings.sql`. On classic
(non-serverless) compute, where the crash above doesn't apply, the
sentence-transformers entries already in that map work as drop-in
alternatives.

**Chunking:** 800 characters, 100 characters of overlap.

Sizing was driven by what the source actually produces:

| Text | Typical length | Chunks at 800 |
| --- | --- | --- |
| `detailedForecast` | 150–400 characters | 1 |
| Alert `description` alone | 400–1,500 characters | 1–2 |
| Alert description + instruction | 800–2,500 characters | 1–4 |

So chunking is a no-op for most forecasts and matters mainly for long alert
bodies. 800 characters comfortably clears `gte-large-en`'s per-chunk needs
without padding requests — a larger window would just mean fewer, larger
chunks with less precise matches, since a hit returns the whole chunk as
context. The 100-character overlap keeps a sentence that straddles a boundary
intact in at least one chunk; that is about one sentence of weather prose,
enough to preserve a protective-action line that would otherwise be cut in
half.

Chunks are stored individually and retrieval returns the matching chunk
alongside its parent document, so a hit shows both the specific passage that
matched and the full context it came from.

---

## Writes

Everything goes through **pg8000** — a pure-Python Postgres driver, not
psycopg2 — using a small `execute_values` replacement (`lakebase.execute_values`)
for batching and an explicit `%s::vector` cast for embeddings.

Two separate decisions are bundled into "pg8000, not psycopg2 or Spark JDBC,"
and it's worth untangling them:

- **Not Spark JDBC**, because it has no mapping for pgvector's `VECTOR` type
  and no `ON CONFLICT` support, which forces a `float8[]` staging column and a
  follow-up `UPDATE … ::vector` pass — a step that is easy to forget and
  leaves the index silently unbuilt when you do. Casting at insert time means
  the column holds real vectors the moment the row lands, and `ON CONFLICT`
  makes every write idempotent.
- **Not psycopg2**, because its compiled C extension is the other package
  (alongside torch) that reliably crashes the kernel on Databricks serverless
  compute with a SIGABRT on import. pg8000 speaks the Postgres wire protocol
  directly in Python — no compiled extension, nothing to conflict with a
  system library the serverless container doesn't ship. Both drivers use the
  same DB-API `%s` placeholder style, so this swap changed zero SQL in the
  project; only `lakebase.py`'s connection-handling internals changed.

The workload does not need distribution anyway: harvesting is network-bound,
and embedding a few thousand short chunks via a hosted endpoint takes seconds
regardless of which node it runs on. The scheduled Job runs on serverless
compute for exactly this reason — no cluster to provision or leave idle
between 30-minute runs.

---

## Running the pipeline

Full setup and deployment steps are in [`README.md`](README.md). The short
version, once secrets and schema exist:

1. **Sync** — `POST /weather/sync {"locations": ["Chicago, IL"], "limit": 50}`,
   or the notebook's harvest cells, or the **Sync weather data** button.
2. **Embed** — `POST /weather/embed`, or the notebook's embedding cells, or the
   **Generate embeddings** button.
3. **Search** — `POST /weather/search {"query": "flash flood risk this weekend", "top_k": 5}`,
   or the search box.

The notebook does all three with inspection between each stage; the web UI does
all three as a linear demo; the scheduled Job does the first two headlessly.

---

## Limitations, and what I would do with more time

**Embedding costs money past the free quota.** `databricks-gte-large-en` is
pay-per-token; a demo-sized corpus (a few hundred documents) costs pennies,
but it's not literally free the way a local model's compute is. There is no
usage cap in this project — a runaway sync loop would keep incurring charges.
A production version should track token spend or wrap the embed step in a
budget check.

**Alerts are sparse by design.** On a calm day a location has zero active alerts,
so a demo may end up with forecasts only. Retrieval still works, but the
severity-coded results are less striking. The honest fix is a backfill from the
NWS alerts archive (`/alerts?start=…&end=…`) to seed a corpus of past severe
weather; right now the table only ever holds what is active at sync time.

**Nothing expires.** Alerts accumulate past their `expires_at` and keep matching
queries about current conditions. `expires_at` is stored precisely so a reaper
job could delete or down-rank expired rows — the `ON DELETE CASCADE` on the
embeddings FK means deleting a document already cleans up its vectors — but that
job is not written.

**Retrieval is purely semantic.** There is no recency weighting and no geographic
filter, so a query about flooding ranks a stale Miami alert alongside a live
Austin one purely on text similarity. The obvious improvement is a hybrid score
that blends cosine similarity with recency and distance from a point of interest,
which pgvector can express in a single `ORDER BY`.

**Chunking is character-based.** A sliding window over characters can split
mid-sentence. Sentence-aware chunking would produce cleaner boundaries; at these
text lengths the difference is small, which is why the simpler approach won, but
it would matter on longer products like area forecast discussions.

**Embedding happens synchronously in the web request.** `POST /weather/embed`
blocks until it finishes. That is fine for a demo-sized corpus and keeps the UI
honest about when work completes, but at scale it belongs in the scheduled Job
with the endpoint returning a job run id instead.

**Geocoding coverage is uneven.** The built-in table covers major U.S. metros;
everything else depends on the Census geocoder, which handles street addresses
well and informal place names poorly. Coordinates always work and are the
reliable escape hatch.

**Single region.** The NWS covers the United States and its territories only.
Extending beyond that means a second source with a different schema — which the
normalized document shape would absorb, but the `source_type` vocabulary and the
severity taxonomy would both need widening.
