# Lakebase schema

Run these against your Lakebase Postgres database before running the notebook
or the app. Any Postgres client works: the Databricks SQL editor pointed at the
Lakebase instance, `psql`, or a notebook cell using the connection details from
`lakebase.connection_parts()`.

| File | What it does |
| --- | --- |
| `01_enable_pgvector.sql` | Enables the `vector` extension and reports its version |
| `02_create_weather_documents.sql` | Creates `weather_documents` (raw narrative text + provenance) |
| `03_create_weather_embeddings.sql` | Creates `weather_embeddings` with a `VECTOR(1024)` column and an HNSW cosine index |
| `04_verify_schema.sql` | Inspection queries — run after syncing and embedding |

Run them in order. `02` must precede `03` because `weather_embeddings.document_id`
is a foreign key into `weather_documents`.

## Changing the embedding model

The `VECTOR(1024)` width in `03` is tied to `databricks-gte-large-en`, the
default embedding endpoint (see `config.py`). If you change `EMBEDDING_MODEL`,
change the column width to match and rebuild the table — pgvector rejects an
insert whose dimension differs from the column declaration, so a mismatch
fails loudly rather than silently corrupting the index.

## Why the vectors are written as `VECTOR`, not `float[]`

Embeddings are inserted with an explicit `%s::vector` cast through
[pg8000](https://github.com/tlocke/pg8000), so the column holds real `VECTOR`
values the moment the row lands. There is no `float8[]` staging column and no
follow-up `UPDATE ... ::vector` pass to remember. Spark JDBC cannot do this —
it has no mapping to pgvector's type and no `ON CONFLICT` support — which is
why the whole write path uses a plain Postgres client instead.

## Why pg8000, not psycopg2

pg8000 is a pure-Python driver: no compiled C extension, no bundled `libpq`.
That matters specifically because psycopg2's compiled extension reliably
crashes the whole Python kernel on Databricks serverless compute — including
Databricks Free Edition, which is serverless-only — with a SIGABRT the moment
it's imported. It's a documented pattern on serverless generally (several
other C-extension packages show the identical symptom there), not something
fixable by editing the importing cell. pg8000 speaks the Postgres wire
protocol directly in Python, so there's nothing to conflict with. Both
drivers use the same DB-API `%s` placeholder style, so the SQL itself is
unaffected by the swap.
