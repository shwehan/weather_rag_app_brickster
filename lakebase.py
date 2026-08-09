"""
Lakebase (Databricks-managed Postgres) connection helper.

Uses **pg8000**, a pure-Python PostgreSQL driver, rather than psycopg2. That
choice is deliberate: psycopg2's compiled C extension (`psycopg2-binary`)
reliably crashes the whole Python kernel on Databricks serverless compute --
including Databricks Free Edition, which is serverless-only -- with a SIGABRT
(exit code 134) the moment it's imported. This is a documented pattern on
serverless generally, not specific to this project: any package with a
compiled C extension linked against system libraries the serverless container
doesn't ship (docling, cfgrib, pymssql all show the identical symptom) can
trigger it. pg8000 speaks the Postgres wire protocol directly in Python, so
there is no compiled extension to conflict with.

The connection string is a standard Postgres URL, e.g.

    postgresql://role:password@host.database.cloud.databricks.com:5432/databricks_postgres?sslmode=require

It is resolved in this order:

1. The ``LAKEBASE_URL`` environment variable, for local development.
2. A base64-encoded Databricks secret (scope ``database``, key ``lakebase-url``
   by default), which is how the deployed Databricks App and the notebook get
   their credentials. Nothing sensitive lives in code, ``.env`` or ``app.yaml``.

Everything in this project talks to Postgres through pg8000 -- there are no
Spark JDBC writes anywhere in the pipeline, because JDBC cannot write to
pgvector's ``VECTOR`` type or use ``ON CONFLICT`` for idempotent upserts.
"""

import base64
import os
import ssl
from contextlib import contextmanager
from functools import lru_cache
from urllib.parse import urlparse

import pg8000.dbapi as pg8000

_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")

# pg8000's DB-API paramstyle is "format" (%s), the same as psycopg2's, so SQL
# written for one reads unchanged against the other.
DatabaseError = pg8000.DatabaseError


@lru_cache(maxsize=1)
def lakebase_url() -> str:
    """Resolve the Lakebase connection URL from the environment or a secret."""
    env_url = os.environ.get("LAKEBASE_URL", "").strip()
    if env_url:
        return env_url

    # Imported lazily so local development without the Databricks SDK
    # configured still works as long as LAKEBASE_URL is set.
    from databricks.sdk import WorkspaceClient

    secret = WorkspaceClient().secrets.get_secret(scope=_SCOPE, key=_KEY)
    return base64.b64decode(secret.value).decode("utf-8")


def connection_parts() -> dict:
    """Split the Lakebase URL into pg8000 connect() keyword arguments.

    Useful in notebooks, where showing the host/database/user (but never the
    password) makes it obvious which instance you are about to write to.
    """
    parsed = urlparse(lakebase_url())
    return {
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "dbname": parsed.path.lstrip("/") or "databricks_postgres",
        "user": parsed.username,
        "password": parsed.password,
        "sslmode": "require",
    }


@contextmanager
def get_connection():
    """Yield a pg8000 DB-API connection.

    Unlike psycopg2, pg8000 has no ``RealDictCursor`` -- see ``run_query()``
    below, which builds the equivalent dict rows from ``cursor.description``.
    """
    parts = connection_parts()
    conn = pg8000.connect(
        user=parts["user"],
        password=parts["password"],
        host=parts["host"],
        port=parts["port"],
        database=parts["dbname"],
        ssl_context=ssl.create_default_context(),
        timeout=15,
    )
    try:
        yield conn
    finally:
        conn.close()


def _rows_as_dicts(cursor) -> list[dict]:
    """Turn a pg8000 cursor's result into a list of dicts, keyed by column name."""
    if cursor.description is None:
        return []
    columns = [col[0] for col in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def run_query(sql: str, params: tuple | None = None) -> list[dict]:
    """Run a read query and return the rows as a list of dicts."""
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(sql, params if params is not None else tuple())
            return _rows_as_dicts(cursor)
        finally:
            cursor.close()


def run_write(sql: str, params: tuple | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE/DDL statement and return the row count."""
    with get_connection() as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(sql, params if params is not None else tuple())
            conn.commit()
            return cursor.rowcount
        finally:
            cursor.close()


def execute_values(
    cursor,
    sql: str,
    argslist: list[tuple],
    template: str | None = None,
    page_size: int = 100,
) -> None:
    """Batch-insert many rows in one round trip per page.

    A minimal, pg8000-compatible replacement for ``psycopg2.extras.execute_values``.
    ``sql`` must contain the literal marker ``VALUES %s`` showing where the
    batch of row tuples goes; everything before it becomes the INSERT prefix
    and everything after it (an ``ON CONFLICT`` clause, typically) is kept as
    a suffix. ``template`` describes one row -- e.g. ``"(%s, %s, %s::vector)"``
    -- exactly as psycopg2's own ``template`` argument does, including SQL
    alongside the placeholders such as the pgvector cast.

    Does not use ``executemany`` because pg8000 (like psycopg2) sends one
    round trip per call there; batching into a single multi-row ``VALUES``
    clause is what actually reduces round trips.
    """
    if not argslist:
        return
    if "VALUES %s" not in sql:
        raise ValueError('sql must contain a literal "VALUES %s" placeholder')
    prefix, _, suffix = sql.partition("VALUES %s")

    if template is None:
        template = "(" + ", ".join(["%s"] * len(argslist[0])) + ")"

    for start in range(0, len(argslist), page_size):
        batch = argslist[start : start + page_size]
        row_sql = ", ".join([template] * len(batch))
        flat_params = [value for row in batch for value in row]
        cursor.execute(f"{prefix}VALUES {row_sql}{suffix}", flat_params)


def sqlstate(error: Exception) -> str | None:
    """Extract the PostgreSQL SQLSTATE code from a pg8000 DatabaseError.

    pg8000 raises ``DatabaseError`` with a single dict argument keyed by the
    single-letter Postgres protocol field codes; ``"C"`` is SQLSTATE. Used to
    detect e.g. undefined-table (``42P01``) without depending on psycopg2's
    exception subclasses.
    """
    if error.args and isinstance(error.args[0], dict):
        return error.args[0].get("C")
    return None


def table_exists(table_name: str) -> bool:
    """Check whether a table is present in the current search path."""
    rows = run_query(
        "SELECT to_regclass(%s) IS NOT NULL AS present", (table_name,)
    )
    return bool(rows and rows[0]["present"])


def ping() -> dict:
    """Return a small health payload describing the Lakebase connection."""
    parts = connection_parts()
    rows = run_query("SELECT version() AS version, current_database() AS database")
    return {
        "host": parts["host"],
        "database": parts["dbname"],
        "user": parts["user"],
        "server_version": rows[0]["version"].split(",")[0] if rows else None,
    }
