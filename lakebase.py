"""
Lakebase (Databricks-managed Postgres) connection helper.

The connection string is a standard Postgres URL, e.g.

    postgresql://role:password@host.database.cloud.databricks.com:5432/databricks_postgres?sslmode=require

It is resolved in this order:

1. The ``LAKEBASE_URL`` environment variable, for local development.
2. A base64-encoded Databricks secret (scope ``database``, key ``lakebase-url``
   by default), which is how the deployed Databricks App and the notebook get
   their credentials. Nothing sensitive lives in code, ``.env`` or ``app.yaml``.

Everything in this project talks to Postgres through psycopg2 -- there are no
Spark JDBC writes anywhere in the pipeline, because JDBC cannot write to
pgvector's ``VECTOR`` type or use ``ON CONFLICT`` for idempotent upserts.
"""

import base64
import os
from contextlib import contextmanager
from functools import lru_cache
from urllib.parse import urlparse

import psycopg2
from psycopg2.extras import RealDictCursor

_SCOPE = os.environ.get("LAKEBASE_SECRET_SCOPE", "database")
_KEY = os.environ.get("LAKEBASE_SECRET_KEY", "lakebase-url")


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
    """Split the Lakebase URL into psycopg2 keyword arguments.

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
    """Yield a psycopg2 connection whose cursors return ``dict`` rows."""
    conn = psycopg2.connect(
        lakebase_url(), cursor_factory=RealDictCursor, connect_timeout=15
    )
    try:
        yield conn
    finally:
        conn.close()


def run_query(sql: str, params: tuple | dict | None = None) -> list[dict]:
    """Run a read query and return the rows as a list of dicts."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()


def run_write(sql: str, params: tuple | dict | None = None) -> int:
    """Run an INSERT/UPDATE/DELETE/DDL statement and return the row count."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            conn.commit()
            return cur.rowcount


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
