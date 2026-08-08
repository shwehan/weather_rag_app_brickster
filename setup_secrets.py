"""Store the Lakebase connection URL for the Databricks App.

Run from an authenticated Databricks terminal. Input is hidden and never
written to a local file. Re-running updates the existing secret value.
"""

import getpass

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import DatabricksError


SCOPE = "database"
KEY = "lakebase-url"
w = WorkspaceClient()

try:
    w.secrets.create_scope(scope=SCOPE)
    print(f"Created secret scope: {SCOPE}")
except DatabricksError as exc:
    if getattr(exc, "error_code", None) != "RESOURCE_ALREADY_EXISTS":
        raise
    print(f"Using existing secret scope: {SCOPE}")

lakebase_url = getpass.getpass("Paste your complete Lakebase URL: ").strip()
if not lakebase_url.startswith(("postgres://", "postgresql://")):
    raise SystemExit("The Lakebase URL must begin with postgres:// or postgresql://")
if "sslmode=require" not in lakebase_url:
    raise SystemExit("The Lakebase URL must include sslmode=require")

w.secrets.put_secret(scope=SCOPE, key=KEY, string_value=lakebase_url)
print(f"Saved the connection URL as {SCOPE}/{KEY}.")
print("Add this secret to the App Authorization page with Can read permission.")

