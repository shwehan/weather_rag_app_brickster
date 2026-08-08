"""
One-time setup: store the Lakebase connection URL in a Databricks secret.

The National Weather Service API needs no key, so this is the only secret the
project uses. Run it once from a machine with the Databricks CLI configured, or
paste it into a notebook cell.

    python setup_secrets.py

Nothing is printed and nothing is written to disk -- the URL goes straight from
the prompt into the secret scope.
"""

import getpass
import sys

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace

SCOPE = "database"
KEY = "lakebase-url"


def main() -> int:
    client = WorkspaceClient()

    existing = {scope.name for scope in client.secrets.list_scopes()}
    if SCOPE not in existing:
        print(f"Creating secret scope {SCOPE!r}")
        client.secrets.create_scope(scope=SCOPE)

    url = getpass.getpass(
        "Lakebase URL "
        "(postgresql://role:password@host:5432/databricks_postgres?sslmode=require): "
    ).strip()

    if not url.startswith("postgres"):
        print("That does not look like a Postgres URL. Nothing was saved.")
        return 1

    client.secrets.put_secret(scope=SCOPE, key=KEY, string_value=url)
    print(f"Stored {SCOPE}/{KEY}")

    # The notebook runs as you, but the deployed App runs as its own service
    # principal. Granting the `users` group READ covers the interactive case;
    # for the App, grant its service principal explicitly:
    #
    #   databricks secrets put-acl database <app-service-principal> READ
    client.secrets.put_acl(
        scope=SCOPE, principal="users", permission=workspace.AclPermission.READ
    )
    print(f"Granted READ on {SCOPE} to the 'users' group")
    print(
        "\nRemember to grant your Databricks App's service principal READ on "
        "this scope as well, or the app cannot open a connection."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
