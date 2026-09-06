"""Shared service dependencies for API routes."""

from src.database.client import DatabaseClient

# One shared client per API process keeps identity and embedding operations in
# the same database connection/store. Production deployment can replace this
# with a dependency container or pool without changing route contracts.
db = DatabaseClient()
