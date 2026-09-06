"""
Database initialization and migration script.
Connects to PostgreSQL + pgvector and applies scripts/init.sql.
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database.client import DatabaseClient


def main():
    parser = argparse.ArgumentParser(description="Initialize PostgreSQL schema for NguyenCuuFacePython")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--dbname", default="face_recognition")
    parser.add_argument("--user", default="face_admin")
    parser.add_argument("--password", default="face_secure_password_2026")
    parser.add_argument("--sql", default="scripts/init.sql")
    args = parser.parse_args()

    sql_path = Path(args.sql)
    if not sql_path.exists():
        print(f"Error: SQL file not found at {sql_path}")
        sys.exit(1)

    print(f"Connecting to {args.user}@{args.host}:{args.port}/{args.dbname}...")
    try:
        client = DatabaseClient(
            host=args.host,
            port=args.port,
            dbname=args.dbname,
            user=args.user,
            password=args.password,
            fallback_to_memory=False,
        )
        print("Applying SQL DDL migrations...")
        client.init_schema(str(sql_path))
        print("[SUCCESS] Schema and HNSW index initialized successfully!")
        client.close()
    except Exception as e:
        print(f"[ERROR] Failed to initialize database: {e}")
        print("Note: Ensure PostgreSQL is running via 'docker compose up -d'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
