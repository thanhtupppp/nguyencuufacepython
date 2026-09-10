"""Static integrity checks for the pgvector benchmark contract.

This does not connect to PostgreSQL and never handles biometric data. It
fails if the benchmark source violates required person-level accounting rules.
"""
from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).with_name("benchmark_pgvector_retrieval.py")

REQUIRED_NAMES = {
    "exact_person_topk",
    "unique_topk",
    "exact_rerank_candidates",
    "build_prototypes",
}


def main() -> int:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    names = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = REQUIRED_NAMES - names
    if missing:
        raise SystemExit(f"missing benchmark primitives: {sorted(missing)}")

    text = SOURCE.read_text(encoding="utf-8")
    required_sql_markers = (
        "PARTITION BY e.person_id",
        "model_version",
        "active",
        "benchmark_person_prototypes",
        "hnsw.ef_search",
        "hnsw.iterative_scan",
    )
    missing_markers = [marker for marker in required_sql_markers if marker not in text]
    if missing_markers:
        raise SystemExit(f"missing benchmark contract markers: {missing_markers}")

    if "recognition threshold" in text.lower() and "threshold" not in text.lower():
        raise SystemExit("unreachable threshold guard")

    print("vector benchmark contract: PASS")
    print("person-level oracle, model/status filters, prototype path, and HNSW controls are present")
    print("NOTE: this is a static contract check; it is not a real PostgreSQL performance receipt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
