"""Fail if runtime code imports the legacy DatabaseClient directly.

The guard intentionally scans only ``src/`` because tests and benchmark
fixtures may import legacy implementations for compatibility/regression
coverage. Production runtime code must import DatabaseClient from
``src.database`` so the person-level PostgreSQL ranking implementation remains
the single entrypoint.
"""

from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = ROOT / "src"
IGNORED_PARTS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache"}


def python_files() -> list[Path]:
    return [
        p
        for p in SOURCE_ROOT.rglob("*.py")
        if not (set(p.parts) & IGNORED_PARTS)
    ]


def find_legacy_imports(path: Path) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError as exc:
        return [f"{path}: syntax error: {exc}"]

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in {"src.database.client", "database.client", ".client"}:
                if any(alias.name == "DatabaseClient" for alias in node.names):
                    violations.append(f"{path}:{node.lineno}: direct legacy DatabaseClient import")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"src.database.client", "database.client"}:
                    violations.append(f"{path}:{node.lineno}: direct legacy database.client import")
    return violations


def main() -> int:
    violations = [v for p in python_files() for v in find_legacy_imports(p)]
    if violations:
        print("Legacy DatabaseClient imports found in production source:")
        print("\n".join(violations))
        return 1
    print("OK: src/ production Python files do not import the legacy DatabaseClient directly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
