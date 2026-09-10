"""Static validator for FAS candidate provenance before artifact verification.

This gate intentionally does not approve a model. It rejects floating refs,
placeholder digests, and unresolved license fields so benchmark inputs remain
reproducible and fail closed.

Usage: python tools/validate_fas_candidate_manifest.py configs/fas_artifacts.yaml
"""
from __future__ import annotations

import pathlib
import re
import sys

import yaml

SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
PLACEHOLDERS = ("REPLACE_WITH_", "REQUIRES_", "TODO", "TBD")


def unresolved(value: object) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return not text or any(text.startswith(token) for token in PLACEHOLDERS)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_fas_candidate_manifest.py MANIFEST")
        return 2

    path = pathlib.Path(sys.argv[1])
    data = yaml.safe_load(path.read_text())
    failures: list[str] = []

    for item in data.get("artifacts", []):
        ident = item.get("id", "<missing-id>")
        upstream = item.get("upstream") or {}
        revision = str(upstream.get("revision", "")).strip()
        digest = str(item.get("sha256", "")).strip()
        license_name = upstream.get("license")

        if not revision or revision in {"main", "master", "HEAD", "latest"}:
            failures.append(f"{ident}: upstream revision must be an immutable commit/tag")
        if not SHA256_RE.fullmatch(digest):
            failures.append(f"{ident}: sha256 is not pinned")
        if unresolved(license_name):
            failures.append(f"{ident}: upstream license is unresolved")

    if failures:
        print("FAS candidate manifest FAILED (fail-closed)")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("FAS candidate manifest PASSED static provenance checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
