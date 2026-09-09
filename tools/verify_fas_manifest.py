"""Fail-closed verifier for pinned FAS artifacts.

Usage: python tools/verify_fas_manifest.py <manifest> <artifact_dir>

The manifest intentionally refuses placeholder SHA-256 values and verifies
that the local artifact bytes match the declared digest before benchmarking.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

import yaml


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify_fas_manifest.py MANIFEST ARTIFACT_DIR")
        return 2

    manifest = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text())
    artifact_dir = pathlib.Path(sys.argv[2])
    failures = []

    for item in manifest.get("artifacts", []):
        expected = item.get("sha256", "")
        filename = pathlib.PurePosixPath(item["upstream"]["path"]).name
        path = artifact_dir / filename
        if not expected or expected.startswith("REPLACE_WITH_"):
            failures.append(f"{item['id']}: SHA-256 is not pinned")
            continue
        if not path.is_file():
            failures.append(f"{item['id']}: artifact missing: {path}")
            continue
        actual = sha256(path)
        if actual.lower() != expected.lower():
            failures.append(f"{item['id']}: SHA-256 mismatch: {actual}")

    if failures:
        print("FAS artifact verification FAILED (fail-closed)")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("FAS artifact verification PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
