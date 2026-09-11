"""Fail-closed validator for the authorized face-image manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

SPLITS = {"calibration", "validation", "locked_test"}
REQUIRED = {"image_id", "path", "person_id", "split", "camera_id", "condition", "authorization_ref", "sha256"}


def _safe_relative(path: str) -> Path:
    if not path or "\\" in path:
        raise ValueError("path must be a non-empty POSIX relative path")
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"unsafe dataset path: {path}")
    return Path(*p.parts)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate(manifest: str, root: str) -> dict[str, Any]:
    manifest_path = Path(manifest)
    root_path = Path(root).resolve()
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    seen_ids: set[str] = set()
    seen_paths: set[str] = set()
    people_by_split: dict[str, set[str]] = {s: set() for s in SPLITS}

    for i, row in enumerate(rows):
        missing = REQUIRED - row.keys()
        if missing:
            raise ValueError(f"row {i}: missing fields: {sorted(missing)}")
        image_id = str(row["image_id"])
        rel = str(row["path"])
        person = str(row["person_id"])
        split = str(row["split"])
        auth = str(row["authorization_ref"])
        expected_sha = str(row["sha256"]).lower()
        if not image_id or image_id in seen_ids:
            raise ValueError(f"row {i}: duplicate/empty image_id")
        if rel in seen_paths:
            raise ValueError(f"row {i}: duplicate path")
        if not person:
            raise ValueError(f"row {i}: empty person_id")
        if split not in SPLITS:
            raise ValueError(f"row {i}: invalid split {split}")
        if not auth:
            raise ValueError(f"row {i}: empty authorization_ref")
        if len(expected_sha) != 64 or any(c not in "0123456789abcdef" for c in expected_sha):
            raise ValueError(f"row {i}: invalid sha256")
        safe = _safe_relative(rel)
        full = (root_path / safe).resolve()
        if root_path not in full.parents:
            raise ValueError(f"row {i}: path escapes dataset root")
        if not full.is_file():
            raise ValueError(f"row {i}: image not found: {rel}")
        actual_sha = _sha256(full)
        if actual_sha != expected_sha:
            raise ValueError(f"row {i}: sha256 mismatch for {rel}")
        seen_ids.add(image_id)
        seen_paths.add(rel)
        people_by_split[split].add(person)

    people_to_splits: dict[str, set[str]] = {}
    for split, people in people_by_split.items():
        for person in people:
            people_to_splits.setdefault(person, set()).add(split)
    leaked = {p: sorted(s) for p, s in people_to_splits.items() if len(s) > 1}
    if leaked:
        raise ValueError(f"person leakage across splits: {leaked}")

    return {
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "image_count": len(rows),
        "person_count": len(people_to_splits),
        "split_counts": {s: sum(1 for r in rows if r["split"] == s) for s in sorted(SPLITS)},
        "person_counts": {s: len(people_by_split[s]) for s in sorted(SPLITS)},
        "status": "VALID",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    receipt = validate(args.manifest, args.root)
    Path(args.output).write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
