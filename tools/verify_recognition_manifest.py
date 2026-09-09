#!/usr/bin/env python3
"""Fail-closed verifier for SCRFD/ArcFace model provenance manifests."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

REQUIRED = ("model_id", "publisher", "upstream_revision", "artifact", "sha256", "weight_license", "commercial_use")
PLACEHOLDERS = {"", "REPLACE_WITH_VERIFIED_SHA256", "pending", "unknown", "tbd"}


def load_manifest(path: Path):
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(text)
    if yaml is None:
        raise RuntimeError("PyYAML is required for YAML manifests")
    return yaml.safe_load(text)


def verify(entry: dict, base: Path) -> list[str]:
    errors = [f"missing:{k}" for k in REQUIRED if k not in entry]
    if errors:
        return errors
    sha = str(entry.get("sha256", "")).strip().lower()
    if sha in PLACEHOLDERS or len(sha) != 64:
        errors.append("unverified_sha256")
    artifact = base / str(entry["artifact"])
    if not artifact.is_file():
        errors.append("artifact_missing")
        return errors
    if "license" in str(entry.get("weight_license", "")).lower() and not entry["weight_license"]:
        errors.append("weight_license_missing")
    if sha not in PLACEHOLDERS and len(sha) == 64:
        actual = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual != sha:
            errors.append(f"sha256_mismatch:{actual}")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    args = ap.parse_args()
    path = Path(args.manifest)
    data = load_manifest(path)
    entries = data.get("models", data) if isinstance(data, dict) else data
    if isinstance(entries, dict):
        entries = list(entries.values())
    failures = {}
    for entry in entries:
        errs = verify(entry, path.parent)
        if errs:
            failures[entry.get("model_id", "unknown")] = errs
    receipt = {"manifest": str(path), "verified": not failures, "failures": failures}
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
