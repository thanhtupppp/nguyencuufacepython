#!/usr/bin/env python3
"""Acquire the official InsightFace v0.7 buffalo_l package and emit an artifact receipt.

This script deliberately does not hard-code a SHA-256 that has not been computed from
bytes. It downloads the official GitHub release asset, hashes the exact bytes, extracts
only det_10g.onnx and w600k_r50.onnx, then hashes each extracted file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
EXPECTED_SIZE = 288_621_354
FILES = ("det_10g.onnx", "w600k_r50.onnx")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("models/artifacts/buffalo_l"))
    ap.add_argument("--receipt", type=Path, default=Path("models/artifacts/buffalo_l_receipt.json"))
    args = ap.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        archive = Path(td) / "buffalo_l.zip"
        urllib.request.urlretrieve(URL, archive)
        actual_size = archive.stat().st_size
        if actual_size != EXPECTED_SIZE:
            raise SystemExit(f"package size mismatch: expected {EXPECTED_SIZE}, got {actual_size}")
        package_sha = sha256(archive)
        with zipfile.ZipFile(archive) as zf:
            names = set(zf.namelist())
            missing = [name for name in FILES if name not in names and f"buffalo_l/{name}" not in names]
            if missing:
                raise SystemExit(f"missing expected model files: {missing}")
            for name in FILES:
                member = name if name in names else f"buffalo_l/{name}"
                target = args.output / name
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    receipt = {
        "schema_version": 1,
        "source": URL,
        "release": "insightface-v0.7",
        "package": {
            "filename": "buffalo_l.zip",
            "size_bytes": EXPECTED_SIZE,
            "sha256": package_sha,
        },
        "artifacts": [
            {
                "filename": name,
                "sha256": sha256(args.output / name),
                "path": str(args.output / name),
            }
            for name in FILES
        ],
        "license_decision": "research-only-until-commercial-license-is-confirmed",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
