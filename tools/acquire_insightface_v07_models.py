#!/usr/bin/env python3
"""Acquire the official InsightFace v0.7 buffalo_l package and emit an artifact receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
EXPECTED_SIZE = 288_621_354
FILES = ("det_10g.onnx", "w600k_r50.onnx")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_member(name: str) -> bool:
    """Allow only relative POSIX ZIP names without traversal."""
    p = PurePosixPath(name)
    return not p.is_absolute() and ".." not in p.parts and "\\" not in name


def locked_member(names: list[str], filename: str) -> str:
    candidates = [n for n in names if n == filename or n == f"buffalo_l/{filename}"]
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one locked member for {filename}, got {candidates}")
    return candidates[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=Path("models/artifacts/buffalo_l"))
    ap.add_argument("--receipt", type=Path, default=Path("models/artifacts/buffalo_l_receipt.json"))
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="insightface-acquire-") as td:
        td_path = Path(td)
        archive = td_path / "buffalo_l.zip"
        urllib.request.urlretrieve(URL, archive)
        actual_size = archive.stat().st_size
        if actual_size != EXPECTED_SIZE:
            raise SystemExit(f"package size mismatch: expected {EXPECTED_SIZE}, got {actual_size}")
        package_sha = sha256(archive)

        staging = td_path / "buffalo_l"
        staging.mkdir()
        with zipfile.ZipFile(archive) as zf:
            infos = zf.infolist()
            unsafe = [info.filename for info in infos if not safe_member(info.filename)]
            if unsafe:
                raise SystemExit(f"unsafe ZIP member(s): {unsafe}")
            names = [info.filename for info in infos]
            for filename in FILES:
                member = locked_member(names, filename)
                target = staging / filename
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
                    "sha256": sha256(staging / name),
                    "path": str(args.output / name),
                }
                for name in FILES
            ],
            "license_decision": "research-only-until-commercial-license-is-confirmed",
        }

        # Publish models and receipt only after all verification/extraction succeeds.
        backup = None
        if args.output.exists():
            backup = td_path / "previous"
            os.replace(args.output, backup)
        try:
            os.replace(staging, args.output)
            tmp_receipt = args.receipt.with_suffix(args.receipt.suffix + ".tmp")
            tmp_receipt.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_receipt, args.receipt)
        except Exception:
            if args.output.exists():
                shutil.rmtree(args.output)
            if backup is not None and backup.exists():
                os.replace(backup, args.output)
            raise

    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
