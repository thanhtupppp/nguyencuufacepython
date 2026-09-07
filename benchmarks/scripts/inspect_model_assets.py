"""Inspect and fingerprint external SCRFD/ArcFace ONNX assets before a real benchmark.

This script never downloads or redistributes model weights. It records only local
metadata and SHA-256 fingerprints so the benchmark can be reproduced against the
exact external artifacts supplied for the run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dim_value(value: Any) -> Any:
    if isinstance(value, (int, str)) or value is None:
        return value
    return str(value)


def inspect(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for ONNX asset inspection") from exc

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "providers": session.get_providers(),
        "inputs": [
            {
                "name": item.name,
                "shape": [dim_value(x) for x in item.shape],
                "type": item.type,
            }
            for item in session.get_inputs()
        ],
        "outputs": [
            {
                "name": item.name,
                "shape": [dim_value(x) for x in item.shape],
                "type": item.type,
            }
            for item in session.get_outputs()
        ],
    }


def validate_arcface(info: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    outputs = info["outputs"]
    if not any(out["shape"] and out["shape"][-1] == 512 for out in outputs):
        errors.append("ArcFace asset has no output whose final dimension is 512")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scrfd", type=Path)
    parser.add_argument("--arcface", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if not args.scrfd and not args.arcface:
        parser.error("provide --scrfd and/or --arcface")

    report: dict[str, Any] = {"schema_version": 1, "assets": {}}
    errors: list[str] = []

    if args.scrfd:
        report["assets"]["scrfd"] = inspect(args.scrfd)

    if args.arcface:
        arcface = inspect(args.arcface)
        report["assets"]["arcface"] = arcface
        errors.extend(validate_arcface(arcface))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
