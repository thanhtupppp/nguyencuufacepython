"""Record ONNX Runtime platform/provider capabilities for edge benchmark preflight.

This does not benchmark model accuracy or latency. It only records the runtime,
platform, architecture and available Execution Providers so later edge runs can
be compared against a known environment.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--model", type=Path, action="append", default=[])
    args = parser.parse_args()

    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime is required") from exc

    report: dict[str, Any] = {
        "schema_version": 1,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "onnxruntime": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "models": [],
    }

    for model in args.model:
        session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        report["models"].append(
            {
                "path": str(model),
                "providers": session.get_providers(),
                "inputs": [x.name for x in session.get_inputs()],
                "outputs": [x.name for x in session.get_outputs()],
            }
        )

    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
