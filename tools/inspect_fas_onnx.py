"""Inspect a local FAS ONNX artifact and emit a machine-readable I/O receipt.

This tool deliberately does not infer Real/Spoof semantics from tensor names alone.
It records graph metadata so the manifest can be updated only after manual/source
verification. Usage:
    python tools/inspect_fas_onnx.py path/to/model.onnx
"""
from __future__ import annotations

import json
import pathlib
import sys

import onnxruntime as ort


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: inspect_fas_onnx.py MODEL.onnx")
        return 2

    path = pathlib.Path(sys.argv[1])
    if not path.is_file():
        print(f"artifact missing: {path}")
        return 1

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = []
    for item in session.get_inputs():
        inputs.append({
            "name": item.name,
            "shape": item.shape,
            "type": item.type,
        })

    outputs = []
    for item in session.get_outputs():
        outputs.append({
            "name": item.name,
            "shape": item.shape,
            "type": item.type,
        })

    receipt = {
        "artifact": str(path),
        "providers": session.get_providers(),
        "inputs": inputs,
        "outputs": outputs,
        "source_semantics": {
            "real_class_index": 0,
            "spoof_class_index": 1,
            "score": "real_logit - spoof_logit",
            "verified_from": "facenox/face-antispoof-onnx/src/inference/inference.py",
        },
        "preprocess": {
            "resize": "letterbox",
            "interpolation": "INTER_LANCZOS4 if upscale else INTER_AREA",
            "padding": "BORDER_REFLECT_101",
            "layout": "CHW",
            "scale": "float32 / 255.0",
            "color_conversion": "none in upstream preprocess.py; caller image channel order must be verified",
        },
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
