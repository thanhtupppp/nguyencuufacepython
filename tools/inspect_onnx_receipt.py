"""Inspect a verified ONNX artifact and emit a reproducible graph contract receipt."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def shape(value):
    dims = []
    for d in value.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            dims.append(d.dim_value)
        elif d.HasField("dim_param"):
            dims.append(d.dim_param)
        else:
            dims.append(None)
    return dims


def tensor_contract(value):
    return {"name": value.name, "shape": shape(value), "elem_type": value.type.tensor_type.elem_type}


def inspect(model: Path) -> dict:
    import onnx
    m = onnx.load(str(model), load_external_data=False)
    return {
        "sha256": sha256(model),
        "ir_version": m.ir_version,
        "opset_import": [{"domain": x.domain, "version": x.version} for x in m.opset_import],
        "inputs": [tensor_contract(x) for x in m.graph.input],
        "outputs": [tensor_contract(x) for x in m.graph.output],
        "node_count": len(m.graph.node),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("model", type=Path)
    p.add_argument("--receipt", type=Path, required=True)
    args = p.parse_args()
    result = inspect(args.model)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
