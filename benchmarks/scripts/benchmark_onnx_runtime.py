"""Benchmark one ONNX Runtime model with an explicit execution contract.

Runtime/latency benchmark only. Recognition accuracy must be measured separately
with the locked SCRFD -> alignment -> ArcFace protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("no measurements")
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--intra-op-threads", type=int, default=0)
    parser.add_argument("--inter-op-threads", type=int, default=0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import onnxruntime as ort

    if not args.model.is_file():
        raise SystemExit(f"model not found: {args.model}")
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("warmup must be >= 0 and iterations must be > 0")
    if args.intra_op_threads < 0 or args.inter_op_threads < 0:
        raise SystemExit("thread counts must be >= 0 (0 means ORT default)")
    available = ort.get_available_providers()
    if args.provider not in available:
        raise SystemExit(f"provider {args.provider!r} unavailable; available={available}")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.intra_op_threads:
        sess_options.intra_op_num_threads = args.intra_op_threads
    if args.inter_op_threads:
        sess_options.inter_op_num_threads = args.inter_op_threads

    session_create_start = time.perf_counter_ns()
    session = ort.InferenceSession(
        str(args.model), sess_options=sess_options, providers=[args.provider]
    )
    session_create_ms = (time.perf_counter_ns() - session_create_start) / 1e6
    actual_providers = session.get_providers()
    if args.provider not in actual_providers:
        raise SystemExit(
            f"requested provider {args.provider!r} was not active; actual={actual_providers}"
        )

    inputs: dict[str, Any] = {}
    for meta in session.get_inputs():
        shape = [1 if isinstance(dim, str) or dim is None else int(dim) for dim in meta.shape]
        dtype = np.float32
        if meta.type == "tensor(float16)":
            dtype = np.float16
        elif meta.type == "tensor(int64)":
            dtype = np.int64
        elif meta.type == "tensor(int32)":
            dtype = np.int32
        inputs[meta.name] = np.zeros(shape, dtype=dtype)

    for _ in range(args.warmup):
        session.run(None, inputs)

    timings_ms: list[float] = []
    for _ in range(args.iterations):
        start = time.perf_counter_ns()
        session.run(None, inputs)
        timings_ms.append((time.perf_counter_ns() - start) / 1e6)

    mean_ms = statistics.fmean(timings_ms)
    report = {
        "schema_version": 2,
        "model": str(args.model),
        "model_sha256": sha256_file(args.model),
        "onnxruntime": ort.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "requested_provider": args.provider,
        "available_providers": available,
        "active_providers": actual_providers,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "thread_config": {
            "intra_op": args.intra_op_threads,
            "inter_op": args.inter_op_threads,
        },
        "session_create_ms": session_create_ms,
        "latency_ms": {
            "mean": mean_ms,
            "p50": percentile(timings_ms, 50),
            "p95": percentile(timings_ms, 95),
            "p99": percentile(timings_ms, 99),
            "min": min(timings_ms),
            "max": max(timings_ms),
        },
        "throughput_fps": 1000.0 / mean_ms,
        "inputs": [
            {"name": x.name, "shape": x.shape, "type": x.type}
            for x in session.get_inputs()
        ],
        "outputs": [
            {"name": x.name, "shape": x.shape, "type": x.type}
            for x in session.get_outputs()
        ],
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
