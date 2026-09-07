"""Benchmark one ONNX Runtime model with a fixed provider and deterministic input.

This is a runtime/latency benchmark only. It intentionally does not claim
recognition accuracy; the face pipeline benchmark must use real images and the
locked SCRFD -> alignment -> ArcFace protocol separately.
"""
from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    import onnxruntime as ort

    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("warmup must be >= 0 and iterations must be > 0")
    if args.provider not in ort.get_available_providers():
        raise SystemExit(
            f"provider {args.provider!r} unavailable; available={ort.get_available_providers()}"
        )

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_create_start = time.perf_counter_ns()
    session = ort.InferenceSession(
        str(args.model), sess_options=sess_options, providers=[args.provider]
    )
    session_create_ms = (time.perf_counter_ns() - session_create_start) / 1e6

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

    report = {
        "schema_version": 1,
        "model": str(args.model),
        "onnxruntime": ort.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "requested_provider": args.provider,
        "actual_providers": session.get_providers(),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "session_create_ms": session_create_ms,
        "latency_ms": {
            "mean": statistics.fmean(timings_ms),
            "p50": percentile(timings_ms, 50),
            "p95": percentile(timings_ms, 95),
            "p99": percentile(timings_ms, 99),
            "min": min(timings_ms),
            "max": max(timings_ms),
        },
        "throughput_fps": 1000.0 / statistics.fmean(timings_ms),
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
