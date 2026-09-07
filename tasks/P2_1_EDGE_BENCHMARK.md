# P2.1 Edge Runtime Benchmark Protocol

## Goal

Measure the same SCRFD + alignment + ArcFace pipeline on PC, Raspberry Pi ARM64 and Android without changing recognition semantics.

## Preflight

Run `python benchmarks/scripts/inspect_runtime.py --out benchmarks/results/runtime.json` on each target. Record Python/ORT version, CPU architecture and available Execution Providers.

For each external model, first run `inspect_model_assets.py` and preserve the SHA-256/model contract.

## Fixed pipeline

`SCRFD -> 5 landmarks -> Umeyama 112x112 -> ArcFace -> 512-D L2`.

Do not introduce FP16/INT8 until the FP32/FP baseline and accuracy threshold are locked.

## Measurements

- cold-start session creation time
- warm-up latency
- p50/p95/p99 inference latency
- throughput (frames/s)
- peak RSS where available
- provider actually used
- CPU utilization where available
- embedding cosine parity against the reference PC CPU run

The repository provides `benchmarks/scripts/benchmark_onnx_runtime.py` for repeatable single-model runtime measurements. It records session creation, p50/p95/p99/mean latency, throughput, model SHA-256, provider metadata, thread configuration and model I/O metadata. The runner now fails closed if the requested provider is not active after session creation, preventing accidental CPU-fallback results from being reported as accelerator measurements. It uses deterministic zero-valued inputs only for runtime timing; these numbers must never be presented as recognition accuracy.

Example:

`python benchmarks/scripts/benchmark_onnx_runtime.py --model /path/model.onnx --provider CPUExecutionProvider --warmup 20 --iterations 100 --out benchmarks/results/model_cpu.json`

Thread-sensitive targets should additionally record explicit `--intra-op-threads` and `--inter-op-threads` values. Compare a small predefined matrix rather than tuning until a single best-looking run appears; the chosen setting must be reproducible.

For a full recognition benchmark, use the real-model runner and real images so SCRFD detection, five-point alignment and ArcFace embedding semantics are measured together.

## Acceptance

A device run is not considered comparable unless model SHA-256, preprocessing, input dimensions and output normalization match the reference.

Accuracy parity is a hard gate: optimized/mobile execution must not materially change recognition decisions at the locked threshold/margin. Any FP16/INT8 proposal requires a paired accuracy report before adoption.

A provider result is invalid if the requested provider is merely available but not active in the created session.

## Platform plan

### PC

Baseline: ONNX Runtime CPU. GPU is an optional secondary measurement.

### Raspberry Pi ARM64

Baseline: ONNX Runtime CPU. XNNPACK may be evaluated only if the selected runtime/build supports it; compare against CPU baseline rather than assuming it is faster.

### Android

Evaluate ORT CPU/XNNPACK first, then NNAPI where the device supports it. Record whether operators are actually delegated; do not claim accelerator usage from configuration alone.

## Model usability

Before Android packaging, run ONNX Runtime's model usability checker for the exact model. It estimates NNAPI/CoreML suitability, but final performance must still be measured on the target device.

## Next

After baseline results exist, evaluate FP16/INT8 and only then model A/B (AdaFace, MagFace, MobileFaceNet) using the same accuracy protocol.
