# P0.1-RUN — Real SCRFD + ArcFace Benchmark

## Status

IN PROGRESS — implementation harness exists; empirical run is blocked until valid licensed ONNX assets and benchmark images are present.

## Objective

Measure the real production recognition path before locking thresholds, margins, or changing models:

`SCRFD -> 5-point landmarks -> Umeyama 112x112 -> ArcFace -> 512-D L2 -> cosine -> score + top1/top2 margin`.

## Preflight

Run:

```bash
python benchmarks/scripts/validate_benchmark_inputs.py --pairs benchmarks/pairs
```

The preflight must pass before any metric is considered valid. It checks missing images, pair labels, duplicate rows, and genuine/impostor availability.

## Model requirements

- Valid licensed SCRFD ONNX asset with 5 landmarks.
- Valid licensed ArcFace/InsightFace ONNX asset.
- Record model SHA-256/fingerprint and version.
- Do not commit proprietary or restricted model weights into the repository.

## Benchmark protocol

1. Generate pairs from gallery/probe.
2. Preflight pairs.
3. Run the real detector and alignment path for raw images; do not replace alignment with a blind 112x112 resize.
4. Extract 512-D L2-normalized embeddings.
5. Calibrate threshold and top-1/top-2 margin on validation only.
6. Lock model, preprocessing, threshold and margin.
7. Evaluate locked test set.
8. Report FAR/FMR, FRR/FNMR, EER, TAR@FAR, 1:N top-1 accuracy and condition breakdown.
9. State insufficient statistical evidence when the number of impostor trials cannot support a requested FAR level.

## Acceptance criteria

- Real model inference completes end-to-end.
- No mock embeddings are used for the final result.
- Validation/test split is explicit and threshold is not tuned on test data.
- Report contains model/version/fingerprint, detector, alignment version, embedding dimension, normalization, metric, threshold, margin and dataset version.
- FAR/FRR/EER/TAR and condition-level results are reproducible from the recorded command/config.

## Next step after DONE

Run P0.3 empirical tracking benchmark with the same real embeddings/video, then calibrate appearance association and temporal-fusion parameters. Only after that proceed to liveness calibration and production integration tests.
