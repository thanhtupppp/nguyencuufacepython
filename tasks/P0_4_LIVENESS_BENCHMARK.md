# P0.4 Liveness / Anti-Spoofing Benchmark Protocol

## Objective

Validate presentation-attack detection before a face is allowed to produce a `person_id`.
The production gate is fail-closed: missing/unverified liveness weights must produce `INCONCLUSIVE`, never `PASS`.

## Attack matrix

Every evaluation must contain bona-fide samples and separate attack classes:

1. Live face — controlled lighting.
2. Live face — low/high illumination within the quality gate.
3. Printed photo — matte and glossy when available.
4. Screen replay — phone and monitor at multiple brightness levels.
5. Video replay — prerecorded moving face.
6. Cut-out / paper mask when available.
7. 3D mask / silicone presentation when available.

Do not mix subjects between train/tuning and locked evaluation. Keep a locked attack set for final reporting.

## Metrics

Report at minimum:

- APCER: attack presentations incorrectly accepted as bona fide.
- BPCER: bona-fide presentations incorrectly rejected.
- ACER: mean(APCER, BPCER).
- ROC-AUC / PR-AUC as secondary diagnostics.
- FAR-like acceptance rate at the identity pipeline operating point.
- `INCONCLUSIVE` rate separately from FAIL.
- Per-attack-class APCER, not only aggregate ACER.

The security gate is driven by APCER at the chosen operating point; overall accuracy is not sufficient for release.

## Runtime protocol

For every face:

```text
SCRFD
  -> quality gate
  -> liveness
  -> alignment
  -> ArcFace
  -> vector search
  -> person_id decision
```

If liveness is `FAIL` or `INCONCLUSIVE`, no successful identity event may be emitted.

For video, evaluate both per-frame decisions and the final temporal decision. Record false accepts caused by temporal smoothing separately.

## Model candidate policy

Current implementation uses the MiniFASNet/Silent-Face family. Recent public implementations provide ONNX variants suitable for edge deployment, including V1SE/V2 and quantized variants, but public accuracy claims are not adopted as project evidence. They must be benchmarked on the project's own attack matrix.

Candidate changes require:

- identical crops/preprocessing;
- identical locked test split;
- model SHA-256;
- latency/memory measurement on target hardware;
- APCER/BPCER/ACER report;
- no regression at the identity-security operating point.

Do not replace the current architecture merely because a candidate has higher aggregate accuracy on a public dataset.

## Acceptance gate

A model is not production-ready until:

- model artifact and preprocessing are fingerprinted;
- all required attack classes have evaluation evidence;
- threshold is selected on validation data only;
- locked-test APCER/BPCER/ACER are reported;
- `INCONCLUSIVE` policy is explicitly tested;
- identity recognition is provably blocked for FAIL/INCONCLUSIVE;
- edge runtime benchmark is collected separately from accuracy metrics.

## Current blocker

The repository still lacks a licensed/validated liveness ONNX artifact and project-specific live/spoof dataset. No threshold or APCER/BPCER/ACER number should be claimed until those assets exist.
