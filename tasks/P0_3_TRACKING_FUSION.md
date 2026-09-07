# P0.3 — Tracking + Temporal Identity Fusion

## Status

Implementation complete; empirical benchmark remains pending.

## Implemented

- IoU + Hungarian association remains the geometry baseline.
- Optional 512-D face-embedding appearance similarity participates in association.
- Quality/similarity-weighted temporal identity fusion replaces equal-weight-only voting.
- `UNKNOWN` remains a rejection state rather than a gallery identity.
- Short-lived lost tracks are retained and can be recovered by embedding similarity.
- Regression tests cover strong-evidence weighting, weak-IoU appearance association, lost-track recovery, and low-quality conflicting identity evidence.

## Safety/accuracy rule

The defaults in `FaceTracker` and `TemporalVotingEngine` are engineering defaults only. They are **not production thresholds** until calibrated on project video and recognition benchmark data.

The identity decision must still respect the recognition layer's locked threshold and top-1/top-2 margin. Tracking must not turn a below-threshold recognition into a confirmed person.

## Verification gate

Before marking P0.3 fully DONE, collect real multi-face video and measure:

- ID switches (IDSW)
- track fragmentation
- track continuity/recovery rate
- false identity assignment rate
- unknown/ambiguous rate
- identity accuracy by pose, blur, lighting and crowding
- latency/FPS impact from embedding association

Compare at minimum:

1. IoU + Hungarian only
2. IoU + Hungarian + embedding appearance
3. weighted temporal fusion enabled/disabled

Tune `appearance_threshold`, `recovery_embedding_threshold`, `appearance_weight`, window size and consensus requirements only from validation data; lock them before the final test set.

## Next task

P0.1-RUN remains the highest-priority blocker because real licensed SCRFD/ArcFace ONNX assets and the locked recognition benchmark are still required. Once those are available, run P0.3 against real embeddings/video and calibrate these tracking parameters.