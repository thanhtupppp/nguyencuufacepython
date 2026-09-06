# Face Recognition Benchmark Protocol

P0.1 is the **real-model baseline gate**. The repository already contains gallery/probe images, genuine/impostor pair lists, metrics, and a runner, but a benchmark run is not considered valid until it uses a real licensed ArcFace/InsightFace ONNX model and the same production preprocessing path.

## Required production path

`SCRFD -> 5-point landmarks -> Umeyama alignment -> 112x112 -> ArcFace -> L2-normalized 512D embedding`

Do not resize an arbitrary raw face image to `112x112` and call it aligned. If landmarks are unavailable, the dataset must be treated as an input fixture only, not as a production-quality recognition benchmark.

## Evaluation gates

1. **Pair verification:** genuine vs impostor cosine distributions.
2. **Threshold calibration:** select operating threshold on validation data only.
3. **False-positive gate:** report FAR/FMR at 1%, 0.1%, and 0.01% targets; do not claim a target that the number of impostor trials cannot statistically resolve.
4. **1:N identification:** evaluate gallery -> probe with per-person aggregation and top-1/top-2 margin.
5. **Condition breakdown:** frontal, angle, blur, low light, distance, and occlusion.
6. **Locked test:** freeze model, preprocessing, threshold, and margin before final test evaluation.

## Model identity

Every benchmark report must record:

- model name and exact version/fingerprint
- detector and detector configuration
- alignment template/version
- embedding dimension
- normalization and distance metric
- threshold and margin
- dataset version and split

Embeddings from different recognition models must never be mixed in one gallery. If the recognition model changes, regenerate embeddings from retained source images.

## Current status

- Dataset/pair generation: **implemented**
- Metric/ROC calculation: **implemented**
- Identity threshold + margin decision engine: **implemented**
- Real-model end-to-end benchmark: **PENDING** until a real model asset is supplied and the production detector/alignment path is exercised
- Quality-gate calibration: **next after baseline benchmark**
