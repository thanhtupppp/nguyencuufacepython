# Model artifacts and reproducibility

This directory is intentionally metadata-first. Do not commit pretrained face-recognition weights until their model/weight license explicitly permits the intended deployment.

## Required manifest

Every evaluated detector/recognizer pair must have a manifest containing:

- model name and exact version/release
- source URL
- artifact filename
- SHA256
- model/weight license and evidence URL
- detector input size and preprocessing
- landmark coordinate convention
- recognizer input size/channel order/normalization
- embedding dimension and L2-normalization rule
- ONNX opset/runtime compatibility
- PC CPU latency and peak memory
- Raspberry Pi/Android deployment notes

## Current architecture contract

SCRFD -> 5-point alignment (112x112) -> ArcFace-compatible 512-D embedding -> quality gate -> person retrieval.

A model change is not accepted from license convenience or benchmark claims alone. It requires the same real-image protocol and must preserve or improve false-match/false-non-match behavior, latency, and reproducibility.

## License rule

Repository/source-code licensing and pretrained-weight licensing are tracked separately. Restricted or research-only weights must not be committed or shipped as production artifacts.
