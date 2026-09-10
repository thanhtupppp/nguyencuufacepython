# In Progress Tasks

Nhiệm vụ đang thực hiện:

## 🚀 P1.2: FastAPI Recognition Backend
- [x] FastAPI app entrypoint (`src/api/main.py`).
- [x] `/api/v1/persons` CRUD danh tính.
- [x] HTTP contracts `/api/v1/faces/enroll`, `/recognize`, `/verify`.
- [x] API unit/contract tests scaffold (`tests/test_api.py`).
- [x] Shared database dependency; API fail-closed nếu model recognition thật chưa được cấu hình.
- [x] Model registry contract: model_version → ONNX asset + SHA-256 fingerprint + 512D contract.
- [x] Recognition pipeline contract: SCRFD → quality gate → 5-point alignment → ArcFace embedding, with fail-closed face/quality checks.
- [x] Wire model registry and pipeline into API runtime; fail-closed until real licensed assets + fingerprint are installed.
- [x] WebSocket `/ws/v1/events` real-time event stream and external event publishing endpoint.
- [x] Optional strict liveness gate wired into the production recognition pipeline when `LIVENESS_MODEL_PATH` is configured.
- [ ] Chạy integration tests với PostgreSQL + pgvector và real model asset.

## 🔬 P0.1-RUN: Real-model benchmark gate
- [ ] Chạy benchmark bằng model ArcFace/InsightFace ONNX hợp lệ.
- [ ] Khóa threshold + top1-top2 margin từ validation.
- [ ] Xuất FAR/FRR/EER/TAR và condition breakdown trên locked test.

## 🔐 P0.1.3: Recognition model provenance + license gate
- [x] Document exact provenance/license requirements for SCRFD + ArcFace.
- [ ] Pin exact SCRFD artifact + SHA-256 + weight license evidence.
- [ ] Pin exact ArcFace artifact + SHA-256 + weight license evidence.
- [x] Implement fail-closed recognition manifest verifier in canonical registry.
- [x] Connect provenance/license gate to `src/recognition/model_registry.py` + `src/api/recognition_runtime.py`.
- [x] Add fail-closed provenance/asset/512-D test matrix.
- [x] Do not count the temporary duplicate `app/model_registry.py` experiment as production integration; it was removed to keep one canonical runtime.

## 🛡️ P0.4: Liveness empirical calibration
- [ ] Benchmark genuine/live vs print/screen/replay attacks bằng model liveness thực.
- [ ] Khóa threshold và inconclusive policy trên validation.
- [ ] Đo APCER/BPCER/ACER hoặc tương đương theo attack condition.

## 🧭 Next selected task
**P0.1.3.3 — pin the actual authorized SCRFD + ArcFace artifacts, then execute the real-model integration/benchmark gate.**

Acceptance: exact model files are available, SHA-256 and weight-license evidence are recorded, the canonical registry verifies both detector and recognizer, FastAPI readiness is `ready` only after both pass, and the benchmark produces reproducible FAR/FRR/EER/TAR results on a locked validation/test protocol.
