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
- [ ] Add liveness gate to production pipeline after a validated anti-spoofing model is configured.
- [ ] WebSocket `/ws/v1/events` stream kết quả realtime.
- [ ] Chạy integration tests với PostgreSQL + pgvector và real model asset.

## 🔬 P0.1-RUN: Real-model benchmark gate
- [ ] Chạy benchmark bằng model ArcFace/InsightFace ONNX hợp lệ.
- [ ] Khóa threshold + top1-top2 margin từ validation.
- [ ] Xuất FAR/FRR/EER/TAR và condition breakdown trên locked test.
