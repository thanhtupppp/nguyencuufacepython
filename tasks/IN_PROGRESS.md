# In Progress Tasks

Nhiệm vụ đang thực hiện:

## 🚀 P1.2: FastAPI Recognition Backend
- [x] FastAPI app entrypoint (`src/api/main.py`).
- [x] `/api/v1/persons` CRUD danh tính.
- [x] HTTP contracts `/api/v1/faces/enroll`, `/recognize`, `/verify`.
- [x] API unit/contract tests scaffold (`tests/test_api.py`).
- [x] Shared database dependency; API fail-closed nếu model recognition thật chưa được cấu hình.
- [ ] Wire real production pipeline: SCRFD → 5-point alignment → ArcFace ONNX → quality/liveness → embedding.
- [ ] Replace model hook with configured model registry and model fingerprint/version validation.
- [ ] WebSocket `/ws/v1/events` stream kết quả realtime.
- [ ] Chạy integration tests với PostgreSQL + pgvector và real model asset.

## 🔬 P0.1-RUN: Real-model benchmark gate
- [ ] Chạy benchmark bằng model ArcFace/InsightFace ONNX hợp lệ.
- [ ] Khóa threshold + top1-top2 margin từ validation.
- [ ] Xuất FAR/FRR/EER/TAR và condition breakdown trên locked test.
