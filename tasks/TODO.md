# Tasks TODO List - NguyenCuuFacePython

Bảng theo dõi tiến độ thực tế. `[x]` chỉ có nghĩa implementation/test contract đã có trong repo; benchmark thực nghiệm chỉ DONE khi có kết quả thật.

## P0 — Core Recognition

### P0.1 Real SCRFD + ArcFace benchmark — IN PROGRESS / BLOCKED ON LICENSED ASSETS + DATASET
- [x] Pair generation + metric suite.
- [x] Dual decision rule: score + top1-top2 margin.
- [x] Benchmark preflight validation, gồm duplicate/reversed-pair leakage và CSV label consistency.
- [x] Real-model runner: SCRFD -> 5 landmarks -> Umeyama 112x112 -> ArcFace -> 512D L2.
- [x] External ONNX asset fingerprint/contract inspector.
- [ ] Cấp SCRFD ONNX hợp lệ + SHA-256.
- [ ] Cấp ArcFace/InsightFace ONNX hợp lệ + SHA-256.
- [ ] Cấp gallery/probe images và validation/locked-test split.
- [ ] Chạy benchmark thật và xuất metrics/ROC/report.
- [ ] Khóa threshold + margin từ validation.

### P0.2 Quality Gate — IMPLEMENTED / CALIBRATION PENDING
- [x] Face size, blur, brightness, pose.
- [x] Unit tests.
- [ ] Calibrate ngưỡng bằng dữ liệu real-model.

### P0.3 Tracking + Temporal Identity Fusion — IMPLEMENTED / EMPIRICAL BENCHMARK PENDING
- [x] Hungarian IoU/geometry association.
- [x] Appearance embedding association.
- [x] Lost-track recovery.
- [x] Quality/similarity-weighted temporal evidence.
- [x] Regression tests.
- [ ] Benchmark ID switches, fragmentation, recovery, identity accuracy trên video thật.

### P0.4 Liveness — RUNTIME INTEGRATED / CALIBRATION PENDING
- [x] Strict PASS/FAIL/INCONCLUSIVE gate.
- [ ] Benchmark live/print/screen/replay với model liveness hợp lệ.
- [ ] Khóa threshold/inconclusive policy và APCER/BPCER/ACER.

## P1 — Data/API/Devices

### P1.1 PostgreSQL + pgvector — IMPLEMENTED / INTEGRATION TEST PENDING
- [x] PostgreSQL + pgvector + HNSW.
- [x] persons/embeddings/devices/access_logs.
- [x] Vector client + unit tests.
- [ ] E2E integration với real model + PostgreSQL/pgvector.

### P1.2 FastAPI — IMPLEMENTED / INTEGRATION TEST PENDING
- [x] Persons CRUD.
- [x] Enroll/recognize/verify.
- [x] Model registry + fail-closed pipeline.
- [x] WebSocket `/ws/v1/events`.
- [x] Strict liveness integration.
- [ ] E2E integration test với PostgreSQL/pgvector + licensed assets.

### P1.3 Multi-device — PARTIAL
- [x] WebSocket event transport.
- [x] MQTT v1 topic/payload contract + validation tests.
- [x] Paho MQTT adapter: TLS, LWT availability, QoS/retained policy, reconnect backoff, idempotent request_id handling.
- [x] MQTT adapter unit tests.
- [ ] MQTT broker integration test.
- [ ] HTTP/WebSocket adapter tests.
- [ ] Multi-camera device/session model.

## P2 — Edge + Model A/B

### P2.1 PC / Raspberry Pi / Android
- [ ] ONNX Runtime CPU baseline benchmark.
- [ ] Raspberry Pi ARM64 benchmark.
- [ ] Android NNAPI/XNNPACK benchmark.
- [ ] FP16/INT8 chỉ sau khi accuracy baseline được khóa.

### P2.2 Model A/B
- [ ] AdaFace vs ArcFace cùng protocol.
- [ ] MagFace vs ArcFace cùng protocol.
- [ ] MobileFaceNet vs ArcFace cho edge trade-off.
- [ ] Chỉ đổi architecture nếu gain có ý nghĩa thống kê và đáp ứng false-match target.

## Current execution order

P0.1 real-model benchmark (blocked on assets/data) -> P0.2 quality calibration -> P0.3 tracking benchmark -> P0.4 liveness calibration -> P1 integration tests -> P1.3 MQTT broker integration + HTTP/WebSocket adapter tests -> multi-camera/device transport -> edge benchmarks -> model A/B.
