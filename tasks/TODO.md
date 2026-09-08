# Tasks TODO List - NguyenCuuFacePython

Bảng theo dõi tiến độ thực tế. `[x]` chỉ có nghĩa implementation/test contract đã có trong repo; benchmark thực nghiệm chỉ DONE khi có kết quả thật.

## P0 — Core Recognition

### P0.1 Real SCRFD + ArcFace benchmark — IN PROGRESS / BLOCKED ON LICENSED ASSETS + DATASET
- [x] Pair generation + metric suite.
- [x] Dual decision rule: score + top1-top2 margin.
- [x] Benchmark preflight validation, gồm duplicate/reversed-pair leakage và CSV label consistency.
- [x] Real-model runner: SCRFD -> 5 landmarks -> Umeyama 112x112 -> ArcFace -> 512D L2.
- [x] External ONNX asset fingerprint/contract inspector.
- [x] Model asset licensing gate/documentation.
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
- [x] Production default now fails closed when liveness model weights are missing.
- [x] Benchmark/release protocol: attack matrix + APCER/BPCER/ACER + INCONCLUSIVE policy.
- [ ] Cấp model liveness ONNX hợp lệ + SHA-256.
- [ ] Cấp live/print/screen/replay locked dataset.
- [ ] Benchmark live/print/screen/replay với model liveness hợp lệ.
- [ ] Khóa threshold/inconclusive policy và APCER/BPCER/ACER.

## P1 — Data/API/Devices

### P1.1 PostgreSQL + pgvector — IMPLEMENTED / CORRECTNESS HARDENING + INTEGRATION TEST PENDING
- [x] PostgreSQL + pgvector + HNSW.
- [x] persons/embeddings/devices/access_logs.
- [x] Vector client + unit tests.
- [x] Regression test defining person-level top-k semantics when one person has many templates.
- [x] Correctness-first SQL specification for best-template-per-person ranking.
- [x] Reusable person-level search helper.
- [x] Package-level `DatabaseClient` now delegates PostgreSQL search to person-level ranking helper.
- [x] Added static guard against direct imports of legacy `DatabaseClient`.
- [x] Added the import-boundary guard to the CI workflow.
- [x] Corrected guard scope to `src/` only; Run #44 proved the corrected import guard passes.
- [x] Added CI-safe `POSTGRES_PASSWORD` to the workflow.
- [x] Replaced fragile Mosquitto `$SYS/broker/uptime` healthcheck with a local `mosquitto_pub` broker-readiness probe.
- [ ] Observe a successful CI broker run and capture result.
  - Run #48 (2026-09-08 02:57 UTC) still failed at the old Mosquitto `$SYS` healthcheck before the new readiness probe commit.
  - New commit `34446ab` contains the corrected readiness probe; a fresh CI run is required.
- [x] Deterministic pgvector retrieval benchmark harness (synthetic, no biometric data).
- [ ] Run benchmark under filtering and record recall/latency results (issue #3).
- [ ] Benchmark candidate-first HNSW vs one-vector-per-person prototype architecture (issue #4).
- [ ] E2E integration với real model + PostgreSQL/pgvector.

### P1.2 FastAPI — IMPLEMENTED / INTEGRATION TEST PENDING
- [x] Persons CRUD.
- [x] Enroll/recognize/verify.
- [x] Model registry + fail-closed pipeline.
- [x] WebSocket `/ws/v1/events`.
- [x] Strict liveness integration.
- [x] API contract tests với fake pipeline/database.
- [ ] E2E integration test với PostgreSQL/pgvector + licensed assets.

### P1.3 Multi-device — PARTIAL
- [x] WebSocket event transport + contract tests.
- [x] MQTT v1 topic/payload contract + validation tests.
- [x] Paho MQTT adapter: TLS, LWT availability, QoS/retained policy, reconnect backoff, idempotent request_id handling.
- [x] MQTT adapter unit tests.
- [x] Local Mosquitto integration harness + opt-in broker tests.
- [x] Shared idempotency abstraction: bounded TTL fallback + PostgreSQL atomic store.
- [x] CI workflow provisions Mosquitto and runs the real-broker test suite.
- [ ] Observe a successful CI broker run and capture result.
- [x] HTTP/WebSocket adapter contract tests.
- [x] Multi-camera device/session runtime model.
- [x] Persist device/session state for multi-worker deployment.

## P2 — Edge + Model A/B

### P2.1 PC / Raspberry Pi / Android — PROTOCOL READY / BENCHMARK PENDING
- [x] Edge runtime capability preflight script.
- [x] Fixed cross-device benchmark protocol and acceptance criteria.
- [x] Repeatable ONNX Runtime latency benchmark runner (session creation, p50/p95/p99, throughput, provider metadata).
- [ ] ONNX Runtime CPU baseline benchmark on PC using the real SCRFD + ArcFace assets.
- [ ] Raspberry Pi ARM64 benchmark.
- [ ] Android NNAPI/XNNPACK benchmark.
- [ ] FP16/INT8 chỉ sau khi accuracy baseline được khóa.

### P2.2 Model A/B
- [ ] AdaFace vs ArcFace cùng protocol.
- [ ] MagFace vs ArcFace cùng protocol.
- [ ] MobileFaceNet vs ArcFace cho edge trade-off.
- [ ] Chỉ đổi architecture nếu gain có ý nghĩa thống kê và đáp ứng false-match target.

## Current execution order

P0.1 real-model benchmark (blocked on assets/data) -> P0.2 quality calibration -> P0.3 tracking benchmark -> P0.4 liveness benchmark (protocol ready; blocked on model/data) -> P1.1 verify successful CI run + vector recall benchmark (issue #3) + retrieval architecture benchmark (issue #4) -> P1.2 PostgreSQL-backed API integration -> P1.3 MQTT broker execution -> P2.1 PC CPU baseline -> P2.1 Raspberry Pi -> P2.1 Android -> P2.2 model A/B.
