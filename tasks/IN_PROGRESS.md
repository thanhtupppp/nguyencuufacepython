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

## 🎚️ P0.2: Real-image quality gate calibration
- [x] Define deterministic quality-feature, decision-state, and reason-code contract.
- [x] Add versioned engineering-default quality configuration; explicitly marked calibration-required.
- [x] P0.2.2 calibration protocol + leakage-safe analyzer implemented.
- [ ] P0.2.2 run with real SCRFD fixture + authorized person-disjoint dataset and calibrate thresholds.
- [x] P0.2.3 authorized dataset manifest contract + fail-closed validator implemented.

## 🧪 P0.4.28: CI collection regression hardening
- [x] Identify root causes in CI run #35: missing repository-root import path and missing `DATABASE_URL` during test collection.
- [x] Add `pythonpath = .` to pytest configuration.
- [x] Provide `DATABASE_URL` from the CI PostgreSQL service.
- [x] Identify second CI failure in run #39: missing `src.database.person_level_query` module required by person-level query regression tests.
- [x] Restore `src/database/person_level_query.py` with parameterized person-level ranking SQL and active-only contract.
- [ ] Fresh CI run on the repair commit must pass unit collection/execution and then run integration tests.

## 🧭 Next selected task
**P0.4.28 — verify the repair commit with a fresh successful CI run; if green, immediately execute the highest-value blocked real-model/data gate.**

Current blockers: CI run #39 failed because `tests/unit/test_person_level_query.py` imported a missing `src.database.person_level_query` module; this has now been restored. Actual licensed SCRFD/ArcFace checkpoint bytes and authorized evaluation image bytes are still not available through the repository connection. P0.2.3 closes the data-manifest validation side, while P0.4.25–P0.4.27 define reproducible model acquisition/provenance/ONNX-contract checks. No recognition threshold, quality threshold, or model architecture change is accepted without real execution evidence.
