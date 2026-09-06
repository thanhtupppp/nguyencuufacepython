# Completed Tasks

Nhật ký các nhiệm vụ đã hoàn thành:

- [x] **2026-09-06**: Tạo file `README.md` định hướng toàn diện dự án.
- [x] **2026-09-06**: Tạo cấu trúc cây thư mục dự án (`docs/`, `tasks/`, `src/`, `benchmarks/`, `configs/`, `scripts/`, `tests/`).
- [x] **2026-09-06**: Thiết lập môi trường ảo Python 3.11 với các dependencies nền tảng.
- [x] **2026-09-06**: Hoàn thành toàn bộ hệ thống tài liệu kỹ thuật từ `docs/00_MASTER_PLAN.md` đến `docs/08_EDGE_DEVICE.md`.
- [x] **2026-09-06**: Hoàn thành implementation **P0.1 - Baseline Recognition + Benchmark Framework**:
  - `src/alignment/aligner.py`: 5-landmark Umeyama alignment 112x112.
  - `src/recognition/base.py`, `src/recognition/arcface.py`: ONNX Runtime ArcFace 512D extractor.
  - `src/detection/scrfd.py`: ONNX Runtime SCRFD detector.
  - `benchmarks/scripts/metrics.py`, `generate_pairs.py`, `run_benchmark.py`: evaluation framework.
  - **Lưu ý:** real-model end-to-end benchmark chưa được đánh dấu hoàn thành; cần model asset hợp lệ và chạy đúng production preprocessing path.
- [x] **2026-09-06**: Hoàn thành **P0.2 - Quality Gate implementation** và unit tests.
- [x] **2026-09-06**: Hoàn thành **P0.3 - Multi-Frame Tracking & Temporal Voting implementation** và unit tests.
- [x] **2026-09-06**: Hoàn thành **P0.4 - Anti-Spoofing implementation** và unit tests.
- [x] **2026-09-06**: Hoàn thành **P1.1 - PostgreSQL + pgvector implementation** và database tests.
- [x] **2026-09-06**: Hoàn thành safety-first identity decision engine với threshold + top1-top2 margin và tests.
- [x] **2026-09-07**: Thêm FastAPI application entrypoint, person CRUD API và HTTP contracts cho enroll/recognize/verify; endpoints fail closed khi real recognition model chưa được cấu hình.
- [x] **2026-09-07**: Bổ sung API dependencies và FastAPI runtime dependencies.
