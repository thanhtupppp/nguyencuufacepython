# Completed Tasks

Nhật ký các nhiệm vụ đã hoàn thành:

- [x] **2026-09-06**: Tạo file `README.md` định hướng toàn diện dự án.
- [x] **2026-09-06**: Tạo cấu trúc cây thư mục dự án (`docs/`, `tasks/`, `src/`, `benchmarks/`, `configs/`, `scripts/`, `tests/`).
- [x] **2026-09-06**: Thiết lập môi trường ảo Python 3.11 (`.venv/`) cài đầy đủ `onnxruntime`, `opencv-python`, `scipy`, `scikit-learn`, `pandas`, `pytest`, `matplotlib`, `psycopg`, `pgvector`.
- [x] **2026-09-06**: Hoàn thành toàn bộ hệ thống tài liệu kỹ thuật chuyên sâu từ [docs/00_MASTER_PLAN.md](file:///d:/nguyencuufacepython/docs/00_MASTER_PLAN.md) đến [docs/08_EDGE_DEVICE.md](file:///d:/nguyencuufacepython/docs/08_EDGE_DEVICE.md).
- [x] **2026-09-06**: Hoàn thành **P0.1 - Baseline Benchmark**:
  - `src/alignment/aligner.py`: Căn chỉnh 5 landmarks Umeyama Affine Transform 112x112 px.
  - `src/recognition/base.py`: Abstract class `BaseFaceRecognizer`.
  - `src/recognition/arcface.py`: ONNX Runtime ArcFace 512D feature extractor.
  - `src/detection/scrfd.py`: ONNX Runtime SCRFD Face Detector.
  - `benchmarks/scripts/metrics.py`: Tính Cosine Similarity, FAR, FRR, EER, ROC, Dual-Threshold Margin.
  - `benchmarks/scripts/generate_pairs.py`: Tạo `genuine.csv` và `impostor.csv`.
  - `benchmarks/scripts/run_benchmark.py`: Xuất báo cáo benchmark và dữ liệu phân bố.
- [x] **2026-09-06**: Hoàn thành **P0.2 - Quality Gate**:
  - `src/quality/quality_gate.py`: Lọc kích thước ($min \ge 60$ px), độ nét Laplacian variance ($Var \ge 50$), ước lượng góc nghiêng Yaw/Pitch/Roll ($\le 30^\circ$), độ sáng $[40, 220]$.
  - `tests/test_quality_gate.py`: 4 unit tests đạt 100% PASSED.
- [x] **2026-09-06**: Hoàn thành **P0.3 - Multi-Frame Tracking & Temporal Voting**:
  - `src/tracking/tracker.py`: IoU Hungarian bipartite matching, Tracklet maintenance.
  - Tự động chọn Best-Frame trong tracklet dựa trên Quality Score.
  - Thuật toán Temporal Voting qua cửa sổ $N = 5$ frames (ngưỡng đồng thuận $\ge 60\%$).
  - `tests/test_tracking.py`: 6 unit tests đạt 100% PASSED.
- [x] **2026-09-06**: Hoàn thành **P0.4 - Anti-Spoofing (Liveness Detection)**:
  - `src/anti_spoofing/liveness.py`: Multi-scale crop (1.0x & 2.7x), phân tích phổ Fourier 2D, MiniFASNet ONNX wrapper.
  - `tests/test_anti_spoofing.py`: 3 unit tests đạt 100% PASSED.
- [x] **2026-09-06**: Hoàn thành **P1.1 - Vector Database (PostgreSQL + pgvector)**:
  - `docker-compose.yml`: Cấu hình container PostgreSQL 16 tích hợp extension `pgvector`.
  - `scripts/init.sql`: DDL bảng `persons`, `face_embeddings`, `devices`, `access_logs` và HNSW index cosine distance.
  - `scripts/init_db.py`: Migration CLI runner.
  - `src/database/client.py`: `DatabaseClient` hỗ trợ PostgreSQL pgvector và in-memory fallback.
  - `tests/test_database.py`: 3 unit tests kiểm tra CRUD, vector search, Dual-Threshold margin decision.
- [x] **2026-09-06**: Toàn bộ **25/25 Unit Tests** đạt trạng thái **PASSED 100%**.
