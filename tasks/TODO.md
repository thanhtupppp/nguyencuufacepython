# Tasks TODO List - NguyenCuuFacePython

Bảng theo dõi tiến độ công việc theo các cấp độ ưu tiên (P0 -> P1 -> P2).

---

## 🎯 P0: Nghiên Cứu & Xác Lập Nền Tảng Nhận Diện (Core Recognition Foundation) - [HOÀN THÀNH 100%]

### P0.1 — Benchmark ArcFace Baseline & Xây Dựng Suite Kiểm Thử Nội Bộ
- [x] Thiết lập cấu trúc thư mục bộ benchmark: `benchmarks/gallery/`, `benchmarks/probe/`, `benchmarks/pairs/`, `benchmarks/results/`.
- [x] Tạo file dependencies `requirements.txt` chuẩn (numpy, opencv-python, onnxruntime, scipy, scikit-learn, pandas, pyyaml, tqdm, matplotlib).
- [x] Xây dựng module trích xuất đặc trưng ArcFace thuần ONNX Runtime (`src/recognition/arcface.py`) với L2-normalization chuẩn.
- [x] Xây dựng module căn chỉnh khuôn mặt 5 landmarks (`src/alignment/aligner.py`) theo thuật toán Umeyama Affine Transform ra ảnh 112x112.
- [x] Xây dựng script sinh danh sách cặp kiểm thử `benchmarks/scripts/generate_pairs.py` (tạo `pairs/genuine.csv` và `pairs/impostor.csv`).
- [x] Xây dựng thư viện tính toán chỉ số `benchmarks/scripts/metrics.py`: Cosine Similarity, FAR, FRR, EER, ROC Curve, Top-1 Accuracy, Top-2 Margin.
- [x] Xây dựng runner hoàn chỉnh `benchmarks/scripts/run_benchmark.py`: xuất ra `similarity_distribution.csv`, `roc.csv`, và `threshold_report.md`.
- [x] Tạo script sinh tập kiểm chuẩn `benchmarks/scripts/create_synthetic_testset.py` để verify harness.
- [x] Xác lập cơ chế phán quyết kép Dual-Threshold (Score + Margin).

### P0.2 — Xây Dựng Quality Gate (Lọc Khuôn Mặt Kém Chất Lượng)
- [x] Xây dựng module `src/quality/quality_gate.py`:
  - [x] Kiểm tra kích thước bounding box ($min \ge 60$ px).
  - [x] Đo độ nét bằng phương sai toán tử Laplacian ($Var \ge 50.0$).
  - [x] Ước lượng góc xoay đầu từ 5 landmarks (Yaw, Pitch, Roll $\le 30^\circ$).
  - [x] Kiểm tra độ sáng trung bình $[40, 220]$.
- [x] Viết unit tests kiểm tra Quality Gate (`tests/test_quality_gate.py` - 4/4 PASSED).

### P0.3 — Multi-Frame Tracking & Temporal Voting
- [x] Xây dựng module tracking `src/tracking/tracker.py` (Hungarian / IoU matching).
- [x] Tự động duy trì Tracklet ID qua video stream.
- [x] Logic Best-Frame Selection theo `overall_quality_score`.
- [x] Triển khai thuật toán Temporal Voting qua cửa sổ $N = 5$ frames (ngưỡng đồng thuận $\ge 60\%$).
- [x] Cơ chế triệt tiêu nhiễu nhận nhầm đơn frame (Noise suppression test).
- [x] Viết unit tests tracking và voting (`tests/test_tracking.py` - 6/6 PASSED).

### P0.4 — Anti-Spoofing (Chống Giả Mạo - Liveness Detection)
- [x] Xây dựng module `src/anti_spoofing/liveness.py` (MiniFASNet / Silent-Face-Anti-Spoofing).
- [x] Tiền xử lý Multi-scale crop: crop 1 (sát mặt 1.0x) và crop 2 (rộng bao quanh 2.7x kèm padding).
- [x] Phân tích phổ Fourier 2D phát hiện hoa văn Moiré màn hình và lưới điểm in ảnh giấy.
- [x] Ngưỡng phát hiện Live vs Spoof ($\ge 0.85$).
- [x] Viết unit tests cho Anti-Spoofing (`tests/test_anti_spoofing.py` - 3/3 PASSED).

---

## 🚀 P1: Hệ Thống Dữ Liệu & Dịch Vụ API

### P1.1 — Cơ Sở Dữ Liệu Vector (PostgreSQL + pgvector) - [HOÀN THÀNH]
- [x] Thiết lập `docker-compose.yml` cho PostgreSQL 16+ với extension `pgvector`.
- [x] Tạo schema DDL chuẩn `scripts/init.sql` (bảng `persons`, `face_embeddings`, `devices`, `access_logs`).
- [x] Thiết lập HNSW index với `vector_cosine_ops`.
- [x] Viết script migration `scripts/init_db.py`.
- [x] Xây dựng module kết nối và truy vấn CSDL vector `src/database/client.py` (hỗ trợ cả PostgreSQL pgvector lẫn in-memory fallback).
- [x] Viết unit tests cho database client (`tests/test_database.py` - 3/3 PASSED).

### P1.2 — FastAPI Recognition Backend (Đang tiếp tục)
- [ ] Xây dựng FastAPI app trong `src/api/main.py`.
- [ ] Endpoint `/api/v1/persons` (CRUD danh tính).
- [ ] Endpoint `/api/v1/faces/enroll` (đăng ký khuôn mặt, trích xuất embedding, lưu pgvector).
- [ ] Endpoint `/api/v1/faces/recognize` (nhận diện khuôn mặt từ ảnh/frame).
- [ ] Endpoint `/api/v1/faces/verify` (xác thực 1:1).
- [ ] WebSocket `/ws/v1/events` stream kết quả realtime.
- [ ] Viết unit tests cho API endpoints (`tests/test_api.py`).

### P1.3 — Giao Thức Đa Thiết Bị & Camera
- [ ] Xây dựng MQTT Client cho ESP32 (lệnh mở cửa relay, phản hồi trạng thái cảm biến).
- [ ] Xây dựng WebSocket Streamer cho camera thời gian thực và dashboard.

---

## 🔬 P2: Tối Ưu Hóa Edge Device & A/B Testing Models

### P2.1 — Tối Ưu Cho Edge Device (Raspberry Pi & Android)
- [ ] Tối ưu hóa mô hình sang ONNX FP16 / INT8 Quantization.
- [ ] Benchmark hiệu năng trên Raspberry Pi (ARM64 CPU / NPU).
- [ ] Xây dựng SDK/Client cho Android.

### P2.2 — A/B Test So Sánh Mô Hình
- [ ] Chạy bộ benchmark P0 trên **AdaFace** (IR-50 / IR-101).
- [ ] Chạy bộ benchmark P0 trên **MagFace** (ResNet-50).
- [ ] Chạy bộ benchmark P0 trên **MobileFaceNet** (cho thiết bị nhúng).
- [ ] Đánh giá định lượng xem AdaFace/MagFace có vượt trội ArcFace đủ lớn để thay thế trong pipeline chính thức hay không.
