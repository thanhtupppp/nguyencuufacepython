# Tasks TODO List - NguyenCuuFacePython

Bảng theo dõi tiến độ công việc theo các cấp độ ưu tiên (P0 -> P1 -> P2).

---

## 🎯 P0: Nghiên Cứu & Xác Lập Nền Tảng Nhận Diện (Core Recognition Foundation)

### P0.1 — Benchmark ArcFace Baseline & Xây Dựng Suite Kiểm Thử Nội Bộ (Đang thực hiện)
- [ ] Thiết lập cấu trúc thư mục bộ benchmark: `benchmarks/gallery/`, `benchmarks/probe/`, `benchmarks/pairs/`, `benchmarks/results/`.
- [ ] Tạo file dependencies `requirements.txt` chuẩn (numpy, opencv-python, onnxruntime, scipy, scikit-learn, pandas, pyyaml, tqdm, matplotlib).
- [ ] Xây dựng module trích xuất đặc trưng ArcFace thuần ONNX Runtime (`src/recognition/arcface.py`) với L2-normalization chuẩn.
- [ ] Xây dựng module căn chỉnh khuôn mặt 5 landmarks (`src/alignment/aligner.py`) theo thuật toán Umeyama Affine Transform ra ảnh 112x112.
- [ ] Xây dựng script sinh danh sách cặp kiểm thử `benchmarks/scripts/generate_pairs.py` (tạo `pairs/genuine.csv` và `pairs/impostor.csv`).
- [ ] Xây dựng thư viện tính toán chỉ số `benchmarks/scripts/metrics.py`: Cosine Similarity, FAR, FRR, EER, ROC Curve, Top-1 Accuracy, Top-2 Margin.
- [ ] Xây dựng runner hoàn chỉnh `benchmarks/scripts/run_benchmark.py`: xuất ra `similarity_distribution.csv`, `roc.csv`, và `threshold_report.md`.
- [ ] Đo đạc thực tế trên các điều kiện kiểm thử:
  - [ ] Frontal (chuẩn)
  - [ ] Góc mặt nghiêng (Angle)
  - [ ] Thiếu sáng / Ngược sáng (Low light)
  - [ ] Mờ do chuyển động (Blur)
  - [ ] Che khuất một phần (Occlusion)
  - [ ] Khoảng cách xa / Mặt nhỏ (Distance)
- [ ] Xác lập cặp tham số tối ưu: Ngưỡng $Threshold$ và $Margin$ tối thiểu chống nhận nhầm.

### P0.2 — Xây Dựng Quality Gate (Lọc Khuôn Mặt Kém Chất Lượng)
- [ ] Xây dựng module `src/quality/quality_gate.py`:
  - [ ] Kiểm tra kích thước bounding box ($min \ge 60$ px).
  - [ ] Đo độ nét bằng phương sai toán tử Laplacian ($Var \ge 50.0$).
  - [ ] Ước lượng góc xoay đầu từ 5 landmarks (Yaw, Pitch, Roll $\le 30^\circ$).
  - [ ] Kiểm tra độ sáng trung bình $[40, 220]$.
- [ ] Benchmark độ trễ bổ sung của Quality Gate (mục tiêu $< 3$ ms trên CPU).
- [ ] Đánh giá độ cải thiện của FAR và FRR sau khi bật Quality Gate.

### P0.3 — Multi-Frame Tracking & Temporal Voting
- [ ] Xây dựng module tracking `src/tracking/tracker.py` dựa trên ByteTrack / SORT.
- [ ] Xây dựng logic chọn frame tốt nhất trong tracklet (Best-frame Selection) dựa trên Quality Gate score.
- [ ] Triển khai thuật toán bỏ phiếu thời gian Temporal Voting qua cửa sổ $N = 5$ frames (ngưỡng đồng thuận $\ge 60\%$).
- [ ] Kiểm nghiệm khả năng chống nhận nhầm đột biến trên video thực tế.

### P0.4 — Anti-Spoofing (Chống Giả Mạo - Liveness Detection)
- [ ] Tích hợp mô hình MiniFASNet (Silent-Face-Anti-Spoofing) thuần ONNX Runtime vào `src/anti_spoofing/liveness.py`.
- [ ] Kiểm thử khả năng phát hiện:
  - [ ] Ảnh in màu / đen trắng 2D.
  - [ ] Video phát lại trên màn hình smartphone/iPad/laptop.
- [ ] Đo đạc tỷ lệ True Liveness Rate và False Liveness Rate.

---

## 🚀 P1: Hệ Thống Dữ Liệu & Dịch Vụ API

### P1.1 — Cơ Sở Dữ Liệu Vector (PostgreSQL + pgvector)
- [ ] Thiết lập Docker Compose cho PostgreSQL 16+ với extension `pgvector`.
- [ ] Tạo schema bảng `persons` và `face_embeddings`.
- [ ] Tạo HNSW index với `vector_cosine_ops`.
- [ ] Viết module `src/database/client.py` xử lý truy vấn top-k theo cosine distance và margin filter.

### P1.2 — FastAPI Recognition Backend
- [ ] Xây dựng FastAPI app trong `src/api/`.
- [ ] Endpoint `/api/v1/persons` (CRUD danh tính).
- [ ] Endpoint `/api/v1/faces/enroll` (đăng ký khuôn mặt, trích xuất embedding, lưu pgvector).
- [ ] Endpoint `/api/v1/faces/recognize` (nhận diện khuôn mặt từ ảnh/frame).
- [ ] Endpoint `/api/v1/faces/verify` (xác thực 1:1).
- [ ] Authentication, API Key và phân quyền thiết bị.

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
