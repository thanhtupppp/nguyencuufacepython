# Tasks TODO List

Danh sách nhiệm vụ của dự án **NguyenCuuFacePython**.

---

## 📌 Phase 1: Research & Benchmarking (Hiện tại)

### 1.1 Nghiên cứu & So sánh Recognition Models (ArcFace vs AdaFace vs MagFace)
- [ ] Xây dựng bộ test dataset đa điều kiện (ánh sáng yếu, góc nghiêng, blur, khoảng cách xa, che một phần).
- [ ] Thiết lập harness benchmark chuẩn: trích xuất embedding 512D với cùng input crop 112x112.
- [ ] Đánh giá phân phối Cosine Similarity (Same Person vs Different Person).
- [ ] Tính toán các chỉ số: FAR (False Accept Rate), FRR (False Reject Rate), EER (Equal Error Rate).
- [ ] Đo lường Candidate Margin (độ chênh lệch giữa candidate #1 và candidate #2).
- [ ] Đo lường độ trễ suy luận (Inference Latency) & RAM/VRAM footprint trên CPU/GPU.

### 1.2 Nghiên cứu Face Detectors (SCRFD vs RetinaFace vs YuNet)
- [ ] Đánh giá độ chính xác phát hiện khuôn mặt nhỏ/nghiêng/mờ.
- [ ] Kiểm tra độ ổn định của 5 facial landmarks khi góc nghiêng thay đổi.
- [ ] Đo tốc độ FPS trên CPU và GPU (SCRFD-500M, SCRFD-2.5G, SCRFD-10G, YuNet ONNX).

### 1.3 Nghiên cứu Anti-Spoofing & Quality Assessment
- [ ] Khảo sát Silent-Face-Anti-Spoofing (MiniFASNetV1/V2).
- [ ] Thiết kế Quality Gate: Blur check (Laplacian variance), Face size threshold, Head pose limit (Yaw/Pitch/Roll).
- [ ] Kiểm tra độ trễ bổ sung của Quality Gate + Anti-Spoofing trước khi đưa vào recognition.

### 1.4 Nghiên cứu Vector Database & Tracking
- [ ] Thử nghiệm PostgreSQL 16+ với `pgvector` (HNSW index vs IVFFlat index).
- [ ] Thiết kế cơ chế Multi-object Face Tracking (ByteTrack/SORT) kết hợp Temporal Voting qua $N$ frames.

---

## 📌 Phase 2: Baseline Engine

- [ ] Xây dựng module `src/detection/` bọc SCRFD ONNX Runtime (hỗ trợ CPU/CUDA).
- [ ] Xây dựng module `src/alignment/` chuẩn hóa khuôn mặt bằng 5 điểm landmark theo affine transform (112x112).
- [ ] Xây dựng module `src/recognition/` với interface thống nhất:
  ```python
  class BaseFaceRecognizer(ABC):
      @abstractmethod
      def extract_embedding(self, face_img: np.ndarray) -> np.ndarray: ...
  ```
- [ ] Triển khai `ArcFaceRecognizer` và `AdaFaceRecognizer`.
- [ ] Module `src/database/`: tạo schema `persons` và `face_embeddings` với pgvector.
- [ ] Triển khai logic Temporal Voting và Margin Check.

---

## 📌 Phase 3: Accuracy & Anti-Spoofing Integration

- [ ] Tích hợp Quality Gate vào pipeline (loại bỏ frame mờ, góc lệch quá lớn trước recognition).
- [ ] Tích hợp MiniFASNet Anti-Spoofing (loại bỏ ảnh tĩnh, video điện thoại).
- [ ] Cơ chế Dynamic Threshold theo chất lượng ảnh / kích thước bounding box.
- [ ] Cơ chế Best-frame Selection trong tracking tracklet.

---

## 📌 Phase 4: Backend API & Service

- [ ] Xây dựng FastAPI app trong `src/api/`.
- [ ] Endpoint `/api/v1/persons` (CRUD danh tính).
- [ ] Endpoint `/api/v1/faces/enroll` (đăng ký khuôn mặt mới, trích xuất embedding, lưu pgvector).
- [ ] Endpoint `/api/v1/faces/recognize` (nhận diện khuôn mặt từ frame gửi lên).
- [ ] Endpoint `/api/v1/faces/verify` (xác thực 1:1 giữa khuôn mặt và person_id).
- [ ] Quản lý cấu hình động (thresholds, margin, model version).
- [ ] Thêm JWT Authentication & API Key cho thiết bị ngoại vi.

---

## 📌 Phase 5: Multi-Device & Edge Integration

- [ ] Dockerize Backend (FastAPI + PostgreSQL + pgvector).
- [ ] Kiểm thử chạy engine trên Raspberry Pi (ONNX Runtime ARM64 / NCNN / TFLite).
- [ ] Thiết kế giao thức MQTT cho ESP32 (gửi tín hiệu trigger cảm biến, nhận lệnh mở khóa/relay).
- [ ] WebSocket streaming cho camera realtime và cập nhật giao diện web.

---

## 📌 Phase 6: Production Hardening

- [ ] Stress-test chạy liên tục 24/7.
- [ ] Kiểm thử nhiều camera đồng thời.
- [ ] Benchmark tài nguyên (CPU, RAM, GPU, độ trễ end-to-end).
- [ ] Viết tài liệu hướng dẫn triển khai hoàn chỉnh.
