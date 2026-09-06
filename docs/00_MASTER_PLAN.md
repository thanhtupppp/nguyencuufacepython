# Kế Hoạch Tổng Thể Kiến Trúc & Triển Khai (Master Plan)

> **Dự án**: NguyenCuuFacePython  
> **Mục tiêu chính**: Xây dựng hệ thống nhận diện khuôn mặt chính xác cao, chống nhận nhầm người (anti false positive), hoạt động ổn định trong thực tế, không cần train lại model khi thêm người mới, và có khả năng mở rộng đa nền tảng (Server, PC, Raspberry Pi, Android, ESP32).

---

## 1. Triết Lý Thiết Kế Cốt Lõi

1. **Accuracy & Anti-False-Positive First**: Thà từ chối nhận diện một frame kém chất lượng (`Unknown` hoặc chờ frame kế tiếp) hơn là nhận nhầm danh tính người khác.
2. **Benchmark Trước — Xây Dựng Hệ Thống Sau**: Tuyệt đối **không vội xây Database (pgvector) hay Backend API** khi chưa có dữ liệu benchmark định lượng thực tế. Nếu nền tảng recognition chưa kiểm soát được tỷ lệ nhận nhầm (FAR) và chưa xác định được ngưỡng (Threshold) + Margin tối ưu, thì hệ thống dù được xây dựng đẹp đến đâu cũng sẽ thất bại trong thực tế.
3. **Giữ Vững Baseline SCRFD + ArcFace**: Không vội vàng thay thế ArcFace bằng AdaFace/MagFace khi chưa có số liệu đo đạc trên dữ liệu camera thực tế của dự án. AdaFace có ưu thế lý thuyết trên ảnh mờ, MagFace mã hóa chất lượng khuôn mặt, nhưng ArcFace vẫn là tiêu chuẩn vàng ổn định nhất để làm cột mốc đo lường baseline.
4. **Độc Lập Runtime & Licensing**: Lưu ý bản cập nhật InsightFace 1.0 / InsightFace Server (2026) và các điều khoản bản quyền của model (như `buffalo_l`). Toàn bộ wrapper suy luận được thiết kế dựa trên **ONNX Runtime** độc lập, tách rời logic nhận diện khỏi phụ thuộc cứng vào bất kỳ thư viện đóng gói nào.
5. **Quyết Định Chuỗi Thời Gian (Temporal Voting)**: Không bao giờ tin tưởng tuyệt đối vào 1 frame đơn lẻ. Danh tính chỉ được xác nhận khi đạt tỷ lệ đồng thuận qua cửa sổ trượt $N$ frames liên tiếp trong cùng một tracklet.

---

## 2. Luồng Pipeline Xử Lý Chi Tiết (End-to-End Architecture)

```text
                                 [ CAMERA STREAM ]
                                (RTSP / USB / File)
                                         │
                                         ▼
                                ┌─────────────────┐
                                │  Face Detector  │ ◄── SCRFD (500M / 2.5G / 10G)
                                │ (Bbox + 5 Lmk)  │
                                └────────┬────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │  Face Tracking  │ ◄── ByteTrack / SORT
                                │ (Tracklet ID)   │     (Duy trì đối tượng qua frames)
                                └────────┬────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │  Quality Gate   │ ◄── Kiểm tra:
                                │ (Đánh giá lọc)  │     - Độ phân giải (min 60x60, opt 112x112)
                                └────────┬────────┘     - Độ mờ (Laplacian variance > threshold)
                                         │              - Góc quay mặt (Yaw, Pitch, Roll <= 30°)
                                         ▼              - Độ sáng / Tương phản hợp lệ
                                ┌─────────────────┐
                                │  Anti-Spoofing  │ ◄── Silent-Face-Anti-Spoofing / MiniFASNet
                                │  (Liveness)     │     (Chống ảnh in, màn hình điện thoại/iPad)
                                └────────┬────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │ Face Alignment  │ ◄── 5 Landmarks Similarity Transform
                                │  (Chuẩn hóa)    │     Crop chuẩn kích thước 112x112 px
                                └────────┬────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │ Feature Extractor│ ◄── ArcFace Baseline (512D)
                                │ (L2-Normalized) │     (Sau này A/B test AdaFace, MagFace)
                                └────────┬────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │ Vector Storage  │ ◄── In-Memory Gallery (P0 Benchmark)
                                │  (pgvector)     │     PostgreSQL 16+ HNSW (P1 Production)
                                └────────┬────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │ Decision Engine │ ◄── 1. Match Threshold (Similarity >= T)
                                │ (Chống nhầm lẫn)│     2. Candidate Margin (Top1 - Top2 >= Margin)
                                └────────┬────────┘     3. Dynamic Threshold theo Face Quality
                                         │
                                         ▼
                                ┌─────────────────┐
                                │ Temporal Voting │ ◄── Tổng hợp kết quả trong tracklet N frames:
                                │ (Bầu chọn chuỗi)│     Đạt tỷ lệ đồng thuận >= 60% -> person_id
                                └────────┬────────┘
                                         │
                                         ▼
                               [ KẾT QUẢ CUỐI CÙNG ]
                            (person_id, name, confidence)
```

---

## 3. Thứ Tự Triển Khai Thực Tế (Phased Roadmap)

Quy trình phát triển tuân thủ nghiêm ngặt nguyên tắc chia nhỏ theo mức độ ưu tiên:

```text
P0.1: Benchmark ArcFace Baseline & Xây Dựng Suite Kiểm Thử Nội Bộ
  │
  ▼
P0.2: Xây Dựng Quality Gate (Blur, Size, Pose, Illumination)
  │
  ▼
P0.3: Multi-Frame Tracking & Temporal Voting
  │
  ▼
P0.4: Tích Hợp Anti-Spoofing (Liveness Detection)
  │
  ▼
P1:   Hệ Thống Dữ Liệu PostgreSQL + pgvector
  │
  ▼
P1:   FastAPI Recognition Service & Quản Lý Danh Tính
  │
  ▼
P1:   Giao Thức Kết Nối Đa Thiết Bị (MQTT / WebSocket / REST)
  │
  ▼
P2:   Tối Ưu Hóa Edge Device (Raspberry Pi, Android, ESP32)
  │
  ▼
P2:   A/B Test So Sánh Mô Hình (AdaFace / MagFace vs ArcFace)
```

---

## 4. Chi Tiết Các Hạng Mục P0

### P0.1 — Benchmark ArcFace Baseline (Ưu tiên số 1)
- Xây dựng bộ test harness: `gallery/` và `probe/` với các biến thể (Frontal, Angle, Low-light, Blur, Occlusion, Distance).
- Trích xuất embedding 512D chuẩn hóa L2 với ArcFace ONNX.
- Tính toán ma trận khoảng cách Cosine Similarity giữa các cặp Genuine và Impostor.
- Vẽ biểu đồ phân bố mật độ và đường ROC.
- Xác định điểm hoạt động tối ưu:
  - Ngưỡng tương đồng $Threshold$ (ví dụ: $0.60 \div 0.65$) tương ứng với $FAR \le 0.01\%$.
  - Ngưỡng Margin tối thiểu $(S_{top1} - S_{top2}) \ge Margin$ (ví dụ: $0.08$).

### P0.2 — Quality Gate Module
- Bộ lọc loại bỏ ảnh xấu trước khi vào mạng trích xuất:
  - Kích thước khuôn mặt $< 60$ px $\rightarrow$ Bỏ qua.
  - Laplacian Variance $< 50$ (mờ do rung hoặc mất nét) $\rightarrow$ Bỏ qua.
  - $|Yaw| > 30^\circ$, $|Pitch| > 30^\circ$ $\rightarrow$ Bỏ qua.
  - Độ sáng trung bình $< 40$ hoặc $> 220$ $\rightarrow$ Bỏ qua.

### P0.3 — Temporal Voting & Tracking
- Duy trì ID khuôn mặt qua các frame liên tục bằng thuật toán ByteTrack.
- Áp dụng cơ chế bỏ phiếu thời gian qua cửa sổ 5 frame để lọc bỏ nhiễu đột biến từ một góc chụp xấu.

### P0.4 — Anti-Spoofing (Liveness)
- Tích hợp mô hình MiniFASNet / Silent-Face-Anti-Spoofing.
- Phân tích tần số Fourier và độ sâu quang học trên 2 tỷ lệ crop khuôn mặt.

---

## 5. Hạng Mục P1 & P2 (Triển Khai Sau Khi P0 Hoàn Tất)

- **P1 — Database**: PostgreSQL + extension `pgvector`, đánh chỉ mục HNSW với `vector_cosine_ops` để tìm kiếm top-k lân cận trong thời gian thực.
- **P1 — Backend API**: FastAPI async framework, cung cấp các endpoint enroll khuôn mặt, nhận diện stream/image, quản lý thiết bị và cấu hình ngưỡng động.
- **P1 — Giao tiếp ngoại vi**: ESP32 điều khiển relay/khóa cửa qua MQTT, Raspberry Pi truyền tải video qua RTSP/WebSocket.
- **P2 — Edge Optimization**: Chuyển đổi mô hình sang định dạng tối ưu trên chip ARM (ONNX Runtime, NCNN, TFLite).
- **P2 — A/B Testing**: Chạy tập benchmark P0 trên AdaFace và MagFace để đưa ra quyết định có nên thay thế ArcFace hay không dựa trên bằng chứng định lượng rõ ràng.
