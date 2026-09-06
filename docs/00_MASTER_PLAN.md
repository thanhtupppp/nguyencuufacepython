# Kế Hoạch Tổng Thể Kiến Trúc & Triển Khai (Master Plan)

> **Dự án**: NguyenCuuFacePython  
> **Mục tiêu chính**: Xây dựng hệ thống nhận diện khuôn mặt chính xác cao, chống nhận nhầm người (anti false positive), hoạt động ổn định trong thực tế, không cần train lại model khi thêm người mới, và có khả năng mở rộng đa nền tảng (Server, PC, Raspberry Pi, Android, ESP32).

---

## 1. Triết Lý Thiết Kế Cốt Lõi

1. **Accuracy & Anti-False-Positive First**: Thà từ chối nhận diện một frame kém chất lượng (`Unknown` hoặc chờ frame kế tiếp) hơn là nhận nhầm danh tính người khác.
2. **Không khóa cứng vào một model (Model-Agnostic Interface)**: Toàn bộ engine nhận diện, trích xuất đặc trưng và detector được trừu tượng hóa bằng Interface chuẩn. Có thể tráo đổi model (ArcFace $\leftrightarrow$ AdaFace $\leftrightarrow$ MagFace) mà không ảnh hưởng tới Database hay REST API.
3. **Quyết định dựa trên chuỗi thời gian (Temporal Voting)**: Không bao giờ tin tưởng tuyệt đối vào 1 frame đơn lẻ trong luồng camera thực tế. Danh tính chỉ được xác nhận khi có sự đồng thuận qua một cửa sổ trượt nhiều frame liên tiếp.
4. **Quản lý phiên bản Embedding (Model Versioning)**: Mỗi vector đặc trưng 512D lưu trong database đều gắn liền với `model_version`. Tuyệt đối không tính similarity giữa các vector sinh ra từ hai model khác nhau.
5. **Độc lập nền tảng tính toán**: Phần logic lõi viết bằng Python chuẩn, hỗ trợ backend ONNX Runtime (CPU, CUDA, TensorRT) và dễ dàng chuyển đổi sang NCNN / TFLite cho các thiết bị nhúng.

---

## 2. Luồng Pipeline Xử Lý Chi Tiết (End-to-End Architecture)

```text
                                 [ CAMERA STREAM ]
                                (RTSP / USB / File)
                                         │
                                         ▼
                                ┌─────────────────┐
                                │  Face Detector  │ ◄── SCRFD / YuNet / RetinaFace
                                │ (Bbox + 5 Lmk)  │
                                └────────┬────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │  Face Tracking  │ ◄── ByteTrack / DeepSORT
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
                                │ Feature Extractor│ ◄── ArcFace / AdaFace / MagFace
                                │ (512D Embedding)│     Chuẩn hóa vector: L2 Norm = 1.0
                                └────────┬────────┘
                                         │
                                         ▼
                                ┌─────────────────┐
                                │   Vector DB     │ ◄── PostgreSQL 16+ kết hợp pgvector
                                │  (pgvector)     │     Index: HNSW với Cosine Distance
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

## 3. Đặc Tả Chi Tiết Từng Module

### 3.1. Face Detection & Landmark Extraction
- **Model chính**: **SCRFD (Sample and Computation Redistribution for Face Detection)**.
  - Tối ưu vượt trội về độ chính xác và tốc độ so với RetinaFace truyền thống.
  - Các biến thể phù hợp từng phần cứng:
    - `SCRFD-500M`: Phù hợp thiết bị CPU yếu, Raspberry Pi, cho tốc độ cao.
    - `SCRFD-2.5G`: Cân bằng giữa tốc độ và độ chính xác cho Server CPU / Mini PC.
    - `SCRFD-10G`: Độ chính xác tối đa trên Server GPU, bắt tốt khuôn mặt xa và nhỏ.
- **Output**:
  - Tọa độ Bounding Box: `[x1, y1, x2, y2]`, confidence score.
  - 5 Facial Landmarks: Mắt trái, mắt phải, đỉnh mũi, khóe miệng trái, khóe miệng phải `[(x1, y1), ..., (x5, y5)]`.

### 3.2. Face Tracking (Đa đối tượng qua video)
- Duy trì tracklet ID cho mỗi khuôn mặt xuất hiện trong khung hình bằng thuật toán **ByteTrack** hoặc **SORT**.
- Gom các frame liên tiếp của cùng một người vào một tập dữ liệu theo thời gian thực để thực hiện:
  - Chọn frame có chất lượng tốt nhất (Best-frame Selection).
  - Áp dụng Temporal Voting.

### 3.3. Quality Gate (Lọc trước khi nhận diện)
Loại bỏ sớm các khuôn mặt không đủ điều kiện để tiết kiệm tài nguyên tính toán và ngăn chặn nhận sai:
- **Kích thước khuôn mặt**: Loại bỏ nếu cạnh nhỏ hơn 60px.
- **Độ nét (Blur Detection)**: Tính phương sai toán tử Laplacian (Laplacian Variance). Nếu $Var < 50$ (mờ/chuyển động rung) $\rightarrow$ loại bỏ.
- **Ước lượng góc nghiêng (Head Pose Estimation)**: Dựa trên 5 điểm landmark, từ chối khuôn mặt nghiêng quá mức:
  - $|Yaw| > 30^\circ$ (quay trái/phải quá nhiều)
  - $|Pitch| > 30^\circ$ (ngửa/cúi quá nhiều)
  - $|Roll| > 20^\circ$ (nghiêng cổ)
- **Độ sáng (Illumination)**: Mean pixel intensity trong khoảng $[40, 220]$.

### 3.4. Anti-Spoofing (Chống giả mạo - Liveness Detection)
- **Model**: **MiniFASNet (Silent-Face-Anti-Spoofing)** chạy trên ONNX Runtime.
- **Phương thức**: Đánh giá kết cấu bề mặt (Fourier frequency spectrum) và phân tích chiều sâu giả định trên 2 thang crop (crop rộng lấy bối cảnh và crop sát mặt).
- **Phân loại**: Ngăn chặn tấn công bằng:
  - Ảnh in màu/đen trắng.
  - Video phát lại trên smartphone, tablet, laptop.
  - Mặt nạ giấy 2D.

### 3.5. Face Alignment (Căn chỉnh khuôn mặt chuẩn hóa)
- Sử dụng phép biến đổi tương đồng Affine (Similarity Transformation - Umeyama algorithm) dựa trên 5 điểm mốc landmarks cố định chuẩn thế giới (ArcFace 112x112 standard landmarks template).
- Đảm bảo mắt luôn nằm ở vị trí ngang cố định, tỷ lệ khuôn mặt đồng nhất trước khi đưa vào mạng trích xuất đặc trưng.

### 3.6. Face Recognition & Feature Extractor
- **Kiến trúc trừu tượng**:
  ```python
  from abc import ABC, abstractmethod
  import numpy as np

  class BaseFaceRecognizer(ABC):
      @property
      @abstractmethod
      def model_version(self) -> str:
          pass

      @property
      @abstractmethod
      def embedding_dim(self) -> int:
          pass

      @abstractmethod
      def extract_embedding(self, aligned_face_112: np.ndarray) -> np.ndarray:
          """
          Trích xuất vector đặc trưng 512 chiều, bắt buộc L2-normalized.
          """
          pass
  ```
- **Các ứng viên nghiên cứu & Benchmark**:
  - **ArcFace (ResNet-50 / ResNet-100)**: Chuẩn công nghiệp, margin loss mạnh mẽ, độ phân biệt danh tính cao.
  - **AdaFace (Adaptive Margin Loss)**: Tối ưu cho ảnh chất lượng thấp, mờ, thiếu sáng bằng cách gán margin theo chất lượng ảnh.
  - **MagFace**: Tự động mã hóa chất lượng ảnh vào độ dài vector (magnitude) trước khi chuẩn hóa.
  - **MobileFaceNet**: Bản siêu nhẹ cho Edge Device (Raspberry Pi / Mobile).

---

## 4. Thiết Kế Cơ Sở Dữ Liệu Vector (PostgreSQL + pgvector)

### 4.1. Bảng `persons` (Danh tính người dùng)
```sql
CREATE TABLE persons (
    person_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    department VARCHAR(128),
    status VARCHAR(32) DEFAULT 'active',
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

### 4.2. Bảng `face_embeddings` (Vector khuôn mặt đa góc chụp)
Mỗi người có thể được đăng ký từ 3 đến 10 ảnh ở nhiều góc độ và điều kiện ánh sáng khác nhau:
```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE face_embeddings (
    id BIGSERIAL PRIMARY KEY,
    person_id VARCHAR(64) REFERENCES persons(person_id) ON DELETE CASCADE,
    embedding vector(512) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    quality_score FLOAT DEFAULT 1.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Index HNSW tăng tốc độ tìm kiếm vector hàng triệu bản ghi với độ trễ dưới 2ms
CREATE INDEX idx_face_embeddings_hnsw 
ON face_embeddings 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

---

## 5. Chiến Lược Chống Nhận Nhầm (Anti-False-Match Strategy)

Nhận nhầm người (False Positive) là vấn đề nghiêm trọng nhất trong các hệ thống sinh trắc học. Dự án giải quyết bằng 3 lớp bảo vệ:

### 1. Phân tích phân phối và ngưỡng động (Dynamic Thresholding)
- Không cố định một con số đơn lẻ. Ngưỡng tương đồng cosine $T$ được điều chỉnh theo chất lượng khuôn mặt $Q$:
  $$T_{effective} = T_{base} + \alpha \times (1.0 - Q)$$
  Nếu khuôn mặt nhỏ hoặc ánh sáng kém, hệ thống tự động thắt chặt ngưỡng nhận diện để tránh nhận bừa.

### 2. Candidate Margin Check (Khoảng cách giữa Top 1 và Top 2)
- Khi thực hiện truy vấn vector:
  - Giả sử kết quả trả về: Ứng viên #1 có score $S_1$, Ứng viên #2 có score $S_2$.
  - Điều kiện chấp nhận danh tính:
    $$S_1 \ge T_{effective} \quad \text{VÀ} \quad (S_1 - S_2) \ge \text{Margin (ví dụ } 0.08\text{)}$$
  - Nếu $S_1$ cao nhưng $S_2$ cũng bám sát $S_1$ (chênh lệch nhỏ hơn Margin), có nguy cơ hai người này có nét mặt tương tự nhau hoặc embedding bị nhiễu $\rightarrow$ Phân loại là `AMBIGUOUS_MATCH` và không mở khóa.

### 3. Cửa sổ bỏ phiếu thời gian (Temporal Voting)
- Một chuỗi gồm $N = 5$ frames liên tiếp trong cùng một tracklet:
  - Nếu ít nhất $M = 3/5$ frames đồng thuận ra cùng một `person_id`, hệ thống mới chính thức xác nhận kết quả.

---

## 6. Kiến Trúc Đa Thiết Bị & Giao Tiếp Ngoại Vi

```text
                  ┌────────────────────────────────────────┐
                  │          TRUNG TÂM XỬ LÝ               │
                  │   FastAPI Server + pgvector + GPU/CPU  │
                  └──────────────────┬─────────────────────┘
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         │ (HTTP REST / WebSocket)   │ (WebSocket / RTSP)        │ (MQTT / HTTP)
         ▼                           ▼                           ▼
┌──────────────────┐       ┌──────────────────┐       ┌──────────────────┐
│ Ứng dụng Desktop │       │   Raspberry Pi   │       │   Thiết bị IoT   │
│ Web Admin UI     │       │   Camera Client  │       │     (ESP32)      │
│ Mobile (Flutter) │       │ (Edge Detection) │       │ Cảm biến / Relay │
└──────────────────┘       └──────────────────┘       └──────────────────┘
```

- **ESP32**: Đóng vai trò là module I/O vật lý (điều khiển khóa từ, chuông cửa, đèn báo, nút bấm, cảm biến hồng ngoại). Giao tiếp với FastAPI Backend qua MQTT hoặc REST API siêu nhẹ.
- **Raspberry Pi**: Có thể chạy client gửi RTSP/ảnh chụp về Server, hoặc tự chạy pipeline phát hiện khuôn mặt nhẹ với `SCRFD-500M` trên CPU trước khi gửi crop về Server trích xuất embedding.
- **PC / Server**: Nơi hội tụ sức mạnh xử lý trích xuất ArcFace/AdaFace và truy vấn vector database với độ trễ cực thấp.

---

## 7. Kế Hoạch Triển Khai Thực Hiện

| Giai đoạn | Nội dung trọng tâm | Đầu ra kiểm thử |
| :--- | :--- | :--- |
| **Phase 1: Research** | Benchmark ArcFace vs AdaFace trên tập dữ liệu suy giảm (blur, pose, light) | File báo cáo so sánh độ chính xác và chọn baseline model |
| **Phase 2: Baseline** | Hoàn thiện Engine lõi (SCRFD + Alignment + Feature Extractor + Vector Search) | Script test nhận diện ảnh tĩnh và video với độ trễ < 50ms/frame |
| **Phase 3: Accuracy** | Tích hợp Quality Gate, Anti-Spoofing, Margin Check và Temporal Voting | Tỷ lệ nhận nhầm (FAR) giảm về mức tiệm cận 0% |
| **Phase 4: Backend** | Xây dựng REST API FastAPI, quản trị người dùng, quản lý camera, WebSocket stream | Bộ API hoàn chỉnh có Swagger UI + Docker Compose |
| **Phase 5: Multi-Device**| Kết nối Raspberry Pi, ESP32 qua MQTT, client Android/Flutter | Hệ thống mẫu tương tác đóng/mở relay thành công |
| **Phase 6: Production** | Chạy thử nghiệm thực tế 24/7 trong nhiều điều kiện ánh sáng ngày/đêm | Hệ thống ổn định, tài nguyên không bị leak bộ nhớ |
