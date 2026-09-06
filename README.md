# NguyenCuuFacePython

> **Nhận diện khuôn mặt chính xác bằng Python**

Dự án nghiên cứu và xây dựng hệ thống Face Detection + Face Recognition có độ chính xác cao, khả năng chống nhận nhầm, hoạt động ổn định trong nhiều điều kiện thực tế và có thể mở rộng cho nhiều thiết bị.

Dự án ưu tiên độ chính xác, độ ổn định và khả năng mở rộng, không chỉ dừng ở việc nhận diện khuôn mặt trong một frame.

---

## 🎯 Mục tiêu

Xây dựng hệ thống có khả năng:

- Phát hiện khuôn mặt chính xác.
- Nhận diện mỗi người bằng một `person_id` duy nhất.
- Không cần train lại model khi thêm người mới.
- Nhận diện ổn định qua nhiều frame.
- Giảm false positive / nhận nhầm người.
- Xử lý khuôn mặt nghiêng, xa, nhỏ, thiếu sáng hoặc chất lượng thấp.
- Phát hiện ảnh/video giả mạo bằng Anti-Spoofing.
- Hỗ trợ nhiều camera.
- Hỗ trợ nhiều thiết bị.
- Có API để các ứng dụng khác sử dụng.
- Có thể triển khai trên PC, Server, Raspberry Pi và Android.
- ESP32 có thể đóng vai trò thiết bị ngoại vi/giao tiếp với hệ thống.

---

## 🧠 Kiến trúc dự kiến

```text
                    CAMERA
                       │
                       ▼
                ┌─────────────┐
                │    SCRFD    │
                │ Face Detect │
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
                │ Quality Gate│
                │ Blur / Size │
                │ Pose / Light│
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
                │ Anti-Spoof  │
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
                │  Alignment  │
                │ 5 Landmarks │
                │    112x112  │
                └──────┬──────┘
                       │
                       ▼
                ┌─────────────┐
                │ ArcFace /   │
                │ InsightFace │
                └──────┬──────┘
                       │
                       ▼
                 512D EMBEDDING
                       │
                       ▼
                ┌─────────────┐
                │ Vector DB   │
                │  pgvector   │
                └──────┬──────┘
                       │
                       ▼
              Similarity + Margin
                       │
                       ▼
                  person_id
                       │
                       ▼
                Temporal Voting
                       │
                       ▼
                 FINAL RESULT
```

---

## 🔬 Công nghệ đang nghiên cứu

### Face Detection

Ưu tiên nghiên cứu:

- **SCRFD**
- **RetinaFace**
- **YuNet**
- Các detector mới có độ chính xác/hiệu năng tốt hơn

### Face Recognition

Baseline:

- **InsightFace**
- **ArcFace**

Các model sẽ được benchmark thêm:

- **AdaFace**
- **MagFace**
- **MobileFaceNet**
- Các model recognition mới nếu có ưu thế thực tế

### Face Alignment

Sử dụng facial landmarks để chuẩn hóa khuôn mặt trước khi tạo embedding.

Mục tiêu:

```text
Face
 ↓
5 Landmarks
 ↓
Alignment
 ↓
112 x 112
 ↓
Recognition
```

### Quality Assessment

Không đưa mọi khuôn mặt vào recognition.

Kiểm tra:

- Kích thước khuôn mặt
- Độ nét
- Độ sáng
- Pose
- Blur
- Occlusion
- Góc nghiêng
- Chất lượng ảnh

---

## 🛡️ Anti-Spoofing

Hệ thống cần phân biệt:

```text
Người thật
   vs
Ảnh
   vs
Video
   vs
Màn hình điện thoại
   vs
Các hình thức giả mạo khác
```

Các phương pháp/model Anti-Spoofing sẽ được nghiên cứu và benchmark riêng.

---

## 👤 Person ID

Hệ thống không train lại model cho từng người.

Ví dụ:

```text
Người A
 ├── Image 1
 ├── Image 2
 ├── Image 3
 └── Image 4
        │
        ▼
    Embedding
        │
        ▼
   person_id = A001
```

Khi có người mới:

```text
Người B
    ↓
Embedding
    ↓
person_id = B001
```

Không cần train lại ArcFace.

---

## 📊 Recognition Pipeline

Hệ thống không quyết định danh tính chỉ dựa trên một frame.

Dự kiến:

```text
Frame 1 → Embedding → Candidate A
Frame 2 → Embedding → Candidate A
Frame 3 → Embedding → Candidate A
Frame 4 → Embedding → Candidate A
Frame 5 → Embedding → Candidate A

              ↓

       Temporal Voting

              ↓

        person_id = A001
```

Điều này giúp giảm nhận nhầm do:

- Frame bị blur
- Ánh sáng thay đổi
- Detector không ổn định
- Pose thay đổi
- Embedding không ổn định trong một frame.

---

## 📐 Threshold và False Match

Không sử dụng một threshold tùy ý.

Cần benchmark:

```text
Same Person
     ↓
Similarity Distribution

Different Person
     ↓
Similarity Distribution
```

Sau đó xác định:

- Recognition threshold
- False Accept Rate (FAR)
- False Reject Rate (FRR)
- Equal Error Rate (EER)
- Candidate margin
- Quality-dependent threshold

Ngoài similarity cao nhất, hệ thống sẽ xem xét khoảng cách giữa:

```text
Best Candidate
        vs
Second Best Candidate
```

để giảm trường hợp hai người có embedding gần nhau.

---

## 🗄️ Vector Database

Dự kiến sử dụng: **PostgreSQL + pgvector**

Ví dụ schema:

```text
persons
├── person_id
├── name
├── status
└── metadata

face_embeddings
├── id
├── person_id
├── embedding
├── quality
├── created_at
└── model_version
```

Mỗi người có thể có nhiều embedding.

---

## 🌐 Backend API

Dự kiến:

```text
Flutter / Web / Device
          │
          ▼
       FastAPI
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
 Users Recognition Devices
          │
          ▼
     Face Engine
```

API sẽ phụ trách:

- Đăng ký người
- Cập nhật người
- Xóa người
- Enrollment khuôn mặt
- Recognition
- Verification
- Quản lý thiết bị
- Quản lý camera
- Logging
- Authentication
- Cấu hình threshold

---

## 📱 Multi Device

Kiến trúc được thiết kế để không phụ thuộc vào một loại chip duy nhất.

Có thể mở rộng:

```text
                 Backend
                    │
       ┌────────────┼────────────┐
       │            │            │
       ▼            ▼            ▼
      PC       Raspberry Pi    Android
       │
       ▼
     ESP32
```

ESP32 có thể đảm nhiệm:

- Cảm biến
- Relay
- Khóa
- Motor
- Nút nhấn
- Trạng thái thiết bị
- Giao tiếp MQTT
- HTTP
- WebSocket

Face Recognition có thể chạy trên:

- PC
- Server
- Raspberry Pi
- Android
- Hoặc edge device phù hợp trong tương lai.

---

## 📡 Device Communication

Các giao thức đang xem xét:

### MQTT
Phù hợp cho:
- ESP32
- Raspberry Pi
- Nhiều thiết bị
- Trạng thái online/offline
- Command/event

### HTTP REST
Phù hợp cho:
- API
- Quản lý người
- Enrollment
- Recognition request
- Configuration

### WebSocket
Phù hợp cho:
- Realtime recognition
- Camera events
- Device status
- Realtime UI

---

## 🧪 Benchmark

Mỗi model phải được kiểm tra bằng cùng một pipeline.

Ví dụ:

```text
Detector
   ↓
Alignment
   ↓
Recognition Model
   ↓
Embedding
   ↓
Vector Search
   ↓
Threshold
```

So sánh:

```text
ArcFace
   vs
AdaFace
   vs
MagFace
   vs
MobileFaceNet
```

Các điều kiện:

- Ánh sáng tốt
- Thiếu sáng
- Khuôn mặt nghiêng
- Khuôn mặt nhỏ
- Blur
- Che một phần khuôn mặt
- Khoảng cách xa
- Nhiều người trong frame
- Camera khác nhau.

> **Lưu ý**: Không thay đổi kiến trúc chỉ dựa vào benchmark của tác giả model; cần benchmark trên dữ liệu và điều kiện thực tế của dự án.

---

## 📁 Project Structure

```text
nguyencuufacepython/
│
├── README.md
│
├── docs/
│   ├── 00_MASTER_PLAN.md
│   ├── 01_GITHUB_RESEARCH.md
│   ├── 02_MODELS.md
│   ├── 03_BENCHMARK.md
│   ├── 04_FACE_ENGINE.md
│   ├── 05_ANTI_SPOOFING.md
│   ├── 06_DATABASE.md
│   ├── 07_API.md
│   └── 08_EDGE_DEVICE.md
│
├── tasks/
│   ├── TODO.md
│   ├── IN_PROGRESS.md
│   └── DONE.md
│
├── src/
│   ├── detection/
│   ├── alignment/
│   ├── recognition/
│   ├── quality/
│   ├── anti_spoofing/
│   ├── tracking/
│   ├── database/
│   └── api/
│
├── tests/
│
├── benchmarks/
│
├── configs/
│
└── scripts/
```

---

## 🗺️ Roadmap

### Phase 1 — Research
- [ ] Nghiên cứu InsightFace
- [ ] Nghiên cứu SCRFD
- [ ] Nghiên cứu ArcFace
- [ ] Nghiên cứu AdaFace
- [ ] Nghiên cứu MagFace
- [ ] Nghiên cứu Anti-Spoofing
- [ ] Nghiên cứu tracking
- [ ] Nghiên cứu vector database
- [ ] Nghiên cứu multi-device architecture

### Phase 2 — Baseline
- [ ] SCRFD detector
- [ ] Face alignment
- [ ] ArcFace recognition
- [ ] Embedding extraction
- [ ] Person enrollment
- [ ] Vector search
- [ ] Threshold
- [ ] Temporal voting

### Phase 3 — Accuracy
- [ ] Quality gate
- [ ] Anti-spoofing
- [ ] Dynamic threshold
- [ ] Best-frame selection
- [ ] Multi-frame recognition
- [ ] Tracking
- [ ] False-match benchmark
- [ ] Model benchmark

### Phase 4 — Backend
- [ ] FastAPI
- [ ] PostgreSQL
- [ ] pgvector
- [ ] Authentication
- [ ] Device management
- [ ] Recognition API

### Phase 5 — Device
- [ ] PC
- [ ] Raspberry Pi
- [ ] Android
- [ ] ESP32
- [ ] MQTT
- [ ] WebSocket
- [ ] Multi-camera

### Phase 6 — Production Test
- [ ] Long-running test
- [ ] Multi-camera test
- [ ] Multi-person test
- [ ] Low-light test
- [ ] Anti-spoofing test
- [ ] False recognition test
- [ ] Performance benchmark
- [ ] CPU/GPU/RAM benchmark

---

## ⚙️ Nguyên tắc phát triển

### 1. Accuracy First

Ưu tiên:

```text
Accuracy
   ↓
Reliability
   ↓
Anti False Match
   ↓
Performance
   ↓
Resource Optimization
```

### 2. Không khóa kiến trúc vào một model

Recognition engine phải có interface để có thể thay đổi:

```text
ArcFace
   │
AdaFace
   │
MagFace
   │
Model khác
```

mà không phải viết lại database hoặc API.

### 3. Mọi thay đổi quan trọng phải benchmark

Không thay đổi model chỉ vì model mới có benchmark tốt trên paper.

### 4. Có version cho embedding

Ví dụ:

```python
model_version = "arcface_v1"
```

Khi thay model:

```python
model_version = "adaface_v1"
```

Điều này giúp tránh trộn embedding của các model khác nhau.

---

## 🔬 Trạng thái hiện tại

**Status**: Research / Development

Baseline dự kiến:

```text
SCRFD
  ↓
Quality Gate
  ↓
Anti-Spoofing
  ↓
Alignment
  ↓
ArcFace / InsightFace
  ↓
Embedding
  ↓
PostgreSQL + pgvector
  ↓
Similarity + Margin
  ↓
Temporal Voting
  ↓
person_id
```

Kiến trúc sẽ được điều chỉnh sau khi benchmark thực tế.

---

## 📚 Research Policy

Dự án sẽ liên tục nghiên cứu:

- GitHub
- Papers
- Model mới
- Face Detection
- Face Recognition
- Face Alignment
- Face Quality
- Anti-Spoofing
- Tracking
- Vector Search
- Edge AI
- Multi-camera
- Multi-device

Mỗi công nghệ mới cần được đánh giá theo:

- Accuracy
- Reliability
- Latency
- Memory
- CPU/GPU
- Dataset performance
- Real-world performance
- Integration complexity

---

## 📄 License

Dự án phục vụ mục đích nghiên cứu và sử dụng cá nhân.

Các model/thư viện bên thứ ba được sử dụng phải tuân theo license riêng của từng dự án.
