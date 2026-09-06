# Đặc Tả Kỹ Thuật Pipeline Nhận Diện Khuôn Mặt (Pipeline Specification)

> **Mục tiêu**: Định nghĩa chuẩn hóa các khâu xử lý trong luồng nhận diện từ Video Frame đến Kết quả định danh cuối cùng.

---

## 1. Sơ Đồ Pipeline Đa Tầng

```text
Input Frame
   │
   ▼
[ Stage 1: Detection ] ────────► SCRFD (Trả về Bounding Box & 5 Landmarks)
   │
   ▼
[ Stage 2: Tracking ] ─────────► ByteTrack (Gán và duy trì Tracklet ID)
   │
   ▼
[ Stage 3: Quality Gate ] ─────► Lọc: Size >= 60px, Blur >= 50, Pose <= 30°, Light [40, 220]
   │ (Pass)
   ▼
[ Stage 4: Anti-Spoofing ] ────► MiniFASNet Liveness Test (Loại bỏ ảnh in/màn hình)
   │ (Live)
   ▼
[ Stage 5: Alignment ] ────────► Umeyama 5-Landmarks Affine Transform -> 112x112 px
   │
   ▼
[ Stage 6: Feature Extraction ] ► ArcFace ONNX -> 512D Vector (L2-Normalized)
   │
   ▼
[ Stage 7: Vector Search ] ────► pgvector Cosine Distance Query (Top-2 candidates)
   │
   ▼
[ Stage 8: Dual-Threshold ] ───► Best_Score >= Threshold VÀ (Best_Score - Second_Score) >= Margin
   │ (Passed)
   ▼
[ Stage 9: Temporal Voting ] ──► Bầu chọn cửa sổ 5 frame liên tiếp (Đồng thuận >= 60%)
   │
   ▼
Final Output: person_id, confidence, name
```

---

## 2. Chi Tiết Từng Khâu Xử Lý

### Khâu 1: Face Detection (SCRFD)
- Resize ảnh đầu vào giữ nguyên tỷ lệ khung hình, pad về kích thước bội số 32 (ví dụ $640 \times 640$).
- Trả về danh sách khuôn mặt với tọa độ `[x1, y1, x2, y2]`, điểm tin cậy `score`, và 5 điểm mốc `landmarks`: mắt trái, mắt phải, mũi, khóe miệng trái, khóe miệng phải.

### Khâu 2: Quality Gate
- **Kích thước khuôn mặt**: Loại bỏ nếu $\min(\text{width}, \text{height}) < 60\text{ px}$.
- **Độ nét (Laplacian Variance)**:
  $$\text{Blur Score} = \text{Var}(\nabla^2 I_{gray})$$
  Nếu $\text{Blur Score} < 50.0 \rightarrow$ Loại bỏ (ảnh bị mờ/chuyển động rung).
- **Góc nghiêng đầu (Head Pose Estimation)**: Ước lượng Yaw, Pitch, Roll dựa trên tỷ lệ khoảng cách giữa 5 điểm landmarks. Nếu $|Yaw| > 30^\circ$ hoặc $|Pitch| > 30^\circ \rightarrow$ Bỏ qua.
- **Độ sáng (Illumination)**: Mean pixel intensity trong khoảng $[40, 220]$.

### Khâu 3: Anti-Spoofing (MiniFASNet)
- Đưa khuôn mặt qua mô hình phân loại Liveness.
- Nếu xác suất $\text{Score}_{live} < 0.85 \rightarrow$ Báo động giả mạo và từ chối nhận diện.

### Khâu 4: Alignment (Umeyama Transform)
- Áp dụng thuật toán bình phương tối thiểu Umeyama ánh xạ 5 điểm phát hiện được vào bộ 5 tọa độ chuẩn của ArcFace:
  - Mắt trái: `(38.2946, 51.6963)`
  - Mắt phải: `(73.5318, 51.5014)`
  - Mũi: `(56.0252, 71.7366)`
  - Miệng trái: `(41.5493, 92.3655)`
  - Miệng phải: `(70.7299, 92.2041)`
- Xuất ra ảnh khuôn mặt đã căn chỉnh kích thước cố định $112 \times 112 \times 3$.

### Khâu 5: Feature Extraction (ArcFace)
- Tiền xử lý: Chuyển BGR sang RGB, chuẩn hóa `(img - 127.5) / 127.5`.
- Chạy qua backbone ArcFace sinh vector 512 chiều.
- Chuẩn hóa L2: $v = \frac{v}{\|v\|_2}$.

### Khâu 6: Phán Quyết Kép (Dual-Threshold Rule)
- Truy vấn lấy Top 1 ($S_1$) và Top 2 ($S_2$).
- Áp dụng quy tắc:
  ```python
  if S1 < threshold:
      result = "UNKNOWN"
  elif (S1 - S2) < margin:
      result = "AMBIGUOUS_MATCH"
  else:
      result = top1_person_id
  ```

### Khâu 7: Bầu Chọn Chuỗi Thời Gian (Temporal Voting)
- Lưu trữ kết quả nhận diện của cùng một Tracklet ID trong cửa sổ trượt $N = 5$ frames.
- Tính tần suất xuất hiện của nhãn dẫn đầu:
  $$\text{Tỷ lệ đồng thuận} = \frac{\text{Count}(\text{person\_id})}{N}$$
- Nếu $\text{Tỷ lệ đồng thuận} \ge 60\% \rightarrow$ Xác nhận mở khóa / ghi nhận danh tính.
