# Quy Trình & Tiêu Chuẩn Benchmark Recognition Nội Bộ (P0)

> **Mục tiêu**: Xây dựng bộ benchmark nhận diện khuôn mặt nội bộ để kiểm chứng thực tế và định lượng chính xác hiệu năng của pipeline baseline (**SCRFD + 5-point alignment + ArcFace**) trước khi tiến hành xây dựng database hay API.
> 
> **Nguyên tắc**: Thà từ chối nhận diện còn hơn nhận nhầm người. Không chọn threshold tùy ý (`if score > 0.5`), mọi ngưỡng và margin phải được xác định từ phân bố điểm số thực tế.

---

## 1. Pipeline Baseline (P0)

```text
camera / image
    │
    ▼
  SCRFD (Face Detection & 5 Landmarks)
    │
    ▼
5-Point Alignment (Similarity Transform 112x112)
    │
    ▼
ArcFace Baseline (Backbone ResNet-50 / buffalo_l)
    │
    ▼
L2-Normalized Embedding (512D)
    │
    ▼
Cosine Similarity Search
    │
    ▼
Dual-Threshold Decision Engine
(Score >= Threshold VÀ Margin >= Top1 - Top2)
    │
    ▼
Same / Different Person (Match / Unknown / Ambiguous)
```

---

## 2. Cấu Trúc Thư Mục Bộ Benchmark (`benchmarks/`)

```text
benchmarks/
├── gallery/                     # Ảnh chuẩn làm mốc nhận diện của từng người
│   ├── person_001/              # 1-3 ảnh chân dung chuẩn, rõ nét
│   ├── person_002/
│   └── ...
│
├── probe/                       # Ảnh kiểm thử theo các kịch bản suy giảm chất lượng
│   ├── frontal/                 # Ảnh chụp thẳng, điều kiện chuẩn
│   ├── angle/                   # Góc quay mặt (Yaw/Pitch/Roll: 15° - 45°)
│   ├── low_light/               # Thiếu sáng, ngược sáng, tương phản kém
│   ├── blur/                    # Mờ do chuyển động hoặc out-of-focus
│   ├── partial_occlusion/       # Che khẩu trang, kính râm, tay che
│   └── distance/                # Khoảng cách xa, khuôn mặt nhỏ (< 50x50 px)
│
├── pairs/                       # Danh sách cặp kiểm thử định danh
│   ├── genuine.csv              # Cặp ảnh của CÙNG một người (Positive pairs)
│   └── impostor.csv             # Cặp ảnh của HAI NGƯỜI KHÁC NHAU (Negative pairs)
│
├── scripts/                     # Mã nguồn chạy đánh giá
│   ├── generate_pairs.py        # Tự động sinh file pairs từ gallery và probe
│   ├── metrics.py               # Thư viện tính toán FAR, FRR, EER, ROC, Margin
│   └── run_benchmark.py         # Runner trích xuất vector, tính toán và xuất báo cáo
│
└── results/                     # Kết quả đo lường thực tế
    ├── similarity_distribution.csv # Bảng phân bố similarity genuine vs impostor
    ├── roc.csv                  # Dữ liệu đường cong ROC (FPR vs TPR)
    └── threshold_report.md      # Báo cáo tổng kết và đề xuất ngưỡng vận hành
```

---

## 3. Phương Pháp Kiểm Chứng & Chỉ Số Đo Lường

### 3.1. Phân Bố Độ Tương Đồng (Similarity Distribution)

So sánh trực tiếp hai tập phân bố cosine similarity:
- **Genuine Distribution**: Tập hợp điểm tương đồng giữa các ảnh của **cùng một người**.
- **Impostor Distribution**: Tập hợp điểm tương đồng giữa các ảnh của **hai người khác nhau**.

```text
      Impostor Density                Genuine Density
          ┌───────┐                      ┌───────┐
          │       │                      │       │
          │       │                      │       │
          │       │                      │       │
      ────┴───────┴──────────────┬───────┴───────┴────► Cosine Similarity
                                 │
                         Optimal Threshold
```

### 3.2. Các Chỉ Số Cốt Lõi (Core Metrics)

1. **FAR (False Accept Rate - Tỷ lệ nhận nhầm)**:
   $$FAR(T) = \frac{\text{Số cặp Impostor có } similarity \ge T}{\text{Tổng số cặp Impostor}}$$
   *Mục tiêu*: FAR $\le 0.01\%$ (hoặc $\le 0.001\%$ cho bảo mật cao).

2. **FRR (False Reject Rate - Tỷ lệ từ chối sai người thật)**:
   $$FRR(T) = \frac{\text{Số cặp Genuine có } similarity < T}{\text{Tổng số cặp Genuine}}$$
   *Mục tiêu*: Càng thấp càng tốt ($< 1\%$).

3. **TAR @ FAR (True Accept Rate tại mức FAR cố định)**:
   $$TAR = 1 - FRR$$
   Đo TAR tại các mốc chuẩn: $TAR @ FAR = 10^{-2}$, $TAR @ FAR = 10^{-3}$, $TAR @ FAR = 10^{-4}$.

4. **EER (Equal Error Rate)**:
   Điểm giao cắt mà tại đó $FAR(T) = FRR(T)$. EER càng nhỏ thì khả năng phân biệt của model càng cao.

5. **Top-1 Accuracy & Top-2 Margin Check**:
   Khi truy vấn gallery, ứng viên tốt nhất (Top 1) có điểm $S_1$ và ứng viên thứ hai (Top 2) có điểm $S_2$:
   - **Quy tắc quyết định 2 lớp (Dual-Threshold Rule)**:
     ```python
     is_matched = (S1 >= threshold) and ((S1 - S2) >= margin)
     ```
   - Nếu $S_1 \ge threshold$ nhưng $(S_1 - S_2) < margin$: Đưa vào diện `AMBIGUOUS_MATCH` (nghi ngờ nhận nhầm do hai người có nét tương đồng).

---

## 4. Tiêu Chí Hoàn Thành Nghiên Cứu Baseline (P0 Sign-off Table)

P0 chỉ được nghiệm thu khi bảng kết quả thực tế sau đây được lấp đầy bằng số liệu kiểm chứng:

| Điều kiện Kiểm Thử | Số lượng mẫu | Top-1 Accuracy (%) | Mean Similarity (Genuine) | Mean Similarity (Impostor) |
| :--- | :---: | :---: | :---: | :---: |
| **Frontal (Chuẩn)** | *Đo thực tế* | *Đo thực tế* | *Đo thực tế* | *Đo thực tế* |
| **Góc mặt (Angle)** | *Đo thực tế* | *Đo thực tế* | *Đo thực tế* | *Đo thực tế* |
| **Thiếu sáng (Low-light)** | *Đo thực tế* | *Đo thực tế* | *Đo thực tế* | *Đo thực tế* |
| **Mờ (Blur)** | *Đo thực tế* | *Đo thực tế* | *Đo thực tế* | *Đo thực tế* |
| **Che khuất (Occlusion)** | *Đo thực tế* | *Đo thực tế* | *Đo thực tế* | *Đo thực tế* |
| **Khoảng cách xa (Distance)** | *Đo thực tế* | *Đo thực tế* | *Đo thực tế* | *Đo thực tế* |

### Thông Số Quyết Định Tối Ưu Được Xác Lập:
- **FAR @ Threshold**: *Xác định qua phân bố*
- **FRR @ Threshold**: *Xác định qua phân bố*
- **EER**: *Xác định qua ROC*
- **Ngưỡng nhận diện tối ưu ($T_{optimal}$)**: *Ví dụ: 0.62*
- **Khoảng cách tối thiểu ($Margin_{optimal}$)**: *Ví dụ: 0.08*

---

## 5. Lưu Ý Về InsightFace & Licensing (2026)

- Bản cập nhật InsightFace 1.0 và InsightFace Server (2026) giới thiệu các cải tiến pipeline và tối ưu tốc độ.
- Các model pretrained sẵn như `buffalo_l` (chứa SCRFD + ArcFace R50) có giấy phép phi thương mại (Non-Commercial / Research License).
- Trong kiến trúc hệ thống, code bọc ONNX Runtime được xây dựng độc lập để có thể nạp weights mã nguồn mở hoặc weights tự train mà không phụ thuộc trực tiếp vào package độc quyền của InsightFace.
