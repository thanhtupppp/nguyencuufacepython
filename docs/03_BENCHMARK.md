# Quy Trình & Tiêu Chuẩn Đánh Giá So Sánh (Benchmark Protocol)

> **Mục tiêu**: So sánh thực tế giữa **ArcFace** và **AdaFace** (và sau đó là MagFace, MobileFaceNet) để chọn ra backbone recognition tối ưu nhất cho bài toán nhận diện thực tế, đặc biệt là trong điều kiện khó (ánh sáng yếu, mờ, góc nghiêng).

---

## 1. Các Ứng Viên So Sánh

1. **ArcFace (Baseline)**:
   - Backbone: ResNet-50 / ResNet-100 (InsightFace pre-trained on Glint360k / MS1MV2).
   - Đặc điểm: Margin loss góc độ cố định $m = 0.5$, $s = 64$. Tiêu chuẩn vàng trong công nghiệp nhận diện.
2. **AdaFace (Adaptive Margin Face Recognition)**:
   - Backbone: IR-50 / IR-101.
   - Đặc điểm: Margin thích ứng dựa theo độ lớn gradient và chất lượng ảnh (ảnh kém chất lượng nhận margin thấp hơn để tránh overfitting). Rất mạnh trong ảnh mờ và ảnh camera giám sát.
3. **MagFace (Magnitude Face)**:
   - Backbone: ResNet-50.
   - Đặc điểm: Độ lớn (magnitude) của vector tỷ lệ thuận với chất lượng khuôn mặt, kết hợp cosine margin.

---

## 2. Tiêu Chí & Chỉ Số Đánh Giá (Metrics)

| Chỉ số | Định nghĩa | Mục tiêu |
| :--- | :--- | :--- |
| **FAR (False Accept Rate)** | Tỷ lệ hai người khác nhau bị nhận diện nhầm là cùng một người | Càng thấp càng tốt (Mục tiêu: $< 0.01\%$ tại ngưỡng chuẩn) |
| **FRR (False Reject Rate)** | Tỷ lệ cùng một người nhưng bị hệ thống từ chối nhận diện | Càng thấp càng tốt ($< 1\%$) |
| **EER (Equal Error Rate)** | Điểm cân bằng nơi $FAR = FRR$ | Càng thấp càng tốt |
| **Similarity Margin** | Khoảng cách trung bình giữa điểm số Top 1 và Top 2 trên tập kiểm thử | Càng lớn càng an toàn (chống nhận nhầm) |
| **Inference Latency** | Thời gian trích xuất embedding 1 khuôn mặt (ms) | $< 15$ ms trên GPU, $< 60$ ms trên CPU |
| **Memory Footprint** | Dung lượng RAM/VRAM khi load model ONNX | $< 500$ MB VRAM |

---

## 3. Các Điều Kiện Kiểm Thử Suy Giảm Thực Tế (Degradation Scenarios)

1. **Khuôn mặt chuẩn (Standard)**: Ảnh chụp thẳng, rõ nét, đủ sáng, kích thước lớn hơn 150x150 px.
2. **Mờ do chuyển động & Out-of-focus (Blurry)**: Áp dụng bộ lọc Gaussian Blur / Motion Blur ($\sigma \in [2, 5]$).
3. **Thiếu sáng & Ngược sáng (Low-light & Backlit)**: Giảm gamma ($\gamma = 0.3 \div 0.6$) hoặc tạo vùng tương phản tối.
4. **Góc nghiêng lớn (Extreme Pose)**: Góc nghiêng Yaw từ $25^\circ$ đến $45^\circ$, Pitch từ $20^\circ$ đến $35^\circ$.
5. **Khuôn mặt kích thước nhỏ (Low Resolution)**: Downscale xuống $30\times 30$ rồi upscale lại $112\times 112$.
6. **Che một phần khuôn mặt (Partial Occlusion)**: Che khẩu trang hoặc kính râm.

---

## 4. Kế Hoạch Triển Khai Script Benchmark

1. Chuẩn bị file script `benchmarks/benchmark_recognizers.py`:
   - Tải weights ONNX chuẩn của ArcFace R50 và AdaFace IR50.
   - Chạy inference trên tập ảnh cặp (Pairs test: Positive pairs và Negative pairs).
   - Xuất biểu đồ phân phối Cosine Similarity:
     - Biểu đồ đường cong phân phối cùng người (Positive).
     - Biểu đồ đường cong phân phối khác người (Negative).
     - Đường ROC (Receiver Operating Characteristic) curve và AUC.
2. Tổng hợp bảng xếp hạng và chọn model làm mặc định cho Phase 2.
