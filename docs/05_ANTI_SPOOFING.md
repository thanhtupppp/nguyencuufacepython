# Thiết Kế Module Chống Giả Mạo (Anti-Spoofing & Liveness Detection)

> **Mục tiêu**: Ngăn chặn tuyệt đối các hình thức tấn công giả mạo khuôn mặt (Presentation Attacks) trước khi thực hiện so khớp danh tính.

---

## 1. Các Hình Thức Tấn Công Giả Mạo Phổ Biến

1. **Print Attack (Ảnh in)**: In màu hoặc đen trắng trên giấy phẳng, giấy ảnh, bìa cứng.
2. **Replay Attack (Màn hình điện tử)**: Phát video hoặc ảnh trên màn hình smartphone, máy tính bảng, laptop, TV.
3. **3D Mask Attack (Mặt nạ 3D)**: Mặt nạ silicon hoặc giấy bồi có hình dạng khối khuôn mặt.

---

## 2. Giải Pháp: MiniFASNet (Silent-Face-Anti-Spoofing)

### 2.1. Nguyên Lý Hoạt Động
- **Phân tích kết cấu bề mặt (Texture Analysis)**: Màn hình điện tử và ảnh in có hiện tượng phản xạ ánh sáng, sọc Moiré, và mật độ điểm ảnh không đồng nhất so với da người thật.
- **Phổ tần số Fourier (Fourier Spectrum)**: Màn hình và giấy in có các dải tần số cao đột biến do lưới pixel.
- **Độ sâu quang học giả định (Pseudo-Depth Map)**: Người thật có độ cong tự nhiên của sống mũi, má, hốc mắt; ảnh in phẳng có bản đồ độ sâu phẳng lì.
- **Multi-Scale Crop**: Đánh giá trên 2 tỷ lệ crop:
  - Crop 1: Cắt sát mặt để phân tích chi tiết lỗ chân lông và mắt.
  - Crop 2: Cắt rộng lấy thêm vùng cổ và phông nền xung quanh để phát hiện viền khung ảnh hoặc viền màn hình điện thoại.

### 2.2. Kiến Trúc & Biến Thể
- **MiniFASNetV1 / MiniFASNetV2**: Mạng nơ-ron tích chập siêu nhẹ dựa trên MobileNetV3 kết hợp cơ chế Squeeze-and-Excitation (SE).
- Kích thước model: $< 2$ MB, thời gian suy luận $< 10$ ms trên CPU thông thường.

---

## 3. Tiêu Chí Đánh Giá

- **TPR (True Positive Rate)**: Tỷ lệ phát hiện đúng người thật (Mục tiêu: $\ge 98\%$).
- **TNR (True Negative Rate)**: Tỷ lệ phát hiện và chặn đứng đối tượng giả mạo (Mục tiêu: $\ge 96\%$).
- **BAPC (Biometric Authentication Presentation Attack Confirmation)**: Đo lường theo tiêu chuẩn ISO/IEC 30107-3.
