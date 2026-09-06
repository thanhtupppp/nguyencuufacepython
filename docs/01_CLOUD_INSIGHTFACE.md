# Nghiên Cứu InsightFace & Xu Hướng 2026 (InsightFace Server & Licensing)

> **Mục tiêu**: Nắm bắt các cập nhật công nghệ mới nhất của hệ sinh thái InsightFace (2026), phân tích giải pháp InsightFace Server và làm rõ các ràng buộc về bản quyền (licensing).

---

## 1. Bản Cập Nhật InsightFace 1.0 & InsightFace Server (2026)

### 1.1. InsightFace Server
- **Khái niệm**: Giải pháp server chuyên dụng được DeepInsight công bố nhằm phục vụ các hệ thống nhận diện khuôn mặt quy mô doanh nghiệp với lưu lượng truy cập lớn.
- **Tính năng nổi bật**:
  - Tích hợp sẵn REST API và gRPC phục vụ trích xuất đặc trưng và tìm kiếm vector.
  - Hỗ trợ tối ưu hóa suy luận bằng TensorRT và INT8 Quantization.
  - Khả năng tìm kiếm trên tập vector khổng lồ (> 50 triệu ảnh) tận dụng phần cứng GPU thế hệ mới (RTX 5090 / H100).
- **Đánh giá với dự án của chúng ta**:
  - Thích hợp cho các cụm máy chủ lớn tập trung.
  - Đối với bài toán của dự án (triển khai linh hoạt từ PC, Server cá nhân đến Raspberry Pi và IoT), việc tự xây dựng pipeline trên **ONNX Runtime** độc lập giúp ta hoàn toàn chủ động, nhẹ tải và không phụ thuộc vào hạ tầng server độc quyền.

### 1.2. Vấn Đề Bản Quyền (Licensing & Model Weights)
- **Mã nguồn InsightFace**: Thư viện InsightFace mã nguồn mở phát hành theo giấy phép MIT.
- **Model Weights Pretrained**:
  - Các gói weights phổ biến như `buffalo_l` (chứa `w600k_r50.onnx`, `det_10g.onnx`) được huấn luyện trên các tập dữ liệu học thuật quy mô lớn (MS1MV2, Glint360k, WebFace600K).
  - Các tập dữ liệu này có điều khoản giới hạn: **Chỉ phục vụ mục đích nghiên cứu phi thương mại (Non-Commercial Research Only)**.
- **Giải pháp cho dự án**:
  - Giai đoạn nghiên cứu & cá nhân: Sử dụng weights pretrained chuẩn để đo đạc baseline và benchmark.
  - Thiết kế kiến trúc trừu tượng `BaseFaceRecognizer`: Cho phép dễ dàng thay thế bằng weights tự train trên tập dữ liệu thương mại mở (hoặc weights từ AdaFace/ArcFace PyTorch mở) mà không cần sửa đổi bất kỳ dòng code pipeline hay database nào.
