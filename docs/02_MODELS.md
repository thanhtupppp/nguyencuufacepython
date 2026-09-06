# So Sánh Chi Tiết Các Mô Hình Nhận Diện & Phát Hiện Khuôn Mặt

> **Mục tiêu**: Phân tích sâu ưu/nhược điểm, cơ chế hoạt động và trường hợp sử dụng phù hợp của các mô hình trong bài toán thực tế.

---

## 1. Các Mô Hình Trích Xuất Đặc Trưng (Face Recognition Backbones)

| Mô hình | Hội nghị / Năm | Cơ chế Margin Loss | Ưu điểm nổi bật | Nhược điểm | Đánh giá áp dụng |
| :--- | :---: | :---: | :--- | :--- | :--- |
| **ArcFace** | CVPR 2019 | Additive Angular Margin ($m = 0.5$) | Độ phân biệt danh tính cực cao, phân cụm chặt chẽ trên hình cầu siêu chiều, ổn định nhất trong ngành. | Kém linh hoạt với ảnh chất lượng rất thấp, dễ overfit vào ảnh xấu. | **Baseline chính thức của hệ thống.** |
| **AdaFace** | CVPR 2022 | Adaptive Margin theo Feature Norm | Margin thích ứng theo chất lượng ảnh: ảnh tốt nhận margin lớn, ảnh xấu nhận margin nhỏ để tránh kéo lệch cụm. Vượt trội trên ảnh mờ, camera an ninh. | Cần điều chỉnh hàm gán margin phù hợp với tập dữ liệu. | **Ứng viên số 1 để A/B test ở Phase P2.** |
| **MagFace** | CVPR 2021 | Magnitude-Aware Margin | Độ lớn (magnitude) của vector đại diện cho chất lượng khuôn mặt; kéo các vector chất lượng cao vào gần tâm lớp. | Tốn nhiều bước tinh chỉnh siêu tham số hơn. | **Ứng viên số 2 để A/B test ở Phase P2.** |
| **MobileFaceNet** | 2018 | ArcFace / CosFace | Kích thước siêu nhỏ (< 4 MB), tốc độ cực nhanh trên CPU / ARM. | Độ chính xác giảm khi kích thước thư viện người dùng $> 1.000$ người. | **Tùy chọn tối ưu cho Raspberry Pi và Mobile.** |

---

## 2. Các Mô Hình Phát Hiện Khuôn Mặt (Face Detectors)

| Detector | Kích thước / FLOPs | Tốc độ | Khả năng phát hiện mặt nhỏ/nghiêng | Landmarks | Khuyến nghị |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **SCRFD-10G** | ~16 MB | Trung bình (GPU/PC) | Rất mạnh, bắt tốt khuôn mặt xa và góc quay lớn | 5 điểm chính xác cao | Dành cho Server GPU và PC mạnh. |
| **SCRFD-500M** | ~2.5 MB | Siêu nhanh (>60 FPS CPU) | Tốt ở khoảng cách vừa và gần | 5 điểm tốt | Dành cho Raspberry Pi, CPU laptop, Android. |
| **YuNet (OpenCV)** | ~350 KB | Cực nhanh | Bắt tốt khuôn mặt trung bình | 5 điểm | Dự phòng khi không có ONNX Runtime. |
| **RetinaFace** | ~28 MB | Chậm hơn SCRFD | Tốt nhưng tốn FLOPs | 5 điểm | Không ưu tiên do SCRFD vượt trội về hiệu năng/độ chính xác. |
