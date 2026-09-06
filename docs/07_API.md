# Đặc Tả Kỹ Thuật Backend REST API & WebSocket (FastAPI)

> **Mục tiêu**: Cung cấp giao diện lập trình ứng dụng (API) cho web, ứng dụng di động và các camera/thiết bị ngoại vi.

---

## 1. Danh Sách Endpoint RESTful

| Phương thức | Endpoint | Mô tả | Đầu vào | Đầu ra |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/persons` | Tạo danh tính người mới | `{ name, department, metadata }` | `{ person_id, created_at }` |
| `GET` | `/api/v1/persons` | Danh sách người dùng | Query params (page, limit) | List persons |
| `DELETE`| `/api/v1/persons/{id}`| Xóa người và toàn bộ embeddings | `person_id` | Status OK |
| `POST` | `/api/v1/faces/enroll`| Đăng ký ảnh khuôn mặt mới | Multipart form (person_id, image) | Embedding ID, quality |
| `POST` | `/api/v1/faces/recognize`| Nhận diện 1:N từ ảnh/frame | Multipart form (image, camera_id) | `{ person_id, similarity, margin, status }` |
| `POST` | `/api/v1/faces/verify` | Xác thực 1:1 | Multipart form (image_1, image_2) | `{ is_same_person, similarity }` |
| `GET` | `/api/v1/devices` | Quản lý danh sách thiết bị | Query params | List devices |
| `POST` | `/api/v1/config` | Cập nhật ngưỡng động | `{ threshold, margin, quality_gate }`| Config updated |

---

## 2. WebSocket Realtime Streaming

- **Endpoint**: `/ws/v1/events`
- **Chức năng**:
  - Phát thông báo nhận diện tức thời (Realtime recognition alert).
  - Cập nhật trạng thái thiết bị online/offline.
  - Đẩy video stream có bounding box phục vụ hiển thị Dashboard Web.
