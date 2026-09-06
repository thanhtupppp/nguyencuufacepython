# Kiến Trúc Triển Khai Thiết Bị Nhúng & Giao Thức Ngoại Vi (Edge Devices & IoT)

> **Mục tiêu**: Định nghĩa vai trò, cấu hình và giao thức kết nối cho Raspberry Pi, Android và ESP32.

---

## 1. Phân Lớp Thiết Bị (Hardware Tiers)

1. **PC / Server Trung Tâm**:
   - Chạy FastAPI Service, PostgreSQL + pgvector, và ArcFace R50.
   - Xử lý các luồng tính toán nặng nhất và quản trị cơ sở dữ liệu.
2. **Raspberry Pi (Edge Gateway / Camera Client)**:
   - Chạy camera module (CSI / USB).
   - Tùy chọn 1: Stream RTSP về Server.
   - Tùy chọn 2: Tự chạy SCRFD-500M trên CPU để phát hiện và crop khuôn mặt, chỉ gửi crop 112x112 về Server để giảm băng thông.
3. **Android (Mobile App / Tablet Kiosk)**:
   - Ứng dụng điểm danh di động hoặc màn hình kiosk gắn tường.
   - Kết nối với FastAPI Backend qua REST và WebSocket.
4. **ESP32 (IoT Peripheral Controller)**:
   - Điều khiển rơ-le (Relay) đóng/mở khóa điện tử từ xa.
   - Nhận tín hiệu cảm biến hồng ngoại (PIR) hoặc nút bấm chuông.

---

## 2. Giao Thức MQTT Cho ESP32

- **Broker**: Eclipse Mosquitto (port 1883).
- **Topics**:
  - `device/{device_id}/status`: ESP32 định kỳ gửi heartbeat (JSON: `{ "online": true, "ip": "...", "uptime": 1200 }`).
  - `device/{device_id}/cmd`: Server gửi lệnh điều khiển xuống ESP32 (JSON: `{ "command": "OPEN_DOOR", "duration_ms": 3000 }`).
  - `device/{device_id}/event`: ESP32 gửi sự kiện cảm biến lên Server (JSON: `{ "event": "DOOR_BELL_PRESSED" }`).
