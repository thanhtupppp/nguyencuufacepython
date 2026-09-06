# In Progress Tasks

Nhiệm vụ đang thực hiện:

## 🚀 P1.2: FastAPI Recognition Backend
- [x] Hoàn thành 100% Phase P0 (Pipeline cốt lõi, Quality Gate, Tracking/Voting, Anti-Spoofing).
- [x] Hoàn thành P1.1 (Docker Compose PostgreSQL 16 pgvector, DDL SQL, DatabaseClient, 25/25 tests PASSED).
- [ ] Xây dựng app FastAPI (`src/api/main.py` và `src/api/routes/`).
- [ ] Endpoints `/api/v1/persons` (CRUD danh tính).
- [ ] Endpoint `/api/v1/faces/enroll` (Multi-part upload ảnh, trích xuất embedding 512D, lưu DB).
- [ ] Endpoint `/api/v1/faces/recognize` (Nhận diện 1:N với Dual-Threshold Score + Margin).
- [ ] Endpoint `/api/v1/faces/verify` (Xác thực 1:1).
- [ ] WebSocket `/ws/v1/events` stream kết quả realtime.
- [ ] Viết unit tests API (`tests/test_api.py`).
