# Thiết Kế Cơ Sở Dữ Liệu Vector (PostgreSQL + pgvector)

> **Mục tiêu**: Lưu trữ danh tính người dùng và vector embedding 512 chiều, đảm bảo tính toàn vẹn dữ liệu (ACID) và khả năng tìm kiếm vector lân cận kNN dưới 5ms.

---

## 1. Schema Thiết Kế Chi Tiết

```sql
-- Kích hoạt extension pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Bảng quản lý danh tính cá nhân
CREATE TABLE IF NOT EXISTS persons (
    person_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    department VARCHAR(128),
    role VARCHAR(64) DEFAULT 'user',
    status VARCHAR(32) DEFAULT 'active',
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Bảng lưu trữ vector embedding khuôn mặt
CREATE TABLE IF NOT EXISTS face_embeddings (
    id BIGSERIAL PRIMARY KEY,
    person_id VARCHAR(64) NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    embedding vector(512) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    quality_score FLOAT DEFAULT 1.0,
    source_image_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Bảng quản lý camera và thiết bị ngoại vi
CREATE TABLE IF NOT EXISTS devices (
    device_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    device_type VARCHAR(64) NOT NULL, -- 'esp32_relay', 'rpi_camera', 'rtsp_stream'
    location VARCHAR(255),
    ip_address VARCHAR(45),
    status VARCHAR(32) DEFAULT 'online',
    last_ping TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Bảng nhật ký điểm danh / mở khóa (Access Logs)
CREATE TABLE IF NOT EXISTS access_logs (
    id BIGSERIAL PRIMARY KEY,
    person_id VARCHAR(64) REFERENCES persons(person_id) ON DELETE SET NULL,
    device_id VARCHAR(64) REFERENCES devices(device_id) ON DELETE SET NULL,
    similarity FLOAT NOT NULL,
    margin FLOAT,
    status VARCHAR(32) NOT NULL, -- 'MATCHED', 'UNKNOWN', 'AMBIGUOUS', 'SPOOF_DETECTED'
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);
```

---

## 2. Chỉ Mục Tối Ưu Hóa (HNSW Indexing)

```sql
-- Tạo HNSW Index tối ưu cho Cosine Distance
CREATE INDEX IF NOT EXISTS idx_face_embeddings_hnsw 
ON face_embeddings 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

---

## 3. Câu Lệnh Truy Vấn Top-2 Ứng Viên (Kèm Margin Check)

```sql
SELECT 
    person_id,
    1.0 - (embedding <=> :query_vector) AS cosine_similarity
FROM face_embeddings
WHERE model_version = :current_model_version
ORDER BY embedding <=> :query_vector ASC
LIMIT 2;
```
