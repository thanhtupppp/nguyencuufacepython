-- Initialize pgvector extension and core database schema for NguyenCuuFacePython

CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Persons Table
CREATE TABLE IF NOT EXISTS persons (
    person_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    department VARCHAR(128),
    role VARCHAR(64) DEFAULT 'user',
    status VARCHAR(32) DEFAULT 'active',
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 2. Face Embeddings Table (512D)
CREATE TABLE IF NOT EXISTS face_embeddings (
    id BIGSERIAL PRIMARY KEY,
    person_id VARCHAR(64) NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    embedding vector(512) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    quality_score FLOAT DEFAULT 1.0,
    source_image_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 3. Devices Table
CREATE TABLE IF NOT EXISTS devices (
    device_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    device_type VARCHAR(64) NOT NULL,
    location VARCHAR(255),
    ip_address VARCHAR(45),
    status VARCHAR(32) DEFAULT 'online',
    last_ping TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 4. Access Logs Table
CREATE TABLE IF NOT EXISTS access_logs (
    id BIGSERIAL PRIMARY KEY,
    person_id VARCHAR(64) REFERENCES persons(person_id) ON DELETE SET NULL,
    device_id VARCHAR(64) REFERENCES devices(device_id) ON DELETE SET NULL,
    similarity FLOAT NOT NULL,
    margin FLOAT,
    status VARCHAR(32) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 5. HNSW Index for ultra-fast Cosine Distance vector search
CREATE INDEX IF NOT EXISTS idx_face_embeddings_hnsw 
ON face_embeddings 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
