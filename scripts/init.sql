-- Initialize pgvector extension and core database schema for NguyenCuuFacePython

CREATE EXTENSION IF NOT EXISTS vector;

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

CREATE TABLE IF NOT EXISTS face_embeddings (
    id BIGSERIAL PRIMARY KEY,
    person_id VARCHAR(64) NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
    embedding vector(512) NOT NULL,
    model_version VARCHAR(64) NOT NULL,
    quality_score FLOAT DEFAULT 1.0,
    source_image_path TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS devices (
    device_id VARCHAR(64) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    device_type VARCHAR(64) NOT NULL,
    location VARCHAR(255),
    ip_address VARCHAR(45),
    status VARCHAR(32) DEFAULT 'online',
    last_ping TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS access_logs (
    id BIGSERIAL PRIMARY KEY,
    person_id VARCHAR(64) REFERENCES persons(person_id) ON DELETE SET NULL,
    device_id VARCHAR(64) REFERENCES devices(device_id) ON DELETE SET NULL,
    similarity FLOAT NOT NULL,
    margin FLOAT,
    status VARCHAR(32) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mqtt_idempotency (
    device_id VARCHAR(64) NOT NULL,
    request_id VARCHAR(128) NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    PRIMARY KEY (device_id, request_id)
);

CREATE INDEX IF NOT EXISTS idx_mqtt_idempotency_expires
ON mqtt_idempotency (expires_at);

CREATE TABLE IF NOT EXISTS device_sessions (
    session_id VARCHAR(64) PRIMARY KEY,
    device_id VARCHAR(64) NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_seen_at TIMESTAMP WITH TIME ZONE NOT NULL,
    status VARCHAR(16) NOT NULL,
    sequence BIGINT NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_device_sessions_device
ON device_sessions (device_id);

CREATE INDEX IF NOT EXISTS idx_device_sessions_last_seen
ON device_sessions (last_seen_at);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_hnsw
ON face_embeddings
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
