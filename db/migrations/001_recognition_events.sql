-- P1.2.3 durable recognition-event boundary
CREATE TABLE IF NOT EXISTS recognition_events (
    event_id UUID PRIMARY KEY,
    payload JSONB NOT NULL,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_recognition_events_received_at
    ON recognition_events (received_at);
