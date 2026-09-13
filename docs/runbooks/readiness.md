# Readiness failure runbook

## Environment

Required production configuration must include `DATABASE_URL` and the full model provenance set used by `src/api/recognition_runtime.py`: `ARCFACE_MODEL_SHA256`, `ARCFACE_MODEL_PUBLISHER`, `ARCFACE_MODEL_REVISION`, `ARCFACE_MODEL_WEIGHT_LICENSE`, `ARCFACE_MODEL_COMMERCIAL_USE`, `ARCFACE_MODEL_PROVENANCE_URL`, and the corresponding SCRFD provenance variables. `LIVENESS_MODEL_PATH` is also required for fail-closed recognition. MQTT credentials/certificates must be supplied through deployment environment when TLS/authentication is enabled.

## Symptoms and actions

**database unavailable**: `/readyz` returns 503. Check PostgreSQL reachability, credentials and pgvector/table migrations. Do not enable memory/SQLite as a production-ready substitute.

**model unavailable or provenance mismatch**: keep recognition blocked. Verify the model file, SHA-256, version, input contract and provenance metadata. Do not silently substitute another model.

**MQTT unavailable**: keep action transport blocked; inspect broker connectivity/TLS and reconnect configuration. Retry only with bounded backoff.

**inference/quality/FAS failure**: reject the recognition attempt. No action side effect may be emitted before identity authorization and durable ledger success.

## Privacy/security limits

Logs and transport error envelopes must not contain passwords, tokens, DSNs, stack traces, raw embeddings, face images or sensitive internal file paths. `person_id` is an identifier and should be treated as personal data according to the deployment's legal/privacy requirements.

## Validation

Run `pytest -q`, the database/API/MQTT/readiness/model/recognition subsets, and the configured type-check/lint commands. CI is the final deployment gate; never merge this branch directly into `main`.
