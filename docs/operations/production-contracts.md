# Production runtime contracts

## Environment

`DATABASE_URL` is required by the production bootstrap. Model provenance variables are required by `build_recognition_pipeline()`; see `src/api/recognition_runtime.py`. MQTT connection settings are deployment-specific and must never be committed as secrets.

## Health

`GET /healthz` is liveness-only and does not depend on database, MQTT or model state.

`GET /readyz` is fail-closed: HTTP 200 requires every required runtime dependency to be explicitly ready; otherwise HTTP 503 is returned. Readiness details contain only safe status labels.

## Correlation

`X-Request-ID` accepts only bounded HTTP-header-safe characters and is generated when absent or invalid. The accepted ID is returned in the HTTP response header and included in action/event envelopes where supported.

## Error safety

Public errors use `VALIDATION_ERROR`, `DEPENDENCY_UNAVAILABLE`, `AUTHORIZATION_FAILURE`, or `INTERNAL_ERROR`. Client responses must not expose credentials, DSNs, stack traces, raw embeddings, face images or sensitive internal paths.

## Action safety

MQTT and other device actions must be authorization-gated, idempotent and durable before physical side effects. A publish acknowledgement only proves broker acceptance, not physical completion on the device.
