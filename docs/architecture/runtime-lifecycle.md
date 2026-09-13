# Runtime lifecycle and readiness

The service keeps runtime I/O out of import-time code. Database, MQTT and model resources are owned by application startup/shutdown and readiness is fail-closed.

## Readiness matrix

| Dependency | Required | Ready when | Failure behavior |
|---|---|---|---|
| database | yes | PostgreSQL + pgvector probe succeeds | `/readyz` returns 503 |
| model_registry | yes | approved model exists, provenance/hash/shape/dimension checks pass | recognition blocked |
| mqtt | yes for action-enabled deployments | broker connection established | action transport blocked |
| recognition | yes | detector, quality, alignment, FAS and embedding contracts are initialized | recognition blocked |

Liveness is independent of these dependencies.

## Recognition order

Detection -> quality gate -> alignment -> anti-spoofing/FAS -> embedding -> identity authorization -> decision -> durable ledger -> side effect.

No physical action is valid before authorization and durable ledger persistence succeed.
