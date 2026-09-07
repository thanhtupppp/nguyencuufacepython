# MQTT v1 device protocol

## Purpose

MQTT is the control/state/event transport for edge devices. It is not the transport for face images or embeddings. Recognition requests may use HTTP/WebSocket; MQTT carries compact commands and lifecycle state so ESP32 and other constrained nodes can participate without moving biometric vectors through the broker.

## Topics

For device `DEVICE_ID`:

- `face/v1/devices/DEVICE_ID/command`
- `face/v1/devices/DEVICE_ID/state` (retained)
- `face/v1/devices/DEVICE_ID/event`
- `face/v1/devices/DEVICE_ID/availability` (retained LWT: `online`/`offline`)

`DEVICE_ID` must not contain `/`, `+`, or `#`.

## Command envelope

```json
{"schema_version":1,"request_id":"req-123","command":"recognize","payload":{}}
```

`request_id` is mandatory so the device can deduplicate QoS-1 redelivery. Commands are intentionally idempotent where possible.

Initial command names:

- `recognize`
- `enroll`
- `health`
- `config`

## State

```json
{"schema_version":1,"device_id":"cam-01","status":"online","firmware_version":"1.0.0","model_version":"arcface-v1","camera_id":"cam-01"}
```

State is safe to retain and must not contain raw frames, embeddings, biometric templates, credentials, or access tokens.

## Recognition events

Events should contain only the minimum result needed by consumers, for example `person_id`, `status`, `similarity`, `margin`, `liveness`, `request_id`, and timestamps. The broker should never be treated as the biometric database.

## Delivery policy

- Commands: QoS 1, no retained command messages.
- State: QoS 1, retained.
- Events: QoS 1 initially; consumers must tolerate duplicate delivery by `request_id`/event id.
- Availability: retained state plus Last Will and Testament.

## Security requirements

Production brokers must use TLS and per-device credentials or certificates with ACLs scoped to that device's topic namespace. A device must not publish or subscribe to another device's command namespace.

## Validation

Contract helpers and tests live in `src/devices/mqtt_contract.py` and `tests/test_mqtt_contract.py`.
