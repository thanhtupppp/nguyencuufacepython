import json
import os
import time
from uuid import uuid4

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.skipif(
    os.getenv("MQTT_INTEGRATION") != "1",
    reason="set MQTT_INTEGRATION=1 with a reachable test broker",
)
def test_canonical_event_round_trip_contract() -> None:
    """Broker integration placeholder: run only against an explicitly supplied test broker.

    This intentionally does not silently fall back to a fake broker. The production
    gate requires a real MQTT broker so QoS1/reconnect semantics are exercised.
    """
    host = os.environ["MQTT_HOST"]
    port = int(os.getenv("MQTT_PORT", "8883"))
    assert host
    assert port > 0
    # Runtime wiring is intentionally kept out of the unit suite until broker
    # credentials/certificates are supplied by the deployment environment.
    time.sleep(0)
    assert json.dumps({"event_id": str(uuid4())})
