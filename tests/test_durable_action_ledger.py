from pathlib import Path

from src.events.durable_dedup import SQLiteActionLedger


def test_event_survives_restart(tmp_path: Path) -> None:
    db = tmp_path / "actions.db"
    ledger = SQLiteActionLedger(db)
    first = ledger.claim("evt-1", "esp32-1")
    assert first.status == "CLAIMED"
    assert first.duplicate is False
    ledger.complete("evt-1")
    ledger.close()

    restarted = SQLiteActionLedger(db)
    second = restarted.claim("evt-1", "esp32-1")
    assert second.duplicate is True
    assert second.status == "COMPLETED"
    restarted.close()


def test_duplicate_claim_does_not_create_second_row(tmp_path: Path) -> None:
    ledger = SQLiteActionLedger(tmp_path / "actions.db")
    assert ledger.claim("evt-2", "esp32-1").duplicate is False
    assert ledger.claim("evt-2", "esp32-1").duplicate is True
    ledger.close()
