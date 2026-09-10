#!/usr/bin/env python3
"""Deterministic track/person identity replay evaluator.

Input JSONL records:
{"frame_index":0,"track_id":"t1","person_id":"A","accepted":true,"score":0.9}

The evaluator deliberately measures tracking identity stability separately from
recognition quality. It can run without camera/model assets and therefore never
creates a false recognition benchmark result.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from src.tracking.confirmation import ConfirmationConfig, IdentityObservation, TrackConfirmation


def load_records(path: Path) -> list[dict]:
    records = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_no}: invalid JSON") from exc
        for key in ("frame_index", "track_id", "person_id", "accepted"):
            if key not in item:
                raise ValueError(f"line {line_no}: missing {key}")
        records.append(item)
    return sorted(records, key=lambda x: (int(x["frame_index"]), str(x["track_id"])))


def evaluate(records: list[dict], min_consecutive: int, max_gap_frames: int) -> dict:
    states: dict[str, TrackConfirmation] = {}
    confirmed: dict[str, str] = {}
    first_confirmation: dict[str, int] = {}
    identity_switches = 0
    events = []

    for item in records:
        tid = str(item["track_id"])
        frame = int(item["frame_index"])
        tracker = states.setdefault(
            tid,
            TrackConfirmation(
                track_id=tid,
                config=ConfirmationConfig(
                    min_consecutive=min_consecutive,
                    max_gap_frames=max_gap_frames,
                ),
            ),
        )
        obs = IdentityObservation(
            person_id=None if item["person_id"] is None else str(item["person_id"]),
            accepted=bool(item["accepted"]),
            score=None if item.get("score") is None else float(item["score"]),
        )
        before = tracker.person_id
        state = tracker.observe(frame, obs)
        after = tracker.person_id
        if before is not None and after is not None and before != after:
            identity_switches += 1
        if before is None and after is not None and tid not in first_confirmation:
            first_confirmation[tid] = frame
        if state.value == "lost":
            confirmed.pop(tid, None)
        elif after is not None:
            confirmed[tid] = after
        events.append({"frame_index": frame, "track_id": tid, "state": state.value, "person_id": after})

    return {
        "records": len(records),
        "tracks": len(states),
        "identity_switches": identity_switches,
        "confirmation_latency_frames": first_confirmation,
        "final_confirmed": dict(sorted(confirmed.items())),
        "events": events,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--min-consecutive", type=int, default=3)
    parser.add_argument("--max-gap-frames", type=int, default=2)
    args = parser.parse_args()
    if args.min_consecutive < 1 or args.max_gap_frames < 0:
        parser.error("min-consecutive must be >= 1 and max-gap-frames must be >= 0")
    result = evaluate(load_records(args.input), args.min_consecutive, args.max_gap_frames)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
