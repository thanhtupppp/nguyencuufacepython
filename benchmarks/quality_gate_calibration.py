"""Leakage-safe quality-gate calibration utilities.

Consumes JSONL quality observations. Thresholds are selected only from the
calibration split; validation is used for model selection and locked_test is
never used to tune thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

FEATURES = {
    "face_width": "min",
    "face_height": "min",
    "bbox_confidence": "min",
    "blur_score": "min",
    "brightness": "range",
    "yaw": "abs_max",
    "pitch": "abs_max",
    "roll": "abs_max",
}


def load_records(path: str) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"person_id", "split", "camera_id", "condition", *FEATURES}
    for i, row in enumerate(rows):
        missing = required - row.keys()
        if missing:
            raise ValueError(f"row {i}: missing fields: {sorted(missing)}")
        if row["split"] not in {"calibration", "validation", "locked_test"}:
            raise ValueError(f"row {i}: invalid split {row['split']}")
    by_person: dict[str, set[str]] = {}
    for row in rows:
        by_person.setdefault(str(row["person_id"]), set()).add(row["split"])
    leaked = {p: sorted(s) for p, s in by_person.items() if len(s) > 1}
    if leaked:
        raise ValueError(f"person leakage across splits: {leaked}")
    return rows


def percentile(rows: list[dict[str, Any]], feature: str, q: float) -> float:
    return float(np.percentile([float(r[feature]) for r in rows], q))


def calibrate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cal = [r for r in rows if r["split"] == "calibration"]
    if not cal:
        raise ValueError("calibration split is empty")
    thresholds = {
        "face_width_min": percentile(cal, "face_width", 5),
        "face_height_min": percentile(cal, "face_height", 5),
        "bbox_confidence_min": percentile(cal, "bbox_confidence", 5),
        "blur_score_min": percentile(cal, "blur_score", 5),
        "brightness_min": percentile(cal, "brightness", 5),
        "brightness_max": percentile(cal, "brightness", 95),
        "yaw_abs_max": percentile(cal, "yaw", 95),
        "pitch_abs_max": percentile(cal, "pitch", 95),
        "roll_abs_max": percentile(cal, "roll", 95),
    }
    # Pose percentiles must be based on absolute values.
    for name in ("yaw", "pitch", "roll"):
        thresholds[f"{name}_abs_max"] = float(np.percentile([abs(float(r[name])) for r in cal], 95))
    return {"method": "calibration_percentiles_v1", "n": len(cal), "thresholds": thresholds}


def evaluate(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> dict[str, Any]:
    counts = {"total": 0, "accept": 0, "reject": 0}
    reasons: dict[str, int] = {}
    by_condition: dict[str, dict[str, int]] = {}
    for r in rows:
        counts["total"] += 1
        rr: list[str] = []
        if float(r["face_width"]) < thresholds["face_width_min"]: rr.append("FACE_TOO_SMALL")
        if float(r["face_height"]) < thresholds["face_height_min"]: rr.append("FACE_TOO_SMALL")
        if float(r["bbox_confidence"]) < thresholds["bbox_confidence_min"]: rr.append("LOW_DETECTION_CONFIDENCE")
        if float(r["blur_score"]) < thresholds["blur_score_min"]: rr.append("BLURRY_FACE")
        if float(r["brightness"]) < thresholds["brightness_min"]: rr.append("UNDER_EXPOSED")
        if float(r["brightness"]) > thresholds["brightness_max"]: rr.append("OVER_EXPOSED")
        for name in ("yaw", "pitch", "roll"):
            if abs(float(r[name])) > thresholds[f"{name}_abs_max"]: rr.append(f"EXTREME_{name.upper()}")
        if rr:
            counts["reject"] += 1
            for reason in sorted(set(rr)): reasons[reason] = reasons.get(reason, 0) + 1
        else:
            counts["accept"] += 1
        cond = str(r["condition"])
        slot = by_condition.setdefault(cond, {"total": 0, "accept": 0, "reject": 0})
        slot["total"] += 1
        slot["reject" if rr else "accept"] += 1
    counts["rejection_rate"] = counts["reject"] / counts["total"] if counts["total"] else None
    return {"counts": counts, "reason_counts": reasons, "by_condition": by_condition}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    rows = load_records(args.jsonl)
    artifact = calibrate(rows)
    artifact["validation"] = evaluate([r for r in rows if r["split"] == "validation"], artifact["thresholds"])
    artifact["locked_test"] = evaluate([r for r in rows if r["split"] == "locked_test"], artifact["thresholds"])
    artifact["input_sha256"] = hashlib.sha256(Path(args.jsonl).read_bytes()).hexdigest()
    Path(args.output).write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
