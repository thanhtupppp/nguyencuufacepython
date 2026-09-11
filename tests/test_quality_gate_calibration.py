import json

import pytest

from benchmarks.quality_gate_calibration import calibrate, evaluate, load_records


def row(person, split, blur=100, brightness=120, size=120, conf=.9, yaw=2, pitch=2, roll=1):
    return {
        "person_id": person, "split": split, "camera_id": "cam1", "condition": "frontal",
        "face_width": size, "face_height": size, "bbox_confidence": conf,
        "blur_score": blur, "brightness": brightness, "yaw": yaw, "pitch": pitch, "roll": roll,
    }


def test_rejects_person_leakage(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(json.dumps(row("p1", "calibration")) + "\n" + json.dumps(row("p1", "validation")) + "\n")
    with pytest.raises(ValueError, match="person leakage"):
        load_records(str(p))


def test_calibration_uses_calibration_split_only():
    rows = [row(f"c{i}", "calibration", blur=100 + i) for i in range(20)]
    rows += [row("v1", "validation", blur=1)]
    artifact = calibrate(rows)
    assert artifact["method"] == "calibration_percentiles_v1"
    assert artifact["thresholds"]["blur_score_min"] > 1


def test_evaluation_has_reason_codes_and_condition_breakdown():
    rows = [row("p1", "validation", blur=1, size=30)]
    out = evaluate(rows, {
        "face_width_min": 60, "face_height_min": 60, "bbox_confidence_min": .5,
        "blur_score_min": 20, "brightness_min": 30, "brightness_max": 230,
        "yaw_abs_max": 35, "pitch_abs_max": 35, "roll_abs_max": 30,
    })
    assert out["counts"]["reject"] == 1
    assert "FACE_TOO_SMALL" in out["reason_counts"]
    assert out["by_condition"]["frontal"]["reject"] == 1
