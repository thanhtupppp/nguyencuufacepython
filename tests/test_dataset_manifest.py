import hashlib
import json
from pathlib import Path

import pytest

from tools.validate_dataset_manifest import validate


def _make(tmp_path: Path, rows):
    root = tmp_path / "images"
    root.mkdir()
    for i, row in enumerate(rows):
        p = root / row["path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        data = f"image-{i}".encode()
        p.write_bytes(data)
        row["sha256"] = hashlib.sha256(data).hexdigest()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return manifest, root


def row(path="a.jpg", person="p1", split="calibration"):
    return {
        "image_id": path,
        "path": path,
        "person_id": person,
        "split": split,
        "camera_id": "cam1",
        "condition": "controlled",
        "authorization_ref": "auth-1",
        "sha256": "0" * 64,
    }


def test_valid_manifest(tmp_path):
    manifest, root = _make(tmp_path, [row()])
    receipt = validate(str(manifest), str(root))
    assert receipt["status"] == "VALID"
    assert receipt["image_count"] == 1


@pytest.mark.parametrize("bad_path", ["../x.jpg", "/x.jpg", "a\\b.jpg"])
def test_unsafe_path_rejected(tmp_path, bad_path):
    r = row(path=bad_path)
    manifest, root = _make(tmp_path, [r])
    with pytest.raises(ValueError):
        validate(str(manifest), str(root))


def test_person_leakage_rejected(tmp_path):
    rows = [row("a.jpg", "p1", "calibration"), row("b.jpg", "p1", "validation")]
    manifest, root = _make(tmp_path, rows)
    with pytest.raises(ValueError, match="person leakage"):
        validate(str(manifest), str(root))


def test_hash_mismatch_rejected(tmp_path):
    manifest, root = _make(tmp_path, [row()])
    data = json.loads(manifest.read_text().strip())
    data["sha256"] = "f" * 64
    manifest.write_text(json.dumps(data) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate(str(manifest), str(root))


def test_duplicate_image_id_rejected(tmp_path):
    rows = [row("a.jpg", "p1"), row("b.jpg", "p2")]
    rows[1]["image_id"] = rows[0]["image_id"]
    manifest, root = _make(tmp_path, rows)
    with pytest.raises(ValueError, match="duplicate"):
        validate(str(manifest), str(root))
