"""Preflight validation for the real face-recognition benchmark."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_pairs(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    required = {"path1", "path2", "person1", "person2", "condition", "is_genuine"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"{path}: missing columns {sorted(required - set(rows[0]))}")
    return rows


def _pair_key(row: dict[str, str]) -> tuple[str, str, str]:
    """Canonicalize pair order so A/B and B/A cannot leak as separate trials."""
    p1, p2 = row["path1"], row["path2"]
    return (min(p1, p2), max(p1, p2), row["condition"])


def validate_pairs(genuine: list[dict[str, str]], impostor: list[dict[str, str]]) -> list[str]:
    errors: list[str] = []
    all_rows = [(r, True) for r in genuine] + [(r, False) for r in impostor]
    seen: set[tuple[str, str, str]] = set()

    for row, expected_genuine in all_rows:
        p1, p2 = row["path1"], row["path2"]
        pid1, pid2 = row["person1"], row["person2"]
        key = _pair_key(row)
        if key in seen:
            errors.append(f"duplicate/reversed pair: {key}")
        seen.add(key)
        if not Path(p1).exists():
            errors.append(f"missing image: {p1}")
        if not Path(p2).exists():
            errors.append(f"missing image: {p2}")
        if Path(p1).suffix.lower() not in IMAGE_EXTS or Path(p2).suffix.lower() not in IMAGE_EXTS:
            errors.append(f"unsupported image extension: {p1}, {p2}")
        actual_genuine = pid1 == pid2
        if actual_genuine != expected_genuine:
            errors.append(f"label mismatch: {p1} / {p2} ({pid1}, {pid2})")
        csv_label = str(row.get("is_genuine", "")).strip().lower()
        if csv_label not in {"0", "1", "true", "false"}:
            errors.append(f"invalid is_genuine label: {p1} / {p2}: {csv_label}")
        elif (csv_label in {"1", "true"}) != expected_genuine:
            errors.append(f"CSV label mismatch: {p1} / {p2}: {csv_label}")

    if not genuine:
        errors.append("no genuine pairs")
    if not impostor:
        errors.append("no impostor pairs")
    return errors


def summarize(genuine: list[dict[str, str]], impostor: list[dict[str, str]]) -> dict:
    persons = Counter()
    conditions = Counter()
    for row in genuine:
        persons[row["person1"]] += 1
        conditions[row["condition"]] += 1
    for row in impostor:
        conditions[f"impostor:{row['condition']}"] += 1
    return {
        "genuine_pairs": len(genuine),
        "impostor_pairs": len(impostor),
        "genuine_persons": len(persons),
        "genuine_pairs_per_person_min": min(persons.values()) if persons else 0,
        "conditions": dict(sorted(conditions.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="benchmarks/pairs")
    args = parser.parse_args()
    root = Path(args.pairs)
    genuine = read_pairs(root / "genuine.csv")
    impostor = read_pairs(root / "impostor.csv")
    errors = validate_pairs(genuine, impostor)
    summary = summarize(genuine, impostor)
    print(summary)
    if errors:
        print("PRECHECK FAILED")
        for error in errors[:100]:
            print(f"- {error}")
        if len(errors) > 100:
            print(f"- ... {len(errors) - 100} more errors")
        return 1
    print("PRECHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
