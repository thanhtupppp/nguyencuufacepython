"""
Pair generator for Face Recognition Benchmark.
Constructs genuine (same person) and impostor (different persons) pairs
from `benchmarks/gallery` and `benchmarks/probe`.
"""

import argparse
import csv
import itertools
from pathlib import Path
import random


def scan_images(directory: Path) -> list[Path]:
    """Scans valid image files."""
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not directory.exists():
        return []
    return [p for p in directory.rglob("*") if p.suffix.lower() in valid_exts]


def build_benchmark_pairs(
    gallery_dir: Path,
    probe_dir: Path,
    output_pairs_dir: Path,
    max_impostors: int = 50000,
    seed: int = 42,
) -> tuple[int, int]:
    """
    Builds genuine and impostor CSV pairs.
    """
    random.seed(seed)
    output_pairs_dir.mkdir(parents=True, exist_ok=True)

    # 1. Map gallery images: person_id -> list of file paths
    gallery_map: dict[str, list[Path]] = {}
    for person_folder in gallery_dir.iterdir():
        if person_folder.is_dir():
            pid = person_folder.name
            imgs = scan_images(person_folder)
            if imgs:
                gallery_map[pid] = imgs

    # 2. Map probe images: condition -> person_id -> list of file paths
    # Expecting probe structure: probe/<condition>/<person_id>/*.jpg OR probe/<condition>/<person_id_image>.jpg
    probe_map: dict[str, dict[str, list[Path]]] = {}
    conditions = ["frontal", "angle", "low_light", "blur", "partial_occlusion", "distance"]
    for cond in conditions:
        cond_dir = probe_dir / cond
        if not cond_dir.exists():
            continue
        probe_map[cond] = {}
        for item in cond_dir.iterdir():
            if item.is_dir():
                pid = item.name
                imgs = scan_images(item)
                if imgs:
                    probe_map[cond].setdefault(pid, []).extend(imgs)
            elif item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                # Format: person_001_probe01.jpg
                pid = item.stem.split("_")[0]
                probe_map[cond].setdefault(pid, []).append(item)

    genuine_pairs = []
    impostor_pairs = []

    # Build genuine pairs: Gallery(P) vs Probe(P, condition)
    for cond, p_map in probe_map.items():
        for pid, probe_imgs in p_map.items():
            if pid in gallery_map:
                for g_img in gallery_map[pid]:
                    for p_img in probe_imgs:
                        genuine_pairs.append({
                            "path1": str(g_img.resolve()),
                            "path2": str(p_img.resolve()),
                            "person1": pid,
                            "person2": pid,
                            "condition": cond,
                            "is_genuine": 1,
                        })

    # Also build within-gallery genuine pairs if multiple images per identity
    for pid, g_imgs in gallery_map.items():
        if len(g_imgs) > 1:
            for g1, g2 in itertools.combinations(g_imgs, 2):
                genuine_pairs.append({
                    "path1": str(g1.resolve()),
                    "path2": str(g2.resolve()),
                    "person1": pid,
                    "person2": pid,
                    "condition": "gallery_ref",
                    "is_genuine": 1,
                })

    # Build impostor pairs: Gallery(P_A) vs Probe(P_B)
    all_persons = list(gallery_map.keys())
    if len(all_persons) >= 2:
        candidate_impostors = []
        for p1, p2 in itertools.combinations(all_persons, 2):
            for g1 in gallery_map[p1]:
                for g2 in gallery_map[p2]:
                    candidate_impostors.append((g1, g2, p1, p2, "gallery_cross"))

        for cond, p_map in probe_map.items():
            for pid_probe, probe_imgs in p_map.items():
                other_gallery_pids = [p for p in all_persons if p != pid_probe]
                for other_pid in other_gallery_pids:
                    for g_img in gallery_map[other_pid]:
                        for p_img in probe_imgs:
                            candidate_impostors.append((g_img, p_img, other_pid, pid_probe, cond))

        if len(candidate_impostors) > max_impostors:
            candidate_impostors = random.sample(candidate_impostors, max_impostors)

        for g1, g2, p1, p2, cond in candidate_impostors:
            impostor_pairs.append({
                "path1": str(g1.resolve()),
                "path2": str(g2.resolve()),
                "person1": p1,
                "person2": p2,
                "condition": cond,
                "is_genuine": 0,
            })

    # Save to CSV
    genuine_csv = output_pairs_dir / "genuine.csv"
    with open(genuine_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path1", "path2", "person1", "person2", "condition", "is_genuine"])
        writer.writeheader()
        writer.writerows(genuine_pairs)

    impostor_csv = output_pairs_dir / "impostor.csv"
    with open(impostor_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path1", "path2", "person1", "person2", "condition", "is_genuine"])
        writer.writeheader()
        writer.writerows(impostor_pairs)

    print(f"Generated {len(genuine_pairs)} genuine pairs -> {genuine_csv}")
    print(f"Generated {len(impostor_pairs)} impostor pairs -> {impostor_csv}")
    return len(genuine_pairs), len(impostor_pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate genuine and impostor pairs for recognition benchmark.")
    parser.add_argument("--gallery", type=str, default="benchmarks/gallery", help="Path to gallery dir")
    parser.add_argument("--probe", type=str, default="benchmarks/probe", help="Path to probe dir")
    parser.add_argument("--out", type=str, default="benchmarks/pairs", help="Output pairs directory")
    args = parser.parse_args()

    build_benchmark_pairs(Path(args.gallery), Path(args.probe), Path(args.out))
