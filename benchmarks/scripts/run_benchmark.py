"""
End-to-end real-model benchmark for SCRFD -> alignment -> ArcFace.

The benchmark intentionally refuses to silently resize a raw image to 112x112.
Each raw image must first pass SCRFD and provide five landmarks, then FaceAligner
creates the canonical 112x112 input for ArcFace.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd

from src.alignment.aligner import FaceAligner
from src.detection.scrfd import SCRFDDetector
from src.recognition.arcface import ArcFaceRecognizer
from benchmarks.scripts.metrics import compute_cosine_similarity, compute_roc_and_rates


def _select_face(detections: list[dict]) -> dict:
    """Select the strongest detected face for the single-face benchmark protocol."""
    valid = [d for d in detections if d.get("landmarks") is not None]
    if not valid:
        raise ValueError("SCRFD found no face with five landmarks")
    return max(valid, key=lambda d: float(d.get("score", 0.0)))


def _make_embedding(
    image_path: str,
    detector: SCRFDDetector,
    aligner: FaceAligner,
    recognizer: ArcFaceRecognizer,
) -> np.ndarray:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")

    detections = detector.detect(image)
    face = _select_face(detections)
    landmarks = np.asarray(face["landmarks"], dtype=np.float32)
    aligned, _ = aligner.align(image, landmarks)
    embedding = recognizer.extract_embedding(aligned)

    embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if embedding.shape != (512,):
        raise ValueError(f"Expected 512-D embedding, got {embedding.shape}")
    norm = float(np.linalg.norm(embedding))
    if not np.isfinite(norm) or norm <= 1e-8:
        raise ValueError("ArcFace returned an invalid embedding norm")
    return embedding / norm


def run_benchmark(
    pairs_dir: Path,
    results_dir: Path,
    scrfd_model_path: Path,
    arcface_model_path: Path,
    scrfd_sha256: str | None = None,
    mock_mode: bool = False,
    threshold_override: float | None = None,
    margin_override: float | None = None,
) -> Path:
    """Run the real-model benchmark. Mock mode is retained only for harness tests."""
    if mock_mode:
        raise ValueError("Mock mode is intentionally disabled for the real-model benchmark gate")
    if not scrfd_model_path.exists():
        raise FileNotFoundError(f"SCRFD model not found: {scrfd_model_path}")
    if not arcface_model_path.exists():
        raise FileNotFoundError(f"ArcFace model not found: {arcface_model_path}")

    results_dir.mkdir(parents=True, exist_ok=True)
    genuine_csv = pairs_dir / "genuine.csv"
    impostor_csv = pairs_dir / "impostor.csv"
    if not genuine_csv.exists() or not impostor_csv.exists():
        raise FileNotFoundError(f"Pairs files not found in {pairs_dir}")

    df_gen = pd.read_csv(genuine_csv)
    df_imp = pd.read_csv(impostor_csv)
    if df_gen.empty or df_imp.empty:
        raise ValueError("Benchmark requires both genuine and impostor pairs")

    detector = SCRFDDetector(model_path=scrfd_model_path, expected_sha256=scrfd_sha256)
    aligner = FaceAligner(output_size=(112, 112))
    recognizer = ArcFaceRecognizer(model_path=arcface_model_path)

    cache: dict[str, np.ndarray] = {}

    def get_embedding(path: str) -> np.ndarray:
        if path not in cache:
            cache[path] = _make_embedding(path, detector, aligner, recognizer)
        return cache[path]

    records: list[dict] = []
    for expected_genuine, frame in ((1, df_gen), (0, df_imp)):
        for _, row in frame.iterrows():
            emb1 = get_embedding(str(row["path1"]))
            emb2 = get_embedding(str(row["path2"]))
            sim = compute_cosine_similarity(emb1, emb2)
            records.append({
                "path1": row["path1"],
                "path2": row["path2"],
                "person1": row["person1"],
                "person2": row["person2"],
                "condition": row["condition"],
                "is_genuine": expected_genuine,
                "similarity": float(sim),
            })

    dist_df = pd.DataFrame(records)
    gen_scores = dist_df.loc[dist_df["is_genuine"] == 1, "similarity"].to_numpy(dtype=np.float32)
    imp_scores = dist_df.loc[dist_df["is_genuine"] == 0, "similarity"].to_numpy(dtype=np.float32)
    if len(gen_scores) == 0 or len(imp_scores) == 0:
        raise ValueError("No usable genuine/impostor scores were produced")

    dist_df.to_csv(results_dir / "similarity_distribution.csv", index=False)
    roc = compute_roc_and_rates(gen_scores, imp_scores)
    pd.DataFrame({
        "threshold": roc["thresholds"],
        "FAR": roc["FAR"],
        "FRR": roc["FRR"],
        "TAR": roc["TAR"],
    }).to_csv(results_dir / "roc.csv", index=False)

    operating = roc["operating_points"]
    target = operating.get("TAR@FAR=0.001", {})
    threshold = threshold_override if threshold_override is not None else target.get("threshold", roc["EER_threshold"])
    margin = margin_override if margin_override is not None else 0.08

    condition_stats: dict[str, dict] = {}
    for condition in sorted(dist_df["condition"].astype(str).unique()):
        sub = dist_df[dist_df["condition"].astype(str) == condition]
        g = sub[sub["is_genuine"] == 1]["similarity"]
        i = sub[sub["is_genuine"] == 0]["similarity"]
        condition_stats[condition] = {
            "genuine_count": int(len(g)),
            "impostor_count": int(len(i)),
            "mean_genuine_similarity": float(g.mean()) if len(g) else None,
            "mean_impostor_similarity": float(i.mean()) if len(i) else None,
            "genuine_recall_at_threshold": float((g >= threshold).mean()) if len(g) else None,
        }

    import json
    metrics = {
        "model": getattr(recognizer, "model_name", arcface_model_path.name),
        "model_version": getattr(recognizer, "model_version", "unknown"),
        "scrfd_model": scrfd_model_path.name,
        "embedding_dim": 512,
        "normalization": "L2",
        "distance_metric": "cosine",
        "alignment": "Umeyama-5-point-112x112",
        "genuine_pairs": int(len(gen_scores)),
        "impostor_pairs": int(len(imp_scores)),
        "eer": float(roc["EER"]),
        "eer_threshold": float(roc["EER_threshold"]),
        "threshold": float(threshold),
        "margin": float(margin),
        "condition_stats": condition_stats,
    }
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    report = results_dir / "threshold_report.md"
    lines = [
        "# Real SCRFD + ArcFace Benchmark",
        "",
        f"- ArcFace model: `{metrics['model']}`",
        f"- ArcFace version: `{metrics['model_version']}`",
        f"- SCRFD model: `{metrics['scrfd_model']}`",
        "- Alignment: `5 landmarks -> Umeyama -> 112x112`",
        "- Embedding: `512-D, L2 normalized`",
        f"- Genuine pairs evaluated: `{len(gen_scores)}`",
        f"- Impostor pairs evaluated: `{len(imp_scores)}`",
        f"- EER: `{roc['EER'] * 100:.4f}%`",
        f"- EER threshold: `{roc['EER_threshold']:.6f}`",
        f"- Selected threshold: `{threshold:.6f}`",
        f"- Selected top1-top2 margin: `{margin:.6f}`",
        "",
        "## Statistical caution",
        "",
        "The benchmark must not claim a FAR operating point more precise than the number of independent impostor trials supports.",
        "Thresholds are calibrated on validation data and must be frozen before a locked test evaluation.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real SCRFD + ArcFace benchmark")
    parser.add_argument("--pairs", type=Path, default=Path("benchmarks/pairs"))
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--scrfd-model", type=Path, required=True)
    parser.add_argument("--arcface-model", type=Path, required=True)
    parser.add_argument("--scrfd-sha256", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--margin", type=float, default=None)
    args = parser.parse_args()
    run_benchmark(
        pairs_dir=args.pairs,
        results_dir=args.results,
        scrfd_model_path=args.scrfd_model,
        arcface_model_path=args.arcface_model,
        scrfd_sha256=args.scrfd_sha256,
        threshold_override=args.threshold,
        margin_override=args.margin,
    )
