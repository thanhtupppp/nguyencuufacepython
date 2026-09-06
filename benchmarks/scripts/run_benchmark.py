"""
Main Benchmark Runner for Face Recognition Evaluation.
Extracts features, computes similarity distributions, ROC curve, and generates threshold report.
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd

from src.alignment.aligner import FaceAligner
from src.recognition.arcface import ArcFaceRecognizer
from benchmarks.scripts.metrics import compute_cosine_similarity, compute_roc_and_rates, evaluate_gallery_probe


def run_benchmark(
    pairs_dir: Path,
    results_dir: Path,
    model_path: Path | None = None,
    mock_mode: bool = False,
    threshold_override: float | None = None,
    margin_override: float | None = None,
) -> Path:
    """
    Executes benchmark evaluation on genuine and impostor pairs.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    genuine_csv = pairs_dir / "genuine.csv"
    impostor_csv = pairs_dir / "impostor.csv"

    if not genuine_csv.exists() or not impostor_csv.exists():
        raise FileNotFoundError(
            f"Pairs files not found in {pairs_dir}. Run generate_pairs.py first."
        )

    df_gen = pd.read_csv(genuine_csv)
    df_imp = pd.read_csv(impostor_csv)

    print(f"Loaded {len(df_gen)} genuine pairs and {len(df_imp)} impostor pairs.")

    # Initialize recognizer
    recognizer = None
    if not mock_mode:
        if model_path is None or not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {model_path}. Provide valid ONNX path or use --mock for testing."
            )
        recognizer = ArcFaceRecognizer(model_path=model_path)
        print(f"Loaded model: {recognizer.model_name} ({recognizer.model_version})")

    # In-memory embedding cache to avoid re-extracting same image
    embedding_cache: dict[str, np.ndarray] = {}

    def get_embedding(img_path_str: str) -> np.ndarray:
        if img_path_str in embedding_cache:
            return embedding_cache[img_path_str]

        if mock_mode:
            # Synthetic reproducible embedding based on person_id hash for unit testing pipeline
            p = Path(img_path_str)
            # Extract identifier part
            stem = p.stem.split("_")[0]
            rng = np.random.RandomState(abs(hash(stem)) % (2**31))
            base_vec = rng.randn(512).astype(np.float32)
            # Add small noise per image
            img_rng = np.random.RandomState(abs(hash(p.stem)) % (2**31))
            noise = img_rng.randn(512).astype(np.float32) * 0.15
            vec = base_vec + noise
            vec = vec / np.linalg.norm(vec)
            embedding_cache[img_path_str] = vec
            return vec

        img = cv2.imread(img_path_str)
        if img is None:
            raise ValueError(f"Cannot read image: {img_path_str}")

        # Assume 112x112 aligned face or resize
        if img.shape[:2] != (112, 112):
            img = cv2.resize(img, (112, 112))
        emb = recognizer.extract_embedding(img)
        embedding_cache[img_path_str] = emb
        return emb

    # Compute similarities for Genuine pairs
    gen_scores = []
    gen_records = []
    print("Computing genuine pair similarities...")
    for idx, row in df_gen.iterrows():
        try:
            emb1 = get_embedding(row["path1"])
            emb2 = get_embedding(row["path2"])
            sim = float(compute_cosine_similarity(emb1, emb2))
            gen_scores.append(sim)
            gen_records.append({
                "path1": row["path1"],
                "path2": row["path2"],
                "person1": row["person1"],
                "person2": row["person2"],
                "condition": row["condition"],
                "is_genuine": 1,
                "similarity": sim,
            })
        except Exception as e:
            print(f"Warning: Failed pair ({row['path1']}, {row['path2']}): {e}")

    # Compute similarities for Impostor pairs
    imp_scores = []
    imp_records = []
    print("Computing impostor pair similarities...")
    for idx, row in df_imp.iterrows():
        try:
            emb1 = get_embedding(row["path1"])
            emb2 = get_embedding(row["path2"])
            sim = float(compute_cosine_similarity(emb1, emb2))
            imp_scores.append(sim)
            imp_records.append({
                "path1": row["path1"],
                "path2": row["path2"],
                "person1": row["person1"],
                "person2": row["person2"],
                "condition": row["condition"],
                "is_genuine": 0,
                "similarity": sim,
            })
        except Exception as e:
            print(f"Warning: Failed pair ({row['path1']}, {row['path2']}): {e}")

    gen_scores_arr = np.array(gen_scores)
    imp_scores_arr = np.array(imp_scores)

    # 1. Save similarity distribution CSV
    dist_df = pd.DataFrame(gen_records + imp_records)
    dist_csv = results_dir / "similarity_distribution.csv"
    dist_df.to_csv(dist_csv, index=False)
    print(f"Saved similarity distributions -> {dist_csv}")

    # 2. Compute ROC, FAR, FRR, EER
    roc_results = compute_roc_and_rates(gen_scores_arr, imp_scores_arr)
    roc_df = pd.DataFrame({
        "threshold": roc_results["thresholds"],
        "FAR": roc_results["FAR"],
        "FRR": roc_results["FRR"],
        "TAR": roc_results["TAR"],
    })
    roc_csv = results_dir / "roc.csv"
    roc_df.to_csv(roc_csv, index=False)
    print(f"Saved ROC curve points -> {roc_csv}")

    # Determine recommended threshold
    # Target FAR <= 0.001 (0.1%) or fallback to EER threshold
    tar_far_001 = roc_results["operating_points"].get("TAR@FAR=0.001", {})
    recommended_threshold = (
        threshold_override
        if threshold_override is not None
        else tar_far_001.get("threshold", roc_results["EER_threshold"])
    )
    recommended_margin = margin_override if margin_override is not None else 0.08

    # 3. Breakdown by Condition
    condition_stats = {}
    for cond in dist_df["condition"].unique():
        sub_gen = dist_df[(dist_df["condition"] == cond) & (dist_df["is_genuine"] == 1)]
        sub_imp = dist_df[(dist_df["condition"] == cond) & (dist_df["is_genuine"] == 0)]
        mean_g = float(sub_gen["similarity"].mean()) if len(sub_gen) > 0 else 0.0
        mean_i = float(sub_imp["similarity"].mean()) if len(sub_imp) > 0 else 0.0
        acc_sub = (
            float((sub_gen["similarity"] >= recommended_threshold).mean() * 100.0)
            if len(sub_gen) > 0
            else 0.0
        )
        condition_stats[cond] = {
            "genuine_count": len(sub_gen),
            "impostor_count": len(sub_imp),
            "mean_genuine_sim": mean_g,
            "mean_impostor_sim": mean_i,
            "genuine_recall_pct": acc_sub,
        }

    # 4. Generate Markdown Report
    report_md = results_dir / "threshold_report.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Báo Cáo Đo Lường Hiệu Năng Nhận Diện Khuôn Mặt (Benchmark Report)\n\n")
        f.write(f"- **Mô hình**: {'Mock Synthetic Recognizer' if mock_mode else recognizer.model_name}\n")
        f.write(f"- **Phiên bản Embedding**: {'mock_v1' if mock_mode else recognizer.model_version}\n")
        f.write(f"- **Tổng số cặp Genuine**: {len(gen_scores_arr)}\n")
        f.write(f"- **Tổng số cặp Impostor**: {len(imp_scores_arr)}\n\n")

        f.write("## 1. Phân Bố Độ Tương Đồng (Cosine Similarity Distribution)\n\n")
        f.write("| Phân bố | Mean | Std | Min | Max |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: |\n")
        f.write(
            f"| **Genuine (Cùng người)** | {roc_results['genuine_mean']:.4f} | {roc_results['genuine_std']:.4f} | {gen_scores_arr.min():.4f} | {gen_scores_arr.max():.4f} |\n"
        )
        f.write(
            f"| **Impostor (Khác người)** | {roc_results['impostor_mean']:.4f} | {roc_results['impostor_std']:.4f} | {imp_scores_arr.min():.4f} | {imp_scores_arr.max():.4f} |\n\n"
        )

        f.write("## 2. Chỉ Số Sai Số & Điểm Cân Bằng (Error Rates & ROC)\n\n")
        f.write(f"- **Equal Error Rate (EER)**: `{roc_results['EER'] * 100.0:.3f}%` tại ngưỡng `{roc_results['EER_threshold']:.4f}`\n")
        for point_name, pt in roc_results["operating_points"].items():
            f.write(
                f"- **{point_name}**: TAR = `{pt['TAR'] * 100.0:.2f}%` (Actual FAR = `{pt['actual_FAR'] * 100.0:.4f}%`) tại Threshold = `{pt['threshold']:.4f}`\n"
            )
        f.write("\n")

        f.write("## 3. Đánh Giá Từng Điều Kiện Kiểm Thử (Degradation Conditions)\n\n")
        f.write("| Điều kiện | Số cặp Genuine | Mean Genuine Sim | Mean Impostor Sim | Recall @ Thresh (%) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---:: |\n")
        for cond, stats in condition_stats.items():
            f.write(
                f"| **{cond}** | {stats['genuine_count']} | {stats['mean_genuine_sim']:.4f} | {stats['mean_impostor_sim']:.4f} | {stats['genuine_recall_pct']:.2f}% |\n"
            )
        f.write("\n")

        f.write("## 4. Tham Số Khuyến Nghị Vận Hành (Recommended Operating Parameters)\n\n")
        f.write(f"- **Ngưỡng nhận diện (Threshold)**: `{recommended_threshold:.4f}`\n")
        f.write(f"- **Khoảng cách tối thiểu (Top-1 vs Top-2 Margin)**: `{recommended_margin:.4f}`\n")
        f.write("- **Quy tắc phán quyết (Decision Rule)**:\n")
        f.write("  ```python\n")
        f.write(f"  if top1_score >= {recommended_threshold:.4f} and (top1_score - top2_score) >= {recommended_margin:.4f}:\n")
        f.write("      status = 'MATCHED'\n")
        f.write(f"  elif top1_score >= {recommended_threshold:.4f} and (top1_score - top2_score) < {recommended_margin:.4f}:\n")
        f.write("      status = 'AMBIGUOUS_MATCH'  # Nghi ngờ hai người tương đồng\n")
        f.write("  else:\n")
        f.write("      status = 'UNKNOWN'  # Từ chối nhận diện\n")
        f.write("  ```\n")

    print(f"Generated benchmark report -> {report_md}")
    return report_md


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Face Recognition Benchmark.")
    parser.add_argument("--pairs", type=str, default="benchmarks/pairs", help="Pairs directory")
    parser.add_argument("--results", type=str, default="benchmarks/results", help="Results directory")
    parser.add_argument("--model", type=str, default=None, help="Path to ArcFace ONNX model")
    parser.add_argument("--mock", action="store_true", help="Run in synthetic mock mode for pipeline verification")
    parser.add_argument("--threshold", type=float, default=None, help="Custom threshold override")
    parser.add_argument("--margin", type=float, default=None, help="Custom margin override")
    args = parser.parse_args()

    run_benchmark(
        pairs_dir=Path(args.pairs),
        results_dir=Path(args.results),
        model_path=Path(args.model) if args.model else None,
        mock_mode=args.mock,
        threshold_override=args.threshold,
        margin_override=args.margin,
    )
