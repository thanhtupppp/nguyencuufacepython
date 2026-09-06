"""
Mathematical and statistical metrics for face recognition evaluation:
- Cosine similarity
- FAR (False Accept Rate), FRR (False Reject Rate), TAR (True Accept Rate)
- EER (Equal Error Rate) and ROC curve computation
- Top-1 Accuracy and Top-2 Margin decision simulation
"""

from typing import Any, Optional
import numpy as np
import pandas as pd


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between two sets of embeddings.
    If 1D arrays: returns scalar float.
    If 2D arrays (N, D) and (M, D): returns (N, M) matrix.
    """
    if emb1.ndim == 1 and emb2.ndim == 1:
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    if emb1.ndim == 1:
        emb1 = emb1.reshape(1, -1)
    if emb2.ndim == 1:
        emb2 = emb2.reshape(1, -1)

    norm1 = np.linalg.norm(emb1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(emb2, axis=1, keepdims=True)
    norm1 = np.maximum(norm1, 1e-10)
    norm2 = np.maximum(norm2, 1e-10)

    emb1_norm = emb1 / norm1
    emb2_norm = emb2 / norm2
    return np.dot(emb1_norm, emb2_norm.T)


def compute_roc_and_rates(
    genuine_scores: np.ndarray,
    impostor_scores: np.ndarray,
    num_thresholds: int = 1000,
) -> dict[str, Any]:
    """
    Calculates FAR, FRR, EER, and TAR @ fixed FAR operating points.

    :param genuine_scores: 1D array of similarity scores for same-person pairs
    :param impostor_scores: 1D array of similarity scores for different-person pairs
    :param num_thresholds: Number of threshold evaluation points
    :return: Dictionary containing metrics and curves
    """
    genuine = np.asarray(genuine_scores, dtype=np.float64)
    impostor = np.asarray(impostor_scores, dtype=np.float64)

    assert len(genuine) > 0, "genuine_scores must not be empty"
    assert len(impostor) > 0, "impostor_scores must not be empty"

    min_val = min(genuine.min(), impostor.min()) - 0.01
    max_val = max(genuine.max(), impostor.max()) + 0.01
    thresholds = np.linspace(min_val, max_val, num_thresholds)

    far_list = []
    frr_list = []
    tar_list = []

    total_gen = len(genuine)
    total_imp = len(impostor)

    # Sort arrays for fast binary search
    sorted_gen = np.sort(genuine)
    sorted_imp = np.sort(impostor)

    for t in thresholds:
        # Impostors with score >= t -> False Accept
        fa_count = total_imp - np.searchsorted(sorted_imp, t, side="left")
        far = fa_count / total_imp

        # Genuine with score < t -> False Reject
        fr_count = np.searchsorted(sorted_gen, t, side="left")
        frr = fr_count / total_gen

        tar = 1.0 - frr

        far_list.append(far)
        frr_list.append(frr)
        tar_list.append(tar)

    far_arr = np.array(far_list)
    frr_arr = np.array(frr_list)
    tar_arr = np.array(tar_list)

    # Find Equal Error Rate (EER) where |FAR - FRR| is minimal
    eer_idx = np.argmin(np.abs(far_arr - frr_arr))
    eer = float((far_arr[eer_idx] + frr_arr[eer_idx]) / 2.0)
    eer_threshold = float(thresholds[eer_idx])

    # Calculate TAR @ fixed FAR targets: 1%, 0.1%, 0.01%
    tar_at_far_targets = {}
    for target_far in [0.01, 0.001, 0.0001]:
        # Find index with largest FAR <= target_far
        valid_indices = np.where(far_arr <= target_far)[0]
        if len(valid_indices) > 0:
            best_idx = valid_indices[0]
            tar_val = float(tar_arr[best_idx])
            t_val = float(thresholds[best_idx])
        else:
            tar_val = float(tar_arr[-1])
            t_val = float(thresholds[-1])
        tar_at_far_targets[f"TAR@FAR={target_far}"] = {
            "TAR": tar_val,
            "threshold": t_val,
            "actual_FAR": float(far_arr[best_idx if len(valid_indices) > 0 else -1]),
        }

    return {
        "thresholds": thresholds,
        "FAR": far_arr,
        "FRR": frr_arr,
        "TAR": tar_arr,
        "EER": eer,
        "EER_threshold": eer_threshold,
        "operating_points": tar_at_far_targets,
        "genuine_mean": float(np.mean(genuine)),
        "genuine_std": float(np.std(genuine)),
        "impostor_mean": float(np.mean(impostor)),
        "impostor_std": float(np.std(impostor)),
    }


def evaluate_gallery_probe(
    gallery_embeddings: dict[str, list[np.ndarray]],
    probe_samples: list[dict[str, Any]],
    threshold: float,
    margin: float = 0.0,
) -> dict[str, Any]:
    """
    Evaluates 1:N recognition performance on Gallery vs Probe with Dual-Threshold rule:
    Match if: best_score >= threshold AND (best_score - second_best_score) >= margin.

    :param gallery_embeddings: Mapping person_id -> list of embeddings
    :param probe_samples: List of dicts with keys: {'person_id', 'condition', 'embedding'}
    :param threshold: Acceptance threshold
    :param margin: Minimum required gap between Top 1 and Top 2 candidate
    :return: Evaluation metrics dictionary
    """
    # Flatten gallery into matrix and label list
    gallery_vectors = []
    gallery_ids = []
    for pid, emb_list in gallery_embeddings.items():
        for emb in emb_list:
            gallery_vectors.append(emb)
            gallery_ids.append(pid)

    gallery_matrix = np.vstack(gallery_vectors)  # (G, D)
    gallery_ids = np.array(gallery_ids)

    total_probes = len(probe_samples)
    correct_matches = 0
    false_matches = 0
    unknown_rejects = 0
    ambiguous_rejects = 0

    condition_stats: dict[str, dict[str, int]] = {}
    margins_recorded = []

    for probe in probe_samples:
        true_pid = probe["person_id"]
        cond = probe.get("condition", "general")
        p_emb = probe["embedding"]

        if cond not in condition_stats:
            condition_stats[cond] = {"total": 0, "correct": 0, "false_match": 0, "rejected": 0, "ambiguous": 0}
        condition_stats[cond]["total"] += 1

        # Calculate cosine similarities against all gallery representations
        sims = np.dot(gallery_matrix, p_emb)  # Assuming both are L2-normalized

        # Aggregate max similarity per person_id
        unique_pids = np.unique(gallery_ids)
        pid_scores = []
        for uid in unique_pids:
            mask = (gallery_ids == uid)
            pid_scores.append((uid, float(np.max(sims[mask]))))

        # Sort candidate persons by score descending
        pid_scores.sort(key=lambda x: x[1], reverse=True)

        top1_id, top1_score = pid_scores[0]
        top2_id, top2_score = pid_scores[1] if len(pid_scores) > 1 else ("None", -1.0)
        curr_margin = top1_score - top2_score
        margins_recorded.append(curr_margin)

        # Dual-Threshold Decision Engine
        if top1_score < threshold:
            # Below threshold -> Rejected as Unknown
            unknown_rejects += 1
            condition_stats[cond]["rejected"] += 1
        elif curr_margin < margin:
            # Top 1 and Top 2 are too close -> Ambiguous
            ambiguous_rejects += 1
            condition_stats[cond]["ambiguous"] += 1
        else:
            # Accepted match
            if top1_id == true_pid:
                correct_matches += 1
                condition_stats[cond]["correct"] += 1
            else:
                false_matches += 1
                condition_stats[cond]["false_match"] += 1

    top1_acc = (correct_matches / total_probes) * 100.0 if total_probes > 0 else 0.0
    false_match_rate = (false_matches / total_probes) * 100.0 if total_probes > 0 else 0.0

    return {
        "total_probes": total_probes,
        "correct_matches": correct_matches,
        "false_matches": false_matches,
        "unknown_rejects": unknown_rejects,
        "ambiguous_rejects": ambiguous_rejects,
        "top1_accuracy_pct": top1_acc,
        "false_match_rate_pct": false_match_rate,
        "mean_margin": float(np.mean(margins_recorded)) if margins_recorded else 0.0,
        "condition_breakdown": condition_stats,
    }
