"""Common anti-spoof benchmark runner.

The runner is deliberately fail-closed: it requires a verified artifact manifest
before loading a model. Dataset adapters are kept explicit so preprocessing and
labels cannot silently drift between candidates.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import time
from dataclasses import dataclass

import cv2
import numpy as np
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

from verify_fas_manifest import sha256


@dataclass
class Sample:
    path: pathlib.Path
    label: int  # 0=real, 1=attack


def load_manifest(path: pathlib.Path, artifact_dir: pathlib.Path) -> dict:
    data = yaml.safe_load(path.read_text())
    for item in data.get("artifacts", []):
        expected = item.get("sha256", "")
        if not expected or expected.startswith("REPLACE_WITH_"):
            raise RuntimeError(f"{item['id']}: SHA-256 is not pinned")
        filename = pathlib.PurePosixPath(item["upstream"]["path"]).name
        artifact = artifact_dir / filename
        if not artifact.is_file():
            raise RuntimeError(f"{item['id']}: artifact missing: {artifact}")
        actual = sha256(artifact)
        if actual.lower() != expected.lower():
            raise RuntimeError(f"{item['id']}: SHA-256 mismatch: {actual}")
    return data


def read_samples(root: pathlib.Path) -> list[Sample]:
    # Explicit directory contract: real/ and attack/ contain image files.
    samples: list[Sample] = []
    for label, name in ((0, "real"), (1, "attack")):
        directory = root / name
        if not directory.is_dir():
            raise RuntimeError(f"missing evaluation directory: {directory}")
        for path in sorted(directory.rglob("*")):
            if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                samples.append(Sample(path, label))
    if not samples:
        raise RuntimeError("no evaluation images found")
    return samples


def prepare_128_rgb(image: np.ndarray) -> np.ndarray:
    # Placeholder for the project-locked face crop/alignment adapter.
    # It intentionally does not invent a detector/alignment policy.
    resized = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


def load_onnx(path: pathlib.Path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def score(session, image: np.ndarray) -> float:
    tensor = image.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    inp = session.get_inputs()[0]
    output = session.run(None, {inp.name: tensor})[0]
    flat = np.asarray(output).reshape(-1)
    if flat.size < 2:
        raise RuntimeError("FAS output must expose at least two scores")
    # Project convention: index 1 is attack/spoof score.
    return float(flat[1])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    parser.add_argument("--dataset", type=pathlib.Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest, args.artifact_dir)
    item = next((x for x in manifest["artifacts"] if x["id"] == args.artifact_id), None)
    if item is None:
        raise RuntimeError(f"unknown artifact id: {args.artifact_id}")
    artifact = args.artifact_dir / pathlib.PurePosixPath(item["upstream"]["path"]).name
    samples = read_samples(args.dataset)
    session = load_onnx(artifact)

    y_true: list[int] = []
    y_score: list[float] = []
    latencies_ms: list[float] = []
    for sample in samples:
        image = cv2.imread(str(sample.path))
        if image is None:
            raise RuntimeError(f"cannot decode image: {sample.path}")
        prepared = prepare_128_rgb(image)
        start = time.perf_counter()
        attack_score = score(session, prepared)
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        y_true.append(sample.label)
        y_score.append(attack_score)

    auc = float(roc_auc_score(y_true, y_score))
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    eer_idx = int(np.argmin(np.abs(fpr - fnr)))
    result = {
        "artifact_id": args.artifact_id,
        "artifact_sha256": sha256(artifact),
        "samples": len(samples),
        "attack_samples": int(sum(y_true)),
        "real_samples": int(len(y_true) - sum(y_true)),
        "roc_auc": auc,
        "eer": float((fpr[eer_idx] + fnr[eer_idx]) / 2.0),
        "eer_threshold": float(thresholds[eer_idx]),
        "latency_ms_p50": float(np.percentile(latencies_ms, 50)),
        "latency_ms_p95": float(np.percentile(latencies_ms, 95)),
        "preprocessing": "TEMPORARY_128_RGB_RESIZE_ONLY__NOT_PRODUCTION_ALIGNMENT",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
