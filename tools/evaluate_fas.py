"""Common anti-spoof benchmark runner.

Fail-closed benchmark runner. Model I/O semantics come from the artifact
manifest; no dtype, normalization, output index, or score direction is
hard-coded in the evaluator.
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


def require_verified(value: object, field: str) -> str:
    text = str(value or "").strip()
    if not text or text.upper().startswith("REQUIRES_") or text.upper().startswith("IMPLEMENTATION_"):
        raise RuntimeError(f"manifest field is not verified: {field}")
    return text


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


def prepare_image(image: np.ndarray, item: dict) -> np.ndarray:
    input_spec = item.get("input", {})
    size = input_spec.get("size")
    if not isinstance(size, list) or len(size) != 2:
        raise RuntimeError("manifest input.size must be [width, height]")
    color = require_verified(input_spec.get("color"), "input.color").upper()
    dtype = require_verified(input_spec.get("dtype"), "input.dtype").lower()
    normalization = require_verified(input_spec.get("normalization"), "input.normalization").lower()

    resized = cv2.resize(image, (int(size[0]), int(size[1])), interpolation=cv2.INTER_LINEAR)
    if color == "RGB":
        prepared = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    elif color == "BGR":
        prepared = resized
    else:
        raise RuntimeError(f"unsupported input color: {color}")

    if normalization == "0_1":
        array = prepared.astype(np.float32) / 255.0
    elif normalization == "-1_1":
        array = prepared.astype(np.float32) / 127.5 - 1.0
    elif normalization == "none":
        array = prepared.astype(np.float32)
    else:
        raise RuntimeError(f"unsupported/unsafe normalization: {normalization}")

    if dtype == "float32":
        array = array.astype(np.float32)
    elif dtype == "uint8":
        if normalization != "none":
            raise RuntimeError("uint8 input requires normalization=none")
        array = array.astype(np.uint8)
    else:
        raise RuntimeError(f"unsupported input dtype: {dtype}")
    return array


def load_onnx(path: pathlib.Path):
    import onnxruntime as ort
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def score(session, image: np.ndarray, item: dict) -> float:
    input_spec = item.get("input", {})
    output_spec = item.get("output", {})
    tensor = np.transpose(image, (2, 0, 1))[None, ...]
    inp = session.get_inputs()[0]
    output = np.asarray(session.run(None, {inp.name: tensor})[0]).reshape(-1)
    output_type = require_verified(output_spec.get("type"), "output.type").lower()
    attack_index = int(require_verified(output_spec.get("attack_index"), "output.attack_index"))
    direction = require_verified(output_spec.get("score_direction"), "output.score_direction").lower()
    if output.size <= attack_index:
        raise RuntimeError("output attack_index exceeds model output size")

    if output_type == "logits":
        if output.size != 2:
            raise RuntimeError("logits output currently requires exactly two classes")
        real_index = 1 - attack_index
        raw = float(output[attack_index] - output[real_index])
        return raw if direction == "higher_is_attack" else -raw
    if output_type == "probability":
        raw = float(output[attack_index])
        return raw if direction == "higher_is_attack" else -raw
    raise RuntimeError(f"unsupported output.type: {output_type}")


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
        prepared = prepare_image(image, item)
        start = time.perf_counter()
        attack_score = score(session, prepared, item)
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
        "preprocessing": {
            "size": item["input"]["size"],
            "color": item["input"]["color"],
            "dtype": item["input"]["dtype"],
            "normalization": item["input"]["normalization"],
        },
        "output_contract": item["output"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
