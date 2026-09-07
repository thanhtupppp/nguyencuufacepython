"""
SCRFD Face Detector module using ONNX Runtime.
Detects face bounding boxes and 5 facial landmarks (left eye, right eye, nose, left mouth, right mouth).
"""

import hashlib
import os
from pathlib import Path
from typing import Any, Optional, Union
import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def distance2bbox(points: np.ndarray, distance: np.ndarray, max_shape: Optional[tuple] = None) -> np.ndarray:
    """Decodes bounding box from anchor centers and predicted distances."""
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points: np.ndarray, distance: np.ndarray, max_shape: Optional[tuple] = None) -> np.ndarray:
    """Decodes 5 keypoints (landmarks) from anchor centers and predicted offsets."""
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, 0] + distance[:, i]
        py = points[:, 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def nms_cpu(bboxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4) -> list[int]:
    """Pure numpy/Python Non-Maximum Suppression."""
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


class SCRFDDetector:
    """
    SCRFD Face Detector supporting multiple backbones (500M, 2.5G, 10G).
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        input_size: tuple[int, int] = (640, 640),
        providers: Optional[list[str]] = None,
        expected_sha256: Optional[str] = None,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.expected_sha256 = expected_sha256
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.session: Any = None

        if providers is None:
            available = ort.get_available_providers() if ort else []
            self.providers = [
                p
                for p in ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
                if p in available
            ]
            if not self.providers and ort:
                self.providers = ["CPUExecutionProvider"]
        else:
            self.providers = providers

        self.fmc = 3  # Feature map count (typically 3 for strides 8, 16, 32)
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True

        if self.model_path and self.model_path.exists():
            if self.expected_sha256:
                self._verify_sha256(self.model_path, self.expected_sha256)
            self._load_model()

    @staticmethod
    def _verify_sha256(path: Path, expected: str) -> None:
        """Verifies SCRFD model file checksum against expected SHA-256 hash."""
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual.lower() != expected.lower():
            raise ValueError(
                f"Model SHA-256 fingerprint mismatch for {path}: expected {expected}, got {actual}"
            )

    def _load_model(self) -> None:
        if ort is None:
            raise ImportError("onnxruntime is required for SCRFDDetector")

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Cap CPU threads to prevent thrashing high-core CPUs (e.g. 48 cores)
        session_options.intra_op_num_threads = min(4, os.cpu_count() or 4)
        session_options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(self.model_path), sess_options=session_options, providers=self.providers
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        """Resizes with preserved aspect ratio and pads to self.input_size."""
        h, w = img.shape[:2]
        target_w, target_h = self.input_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        blob = cv2.dnn.blobFromImage(
            padded, 1.0 / 128.0, (target_w, target_h), (127.5, 127.5, 127.5), swapRB=True
        )
        return blob, scale

    def detect(self, img: np.ndarray) -> list[dict]:
        """
        Detects faces in input image.

        :param img: BGR numpy image
        :return: List of dicts with keys:
                 - 'bbox': [x1, y1, x2, y2]
                 - 'score': float confidence
                 - 'landmarks': [(x1, y1), ..., (x5, y5)] 5 keypoints
        """
        if self.session is None:
            raise RuntimeError("Model session not initialized.")

        blob, scale = self.preprocess(img)
        net_outs = self.session.run(self.output_names, {self.input_name: blob})

        # Process multi-scale heads
        scores_list = []
        bboxes_list = []
        kps_list = []

        input_h, input_w = self.input_size

        for idx, stride in enumerate(self._feat_stride_fpn):
            score = net_outs[idx]
            bbox = net_outs[idx + self.fmc] * stride
            kps = net_outs[idx + self.fmc * 2] * stride if len(net_outs) > self.fmc * 2 else None

            height, width = input_h // stride, input_w // stride
            anchor_centers = np.stack(list(np.mgrid[:height, :width][::-1]), axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if self._num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))

            pos_inds = np.where(score >= self.conf_threshold)[0]
            if len(pos_inds) == 0:
                continue

            bboxes = distance2bbox(anchor_centers, bbox)
            scores_list.append(score[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            if kps is not None:
                kpss = distance2kps(anchor_centers, kps)
                kps_list.append(kpss[pos_inds])

        if not scores_list:
            return []

        scores = np.vstack(scores_list).ravel()
        bboxes = np.vstack(bboxes_list)
        all_kpss: Optional[np.ndarray] = np.vstack(kps_list) if kps_list else None

        # Re-scale back to original image size
        bboxes[:, :4] /= scale
        if all_kpss is not None:
            all_kpss /= scale

        keep = nms_cpu(bboxes, scores, self.nms_threshold)

        results = []
        for k in keep:
            landmarks = None
            if all_kpss is not None:
                landmarks = all_kpss[k].reshape(5, 2)
            results.append({
                "bbox": bboxes[k].tolist(),
                "score": float(scores[k]),
                "landmarks": landmarks,
            })
        return results
