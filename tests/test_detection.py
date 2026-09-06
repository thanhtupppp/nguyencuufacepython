"""
Unit tests for SCRFD detection preprocessing and decoding utilities.
"""

import numpy as np
import pytest
from src.detection.scrfd import distance2bbox, distance2kps, nms_cpu


def test_distance2bbox():
    points = np.array([[100.0, 100.0]], dtype=np.float32)
    distance = np.array([[20.0, 20.0, 30.0, 30.0]], dtype=np.float32)
    bboxes = distance2bbox(points, distance)
    # x1 = 100 - 20 = 80, y1 = 100 - 20 = 80, x2 = 100 + 30 = 130, y2 = 100 + 30 = 130
    expected = np.array([[80.0, 80.0, 130.0, 130.0]], dtype=np.float32)
    np.testing.assert_allclose(bboxes, expected)


def test_nms_cpu():
    bboxes = np.array([
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 12.0, 48.0, 48.0],  # High overlap with box 0
        [100.0, 100.0, 150.0, 150.0],  # Completely disjoint box
    ], dtype=np.float32)
    scores = np.array([0.9, 0.7, 0.85], dtype=np.float32)

    keep = nms_cpu(bboxes, scores, iou_threshold=0.5)
    # Should keep box 0 (0.9) and box 2 (0.85), suppressing box 1
    assert keep == [0, 2]
