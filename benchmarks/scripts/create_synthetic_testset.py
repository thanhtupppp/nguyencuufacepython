"""
Generates synthetic calibration face images across 6 degradation conditions
to verify the benchmark harness end-to-end.
"""

from pathlib import Path
import cv2
import numpy as np


def draw_face_canvas(seed_id: int, variant_name: str) -> np.ndarray:
    """Draws a 112x112 synthetic face canvas with identifiable geometric patterns."""
    canvas = np.zeros((112, 112, 3), dtype=np.uint8)
    rng = np.random.RandomState(seed_id)
    base_color = rng.randint(50, 200, size=3).tolist()

    # Head ellipse
    cv2.ellipse(canvas, (56, 56), (36, 48), 0, 0, 360, base_color, -1)
    # Eyes
    cv2.circle(canvas, (38, 51), 6, (255, 255, 255), -1)
    cv2.circle(canvas, (74, 51), 6, (255, 255, 255), -1)
    eye_pupil_color = rng.randint(0, 100, size=3).tolist()
    cv2.circle(canvas, (38, 51), 3, eye_pupil_color, -1)
    cv2.circle(canvas, (74, 51), 3, eye_pupil_color, -1)
    # Nose
    cv2.circle(canvas, (56, 71), 4, (100, 100, 100), -1)
    # Mouth
    cv2.ellipse(canvas, (56, 92), (15, 6), 0, 0, 180, (50, 50, 200), 2)

    # Apply degradation variations
    if variant_name == "frontal":
        pass  # Clean
    elif variant_name == "angle":
        # Slight perspective shift
        pts1 = np.float32([[0, 0], [112, 0], [0, 112], [112, 112]])
        pts2 = np.float32([[15, 5], [105, 0], [15, 105], [105, 112]])
        m = cv2.getPerspectiveTransform(pts1, pts2)
        canvas = cv2.warpPerspective(canvas, m, (112, 112))
    elif variant_name == "low_light":
        # Darken significantly
        canvas = (canvas * 0.35).astype(np.uint8)
    elif variant_name == "blur":
        # Motion / Gaussian blur
        canvas = cv2.GaussianBlur(canvas, (11, 11), 3.0)
    elif variant_name == "partial_occlusion":
        # Add mask over mouth & nose
        cv2.rectangle(canvas, (20, 70), (92, 108), (220, 220, 220), -1)
    elif variant_name == "distance":
        # Downscale and upscale (pixelation)
        small = cv2.resize(canvas, (24, 24), interpolation=cv2.INTER_LINEAR)
        canvas = cv2.resize(small, (112, 112), interpolation=cv2.INTER_NEAREST)

    return canvas


def generate_test_dataset(base_dir: Path, num_persons: int = 5) -> None:
    gallery_dir = base_dir / "gallery"
    probe_dir = base_dir / "probe"
    conditions = ["frontal", "angle", "low_light", "blur", "partial_occlusion", "distance"]

    for i in range(1, num_persons + 1):
        pid = f"person_{i:03d}"
        p_gallery = gallery_dir / pid
        p_gallery.mkdir(parents=True, exist_ok=True)

        # 2 gallery reference images
        for g_idx in [1, 2]:
            img = draw_face_canvas(i * 100 + g_idx, "frontal")
            cv2.imwrite(str(p_gallery / f"{pid}_ref{g_idx}.jpg"), img)

        # 1 probe image per condition
        for cond in conditions:
            p_probe = probe_dir / cond / pid
            p_probe.mkdir(parents=True, exist_ok=True)
            probe_img = draw_face_canvas(i * 100 + 10, cond)
            cv2.imwrite(str(p_probe / f"{pid}_{cond}.jpg"), probe_img)

    print(f"Generated synthetic testset for {num_persons} identities across {len(conditions)} conditions.")


if __name__ == "__main__":
    generate_test_dataset(Path("benchmarks"), num_persons=5)
