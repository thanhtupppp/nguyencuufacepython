"""
Utility script to download official pretrained ONNX models for:
- SCRFD Face Detectors (SCRFD 500M / 10G)
- ArcFace Recognition Backbones (w600k_r50, glint360k_r50)
- MiniFASNet Anti-Spoofing
"""

import argparse
from pathlib import Path
import urllib.request
import zipfile
import sys


MODELS_CONFIG = {
    "arcface_r50": {
        "filename": "w600k_r50.onnx",
        "url": "https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx",
        "description": "ArcFace ResNet-50 trained on WebFace600K (512D output, ~174MB)",
    },
    "scrfd_10g": {
        "filename": "det_10g.onnx",
        "url": "https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/buffalo_l/det_10g.onnx",
        "description": "SCRFD 10G detector with 5 facial landmarks (~16.9MB)",
    },
}


def download_file(url: str, dest_path: Path) -> bool:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 1000:
        print(f"[OK] Model already exists: {dest_path} ({dest_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return True

    print(f"Downloading from {url} to {dest_path}...")
    try:
        def reporthook(count, block_size, total_size):
            if total_size > 0:
                percent = int(count * block_size * 100 / total_size)
                sys.stdout.write(f"\rProgress: {percent}% ({count * block_size / 1024 / 1024:.1f}/{total_size / 1024 / 1024:.1f} MB)")
                sys.stdout.flush()

        urllib.request.urlretrieve(url, str(dest_path), reporthook=reporthook)
        print("\nDownload complete.")
        return True
    except Exception as e:
        print(f"\n[ERROR] Failed to download {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False


def main():
    parser = argparse.ArgumentParser(description="Download pretrained ONNX models for NguyenCuuFacePython")
    parser.add_argument("--model", type=str, default="all", choices=["all", "arcface_r50", "scrfd_10g", "scrfd_500m"])
    parser.add_argument("--output_dir", type=str, default="models/checkpoints")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    target_models = MODELS_CONFIG.keys() if args.model == "all" else [args.model]

    for m_key in target_models:
        cfg = MODELS_CONFIG[m_key]
        dest = out_dir / cfg["filename"]
        print(f"\nTarget: {m_key} - {cfg['description']}")
        download_file(cfg["url"], dest)


if __name__ == "__main__":
    main()
