"""
Server launcher for NguyenCuuFacePython FastAPI Recognition Backend & WebSocket Event Hub.
"""

import argparse
import os
from pathlib import Path
import sys

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NguyenCuuFacePython - FastAPI Recognition Backend & WebSocket Hub"
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change (development)")
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes (default: 1)")
    args = parser.parse_args()

    # Automatic fallback configuration for local checkpoint models
    chk_arcface = Path("models/checkpoints/w600k_r50.onnx")
    chk_scrfd = Path("models/checkpoints/det_10g.onnx")

    if chk_arcface.is_file() and not os.getenv("ARCFACE_MODEL_PATH"):
        os.environ["ARCFACE_MODEL_PATH"] = str(chk_arcface)
        if not os.getenv("ARCFACE_MODEL_SHA256"):
            os.environ["ARCFACE_MODEL_SHA256"] = "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43"

    if chk_scrfd.is_file() and not os.getenv("SCRFD_MODEL_PATH"):
        os.environ["SCRFD_MODEL_PATH"] = str(chk_scrfd)

    if sys.platform == "win32":
        try:
            reconfig_out = getattr(sys.stdout, "reconfigure", None)
            if callable(reconfig_out):
                reconfig_out(encoding="utf-8")
            reconfig_err = getattr(sys.stderr, "reconfigure", None)
            if callable(reconfig_err):
                reconfig_err(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("NguyenCuuFacePython - Khoi dong FastAPI Backend & WebSocket Hub")
    print("=" * 70)
    print(f" * Server address:         http://{args.host}:{args.port}")
    print(f" * Swagger UI (API Docs):  http://localhost:{args.port}/docs")
    print(f" * Redoc documentation:    http://localhost:{args.port}/redoc")
    print(f" * WebSocket Event Stream: ws://localhost:{args.port}/ws/v1/events")
    print(" * Quan ly danh tinh:      /api/v1/persons")
    print(" * Nhan dien & Xac thuc:   /api/v1/faces (enroll, recognize, verify)")
    print(" * Ban su kien Edge:       /api/v1/events")
    print("-" * 70)

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
        app_dir=str(PROJECT_ROOT),
    )


if __name__ == "__main__":
    main()
