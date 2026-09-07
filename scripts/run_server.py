"""
Server launcher for NguyenCuuFacePython FastAPI Recognition Backend & WebSocket Event Hub.
"""

import argparse
import os
from pathlib import Path
import sys
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

    print("=" * 70)
    print("NguyenCuuFacePython - Khởi động FastAPI Backend & WebSocket Hub")
    print("=" * 70)
    print(f" * Máy chủ đang chạy tại:   http://{args.host}:{args.port}")
    print(f" * Swagger UI (API Docs):  http://localhost:{args.port}/docs")
    print(f" * Redoc:                  http://localhost:{args.port}/redoc")
    print(f" * WebSocket Event Stream: ws://localhost:{args.port}/ws/v1/events")
    print(" * Quản lý danh tính:     /api/v1/persons")
    print(" * Nhận diện & Xác thực:   /api/v1/faces (enroll, recognize, verify)")
    print(" * Bắn sự kiện Edge:       /api/v1/events")
    print("-" * 70)

    uvicorn.run(
        "src.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
