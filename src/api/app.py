"""Minimal FastAPI boundary for the face-recognition service."""
from __future__ import annotations

from fastapi import FastAPI


def create_app(*, engine_ready: bool = False) -> FastAPI:
    app = FastAPI(title="NguyenCuuFacePython", version="0.1.0")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> dict[str, object]:
        return {"ready": engine_ready}

    return app


app = create_app()
