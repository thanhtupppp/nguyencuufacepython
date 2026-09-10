import os
from fastapi import FastAPI, Response, status
from fastapi.middleware.cors import CORSMiddleware

from .routes.persons import router as persons_router
from .routes.faces import router as faces_router
from .routes.events import router as events_router
from .routes.dispenser import router as dispenser_router
from .dependencies import db, recognition_pipeline
from .readiness import ReadinessGate
from src.database.readiness import probe_postgres_pgvector


def build_readiness_gate() -> ReadinessGate:
    """Build a fail-closed readiness snapshot from required runtime dependencies."""
    gate = ReadinessGate((
        "scrfd", "arcface", "alignment", "fas", "vector_db", "model_provenance"
    ))
    pipeline_ready = recognition_pipeline is not None
    detail = "initialized" if pipeline_ready else "real_model_asset_or_fingerprint_unavailable"
    for name in ("scrfd", "arcface", "alignment", "fas", "model_provenance"):
        gate.set_status(name, pipeline_ready, detail)

    db_ready, db_detail = probe_postgres_pgvector(db)
    gate.set_status("vector_db", db_ready, db_detail)
    return gate


def create_app() -> FastAPI:
    app = FastAPI(
        title="NguyenCuuFacePython - Recognition Backend & Event Hub",
        version="1.2.0",
        description="Production Face Recognition Backend: Identity Management, 1:N Recognition, 1:1 Verification, Real-time WebSocket Event Stream, and Smart Anti-Abuse Dispenser.",
    )
    cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(CORSMiddleware, allow_origins=cors_origins or ["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.include_router(persons_router, prefix="/api/v1/persons", tags=["persons"])
    app.include_router(faces_router, prefix="/api/v1/faces", tags=["faces"])
    app.include_router(dispenser_router, prefix="/api/v1/dispenser", tags=["dispenser"])
    app.include_router(events_router, tags=["events"])

    @app.get("/healthz", tags=["health"])
    def healthz() -> dict[str, str]:
        return {"status": "ok", "recognition": "ready" if recognition_pipeline is not None else "blocked"}

    @app.get("/readyz", tags=["health"])
    def readyz(response: Response) -> dict[str, object]:
        gate = build_readiness_gate()
        if not gate.ready:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return gate.as_dict()

    @app.get("/v1/models/status", tags=["health"])
    def model_status() -> dict[str, str]:
        if recognition_pipeline is None:
            return {"status": "blocked", "reason": "real_model_asset_or_fingerprint_unavailable"}
        return {"status": "ready"}

    return app


app = create_app()
