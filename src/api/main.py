import os
from typing import Any, Callable

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
    """Build a fail-closed readiness snapshot from the legacy runtime."""
    gate = ReadinessGate(("scrfd", "arcface", "alignment", "fas", "vector_db", "model_provenance"))
    pipeline_ready = recognition_pipeline is not None
    detail = "initialized" if pipeline_ready else "real_model_asset_or_fingerprint_unavailable"
    for name in ("scrfd", "arcface", "alignment", "fas", "model_provenance"):
        gate.set_status(name, pipeline_ready, detail)
    db_ready, db_detail = probe_postgres_pgvector(db)
    gate.set_status("vector_db", db_ready, db_detail)
    return gate


def build_production_probes(repository: Any, lifecycle: Any) -> dict[str, Callable[[], tuple[bool, str]]]:
    """Return real production probes; test doubles are injected by create_app()."""
    def probe_database() -> tuple[bool, str]:
        return probe_postgres_pgvector(repository)

    def probe_model_registry() -> tuple[bool, str]:
        pipeline = recognition_pipeline
        if pipeline is None:
            return False, "model_registry_not_ready"
        return True, "validated_model_registry"

    def probe_recognition() -> tuple[bool, str]:
        pipeline = recognition_pipeline
        if pipeline is None:
            return False, "recognition_pipeline_not_ready"
        required = (pipeline.detector, pipeline.quality_gate, pipeline.aligner, pipeline.liveness, pipeline.recognizer)
        if any(item is None for item in required):
            return False, "recognition_component_missing"
        return True, "detector_quality_alignment_fas_embedding_ready"

    def probe_mqtt() -> tuple[bool, str]:
        enabled = os.getenv("ACTION_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        if not enabled:
            return True, "action_disabled"
        return False, "mqtt_transport_not_registered"

    return {
        "database": probe_database,
        "model_registry": probe_model_registry,
        "recognition": probe_recognition,
        "mqtt": probe_mqtt,
    }


def create_app() -> FastAPI:
    app = FastAPI(
        title="NguyenCuuFacePython - Recognition Backend & Event Hub",
        version="1.3.0",
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

    return app


app = create_app()
