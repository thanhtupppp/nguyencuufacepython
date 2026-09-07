import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes.persons import router as persons_router
from .routes.faces import router as faces_router
from .routes.events import router as events_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="NguyenCuuFacePython - Recognition Backend & Event Hub",
        version="1.2.0",
        description="Production Face Recognition Backend: Identity Management, 1:N Recognition, 1:1 Verification, and Real-time WebSocket Event Stream.",
    )

    # Enable CORS for web dashboards and mobile clients
    cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(persons_router, prefix="/api/v1/persons", tags=["persons"])
    app.include_router(faces_router, prefix="/api/v1/faces", tags=["faces"])
    app.include_router(events_router, tags=["events"])

    @app.get("/healthz", tags=["health"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
