"""FastAPI application entrypoint for the face recognition service."""

from fastapi import FastAPI

from .routes.persons import router as persons_router
from .routes.faces import router as faces_router


def create_app() -> FastAPI:
    app = FastAPI(
        title="Personal Face Recognition API",
        version="1.0.0",
        description="Enrollment, 1:N recognition and 1:1 verification API.",
    )
    app.include_router(persons_router, prefix="/api/v1/persons", tags=["persons"])
    app.include_router(faces_router, prefix="/api/v1/faces", tags=["faces"])

    @app.get("/healthz", tags=["health"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()
