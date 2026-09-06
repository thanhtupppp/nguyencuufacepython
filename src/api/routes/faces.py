"""Face enrollment, recognition and verification endpoints.

The production pipeline is intentionally dependency-injected through module-level
factories so the API can later be wired to SCRFD + alignment + ArcFace without
changing the HTTP contract.
"""

from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.database.client import DatabaseClient

router = APIRouter()
_db = DatabaseClient()


class VerifyRequest(BaseModel):
    person_id: str = Field(min_length=1)
    threshold: float = Field(default=0.60, ge=-1.0, le=1.0)
    margin: float = Field(default=0.08, ge=0.0, le=2.0)


def _decode_image(data: bytes) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image upload")
    return image


def _extract_embedding(image: np.ndarray) -> tuple[np.ndarray, str, float]:
    """Production hook: SCRFD -> 5-point alignment -> ArcFace.

    Kept explicit so a future model registry can replace this function while the
    API schema remains stable. Refuse recognition/enrollment until a real model
    is configured instead of silently generating invalid embeddings.
    """
    raise HTTPException(
        status_code=503,
        detail="Recognition model is not configured; install/configure SCRFD + ArcFace ONNX assets.",
    )


@router.post("/enroll", status_code=201)
async def enroll_face(
    person_id: str = Form(..., min_length=1, max_length=128),
    image: UploadFile = File(...),
    quality_score: float = Form(default=1.0, ge=0.0, le=1.0),
) -> dict:
    person = _db.get_person(person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="person_id not found; create the person first")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")
    decoded = _decode_image(data)
    embedding, model_version, measured_quality = _extract_embedding(decoded)
    embedding_id = _db.add_embedding(
        person_id=person_id,
        embedding=embedding,
        model_version=model_version,
        quality_score=min(quality_score, measured_quality),
        source_image_path=None,
    )
    return {
        "embedding_id": embedding_id,
        "person_id": person_id,
        "model_version": model_version,
        "quality_score": min(quality_score, measured_quality),
    }


@router.post("/recognize")
async def recognize_face(
    image: UploadFile = File(...),
    model_version: str = Form(default="arcface_v1"),
    threshold: float = Form(default=0.60, ge=-1.0, le=1.0),
    margin: float = Form(default=0.08, ge=0.0, le=2.0),
    device_id: Optional[str] = Form(default=None),
) -> dict:
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")
    decoded = _decode_image(data)
    embedding, extracted_version, _ = _extract_embedding(decoded)
    if extracted_version != model_version:
        raise HTTPException(status_code=409, detail="Requested model_version does not match loaded model")

    decision = _db.recognize_with_margin(
        embedding,
        model_version=model_version,
        threshold=threshold,
        margin=margin,
    )
    _db.log_access(
        person_id=decision.person_id,
        device_id=device_id,
        similarity=decision.similarity,
        margin=decision.margin,
        status=decision.status,
    )
    return {
        "status": decision.status,
        "person_id": decision.person_id,
        "similarity": decision.similarity,
        "second_similarity": decision.second_similarity,
        "margin": decision.margin,
        "candidates": [c.__dict__ for c in decision.top_candidates],
        "model_version": model_version,
    }


@router.post("/verify")
async def verify_face(
    image: UploadFile = File(...),
    person_id: str = Form(..., min_length=1, max_length=128),
    model_version: str = Form(default="arcface_v1"),
    threshold: float = Form(default=0.60, ge=-1.0, le=1.0),
) -> dict:
    if _db.get_person(person_id) is None:
        raise HTTPException(status_code=404, detail="person_id not found")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")
    decoded = _decode_image(data)
    embedding, extracted_version, _ = _extract_embedding(decoded)
    if extracted_version != model_version:
        raise HTTPException(status_code=409, detail="Requested model_version does not match loaded model")

    candidates = _db.search_top_k(embedding, model_version=model_version, top_k=32)
    candidate = next((c for c in candidates if c.person_id == person_id), None)
    similarity = candidate.similarity if candidate else -1.0
    return {
        "verified": similarity >= threshold,
        "person_id": person_id,
        "similarity": similarity,
        "threshold": threshold,
        "model_version": model_version,
    }
