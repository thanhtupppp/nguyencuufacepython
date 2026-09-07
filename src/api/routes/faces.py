from datetime import datetime, timezone
from typing import Optional

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.api.dependencies import db, recognition_pipeline, verify_api_key
from src.api.websocket_manager import ws_manager

router = APIRouter(dependencies=[Depends(verify_api_key)])

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB


def _validate_image_bytes(data: bytes) -> None:
    """Validates raw image bytes for maximum size and authentic MIME magic bytes."""
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Payload too large. Maximum image size is {MAX_IMAGE_BYTES // (1024 * 1024)}MB.",
        )

    is_jpeg = data.startswith(b"\xff\xd8\xff")
    is_png = data.startswith(b"\x89PNG\r\n\x1a\n")
    is_webp = data.startswith(b"RIFF") and len(data) >= 12 and data[8:12] == b"WEBP"

    if not (is_jpeg or is_png or is_webp):
        raise HTTPException(
            status_code=400,
            detail="Invalid image format. Allowed formats: JPEG, PNG, WEBP.",
        )


def _decode_image(data: bytes) -> np.ndarray:
    import cv2

    _validate_image_bytes(data)
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image upload: failed to decode image")
    return image


def _extract_embedding(image: np.ndarray) -> tuple[np.ndarray, str, float]:
    """Run SCRFD -> quality gate -> alignment -> ArcFace."""
    if recognition_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Recognition model is not configured; install/configure SCRFD + ArcFace ONNX assets and SHA-256.",
        )
    try:
        result = recognition_pipeline.extract_best_face(image)
    except ValueError as exc:
        code = str(exc)
        status = 422 if code in {"NO_FACE_DETECTED", "NO_FACE_PASSED_QUALITY_GATE", "MASK_DETECTED", "OCCLUSION_DETECTED"} else 503
        detail_msg = code
        if code == "MASK_DETECTED":
            detail_msg = "Face mask detected. Please remove mask for enrollment or recognition."
        elif code == "OCCLUSION_DETECTED":
            detail_msg = "Face is occluded (hands/object). Please uncover your face for enrollment or recognition."
        raise HTTPException(status_code=status, detail=detail_msg) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail="Recognition runtime unavailable") from exc
    return result.embedding, result.model_version, result.quality_score


@router.post("/enroll", status_code=201)
async def enroll_face(
    person_id: str = Form(..., min_length=1, max_length=128),
    image: UploadFile = File(...),
    quality_score: float = Form(default=1.0, ge=0.0, le=1.0),
) -> dict:
    if db.get_person(person_id) is None:
        raise HTTPException(status_code=404, detail="person_id not found; create the person first")
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")
    decoded = _decode_image(data)
    embedding, model_version, measured_quality = _extract_embedding(decoded)
    score = min(quality_score, measured_quality)
    embedding_id = db.add_embedding(person_id, embedding, model_version, score, None)
    return {"embedding_id": embedding_id, "person_id": person_id, "model_version": model_version, "quality_score": score}


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
    decision = db.recognize_with_margin(embedding, model_version, threshold, margin)
    db.log_access(decision.person_id, device_id, decision.similarity, decision.margin, decision.status)

    # Broadcast real-time recognition event to WebSocket subscribers
    person_record = db.get_person(decision.person_id) if decision.person_id else None
    event_payload = {
        "event_type": "ACCESS_EVENT",
        "person_id": decision.person_id,
        "name": person_record.get("name") if person_record else None,
        "status": decision.status,
        "similarity": decision.similarity,
        "margin": decision.margin,
        "device_id": device_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    await ws_manager.broadcast(event_payload)

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
    if db.get_person(person_id) is None:
        raise HTTPException(status_code=404, detail="person_id not found")
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty image upload")
    decoded = _decode_image(data)
    embedding, extracted_version, _ = _extract_embedding(decoded)
    if extracted_version != model_version:
        raise HTTPException(status_code=409, detail="Requested model_version does not match loaded model")
    candidates = db.search_top_k(embedding, model_version=model_version, top_k=32)
    candidate = next((c for c in candidates if c.person_id == person_id), None)
    similarity = candidate.similarity if candidate else -1.0
    return {"verified": similarity >= threshold, "person_id": person_id, "similarity": similarity, "threshold": threshold, "model_version": model_version}
