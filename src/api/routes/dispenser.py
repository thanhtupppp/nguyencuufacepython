from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Optional
import urllib.parse
import urllib.request
import uuid

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

from src.api.dependencies import db, recognition_pipeline, verify_api_key
from src.api.websocket_manager import ws_manager

router = APIRouter(dependencies=[Depends(verify_api_key)])

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB

_dispenser_config = {
    "cooldown_minutes": 5.0,
    "dismiss_seconds": 3,
    "pulse_ms": 2500,
    "device_id": "dispenser_01",
    "device_name": "Máy Cấp Giấy Vệ Sinh #1",
    "viewfinder_style": "hud",
    "voice_enabled": True,
    "voice_volume": 1.0,
    "voice_rate": 1.0,
    "touchless_enabled": True,
    "touchless_delay": 1.5,
    "welcome_voice_enabled": True,
}


class DispenserConfigModel(BaseModel):
    cooldown_minutes: float = Field(default=5.0, ge=0.1, le=1440.0)
    dismiss_seconds: int = Field(default=3, ge=1, le=60)
    pulse_ms: int = Field(default=2500, ge=500, le=10000)
    device_id: str = Field(default="dispenser_01", max_length=64)
    device_name: str = Field(default="Máy Cấp Giấy Vệ Sinh #1", max_length=128)
    viewfinder_style: str = Field(default="hud", max_length=32)
    voice_enabled: bool = Field(default=True)
    voice_volume: float = Field(default=1.0, ge=0.0, le=1.0)
    voice_rate: float = Field(default=1.0, ge=0.5, le=2.0)
    touchless_enabled: bool = Field(default=True)
    touchless_delay: float = Field(default=1.5, ge=0.5, le=10.0)
    welcome_voice_enabled: bool = Field(default=True)



def _validate_image_bytes(data: bytes) -> None:
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
        raise HTTPException(status_code=400, detail="Failed to decode image data")
    return image


@router.post("/request-paper")
async def request_toilet_paper(
    image: UploadFile = File(...),
    cooldown_minutes: float = Form(default=5.0, ge=0.1, le=1440.0),
    device_id: str = Form(default="dispenser_01"),
    threshold: float = Form(default=0.60, ge=-1.0, le=1.0),
    margin: float = Form(default=0.08, ge=0.0, le=2.0),
) -> dict:
    """
    Core AI Dispenser Logic:
    1. Detect face & validate Quality Gate (Mask, Occlusion, Blur, Spoof).
    2. Search face vector in DB:
       - Unknown face: Auto-enroll as new user and grant paper.
       - Known face: Check last dispense time. If within cooldown_minutes, DENY to prevent abuse.
         If past cooldown, GRANT paper.
    3. Broadcast hardware trigger via WebSocket.
    """
    data = await image.read()
    if not data:
        return {
            "granted": False,
            "status": "EMPTY_IMAGE",
            "message": "Không nhận được hình ảnh từ camera!",
            "cooldown_remaining_seconds": 0,
        }

    try:
        decoded = _decode_image(data)
    except HTTPException as e:
        return {
            "granted": False,
            "status": "INVALID_IMAGE",
            "message": e.detail,
            "cooldown_remaining_seconds": 0,
        }

    # 1. Quality & Face extraction
    if recognition_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Hệ thống AI nhận diện khuôn mặt chưa được khởi tạo đầy đủ.",
        )

    try:
        extract_result = recognition_pipeline.extract_best_face(decoded)
    except ValueError as exc:
        code = str(exc)
        msg_map = {
            "NO_FACE_DETECTED": "Không tìm thấy khuôn mặt! Vui lòng đứng đối diện camera.",
            "NO_FACE_PASSED_QUALITY_GATE": "Ảnh khuôn mặt quá mờ hoặc góc nghiêng quá lớn. Vui lòng nhìn thẳng!",
            "MASK_DETECTED": "Phát hiện khẩu trang! Vui lòng tháo khẩu trang để nhận giấy.",
            "OCCLUSION_DETECTED": "Khuôn mặt bị che khuất (tay che mặt hoặc vật cản). Vui lòng để lộ rõ mặt!",
            "SPOOF_DETECTED": "Cảnh báo bảo mật: Phát hiện hình ảnh giả mạo hoặc màn hình điện thoại!",
        }
        user_msg = msg_map.get(code, f"Chất lượng ảnh không đạt yêu cầu ({code})")
        status_code = code

        # Log failed attempt
        db.log_dispense(
            person_id="ANONYMOUS",
            device_id=device_id,
            status=status_code,
            similarity=0.0,
            cooldown_seconds_remaining=0,
            message=user_msg,
        )

        await ws_manager.broadcast({
            "event_type": "DISPENSER_BLOCKED",
            "status": status_code,
            "message": user_msg,
            "device_id": device_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return {
            "granted": False,
            "status": status_code,
            "message": user_msg,
            "cooldown_remaining_seconds": 0,
        }

    embedding = extract_result.embedding
    model_version = extract_result.model_version
    measured_quality = extract_result.quality_score

    # 2. Vector search in database
    decision = db.recognize_with_margin(embedding, model_version, threshold, margin)
    now = datetime.now(timezone.utc)

    # -----------------------------------------------------------------
    # Case A: KHUÔN MẶT ĐÃ CÓ TRONG CSDL (MATCHED) -> Kiểm tra Cooldown
    # -----------------------------------------------------------------
    if decision.status == "MATCHED" and decision.person_id:
        person_id = decision.person_id
        person_info = db.get_person(person_id) or {}
        person_name = person_info.get("name", person_id)

        last_dispense = db.get_last_successful_dispense(person_id)
        cooldown_seconds = int(cooldown_minutes * 60)

        if last_dispense:
            # Normalize timezone if needed
            if last_dispense.tzinfo is None:
                last_dispense = last_dispense.replace(tzinfo=timezone.utc)
            elapsed_seconds = (now - last_dispense).total_seconds()

            if elapsed_seconds < cooldown_seconds:
                remaining_seconds = int(cooldown_seconds - elapsed_seconds)
                rem_min = remaining_seconds // 60
                rem_sec = remaining_seconds % 60
                time_str = f"{rem_min} phút {rem_sec} giây" if rem_min > 0 else f"{rem_sec} giây"

                denial_msg = (
                    f"Bạn vừa nhận giấy cách đây ít phút. "
                    f"Vui lòng đợi thêm {time_str} để tránh lãng phí giấy vệ sinh!"
                )

                db.log_dispense(
                    person_id=person_id,
                    device_id=device_id,
                    status="COOLDOWN_BLOCKED",
                    similarity=decision.similarity,
                    cooldown_seconds_remaining=remaining_seconds,
                    message=denial_msg,
                    timestamp=now,
                )

                await ws_manager.broadcast({
                    "event_type": "DISPENSER_BLOCKED",
                    "status": "COOLDOWN_BLOCKED",
                    "person_id": person_id,
                    "name": person_name,
                    "cooldown_seconds_remaining": remaining_seconds,
                    "message": denial_msg,
                    "device_id": device_id,
                    "timestamp": now.isoformat(),
                })

                return {
                    "granted": False,
                    "status": "COOLDOWN_BLOCKED",
                    "person_id": person_id,
                    "name": person_name,
                    "similarity": decision.similarity,
                    "cooldown_remaining_seconds": remaining_seconds,
                    "last_dispensed_at": last_dispense.isoformat(),
                    "message": denial_msg,
                }

        # Cooldown passed or first time taking paper: GRANT
        success_msg = f"Xác thực thành công! Xin chào {person_name}. Đang cấp giấy vệ sinh..."
        db.log_dispense(
            person_id=person_id,
            device_id=device_id,
            status="GRANTED",
            similarity=decision.similarity,
            cooldown_seconds_remaining=0,
            message=success_msg,
            timestamp=now,
        )

        await ws_manager.broadcast({
            "event_type": "DISPENSER_TRIGGER",
            "status": "GRANTED",
            "person_id": person_id,
            "name": person_name,
            "pulse_ms": 2500,
            "device_id": device_id,
            "timestamp": now.isoformat(),
        })

        return {
            "granted": True,
            "status": "GRANTED",
            "person_id": person_id,
            "name": person_name,
            "is_new_user": False,
            "similarity": decision.similarity,
            "cooldown_remaining_seconds": 0,
            "dispensed_at": now.isoformat(),
            "pulse_ms": 2500,
            "message": success_msg,
        }

    # -----------------------------------------------------------------
    # Case B: KHUÔN MẶT CHƯA CÓ TRONG CSDL (UNKNOWN) -> Tự động đăng ký
    # -----------------------------------------------------------------
    short_uuid = uuid.uuid4().hex[:6].upper()
    new_person_id = f"USER_{short_uuid}"
    new_name = f"Khách #{short_uuid}"

    # Auto-enroll new person into DB
    db.create_person(
        person_id=new_person_id,
        name=new_name,
        department="Khu công cộng",
        role="Khách",
        metadata={"auto_enrolled": True, "created_at": now.isoformat()},
    )
    db.add_embedding(
        person_id=new_person_id,
        embedding=embedding,
        model_version=model_version,
        quality_score=measured_quality,
        source_image_path=None,
    )

    new_user_msg = f"Chào mừng người dùng mới ({new_name})! Đã ghi nhận khuôn mặt và đang cấp giấy..."
    db.log_dispense(
        person_id=new_person_id,
        device_id=device_id,
        status="NEW_USER_GRANTED",
        similarity=1.0,
        cooldown_seconds_remaining=0,
        message=new_user_msg,
        timestamp=now,
    )

    await ws_manager.broadcast({
        "event_type": "DISPENSER_TRIGGER",
        "status": "NEW_USER_GRANTED",
        "person_id": new_person_id,
        "name": new_name,
        "is_new_user": True,
        "pulse_ms": 2500,
        "device_id": device_id,
        "timestamp": now.isoformat(),
    })

    return {
        "granted": True,
        "status": "NEW_USER_GRANTED",
        "person_id": new_person_id,
        "name": new_name,
        "is_new_user": True,
        "similarity": 1.0,
        "cooldown_remaining_seconds": 0,
        "dispensed_at": now.isoformat(),
        "pulse_ms": 2500,
        "message": new_user_msg,
    }


@router.get("/stats")
async def get_dispenser_stats(device_id: Optional[str] = None) -> dict:
    """Returns dispenser metrics: total rolls dispensed, abuse attempts blocked, unique users."""
    stats = db.get_dispense_stats(device_id=device_id)
    return stats


@router.get("/logs")
async def get_dispenser_logs(limit: int = 50) -> list[dict]:
    """Returns recent dispense history."""
    return db.get_dispense_logs(limit=limit)


@router.delete("/logs")
async def clear_dispenser_logs(clear_test_users: bool = Query(default=True)) -> dict:
    """Clears all dispense history logs and optionally auto-enrolled guest test users."""
    deleted_count = db.clear_dispense_logs(clear_auto_enrolled_test_users=clear_test_users)
    stats = db.get_dispense_stats()

    await ws_manager.broadcast({
        "event_type": "DISPENSER_LOGS_CLEARED",
        "deleted_count": deleted_count,
        "stats": stats,
    })

    return {
        "status": "ok",
        "deleted_count": deleted_count,
        "stats": stats,
        "message": "Đã xóa toàn bộ nhật ký cấp giấy và đặt lại dữ liệu thử nghiệm.",
    }


@router.get("/config")
async def get_dispenser_config() -> dict:
    """Returns current dispenser settings."""
    return _dispenser_config


@router.post("/config")
async def update_dispenser_config(config: DispenserConfigModel) -> dict:
    """Updates dispenser settings (cooldown, pulse, device name)."""
    _dispenser_config["cooldown_minutes"] = config.cooldown_minutes
    _dispenser_config["dismiss_seconds"] = config.dismiss_seconds
    _dispenser_config["pulse_ms"] = config.pulse_ms
    _dispenser_config["device_id"] = config.device_id
    _dispenser_config["device_name"] = config.device_name
    _dispenser_config["viewfinder_style"] = config.viewfinder_style
    _dispenser_config["voice_enabled"] = config.voice_enabled
    _dispenser_config["voice_volume"] = config.voice_volume
    _dispenser_config["voice_rate"] = config.voice_rate
    _dispenser_config["touchless_enabled"] = config.touchless_enabled
    _dispenser_config["touchless_delay"] = config.touchless_delay
    _dispenser_config["welcome_voice_enabled"] = config.welcome_voice_enabled
    return {"status": "ok", "config": _dispenser_config}


@router.post("/presence-check")
async def check_presence(image: UploadFile = File(...)) -> dict:
    """Fast presence check for Kiosk with strict anti-false-positive filtering.
    
    Filters out background clutter, tiny distant faces, and non-face objects:
    - Confidence threshold >= 0.70
    - Face bounding box >= 45x45 px (must occupy significant portion of kiosk frame)
    - Face center must be within central interactive viewing area
    - Aspect ratio must match human face proportions (0.75 <= h/w <= 2.2)
    - Facial eye distance validation
    """
    if recognition_pipeline is None:
        raise HTTPException(status_code=503, detail="Recognition service is unavailable")

    data = await image.read()
    img_bgr = _decode_image(data)
    h_img, w_img = img_bgr.shape[:2]

    try:
        if hasattr(recognition_pipeline, "detector") and recognition_pipeline.detector:
            faces = recognition_pipeline.detector.detect(img_bgr)
            valid_faces = []
            for f in faces:
                score = float(f.get("score", 0.0))
                bbox = f.get("bbox", [])
                if len(bbox) != 4 or score < 0.70:
                    continue

                x1, y1, x2, y2 = bbox
                fw = x2 - x1
                fh = y2 - y1

                # 1. Size check: Must be person standing in front of kiosk (not distant background)
                min_w = max(45.0, w_img * 0.14)
                min_h = max(45.0, h_img * 0.16)
                if fw < min_w or fh < min_h:
                    continue

                # 2. Aspect ratio check
                aspect = fh / max(1.0, fw)
                if aspect < 0.75 or aspect > 2.2:
                    continue

                # 3. Position check: Center must be in interactive camera zone
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                if cx < 0.10 * w_img or cx > 0.90 * w_img:
                    continue
                if cy < 0.08 * h_img or cy > 0.92 * h_img:
                    continue

                # 4. Landmark check if available
                landmarks = f.get("landmarks")
                if landmarks is not None and len(landmarks) == 5:
                    lm = np.asarray(landmarks, dtype=np.float32)
                    eye_dist = float(np.linalg.norm(lm[1] - lm[0]))
                    if eye_dist < 0.18 * fw:
                        continue

                valid_faces.append((score, f))

            if valid_faces:
                best_score, best_face = max(valid_faces, key=lambda item: item[0])
                return {
                    "face_detected": True,
                    "confidence": best_score,
                    "bbox": [float(v) for v in best_face.get("bbox", [])],
                }
        else:
            best_face = recognition_pipeline.extract_best_face(img_bgr)
            if best_face is not None and getattr(best_face, "detector_score", 0.0) >= 0.70:
                bbox = getattr(best_face, "bbox", [])
                if len(bbox) == 4:
                    fw = bbox[2] - bbox[0]
                    fh = bbox[3] - bbox[1]
                    if fw >= 45.0 and fh >= 45.0:
                        return {
                            "face_detected": True,
                            "confidence": float(best_face.detector_score),
                            "bbox": [float(v) for v in bbox],
                        }
    except Exception:
        pass

    return {
        "face_detected": False,
        "confidence": 0.0,
        "bbox": None,
    }


TTS_CACHE_DIR = Path("data/tts_cache")
TTS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/tts")
async def get_vietnamese_tts(text: str = Query(..., min_length=1, max_length=500)):
    """Returns crystal-clear, authentic native Vietnamese female voice (Google TTS) audio with disk caching."""
    clean_text = text.strip()
    cache_key = hashlib.md5(clean_text.encode("utf-8")).hexdigest()
    cache_file = TTS_CACHE_DIR / f"{cache_key}.mp3"

    if cache_file.exists():
        return Response(content=cache_file.read_bytes(), media_type="audio/mpeg")

    # Fetch from Google Vietnamese TTS
    encoded = urllib.parse.quote(clean_text)
    url = f"https://translate.google.com/translate_tts?ie=UTF-8&tl=vi&client=tw-ob&q={encoded}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read()
            cache_file.write_bytes(data)
            return Response(content=data, media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch Vietnamese TTS: {e}")

