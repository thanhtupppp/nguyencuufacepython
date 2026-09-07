from datetime import datetime, timezone
from typing import Optional
import uuid

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.api.dependencies import db, recognition_pipeline, verify_api_key
from src.api.websocket_manager import ws_manager

router = APIRouter(dependencies=[Depends(verify_api_key)])

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB


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
