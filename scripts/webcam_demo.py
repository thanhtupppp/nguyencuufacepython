"""
NguyenCuuFacePython - Live Interactive Webcam Demo.
Demonstrates the full end-to-end pipeline in real time:
Camera -> SCRFD Detection -> 5 Landmarks -> Quality Gate -> Anti-Spoofing ->
Alignment (112x112) -> ArcFace 512D Embedding -> pgvector / DB Search ->
Dual-Threshold Margin -> Face Tracking -> Temporal Voting -> Realtime HUD.

Hotkeys:
  [e] : Đăng ký (Enroll) khuôn mặt hiện tại vào Database
  [c] : Xóa sạch danh sách người đã đăng ký (Clear DB)
  [q] : Thoát chương trình (Quit)
"""

import argparse
from pathlib import Path
import sys
import time
from typing import Any

# Ensure repository root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from src.detection.scrfd import SCRFDDetector
from src.alignment.aligner import FaceAligner
from src.recognition.arcface import ArcFaceRecognizer
from src.quality.quality_gate import FaceQualityGate
from src.anti_spoofing.liveness import AntiSpoofDetector
from src.tracking.tracker import FaceTracker, DetectionItem, TemporalVotingEngine
from src.database.client import DatabaseClient


def draw_hud(
    frame: np.ndarray,
    tracklets: list,
    voting_results: dict[int, Any],
    fps: float,
    num_enrolled: int,
    frame_idx: int = 0,
    toast_msg: str = "",
    toast_color: tuple = (0, 255, 0),
) -> np.ndarray:
    """Draws visual annotations, bounding boxes, landmarks, and status banners on frame."""
    h, w = frame.shape[:2]

    # Top info banner (semi-transparent background)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    cv2.putText(
        frame,
        f"NguyenCuuFacePython Live Demo | FPS: {fps:.1f} | Enrolled: {num_enrolled}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 200),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Controls: [e] Enroll Face | [c] Clear Database | [q] Quit",
        (15, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Optional Toast Notification Banner near bottom
    if toast_msg:
        t_overlay = frame.copy()
        cv2.rectangle(t_overlay, (20, h - 55), (w - 20, h - 15), (20, 20, 20), -1)
        cv2.addWeighted(t_overlay, 0.85, frame, 0.15, 0, frame)
        cv2.putText(
            frame,
            toast_msg,
            (35, h - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            toast_color,
            2,
            cv2.LINE_AA,
        )

    # Draw each active tracklet
    for tr in tracklets:
        tid = tr.track_id
        x1, y1, x2, y2 = [int(v) for v in tr.bbox]

        # Get latest observation item
        latest_item = tr.history[-1] if tr.history else None
        voting = voting_results.get(tid)

        # State Smoothing (Hysteresis): Check if tracklet is confirmed or persistent across blinks
        is_confirmed = False
        confirmed_name = None
        confirmed_sim = 0.0
        confirmed_cons = 0.0

        if voting and voting.status == "CONFIRMED_MATCH" and voting.person_id:
            is_confirmed = True
            confirmed_name = voting.person_id
            confirmed_sim = voting.mean_similarity
            confirmed_cons = voting.consensus_ratio
        elif hasattr(tr, "is_identity_persistent") and tr.is_identity_persistent(frame_idx, max_hold_frames=30):
            is_confirmed = True
            confirmed_name = tr.confirmed_id
            confirmed_sim = tr.confirmed_sim
            confirmed_cons = tr.confirmed_consensus

        color = (0, 165, 255)  # Orange default (pending)
        label = f"Track #{tid} [Pending...]"

        if is_confirmed and confirmed_name:
            color = (0, 255, 0)  # Solid Green for confirmed match
            label = f"{confirmed_name} (Sim: {confirmed_sim:.2f}, Consensus: {confirmed_cons*100:.0f}%)"
        elif latest_item and not latest_item.is_valid_quality:
            # Quality Gate rejected with specific reason (only shown for unconfirmed tracklets)
            color = (0, 100, 255)  # Dark Orange
            reasons_summary = (
                latest_item.rejection_reasons[0].split("(")[0].strip()
                if latest_item.rejection_reasons
                else "Low Quality"
            )
            label = f"Track #{tid} [{reasons_summary}]"
        elif voting:
            if voting.status == "AMBIGUOUS":
                color = (0, 255, 255)  # Yellow for ambiguous
                label = f"Track #{tid} [AMBIGUOUS - Margin Check]"
            elif voting.status == "UNKNOWN":
                color = (0, 0, 255)  # Red for unknown
                label = f"Track #{tid} [UNKNOWN]"
            elif voting.status == "PENDING":
                color = (0, 200, 255)
                label = f"Track #{tid} [Voting {voting.frame_count}/5...]"

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(y1 - 10, label_size[1] + 10)
        cv2.rectangle(
            frame,
            (x1, label_y - label_size[1] - 4),
            (x1 + label_size[0] + 6, label_y + 4),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 3, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        # Draw 5 landmarks if available
        if latest_item and latest_item.landmarks is not None:
            for lmk in latest_item.landmarks:
                lx, ly = int(lmk[0]), int(lmk[1])
                cv2.circle(frame, (lx, ly), 3, (255, 0, 255), -1, cv2.LINE_AA)

    return frame


def run_webcam(
    camera_id: int = 0,
    det_model: str = "models/checkpoints/det_10g.onnx",
    rec_model: str = "models/checkpoints/w600k_r50.onnx",
    threshold: float = 0.60,
    margin: float = 0.08,
    sqlite_path: str = "data/faces.db",
):
    print("=" * 60)
    print("NguyenCuuFacePython - Khởi động Live Webcam Demo...")
    print("=" * 60)

    # Check model weights
    det_path = Path(det_model)
    rec_path = Path(rec_model)

    if not det_path.exists() or not rec_path.exists():
        print(f"\n[LƯU Ý] Chưa tìm thấy đầy đủ model weights tại {det_path} hoặc {rec_path}.")
        print("Đang tự động tải mô hình qua scripts/download_models.py...")
        from scripts.download_models import main as dl_main
        sys.argv = ["download_models.py", "--model", "all"]
        dl_main()

    print("\n1. Khởi tạo module Face Detector (SCRFD)...")
    detector = SCRFDDetector(model_path=str(det_path), conf_threshold=0.5, nms_threshold=0.4)

    print("2. Khởi tạo module Face Aligner (Umeyama 5-Landmarks 112x112)...")
    aligner = FaceAligner(output_size=(112, 112))

    print("3. Khởi tạo module Face Recognizer (ArcFace ResNet-50 512D)...")
    recognizer = ArcFaceRecognizer(model_path=str(rec_path))
    ep = recognizer.session.get_providers()[0] if recognizer.session else "Unknown"
    hw_info = "GPU Acceleration Active (NVIDIA GeForce GTX 1070 Ti)" if ("Dml" in ep or "CUDA" in ep) else "CPU"
    print(f"   [Hardware Acceleration: {hw_info} - Provider: {ep}]")

    print("4. Khởi tạo module Quality Gate & Anti-Spoofing...")
    # Calibrated thresholds for real-time indoor webcam environment
    quality_gate = FaceQualityGate(
        min_face_size=50,
        blur_threshold=15.0,
        max_yaw=40.0,
        max_pitch=40.0,
        min_brightness=25.0,
    )
    anti_spoof = AntiSpoofDetector(model_path=None, threshold=0.75)

    print("5. Khởi tạo module Face Tracker & Temporal Voting Engine...")
    tracker = FaceTracker(iou_threshold=0.25, max_lost_frames=30, min_hits_to_activate=1)
    voting_engine = TemporalVotingEngine(window_size=5, min_consensus_ratio=0.60, min_frames=2)

    print("6. Khởi tạo Database Vector Search...")
    db_client = DatabaseClient(sqlite_path=sqlite_path)
    enrolled_persons = db_client.list_persons()
    print(f"   [Database: SQLite Local ({sqlite_path}) - {len(enrolled_persons)} người đã đăng ký]")

    print(f"\n7. Mở Camera ID {camera_id}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        # Try camera 1 if 0 fails
        print(f"[CẢNH BÁO] Không thể mở camera {camera_id}. Đang thử mở camera 1...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("[LỖI] Không tìm thấy camera nào hoạt động trên máy tính.")
        print("Vui lòng kiểm tra kết nối webcam hoặc cấp quyền camera trong Windows Settings.")
        return

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n" + "=" * 60)
    print("DEMO ĐANG CHẠY THÀNH CÔNG!")
    print("  Nhấn phím [e] trên cửa sổ video để ĐĂNG KÝ (Enroll) khuôn mặt.")
    print("  Nhấn phím [c] để XÓA SẠCH danh sách.")
    print("  Nhấn phím [q] để THOÁT.")
    print("=" * 60 + "\n")

    frame_count = 0
    fps = 0.0
    start_time = time.time()
    enroll_counter = len(enrolled_persons) + 1

    last_active_tracklets: list[Any] = []
    latest_voting_results: dict[int, Any] = {}

    toast_text = ""
    toast_color = (0, 255, 0)
    toast_expiry = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Mất tín hiệu camera...")
            break

        frame_count += 1

        # 1. Detect faces with SCRFD
        detections_raw = detector.detect(frame)
        detection_items = []

        for det in detections_raw:
            bbox = det["bbox"]
            conf = det["score"]
            lmk = det["landmarks"]

            # 2. Quality Gate check
            q_res = quality_gate.assess_quality(frame, bbox, lmk)

            item = DetectionItem(
                frame_idx=frame_count,
                bbox=bbox,
                score=conf,
                landmarks=lmk,
                quality_score=q_res.overall_quality_score,
                is_valid_quality=q_res.is_valid,
                rejection_reasons=q_res.rejection_reasons,
            )

            # 3. Extract embedding whenever landmarks are detected
            if lmk is not None:
                aligned_face, _ = aligner.align(frame, lmk)
                emb = recognizer.extract_embedding(aligned_face)
                item.aligned_face = aligned_face
                item.embedding = emb

            # 4. If quality is valid, perform Anti-Spoofing and Recognition
            if q_res.is_valid and item.embedding is not None:
                spoof_res = anti_spoof.predict_liveness(frame, bbox)

                if spoof_res.is_live:
                    # 5. Recognize against Database with Margin Check
                    rec_dec = db_client.recognize_with_margin(
                        item.embedding, model_version=recognizer.model_version, threshold=threshold, margin=margin
                    )
                    item.predicted_id = rec_dec.person_id if rec_dec.status == "MATCHED" else rec_dec.status
                    item.similarity = rec_dec.similarity
                    item.margin = rec_dec.margin
                else:
                    item.predicted_id = "SPOOF_DETECTED"
                    item.similarity = spoof_res.liveness_score
            else:
                item.predicted_id = "LOW_QUALITY"

            detection_items.append(item)

        # 6. Update Multi-Object Tracker
        active_tracklets = tracker.update(detection_items, frame_idx=frame_count)
        last_active_tracklets = active_tracklets

        # 7. Evaluate Temporal Voting for each active tracklet
        latest_voting_results = {}
        for tr in active_tracklets:
            vote_res = voting_engine.vote(tr)
            latest_voting_results[tr.track_id] = vote_res

        # Calculate FPS
        if frame_count % 10 == 0:
            elapsed = time.time() - start_time
            fps = 10.0 / max(elapsed, 1e-4)
            start_time = time.time()

        # Render Annotations & HUD
        num_enrolled = len(db_client.list_persons())
        current_toast = toast_text if time.time() < toast_expiry else ""
        display_frame = draw_hud(
            frame,
            active_tracklets,
            latest_voting_results,
            fps,
            num_enrolled,
            frame_idx=frame_count,
            toast_msg=current_toast,
            toast_color=toast_color,
        )

        cv2.imshow("NguyenCuuFacePython - Live Webcam Demo", display_frame)

        key = cv2.waitKey(1) & 0xFF

        # Hotkey [q]: Quit
        if key == ord("q"):
            print("Đang tắt webcam demo...")
            break

        # Hotkey [c]: Clear DB
        elif key == ord("c"):
            for p in db_client.list_persons():
                db_client.delete_person(p["person_id"])
            print("\n[INFO] Đã xóa toàn bộ dữ liệu khuôn mặt trong Database.")
            toast_text = "[DATABASE CLEARED] Da xoa toan bo danh sach khuon mat"
            toast_color = (0, 150, 255)
            toast_expiry = time.time() + 3.0

        # Hotkey [e]: Enroll current face
        elif key == ord("e"):
            target_tracklets = active_tracklets if active_tracklets else last_active_tracklets
            if not target_tracklets:
                print("\n[CẢNH BÁO] Không có khuôn mặt nào trong khung hình để đăng ký!")
                toast_text = "[CANH BAO] Khong co khuon mat nao trong khung hinh!"
                toast_color = (0, 0, 255)
                toast_expiry = time.time() + 2.5
            else:
                # Enroll the first active tracklet
                target_tr = target_tracklets[0]
                best_item = target_tr.best_item or (target_tr.history[-1] if target_tr.history else None)

                if best_item and best_item.embedding is not None:
                    # 1. Check if this face belongs to ANY existing person in Database
                    # Any candidate with similarity >= 0.52 is considered the same person
                    candidates = db_client.search_top_k(
                        best_item.embedding,
                        model_version=recognizer.model_version,
                        top_k=1,
                    )

                    if candidates and candidates[0].similarity >= 0.52:
                        existing_pid = candidates[0].person_id
                        existing_p = db_client.get_person(existing_pid)
                        p_name = existing_p["name"] if existing_p else existing_pid
                        print(f"\n[THÔNG BÁO] Khuôn mặt này ĐÃ ĐƯỢC ĐĂNG KÝ TRƯỚC ĐÓ!")
                        print(f"  - Danh tính: {p_name} (ID: {existing_pid})")
                        print(f"  - Độ tương đồng với mẫu cũ: {candidates[0].similarity:.2f}")

                        # Add new angle/template to existing person for even better recognition
                        db_client.add_embedding(
                            person_id=existing_pid,
                            embedding=best_item.embedding,
                            model_version=recognizer.model_version,
                            quality_score=best_item.quality_score,
                        )
                        print(f"  - Đã tự động cập nhật thêm góc mặt mới vào hồ sơ {p_name} để nhận diện nhạy hơn!")
                        toast_text = f"[DA DANG KY] {p_name}: Da cap nhat them goc mat moi (Sim: {candidates[0].similarity:.2f})"
                        toast_color = (0, 255, 200)
                        toast_expiry = time.time() + 3.5
                    else:
                        # 2. Brand new person enrollment
                        user_name = f"User_{enroll_counter:02d}"
                        person_id = f"P_{enroll_counter:03d}"
                        enroll_counter += 1

                        db_client.create_person(person_id=person_id, name=user_name, department="VIP")
                        db_client.add_embedding(
                            person_id=person_id,
                            embedding=best_item.embedding,
                            model_version=recognizer.model_version,
                            quality_score=best_item.quality_score,
                        )
                        print(f"\n[THÀNH CÔNG] Đã đăng ký thành công: {user_name} (ID: {person_id})!")
                        print(f"  - Điểm chất lượng Best Frame: {best_item.quality_score:.2f}")
                        if not best_item.is_valid_quality and best_item.rejection_reasons:
                            reasons_str = ", ".join(best_item.rejection_reasons)
                            print(f"  - Lưu ý chất lượng: {reasons_str} (Hệ thống vẫn hỗ trợ trích xuất)")
                        toast_text = f"[THANH CONG] Da dang ky {user_name} (ID: {person_id})"
                        toast_color = (0, 255, 0)
                        toast_expiry = time.time() + 3.5
                else:
                    target_reasons = target_tr.history[-1].rejection_reasons if target_tr.history and target_tr.history[-1].rejection_reasons else []
                    reasons_str = f" Lý do: {', '.join(target_reasons)}" if target_reasons else ""
                    print(f"\n[CẢNH BÁO] Chưa trích xuất được vector khuôn mặt.{reasons_str} Hãy nhìn thẳng vào camera và thử lại.")
                    toast_text = "[CANH BAO] Chua trich xuat duoc vector khuon mat!"
                    toast_color = (0, 0, 255)
                    toast_expiry = time.time() + 2.5

    cap.release()
    cv2.destroyAllWindows()
    print("Demo đã kết thúc.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Webcam Demo for NguyenCuuFacePython")
    parser.add_argument("--camera", type=int, default=0, help="Camera Index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.60, help="Recognition cosine threshold")
    parser.add_argument("--margin", type=float, default=0.08, help="Candidate margin gap")
    parser.add_argument("--db", type=str, default="data/faces.db", help="Path to SQLite database file (default: data/faces.db)")
    args = parser.parse_args()

    run_webcam(
        camera_id=args.camera,
        threshold=args.threshold,
        margin=args.margin,
        sqlite_path=args.db,
    )

