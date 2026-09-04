"""Công cụ dòng lệnh CLI thu thập dữ liệu MediaPipe Landmarks xuất tệp CSV tuân thủ Dataset Schema."""

import argparse
import csv
import logging
import os
import sys
import time
import uuid

import cv2

# Import hand_detector từ gói src.hand_gesture_controller
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from hand_gesture_controller.hand_detector import HandDetector

logger = logging.getLogger("collect_landmarks")

ALLOWED_GESTURES = (
    "Fist",
    "Stop",
    "OK",
    "Peace",
    "Select",
    "Options",
    "Thumbs Up",
    "Thumbs Down",
    "Unknown",
)

CSV_HEADER = [
    "sample_id",
    "subject_id",
    "session_id",
    "gesture",
    "frame_index",
    "timestamp_ms",
    "handedness",
    "camera_width",
    "camera_height",
    "device_id",
    "lighting",
] + [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def init_dataset_csv(csv_path: str) -> None:
    """Khởi tạo tệp CSV chứa tiêu đề chuẩn nếu chưa tồn tại."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


def collect_landmarks(
    subject_id: str,
    session_id: str,
    label: str,
    output_csv: str = "data/raw/landmarks_dataset.csv",
    camera_index: int = 0,
    max_samples: int = 200,
    lighting: str = "normal",
    device_id: str = "cam_01",
) -> None:
    """Vòng lặp thu thập mẫu landmark 3D từ webcam và ghi vào CSV."""
    if not subject_id or not subject_id.strip():
        raise ValueError("Mã người tham gia (--subject-id) không được để trống.")
    if not session_id or not session_id.strip():
        raise ValueError("Mã phiên (--session-id) không được để trống.")
    if max_samples <= 0:
        raise ValueError("Số lượng mẫu (--samples) phải lớn hơn 0.")
    if label not in ALLOWED_GESTURES:
        logger.warning("Nhãn '%s' không nằm trong danh sách khuyến nghị: %s", label, ALLOWED_GESTURES)

    init_dataset_csv(output_csv)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Không thể mở camera index {camera_index}.")

    detector = HandDetector(detectionCon=0.7, maxHands=1)
    logger.info("=== BẮT ĐẦU THU THẬP DỮ LIỆU CỬ CHỈ: '%s' ===", label)
    logger.info("Subject: %s | Session: %s | Mục tiêu: %d mẫu", subject_id, session_id, max_samples)
    logger.info("Nhấn 'S' để ghi mẫu, 'Q' để hủy và thoát.")

    recording = False
    samples_count = 0
    frame_index = 0

    try:
        output_file = open(output_csv, mode="a", newline="", encoding="utf-8")
        writer = csv.writer(output_file)

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.error("Không thể kết nối với camera.")
                break

            frame = cv2.flip(frame, 1)
            frame = detector.findHands(frame, draw=True)
            results = detector.results

            frame_h, frame_w = frame.shape[:2]
            frame_index += 1

            if results and results.multi_hand_landmarks and results.multi_handedness:
                primary_landmarks = results.multi_hand_landmarks[0]
                handedness = results.multi_handedness[0].classification[0].label

                if recording:
                    sample_id = f"{subject_id}_{session_id}_{samples_count + 1:04d}_{uuid.uuid4().hex[:6]}"
                    timestamp_ms = round(time.time() * 1000.0, 2)

                    row = [
                        sample_id,
                        subject_id,
                        session_id,
                        label,
                        frame_index,
                        timestamp_ms,
                        handedness,
                        frame_w,
                        frame_h,
                        device_id,
                        lighting,
                    ]

                    for lm in primary_landmarks.landmark:
                        row.extend([lm.x, lm.y, lm.z])

                    writer.writerow(row)
                    samples_count += 1

                    if samples_count >= max_samples:
                        logger.info("Hoàn tất thu thập %d mẫu cho nhãn '%s'!", max_samples, label)
                        break

            status_text = f"RECORDING ({samples_count}/{max_samples})" if recording else "PRESS 'S' TO START"
            status_color = (0, 255, 0) if recording else (0, 255, 255)

            cv2.putText(
                frame,
                f"Label: {label} | Status: {status_text}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )
            cv2.imshow("Landmark Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s") or key == ord("S"):
                recording = True
            elif key == ord("q") or key == ord("Q"):
                logger.info("Đã hủy quá trình thu thập.")
                break

    finally:
        if "output_file" in locals():
            output_file.close()
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


def main() -> None:
    """Khởi tạo CLI Parser cho tool collect_landmarks."""
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
    parser = argparse.ArgumentParser(description="Công cụ thu thập dữ liệu MediaPipe Landmarks xuất CSV")
    parser.add_argument("--subject-id", type=str, required=True, help="Mã người tham gia (ví dụ: subject_001)")
    parser.add_argument("--session-id", type=str, required=True, help="Mã phiên thu thập (ví dụ: session_001)")
    parser.add_argument("--label", type=str, required=True, help="Tên nhãn cử chỉ (Fist, Stop, Select, Peace, ...)")
    parser.add_argument("--output", type=str, default="data/raw/landmarks_dataset.csv", help="Đường dẫn tệp CSV lưu")
    parser.add_argument("--camera", type=int, default=0, help="Index camera thiết bị")
    parser.add_argument("--samples", type=int, default=200, help="Số lượng mẫu tối đa cần thu thập")
    parser.add_argument("--lighting", type=str, default="normal", help="Điều kiện ánh sáng (normal, dim, bright)")

    args = parser.parse_args()

    collect_landmarks(
        subject_id=args.subject_id,
        session_id=args.session_id,
        label=args.label,
        output_csv=args.output,
        camera_index=args.camera,
        max_samples=args.samples,
        lighting=args.lighting,
    )


if __name__ == "__main__":
    main()
