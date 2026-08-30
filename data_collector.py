"""Công cụ dòng lệnh (CLI Tool) thu thập dữ liệu 21 điểm mốc 3D của MediaPipe Hands xuất ra tệp CSV.

Sử dụng để ghi nhận dữ liệu thực nghiệm cho các cử chỉ tay khác nhau nhằm mục đích:
- Xây dựng dataset đánh giá độc lập (Evaluation Dataset).
- Huấn luyện các mô hình Machine Learning (KNN, SVM, Random Forest, MLP, LSTM).
"""

import argparse
import csv
import os
import time
from typing import List, Optional

import cv2
import Hand_Detector as htm


def init_dataset_csv(csv_path: str) -> None:
    """Khởi tạo tệp CSV chứa tiêu đề các cột landmark (x0, y0, z0 -> x20, y20, z20, label).

    Args:
        csv_path (str): Đường dẫn tệp CSV lưu trữ.
    """
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    if not os.path.exists(csv_path):
        header: List[str] = []
        for i in range(21):
            header.extend([f"x{i}", f"y{i}", f"z{i}"])
        header.append("label")

        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def collect_landmarks(
    label: str,
    output_csv: str = "data/landmarks_dataset.csv",
    camera_index: int = 0,
    max_samples: int = 200,
) -> None:
    """Vòng lặp thu thập mẫu landmark 3D từ webcam và ghi vào CSV.

    Args:
        label (str): Tên nhãn cử chỉ đang thu thập (ví dụ: Fist, Peace, Stop, Wave).
        output_csv (str): Đường dẫn tệp CSV xuất ra.
        camera_index (int): Index của camera.
        max_samples (int): Số lượng mẫu tối đa cần thu thập.
    """
    init_dataset_csv(output_csv)
    cap = cv2.VideoCapture(camera_index)
    detector = htm.HandDetector(detectionCon=0.7, maxHands=1)

    print(f"=== ĐANG THU THẬP DỮ LIỆU CỬ CHỈ: '{label}' ===")
    print(f"Mục tiêu: {max_samples} mẫu -> Tệp xuất: {output_csv}")
    print("Nhấn 's' để bắt đầu ghi mẫu, nhấn 'q' để hủy và thoát.")

    recording = False
    samples_count = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Lỗi: Không thể kết nối với camera.")
                break

            frame = detector.findHands(cv2.flip(frame, 1))
            results = detector.results

            frame_h, frame_w = frame.shape[:2]

            if results and results.multi_hand_landmarks:
                primary = results.multi_hand_landmarks[0]
                if recording:
                    row: List[float] = []
                    for lm in primary.landmark:
                        row.extend([lm.x, lm.y, lm.z])
                    row.append(label)

                    with open(output_csv, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        writer.writerow(row)

                    samples_count += 1
                    if samples_count >= max_samples:
                        print(f"Đã hoàn tất thu thập {max_samples} mẫu cho nhãn '{label}'!")
                        break

            # Hiển thị HUD thông tin thu thập
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
            if key == ord("s"):
                recording = True
            elif key == ord("q"):
                print("Đã hủy quá trình thu thập.")
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


def main() -> None:
    """Hàm nạp tham số CLI cho công cụ thu thập dữ liệu."""
    parser = argparse.ArgumentParser(
        description="Công cụ thu thập dữ liệu MediaPipe Landmark 3D xuất CSV"
    )
    parser.add_argument(
        "--label", type=str, required=True, help="Tên nhãn cử chỉ (ví dụ: Fist, Stop, OK, Peace)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/landmarks_dataset.csv",
        help="Đường dẫn tệp CSV lưu dữ liệu",
    )
    parser.add_argument("--camera", type=int, default=0, help="Chỉ số camera thiết bị")
    parser.add_argument(
        "--samples", type=int, default=200, help="Số lượng mẫu tối đa cần thu thập"
    )

    args = parser.parse_args()
    collect_landmarks(
        label=args.label,
        output_csv=args.output,
        camera_index=args.camera,
        max_samples=args.samples,
    )


if __name__ == "__main__":
    main()
