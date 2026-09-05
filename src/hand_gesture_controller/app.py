"""Ứng dụng chính điều phối camera, nhận diện cử chỉ tay và quản lý tương tác vật thể kéo thả."""

import argparse
import logging
import sys
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np

from .event_mapper import GestureEventMapper
from .finger_number import FingerNumber
from .gesture_detector import GestureDetector
from .gesture_smoother import GestureSmoother
from .hand_detector import HandDetector
from .object_manager import DraggableObjectManager, ShapeMenu
from .performance_monitor import PerformanceMonitor

logger = logging.getLogger("hand_gesture_controller")


def setup_logging(log_level: str = "INFO") -> None:
    """Cấu hình định dạng và mức ghi log hệ thống."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def overlay_image(
    background: np.ndarray, overlay: Optional[np.ndarray], x: int, y: int
) -> None:
    """Chèn ảnh BGR/BGRA (kèm kênh Alpha trong suốt) lên nền background tại tọa độ (x, y).

    Args:
        background (np.ndarray): Ảnh nền BGR.
        overlay (Optional[np.ndarray]): Ảnh phủ BGR (3 kênh) hoặc BGRA (4 kênh).
        x (int): Tọa độ X góc trên bên trái của overlay.
        y (int): Tọa độ Y góc trên bên trái của overlay.
    """
    if background is None or overlay is None or overlay.ndim != 3:
        return
    bg_h, bg_w = background.shape[:2]
    ol_h, ol_w = overlay.shape[:2]
    if x >= bg_w or y >= bg_h or x + ol_w <= 0 or y + ol_h <= 0:
        return

    x_start, y_start = max(0, x), max(0, y)
    x_end, y_end = min(bg_w, x + ol_w), min(bg_h, y + ol_h)
    ox, oy = x_start - x, y_start - y
    crop = overlay[oy : oy + y_end - y_start, ox : ox + x_end - x_start]

    if crop.shape[2] == 4:
        alpha = crop[:, :, 3:4].astype(float) / 255.0
        bg_crop = background[y_start:y_end, x_start:x_end].astype(float)
        background[y_start:y_end, x_start:x_end] = (
            alpha * crop[:, :, :3] + (1 - alpha) * bg_crop
        ).astype(background.dtype)
    elif crop.shape[2] == 3:
        background[y_start:y_end, x_start:x_end] = crop


class HandGestureApp:
    """Ứng dụng chính điều phối camera, nhận diện cử chỉ tay và quản lý tương tác vật thể kéo thả."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        smoothing_window: int = 5,
        smoothing_votes: int = 3,
        benchmark_output: Optional[str] = None,
    ) -> None:
        """Khởi tạo toàn bộ mô-đun ứng dụng và mở camera."""
        if width <= 0:
            raise ValueError("Chiều rộng khung hình (--width) phải lớn hơn 0.")
        if height <= 0:
            raise ValueError("Chiều cao khung hình (--height) phải lớn hơn 0.")
        if camera_index < 0:
            raise ValueError("Chỉ số camera (--camera) phải lớn hơn hoặc bằng 0.")

        self.width, self.height = width, height
        self.show_debug: bool = True
        self.cap = self._open_camera(camera_index)
        self.detector = HandDetector(detectionCon=0.7, maxHands=1)
        self.finger_counter = FingerNumber()
        self.gesture_detector = GestureDetector()
        self.event_mapper = GestureEventMapper()
        self.gesture_smoother = GestureSmoother(
            window_size=smoothing_window,
            minimum_votes=smoothing_votes,
        )
        self.performance = PerformanceMonitor()
        self.benchmark_output = benchmark_output
        self.object_manager = DraggableObjectManager(smooth_alpha=0.4)
        self.shape_menu = ShapeMenu()
        self.missing_hand_frames: int = 0

    def _open_camera(self, camera_index: int) -> cv2.VideoCapture:
        """Mở camera với thử lại và fallback."""
        for index in dict.fromkeys([camera_index, 0, 1]):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                logger.info("Đã kết nối thành công với camera index %d", index)
                return cap
            cap.release()
        raise RuntimeError("Không tìm thấy camera khả dụng trên hệ thống.")

    def _draw_gesture_labels(
        self,
        frame: np.ndarray,
        hand_landmarks: Any,
        static_result: Tuple[str, Tuple[int, int, int]],
        motion_result: Tuple[str, Tuple[int, int, int]],
    ) -> None:
        """Hiển thị nhãn cử chỉ tĩnh và cử chỉ chuyển động mượt mà gần cổ tay."""
        frame_height, frame_width = frame.shape[:2]
        wrist = hand_landmarks.landmark[0]
        x = int(wrist.x * frame_width)
        y = max(25, int(wrist.y * frame_height) - 30)
        static_gesture, static_color = static_result
        motion_gesture, motion_color = motion_result

        cv2.putText(
            frame,
            f"Static: {static_gesture}",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            static_color,
            2,
        )
        cv2.putText(
            frame,
            f"Motion: {motion_gesture}",
            (x, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            motion_color,
            2,
        )

    def _draw_hud(self, frame: np.ndarray) -> None:
        """Vẽ card HUD hiển thị thông số FPS và độ trễ Latency ms."""
        if not self.show_debug:
            return

        summary = self.performance.summary()
        fps_text = f"FPS: {summary['average_fps']:.1f}"
        lat_text = f"Latency: {summary['average_latency_ms']:.1f}ms (P95: {summary['p95_latency_ms']:.1f}ms)"
        frames_text = f"Frames: {summary['total_frames']}"

        cv2.rectangle(frame, (10, 10), (320, 90), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (320, 90), (0, 255, 0), 1)

        cv2.putText(
            frame, fps_text, (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        cv2.putText(
            frame, lat_text, (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        cv2.putText(
            frame, frames_text, (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )

    def run(self) -> None:
        """Vòng lặp sự kiện chính của ứng dụng."""
        logger.info("Đang khởi tạo ứng dụng Hand Gesture Controller...")
        logger.info("Nhấn 'Q' để thoát, 'D' để bật/tắt HUD, 'C' để xóa canvas.")

        try:
            while True:
                started_at = time.perf_counter()
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    logger.warning("Không thể đọc khung hình từ camera. Đang dừng...")
                    break

                frame = cv2.flip(frame, 1)
                frame = self.detector.findHands(frame, draw=True)
                results = self.detector.results

                primary_landmarks = None
                primary_label = "Right"

                if (
                    results
                    and results.multi_hand_landmarks
                    and results.multi_handedness
                ):
                    primary_landmarks = results.multi_hand_landmarks[0]
                    primary_label = results.multi_handedness[0].classification[0].label
                    self.missing_hand_frames = 0
                else:
                    self.missing_hand_frames += 1
                    if self.missing_hand_frames > 3:
                        self.gesture_detector.reset()
                        self.gesture_smoother.reset()
                        self.event_mapper.reset()

                if primary_landmarks is not None:
                    raw_static_gesture, static_color = self.gesture_detector.detect_static_gesture(
                        primary_landmarks
                    )
                    smoothed_gesture, smoothed_color = self.gesture_smoother.update(
                        raw_static_gesture, static_color
                    )
                    motion_result, _ = self.gesture_detector.detect_motion_gesture(
                        primary_landmarks
                    )
                    motion_gesture, _ = motion_result

                    # Chuyển nhãn cử chỉ thành sự kiện ứng dụng (GestureEvent)
                    event = self.event_mapper.map_gesture_to_event(
                        smoothed_gesture, motion_gesture
                    )

                    self.object_manager.update_event(
                        primary_landmarks,
                        event,
                        self.width,
                        self.height,
                    )
                    self.shape_menu.update(
                        primary_landmarks,
                        smoothed_gesture,
                        self.width,
                        self.height,
                        self.object_manager,
                    )

                    _, icon_img = self.finger_counter.detect_gesture(
                        primary_landmarks, primary_label
                    )
                    if icon_img is not None:
                        overlay_image(frame, icon_img, self.width - 160, 20)

                    self._draw_gesture_labels(
                        frame,
                        primary_landmarks,
                        (smoothed_gesture, smoothed_color),
                        motion_result,
                    )
                    self.object_manager.draw_cursor(
                        frame, primary_landmarks, self.width, self.height
                    )

                self.object_manager.draw_all(frame)
                self.shape_menu.draw(frame, visible=self.object_manager.visible)

                self.performance.record_frame(started_at)
                self._draw_hud(frame)

                cv2.imshow("Hand Gesture Controller", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    logger.info("Nhận tín hiệu thoát từ phím Q.")
                    break
                if key == ord("d") or key == ord("D"):
                    self.show_debug = not self.show_debug
                if key == ord("c") or key == ord("C"):
                    self.object_manager.objects.clear()
                    logger.info("Đã xóa toàn bộ vật thể trên canvas.")

        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Giải phóng tài nguyên và lưu báo cáo benchmark nếu có."""
        logger.info("Đang giải phóng tài nguyên hệ thống...")
        if self.benchmark_output:
            self.performance.save(self.benchmark_output)
        self.cap.release()
        self.detector.close()
        cv2.destroyAllWindows()
        logger.info("Ứng dụng kết thúc an toàn.")


def main() -> None:
    """Đọc tham số dòng lệnh CLI và khởi chạy ứng dụng HandGestureApp."""
    parser = argparse.ArgumentParser(
        description="Hand Gesture Controller - Ứng dụng tương tác đồ họa bằng cử chỉ tay"
    )
    parser.add_argument("--camera", type=int, default=0, help="Chỉ số camera (mặc định 0)")
    parser.add_argument("--width", type=int, default=640, help="Chiều rộng khung hình (mặc định 640)")
    parser.add_argument("--height", type=int, default=480, help="Chiều cao khung hình (mặc định 480)")
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Kích thước cửa sổ làm mượt cử chỉ (mặc định 5)",
    )
    parser.add_argument(
        "--smoothing-votes",
        type=int,
        default=3,
        help="Số phiếu tối thiểu đồng thuận cử chỉ (mặc định 3)",
    )
    parser.add_argument(
        "--benchmark-output",
        type=str,
        default=None,
        help="Đường dẫn xuất file JSON kết quả FPS/Latency",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Mức ghi log hệ thống",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    try:
        app = HandGestureApp(
            camera_index=args.camera,
            width=args.width,
            height=args.height,
            smoothing_window=args.smoothing_window,
            smoothing_votes=args.smoothing_votes,
            benchmark_output=args.benchmark_output,
        )
        app.run()
    except Exception as e:
        logger.error("Lỗi khởi chạy ứng dụng: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
