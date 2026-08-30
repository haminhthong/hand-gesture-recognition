import argparse
import time
from typing import Any, Optional, Tuple

import cv2
import numpy as np

import Hand_Detector as htm
from DraggableObject import DraggableObjectManager, ShapeMenu
from Finger_Number import FingerNumber
from GestureDetector import GestureDetector
from GestureSmoother import GestureSmoother
from PerformanceMonitor import PerformanceMonitor


def overlay_image(
    background: np.ndarray, overlay: Optional[np.ndarray], x: int, y: int
) -> None:
    """Chèn ảnh BGR/BGRA (kèm kênh Alpha trong suốt) lên nền background tại tọa độ (x, y).

    Tự động xử lý cắt tỉa (cropping) nếu ảnh overlay nằm một phần ngoài biên khung hình background.

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
    """Ứng dụng chính điều phối camera, nhận diện cử chỉ tay và quản lý tương tác vật thể kéo thả.

    Attributes:
        width (int): Chiều rộng khung hình camera.
        height (int): Chiều cao khung hình camera.
        show_debug (bool): Trạng thái bật/tắt HUD thông số kỹ thuật chi tiết.
        cap (cv2.VideoCapture): Đối tượng kết nối camera OpenCV.
        detector (htm.HandDetector): Bộ nhận diện bàn tay MediaPipe.
        finger_counter (FingerNumber): Bộ đếm số ngón và lấy ảnh icon đại diện.
        gesture_detector (GestureDetector): Trình luận quy tắc cử chỉ tĩnh & động.
        gesture_smoother (GestureSmoother): Bộ làm mượt nhãn cử chỉ tĩnh theo thời gian.
        performance (PerformanceMonitor): Bộ theo dõi và đo FPS/latency.
        object_manager (DraggableObjectManager): Quản lý danh sách vật thể kéo thả trên canvas.
        shape_menu (ShapeMenu): Trình đơn chọn hình dạng tạo mới.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
        smoothing_window: int = 5,
        smoothing_votes: int = 3,
        benchmark_output: Optional[str] = None,
    ) -> None:
        """Khởi tạo toàn bộ mô-đun ứng dụng và mở camera.

        Args:
            camera_index: Index của camera thiết bị.
            width: Chiều rộng khung hình mong muốn.
            height: Chiều cao khung hình mong muốn.
            smoothing_window: Kích thước cửa sổ làm mượt cử chỉ.
            smoothing_votes: Số phiếu tối thiểu để chấp nhận nhãn cử chỉ.
            benchmark_output: Đường dẫn xuất file JSON hiệu năng (nếu có).
        """
        self.width, self.height = width, height
        self.show_debug: bool = True
        self.cap = self._open_camera(camera_index)
        self.detector = htm.HandDetector(detectionCon=0.7, maxHands=2)
        self.finger_counter = FingerNumber()
        self.gesture_detector = GestureDetector()
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
        """Mở camera theo index yêu cầu hoặc thử các index phổ biến (0, 1) làm dự phòng.

        Args:
            camera_index: Index camera ưu tiên.

        Returns:
            cv2.VideoCapture: Đối tượng camera đã sẵn sàng đọc khung hình.

        Raises:
            RuntimeError: Nếu không tìm thấy bất kỳ camera khả dụng nào.
        """
        for index in dict.fromkeys([camera_index, 0, 1]):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
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

    def _draw_finger_icons(self, frame: np.ndarray, results: Any) -> None:
        """Hiển thị ảnh icon minh họa trạng thái đếm ngón cho bàn tay trái / phải."""
        frame_width = frame.shape[1]
        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            label = handedness.classification[0].label
            _, icon = self.finger_counter.detect_gesture(hand_landmarks, label)
            if icon is None:
                continue

            icon_height, icon_width = icon.shape[:2]
            icon = cv2.resize(icon, (icon_width * 2, icon_height * 2))
            icon_x = 10 if label == "Left" else frame_width - icon.shape[1] - 10
            overlay_image(frame, icon, icon_x, 10)

    def _handle_missing_hand(self, frame_width: int, frame_height: int) -> None:
        """Đặt lại tất cả các trạng thái tương tác và FSM khi mất dấu bàn tay."""
        self.missing_hand_frames += 1
        self.object_manager.update(
            None,
            "Unknown",
            "Still",
            frame_width,
            frame_height,
        )
        self.shape_menu.update(
            None,
            "Unknown",
            frame_width,
            frame_height,
            self.object_manager,
        )
        if self.missing_hand_frames == 5:
            self.gesture_detector.reset()
            self.gesture_smoother.reset()

    def _draw_hud(self, frame: np.ndarray) -> None:
        """Vẽ lớp giao diện HUD hiện đại (Performance Card & Control Shortcuts)."""
        frame_height, frame_width = frame.shape[:2]

        # 1. Bảng phím tắt điều khiển ở góc trên giữa
        shortcut_text = "[Q] Exit  |  [D] Debug HUD  |  [C] Clear Canvas"
        cv2.rectangle(
            frame,
            (frame_width // 2 - 190, 5),
            (frame_width // 2 + 190, 30),
            (30, 30, 30),
            -1,
        )
        cv2.rectangle(
            frame,
            (frame_width // 2 - 190, 5),
            (frame_width // 2 + 190, 30),
            (100, 100, 100),
            1,
        )
        cv2.putText(
            frame,
            shortcut_text,
            (frame_width // 2 - 180, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 255, 200),
            1,
        )

        # 2. Card hiển thị FPS và Latency góc dưới phải (khi bật Debug)
        if self.show_debug:
            card_w, card_h = 190, 55
            card_x = frame_width - card_w - 10
            card_y = frame_height - card_h - 10

            # Nền mờ cho card HUD
            cv2.rectangle(
                frame,
                (card_x, card_y),
                (card_x + card_w, card_y + card_h),
                (20, 20, 20),
                -1,
            )
            cv2.rectangle(
                frame,
                (card_x, card_y),
                (card_x + card_w, card_y + card_h),
                (0, 255, 255),
                1,
            )

            cv2.putText(
                frame,
                f"FPS: {self.performance.average_fps:.1f}",
                (card_x + 12, card_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Latency: {self.performance.average_latency_ms:.1f} ms",
                (card_x + 12, card_y + 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1,
            )

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Xử lý một khung hình video và trả về khung hình đã được vẽ toàn bộ giao diện.

        Args:
            frame: Khung hình BGR từ camera.

        Returns:
            np.ndarray: Khung hình BGR đã xử lý xong.
        """
        processing_started_at = time.perf_counter()
        frame = self.detector.findHands(cv2.flip(frame, 1))
        frame_h, frame_w = frame.shape[:2]
        results = self.detector.results

        if results and results.multi_hand_landmarks and results.multi_handedness:
            self.missing_hand_frames = 0
            primary = results.multi_hand_landmarks[0]
            result = self.gesture_detector.detect_gesture(primary, mode="both")

            static_gesture, static_color = self.gesture_smoother.update(
                *result["static"]
            )
            (motion_gesture, motion_color), _ = result["motion"]

            self.object_manager.update(
                primary,
                static_gesture,
                motion_gesture,
                frame_w,
                frame_h,
            )
            self.shape_menu.update(
                primary,
                static_gesture,
                frame_w,
                frame_h,
                self.object_manager,
            )

            if self.object_manager.visible:
                self.object_manager.draw_cursor(
                    frame,
                    primary,
                    frame_w,
                    frame_h,
                )

            self._draw_gesture_labels(
                frame,
                primary,
                (static_gesture, static_color),
                (motion_gesture, motion_color),
            )
            self._draw_finger_icons(frame, results)
        else:
            self._handle_missing_hand(frame_w, frame_h)

        self.object_manager.draw_all(frame)
        self.shape_menu.draw(frame, self.object_manager.visible)
        self.performance.record_frame(processing_started_at)
        self._draw_hud(frame)
        return frame

    def run(self) -> None:
        """Khởi chạy vòng lặp xử lý camera realtime cho đến khi nhấn phím 'q'."""
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Không thể lấy dữ liệu khung hình từ camera.")

                processed_frame = self.process_frame(frame)
                cv2.imshow("Hand Gesture Controller", processed_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("d"):
                    self.show_debug = not self.show_debug
                elif key == ord("c"):
                    self.object_manager.objects.clear()
        finally:
            self.cap.release()
            self.detector.close()
            cv2.destroyAllWindows()
            if self.benchmark_output:
                self.performance.save(self.benchmark_output)


def main() -> None:
    """Hàm nạp và khởi chạy ứng dụng từ dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="Ứng dụng điều khiển vật thể bằng cử chỉ tay realtime"
    )
    parser.add_argument("--camera", type=int, default=0, help="Chỉ số camera thiết bị")
    parser.add_argument("--width", type=int, default=640, help="Chiều rộng khung hình")
    parser.add_argument("--height", type=int, default=480, help="Chiều cao khung hình")
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=5,
        help="Số khung hình dùng để làm mượt cử chỉ",
    )
    parser.add_argument(
        "--smoothing-votes",
        type=int,
        default=3,
        help="Số phiếu tối thiểu để chấp nhận một cử chỉ",
    )
    parser.add_argument(
        "--benchmark-output",
        help="Đường dẫn JSON để lưu kết quả FPS và độ trễ",
    )
    args = parser.parse_args()

    HandGestureApp(
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        smoothing_window=args.smoothing_window,
        smoothing_votes=args.smoothing_votes,
        benchmark_output=args.benchmark_output,
    ).run()


if __name__ == "__main__":
    main()

