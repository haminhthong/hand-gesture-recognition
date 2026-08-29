import argparse
import time

import cv2

import Hand_Detector as htm
from DraggableObject import DraggableObjectManager, ShapeMenu
from Finger_Number import FingerNumber
from GestureDetector import GestureDetector


def overlay_image(background, overlay, x, y):
    """Chèn ảnh BGR/BGRA và cắt phần nằm ngoài khung hình."""
    if background is None or overlay is None or overlay.ndim != 3:
        return
    bg_h, bg_w = background.shape[:2]
    ol_h, ol_w = overlay.shape[:2]
    if x >= bg_w or y >= bg_h or x + ol_w <= 0 or y + ol_h <= 0:
        return

    x_start, y_start = max(0, x), max(0, y)
    x_end, y_end = min(bg_w, x + ol_w), min(bg_h, y + ol_h)
    ox, oy = x_start - x, y_start - y
    crop = overlay[oy:oy + y_end - y_start, ox:ox + x_end - x_start]
    if crop.shape[2] == 4:
        alpha = crop[:, :, 3:4].astype(float) / 255.0
        bg_crop = background[y_start:y_end, x_start:x_end].astype(float)
        background[y_start:y_end, x_start:x_end] = (
            alpha * crop[:, :, :3] + (1 - alpha) * bg_crop
        ).astype(background.dtype)
    elif crop.shape[2] == 3:
        background[y_start:y_end, x_start:x_end] = crop


class HandGestureApp:
    """Điều phối camera, nhận diện cử chỉ và giao diện kéo thả."""

    def __init__(self, camera_index=0, width=640, height=480):
        self.width, self.height = width, height
        self.cap = self._open_camera(camera_index)
        self.detector = htm.HandDetector(detectionCon=0.7, maxHands=2)
        self.finger_counter = FingerNumber()
        self.gesture_detector = GestureDetector()
        self.object_manager = DraggableObjectManager()
        self.shape_menu = ShapeMenu()
        self.previous_time = time.perf_counter()
        self.missing_hand_frames = 0

    def _open_camera(self, camera_index):
        """Mở camera yêu cầu hoặc thử các camera phổ biến làm phương án dự phòng."""
        for index in dict.fromkeys([camera_index, 0, 1]):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                return cap
            cap.release()
        raise RuntimeError("Không tìm thấy camera khả dụng.")

    def _draw_gesture_labels(
        self,
        frame,
        hand_landmarks,
        static_result,
        motion_result,
    ):
        """Hiển thị tên cử chỉ gần cổ tay."""
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

    def _draw_finger_icons(self, frame, results):
        """Hiển thị ảnh minh họa số ngón cho từng bàn tay."""
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

    def _handle_missing_hand(self, frame_width, frame_height):
        """Đặt lại các trạng thái phụ thuộc vào bàn tay khi mất dấu."""
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

    def _draw_fps(self, frame):
        """Tính và hiển thị số khung hình xử lý mỗi giây."""
        now = time.perf_counter()
        elapsed = now - self.previous_time
        self.previous_time = now
        fps = 1 / elapsed if elapsed > 0 else 0
        frame_height, frame_width = frame.shape[:2]
        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (frame_width - 120, frame_height - 15),
            cv2.FONT_HERSHEY_PLAIN,
            1.3,
            (0, 0, 255),
            2,
        )

    def process_frame(self, frame):
        """Xử lý một khung hình và trả về ảnh đã vẽ kết quả."""
        frame = self.detector.findHands(cv2.flip(frame, 1))
        frame_h, frame_w = frame.shape[:2]
        results = self.detector.results

        if results and results.multi_hand_landmarks and results.multi_handedness:
            self.missing_hand_frames = 0
            primary = results.multi_hand_landmarks[0]
            result = self.gesture_detector.detect_gesture(primary, mode="both")
            static_gesture, static_color = result["static"]
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
        self._draw_fps(frame)
        return frame

    def run(self):
        """Chạy vòng lặp camera cho đến khi người dùng nhấn phím q."""
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Không thể lấy dữ liệu từ camera.")
                cv2.imshow("Hand Gesture Controller", self.process_frame(frame))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            self.cap.release()
            self.detector.close()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Điều khiển vật thể bằng cử chỉ tay")
    parser.add_argument("--camera", type=int, default=0, help="Chỉ số camera")
    args = parser.parse_args()
    HandGestureApp(camera_index=args.camera).run()


if __name__ == "__main__":
    main()
