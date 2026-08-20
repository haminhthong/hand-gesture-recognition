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
        for index in dict.fromkeys([camera_index, 0, 1]):
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                return cap
            cap.release()
        raise RuntimeError("Không tìm thấy camera khả dụng.")

    def process_frame(self, frame):
        frame = self.detector.findHands(cv2.flip(frame, 1))
        frame_h, frame_w = frame.shape[:2]
        results = self.detector.results

        if results and results.multi_hand_landmarks and results.multi_handedness:
            self.missing_hand_frames = 0
            primary = results.multi_hand_landmarks[0]
            result = self.gesture_detector.detect_gesture(primary, mode="both")
            static_gesture, static_color = result["static"]
            (motion_gesture, motion_color), _ = result["motion"]
            self.object_manager.update(primary, static_gesture, motion_gesture, frame_w, frame_h)
            self.shape_menu.update(primary, static_gesture, frame_w, frame_h, self.object_manager)
            if self.object_manager.visible:
                self.object_manager.draw_cursor(frame, primary, frame_w, frame_h)

            wrist = primary.landmark[0]
            x, y = int(wrist.x * frame_w), max(25, int(wrist.y * frame_h) - 30)
            cv2.putText(frame, f"Static: {static_gesture}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, static_color, 2)
            cv2.putText(frame, f"Motion: {motion_gesture}", (x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, motion_color, 2)

            for hand_idx, landmarks in enumerate(results.multi_hand_landmarks):
                if hand_idx >= len(results.multi_handedness):
                    break
                label = results.multi_handedness[hand_idx].classification[0].label
                _, icon = self.finger_counter.detect_gesture(landmarks, label)
                if icon is not None:
                    ih, iw = icon.shape[:2]
                    icon = cv2.resize(icon, (iw * 2, ih * 2))
                    icon_x = 10 if label == "Left" else frame_w - icon.shape[1] - 10
                    overlay_image(frame, icon, icon_x, 10)
        else:
            self.missing_hand_frames += 1
            self.object_manager.update(None, "Unknown", "Still", frame_w, frame_h)
            self.shape_menu.update(None, "Unknown", frame_w, frame_h, self.object_manager)
            if self.missing_hand_frames == 5:
                self.gesture_detector.reset()

        self.object_manager.draw_all(frame)
        self.shape_menu.draw(frame, self.object_manager.visible)
        now = time.perf_counter()
        elapsed, self.previous_time = now - self.previous_time, now
        fps = 1 / elapsed if elapsed > 0 else 0
        cv2.putText(frame, f"FPS: {int(fps)}", (frame_w - 120, frame_h - 15), cv2.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2)
        return frame

    def run(self):
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
