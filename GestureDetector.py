import math
import time


class GestureDetector:
    """Nhận diện cử chỉ tĩnh và chuyển động từ 21 điểm mốc bàn tay."""

    STATIC_GESTURES = (
        "Fist", "OK", "Thumbs Up", "Thumbs Down",
        "Peace", "Stop", "Select", "Options",
    )
    MOTION_GESTURES = (
        "SOS", "Wave", "Move Left", "Move Right",
        "Move Up", "Move Down", "Still", "On/Off",
    )
    GESTURES = STATIC_GESTURES + MOTION_GESTURES
    COLORS = {
        "Fist": (0, 0, 255),
        "OK": (0, 255, 0),
        "Thumbs Up": (255, 255, 0),
        "Thumbs Down": (255, 128, 0),
        "Stop": (0, 165, 255),
        "Peace": (255, 0, 255),
        "Select": (128, 0, 128),
        "Options": (0, 128, 128),
        "Unknown": (255, 255, 255),
        "SOS": (0, 0, 255),
        "Wave": (255, 255, 0),
        "Move Left": (255, 0, 255),
        "Move Right": (0, 255, 255),
        "Move Up": (0, 255, 0),
        "Move Down": (255, 165, 0),
        "Still": (255, 255, 255),
        "On/Off": (128, 128, 0),
    }
    MOVE_THRESHOLD = 0.015
    SOS_TIMEOUT_SECONDS = 1.5
    WAVE_TIMEOUT_SECONDS = 2.0
    WAVE_DIRECTION_CHANGES = 3

    def __init__(self):
        self.move_threshold = self.MOVE_THRESHOLD
        self.reset()

    def reset(self):
        """Xóa trạng thái theo thời gian khi mất dấu bàn tay."""
        self.last_detect_gesture = ("Still", self.COLORS["Still"])
        self.gestureTracking = ""
        self.sTime = 0.0
        self.eTime = time.perf_counter()
        self.wave_direction = ""
        self.wave_count = 0
        self.prev_center = None

    @staticmethod
    def distance(point_a, point_b):
        """Tính khoảng cách Euclid giữa hai điểm mốc."""
        return math.hypot(point_a.x - point_b.x, point_a.y - point_b.y)

    @staticmethod
    def get_hand_center(landmarks):
        """Tính tâm bàn tay từ cổ tay và năm đầu ngón."""
        points = landmarks.landmark
        center_ids = (0, 4, 8, 12, 16, 20)
        center_x = sum(points[index].x for index in center_ids) / len(center_ids)
        center_y = sum(points[index].y for index in center_ids) / len(center_ids)
        return center_x, center_y

    def get_move_direction(self, current_center, previous_center):
        """Xác định hướng di chuyển giữa hai khung hình liên tiếp."""
        if previous_center is None:
            return "Still"

        delta_x = current_center[0] - previous_center[0]
        delta_y = current_center[1] - previous_center[1]
        if math.hypot(delta_x, delta_y) < self.move_threshold:
            return "Still"

        # Trục y của ảnh tăng theo chiều đi xuống nên cần đảo dấu delta_y.
        angle = math.degrees(math.atan2(delta_x, -delta_y))
        if -45 <= angle <= 45:
            return "Move Up"
        if 45 < angle <= 135:
            return "Move Right"
        if angle > 135 or angle < -135:
            return "Move Down"
        return "Move Left"

    def is_finger_up(self, landmarks, tip_id, joint_id, wrist_id=0):
        """Kiểm tra một ngón có duỗi ra xa cổ tay hay không."""
        return self.distance(
            landmarks[tip_id], landmarks[wrist_id]
        ) > self.distance(landmarks[joint_id], landmarks[wrist_id])

    def is_thumb_up(self, landmarks, tip_id, joint_id, wrist_id=17):
        """Kiểm tra ngón cái có duỗi ra khỏi lòng bàn tay hay không."""
        return self.distance(
            landmarks[tip_id], landmarks[wrist_id]
        ) > self.distance(landmarks[joint_id], landmarks[wrist_id])

    def _finger_states(self, landmarks):
        """Trả về trạng thái duỗi của năm ngón theo thứ tự từ cái đến út."""
        return (
            self.is_thumb_up(landmarks, 4, 3),
            self.is_finger_up(landmarks, 8, 6),
            self.is_finger_up(landmarks, 12, 10),
            self.is_finger_up(landmarks, 16, 14),
            self.is_finger_up(landmarks, 20, 18),
        )

    def _is_on_off_start(self, landmarks, finger_states):
        _, index_up, middle_up, ring_up, pinky_up = finger_states
        return (
            index_up
            and middle_up
            and not ring_up
            and not pinky_up
            and self.distance(landmarks[8], landmarks[4]) >= 0.1
            and self.distance(landmarks[12], landmarks[4]) < 0.06
        )

    def _is_on_off_end(self, landmarks, finger_states):
        _, index_up, middle_up, ring_up, pinky_up = finger_states
        return (
            index_up
            and not middle_up
            and not ring_up
            and not pinky_up
            and self.distance(landmarks[8], landmarks[4]) >= 0.1
            and self.distance(landmarks[12], landmarks[4]) >= 0.06
        )

    def detect_static_gesture(self, landmarks):
        """Nhận diện cử chỉ dựa trên tư thế hiện tại của bàn tay."""
        if not landmarks:
            return "Unknown", self.COLORS["Unknown"]

        points = landmarks.landmark
        wrist = points[0]
        thumb_tip = points[4]
        index_tip = points[8]
        middle_tip = points[12]
        finger_states = self._finger_states(points)
        fingers_up = sum(finger_states)
        thumb_up, index_up, middle_up, _, _ = finger_states

        gesture = "Unknown"
        if fingers_up == 0:
            gesture = "Fist"
        elif fingers_up == 5:
            gesture = "Stop"
        elif fingers_up == 2 and index_up and middle_up and not thumb_up:
            gesture = "Peace"
        elif self.distance(thumb_tip, index_tip) < 0.06:
            if self.distance(middle_tip, index_tip) < 0.06:
                gesture = "Options"
            elif self.distance(middle_tip, index_tip) > 0.1 and fingers_up == 2:
                gesture = "Select"
            elif fingers_up >= 3:
                gesture = "OK"
        elif fingers_up == 1 and self.is_thumb_up(points, 4, 2):
            gesture = "Thumbs Up" if thumb_tip.y < wrist.y else "Thumbs Down"

        return gesture, self.COLORS.get(gesture, self.COLORS["Unknown"])

    def _reset_wave(self):
        """Dừng theo dõi cử chỉ vẫy tay."""
        self.gestureTracking = ""
        self.wave_direction = ""
        self.wave_count = 0
        self.prev_center = None

    def _track_wave(self, landmarks):
        """Theo dõi hướng di chuyển và nhận diện một lần vẫy tay."""
        current_center = self.get_hand_center(landmarks)
        move_direction = self.get_move_direction(current_center, self.prev_center)
        now = time.perf_counter()

        if self.gestureTracking != "Start Wave":
            self.gestureTracking = "Start Wave"
            self.wave_direction = ""
            self.wave_count = 0
            self.sTime = now
        else:
            direction = {
                "Move Left": "left",
                "Move Right": "right",
            }.get(move_direction)
            if direction and direction != self.wave_direction:
                self.wave_direction = direction
                self.wave_count += 1

        if now - self.sTime >= self.WAVE_TIMEOUT_SECONDS:
            self._reset_wave()
            return "Still"

        self.prev_center = current_center
        if self.wave_count >= self.WAVE_DIRECTION_CHANGES:
            return "Wave"
        return move_direction

    def detect_motion_gesture(self, landmarks):
        """Nhận diện cử chỉ cần theo dõi qua nhiều khung hình."""
        if not landmarks:
            return self.last_detect_gesture, self.gestureTracking

        points = landmarks.landmark
        finger_states = self._finger_states(points)
        fingers_up = sum(finger_states)
        thumb_up = finger_states[0]
        gesture = "Still"

        if not thumb_up and fingers_up == 4:
            if self.gestureTracking != "Start SOS":
                self.sTime = time.perf_counter()
                self.gestureTracking = "Start SOS"
        elif fingers_up == 0 and self.gestureTracking == "Start SOS":
            elapsed = time.perf_counter() - self.sTime
            if elapsed < self.SOS_TIMEOUT_SECONDS:
                gesture = "SOS"
            self.gestureTracking = ""
        elif self._is_on_off_start(points, finger_states):
            self.gestureTracking = "Start onoff"
        elif (
            self.gestureTracking == "Start onoff"
            and self._is_on_off_end(points, finger_states)
        ):
            gesture = "On/Off"
            self.gestureTracking = ""
        elif fingers_up == 5:
            gesture = self._track_wave(landmarks)
        elif self.gestureTracking == "Start Wave":
            self._reset_wave()

        color = self.COLORS.get(gesture, self.COLORS["Still"])
        self.last_detect_gesture = (gesture, color)
        return self.last_detect_gesture, self.gestureTracking

    def detect_gesture(self, landmarks, mode="motion"):
        """Nhận diện cử chỉ tĩnh, chuyển động hoặc cả hai."""
        if mode not in {"static", "motion", "both"}:
            raise ValueError("Chế độ phải là 'static', 'motion' hoặc 'both'.")

        if not landmarks:
            if mode == "static":
                return "Unknown", self.COLORS["Unknown"]
            if mode == "motion":
                return self.last_detect_gesture, self.gestureTracking
            return {
                "static": ("Unknown", self.COLORS["Unknown"]),
                "motion": (self.last_detect_gesture, self.gestureTracking),
            }

        if mode == "static":
            return self.detect_static_gesture(landmarks)
        if mode == "motion":
            return self.detect_motion_gesture(landmarks)
        return {
            "static": self.detect_static_gesture(landmarks),
            "motion": self.detect_motion_gesture(landmarks),
        }
