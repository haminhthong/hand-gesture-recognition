import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union


class GestureDetector:
    """Hệ thống nhận diện cử chỉ bàn tay tĩnh và chuyển động dựa trên 21 điểm mốc (landmarks).

    Sử dụng bộ luật hình học (Rule-based Geometry Engine) dựa trên khoảng cách Euclidean,
    hướng góc vector (trên hệ tọa độ OpenCV) và Máy trạng thái hữu hạn (FSM) theo thời gian.
    """

    STATIC_GESTURES: Tuple[str, ...] = (
        "Fist",
        "OK",
        "Thumbs Up",
        "Thumbs Down",
        "Peace",
        "Stop",
        "Select",
        "Options",
    )
    MOTION_GESTURES: Tuple[str, ...] = (
        "SOS",
        "Wave",
        "Move Left",
        "Move Right",
        "Move Up",
        "Move Down",
        "Still",
        "On/Off",
    )
    GESTURES: Tuple[str, ...] = STATIC_GESTURES + MOTION_GESTURES

    COLORS: Dict[str, Tuple[int, int, int]] = {
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

    MOVE_THRESHOLD: float = 0.015
    SOS_TIMEOUT_SECONDS: float = 1.5
    WAVE_TIMEOUT_SECONDS: float = 2.0
    WAVE_DIRECTION_CHANGES: int = 3

    def __init__(self) -> None:
        """Khởi tạo bộ nhận diện cử chỉ và đặt lại trạng thái FSM ban đầu."""
        self.move_threshold = self.MOVE_THRESHOLD
        self.reset()

    def reset(self) -> None:
        """Đặt lại tất cả các biến đếm thời gian và trạng thái theo dõi khi mất dấu bàn tay."""
        self.last_detect_gesture: Tuple[str, Tuple[int, int, int]] = (
            "Still",
            self.COLORS["Still"],
        )
        self.gestureTracking: str = ""
        self.sTime: float = 0.0
        self.eTime: float = time.perf_counter()
        self.wave_direction: str = ""
        self.wave_count: int = 0
        self.prev_center: Optional[Tuple[float, float]] = None

    @staticmethod
    def distance(point_a: Any, point_b: Any) -> float:
        """Tính khoảng cách Euclidean 2D giữa hai điểm mốc (landmarks).

        Args:
            point_a: Điểm mốc thứ nhất có thuộc tính x, y (chuẩn hóa 0.0 - 1.0).
            point_b: Điểm mốc thứ hai có thuộc tính x, y.

        Returns:
            float: Khoảng cách Euclidean giữa hai điểm.
        """
        return math.hypot(point_a.x - point_b.x, point_a.y - point_b.y)

    @staticmethod
    def get_hand_center(landmarks: Any) -> Tuple[float, float]:
        """Tính trọng tâm 2D (Centroid) của bàn tay dựa trên cổ tay và 5 đầu ngón tay.

        Args:
            landmarks: Đối tượng chứa thuộc tính landmark (danh sách 21 điểm mốc).

        Returns:
            Tuple[float, float]: Tọa độ trọng tâm (center_x, center_y).
        """
        points = landmarks.landmark
        center_ids = (0, 4, 8, 12, 16, 20)
        center_x = sum(points[idx].x for idx in center_ids) / len(center_ids)
        center_y = sum(points[idx].y for idx in center_ids) / len(center_ids)
        return center_x, center_y

    def get_move_direction(
        self,
        current_center: Tuple[float, float],
        previous_center: Optional[Tuple[float, float]],
    ) -> str:
        """Xác định hướng chuyển động giữa hai vị trí trọng tâm bàn tay liên tiếp.

        Sử dụng góc lượng giác atan2 trong hệ tọa độ đồ họa (gốc ở góc trên trái).

        Args:
            current_center: Tọa độ trọng tâm khung hình hiện tại.
            previous_center: Tọa độ trọng tâm khung hình trước đó.

        Returns:
            str: Hướng di chuyển: "Still", "Move Up", "Move Down", "Move Left", "Move Right".
        """
        if previous_center is None:
            return "Still"

        delta_x = current_center[0] - previous_center[0]
        delta_y = current_center[1] - previous_center[1]

        # Kiểm tra nếu dịch chuyển quá nhỏ thì xem như đứng yên (chống nhiễu)
        if math.hypot(delta_x, delta_y) < self.move_threshold:
            return "Still"

        # Trục Y của ảnh OpenCV tăng từ trên xuống dưới -> Đảo dấu delta_y khi tính góc
        angle = math.degrees(math.atan2(delta_x, -delta_y))
        if -45 <= angle <= 45:
            return "Move Up"
        if 45 < angle <= 135:
            return "Move Right"
        if angle > 135 or angle < -135:
            return "Move Down"
        return "Move Left"

    def is_finger_up(
        self, landmarks: List[Any], tip_id: int, joint_id: int, wrist_id: int = 0
    ) -> bool:
        """Kiểm tra một ngón tay (trỏ, giữa, áp út, út) có đang duỗi ra xa cổ tay hay không.

        Args:
            landmarks: Danh sách 21 điểm mốc.
            tip_id: ID điểm mốc đầu ngón tay.
            joint_id: ID điểm mốc khớp ngón tay.
            wrist_id: ID điểm mốc cổ tay (mặc định ID 0).

        Returns:
            bool: True nếu ngón tay duỗi thẳng.
        """
        return self.distance(
            landmarks[tip_id], landmarks[wrist_id]
        ) > self.distance(landmarks[joint_id], landmarks[wrist_id])

    def is_thumb_up(
        self, landmarks: List[Any], tip_id: int, joint_id: int, wrist_id: int = 17
    ) -> bool:
        """Kiểm tra ngón cái có duỗi ra khỏi lòng bàn tay hay không.

        Dùng gốc ngón út (ID 17) làm tham chiếu khoảng cách để tính độ giang rộng của ngón cái.

        Args:
            landmarks: Danh sách 21 điểm mốc.
            tip_id: ID đầu ngón cái (ID 4).
            joint_id: ID khớp ngón cái (ID 3).
            wrist_id: ID gốc ngón út (ID 17).

        Returns:
            bool: True nếu ngón cái dang ra mở rộng.
        """
        return self.distance(
            landmarks[tip_id], landmarks[wrist_id]
        ) > self.distance(landmarks[joint_id], landmarks[wrist_id])

    def _finger_states(self, landmarks: List[Any]) -> Tuple[bool, bool, bool, bool, bool]:
        """Trả về trạng thái duỗi/gập của 5 ngón tay theo thứ tự: [Cái, Trỏ, Giữa, Áp Út, Út]."""
        return (
            self.is_thumb_up(landmarks, 4, 3),
            self.is_finger_up(landmarks, 8, 6),
            self.is_finger_up(landmarks, 12, 10),
            self.is_finger_up(landmarks, 16, 14),
            self.is_finger_up(landmarks, 20, 18),
        )

    def _is_on_off_start(
        self, landmarks: List[Any], finger_states: Tuple[bool, bool, bool, bool, bool]
    ) -> bool:
        """Trạng thái bắt đầu của cử chỉ On/Off: ngón trỏ & giữa duỗi, trỏ xa cái, giữa gần cái."""
        _, index_up, middle_up, ring_up, pinky_up = finger_states
        return (
            index_up
            and middle_up
            and not ring_up
            and not pinky_up
            and self.distance(landmarks[8], landmarks[4]) >= 0.1
            and self.distance(landmarks[12], landmarks[4]) < 0.06
        )

    def _is_on_off_end(
        self, landmarks: List[Any], finger_states: Tuple[bool, bool, bool, bool, bool]
    ) -> bool:
        """Trạng thái kết thúc của cử chỉ On/Off: ngón giữa gập lại."""
        _, index_up, middle_up, ring_up, pinky_up = finger_states
        return (
            index_up
            and not middle_up
            and not ring_up
            and not pinky_up
            and self.distance(landmarks[8], landmarks[4]) >= 0.1
            and self.distance(landmarks[12], landmarks[4]) >= 0.06
        )

    def detect_static_gesture(
        self, landmarks: Any
    ) -> Tuple[str, Tuple[int, int, int]]:
        """Nhận diện cử chỉ tĩnh dựa trên hình học bàn tay hiện tại.

        Args:
            landmarks: Đối tượng chứa 21 điểm mốc của bàn tay.

        Returns:
            Tuple[str, Tuple[int, int, int]]: (Tên cử chỉ tĩnh, Màu RGB tương ứng).
        """
        if not landmarks or not hasattr(landmarks, "landmark"):
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

    def _reset_wave(self) -> None:
        """Dừng theo dõi và xóa trạng thái cử chỉ vẫy tay."""
        self.gestureTracking = ""
        self.wave_direction = ""
        self.wave_count = 0
        self.prev_center = None

    def _track_wave(self, landmarks: Any) -> str:
        """Theo dõi chuỗi chuyển động vẫy tay qua nhiều khung hình."""
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

    def detect_motion_gesture(
        self, landmarks: Any
    ) -> Tuple[Tuple[str, Tuple[int, int, int]], str]:
        """Nhận diện cử chỉ chuyển động cần Máy Trạng Thái FSM qua nhiều khung hình.

        Args:
            landmarks: Đối tượng chứa 21 điểm mốc của bàn tay.

        Returns:
            Tuple[Tuple[str, Tuple[int, int, int]], str]: ((Nhãn cử chỉ, Màu BGR), Trạng thái tracking).
        """
        if not landmarks or not hasattr(landmarks, "landmark"):
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

    def detect_gesture(
        self, landmarks: Any, mode: str = "motion"
    ) -> Union[
        Tuple[str, Tuple[int, int, int]],
        Tuple[Tuple[str, Tuple[int, int, int]], str],
        Dict[str, Any],
    ]:
        """Nhận diện cử chỉ tay theo chế độ tĩnh ("static"), chuyển động ("motion") hoặc cả hai ("both").

        Args:
            landmarks: 21 điểm mốc bàn tay.
            mode: Chế độ nhận diện ("static", "motion", hoặc "both").

        Raises:
            ValueError: Nếu mode không thuộc 3 giá trị trên.

        Returns:
            Kết quả cử chỉ tương ứng với mode được chọn.
        """
        if mode not in {"static", "motion", "both"}:
            raise ValueError("Chế độ phải là 'static', 'motion' hoặc 'both'.")

        if not landmarks or not hasattr(landmarks, "landmark"):
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

