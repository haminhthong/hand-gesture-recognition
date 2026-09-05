"""Bộ nhận diện cử chỉ (GestureDetector) dựa trên luật hình học chuẩn hóa và FSM."""

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .config import DEFAULT_THRESHOLDS, GestureThresholds
from .gesture_features import (
    FINGER_LANDMARKS,
    calculate_distance_2d,
    calculate_palm_size,
    is_finger_extended,
    normalized_distance,
)


@dataclass(frozen=True)
class GestureResult:
    """Kết quả phân loại cử chỉ kèm điểm độ khớp quy tắc hình học (rule_score) hoặc xác suất ML.

    Attributes:
        label (str): Nhãn cử chỉ được phân loại.
        confidence (float): Độ tin cậy (alias tương thích ngược cho rule_score) [0.0, 1.0].
        rule_score (float): Điểm độ khớp heuristic phản ánh margin điều kiện hình học [0.0, 1.0].
        source (str): Nguồn phân loại ("rule_based" hoặc "ml").
    """

    label: str
    confidence: float
    rule_score: float = 0.0
    source: str = "rule_based"

    def __post_init__(self) -> None:
        if self.rule_score == 0.0 and self.confidence != 0.0:
            object.__setattr__(self, "rule_score", self.confidence)
        elif self.confidence == 0.0 and self.rule_score != 0.0:
            object.__setattr__(self, "confidence", self.rule_score)


class GestureDetector:
    """Hệ thống nhận diện cử chỉ bàn tay dựa trên luật hình học đã chuẩn hóa theo kích thước lòng bàn tay (Palm-size Normalized) và FSM.

    Attributes:
        thresholds (GestureThresholds): Đối tượng lưu trữ các ngưỡng phân loại.
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

    def __init__(self, thresholds: Optional[GestureThresholds] = None) -> None:
        """Khởi tạo GestureDetector.

        Args:
            thresholds: Cấu hình các ngưỡng hình học. Nếu None, sử dụng DEFAULT_THRESHOLDS.
        """
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.move_threshold = self.thresholds.movement_distance
        self.reset()

    def reset(self) -> None:
        """Đặt lại tất cả các biến FSM khi mất dấu bàn tay."""
        self.last_detect_gesture: Tuple[str, Tuple[int, int, int]] = (
            "Still",
            self.COLORS["Still"],
        )
        self.gestureTracking: str = ""
        self.sTime: float = 0.0
        self.wave_direction: str = ""
        self.wave_count: int = 0
        self.prev_center: Optional[Tuple[float, float]] = None

    @staticmethod
    def distance(point_a: Any, point_b: Any) -> float:
        """Tính khoảng cách Euclidean 2D giữa hai điểm mốc."""
        return calculate_distance_2d(point_a, point_b)

    @staticmethod
    def get_hand_center(landmarks: Any) -> Tuple[float, float]:
        """Tính trọng tâm 2D của bàn tay."""
        points = landmarks.landmark
        center_ids = (0, 4, 8, 12, 16, 20)
        center_x = sum(points[idx].x for idx in center_ids) / len(center_ids)
        center_y = sum(points[idx].y for idx in center_ids) / len(center_ids)
        return center_x, center_y

    def get_move_direction(
        self,
        current_center: Tuple[float, float],
        previous_center: Optional[Tuple[float, float]],
        palm_size: float = 0.2,
    ) -> str:
        """Xác định hướng di chuyển bàn tay."""
        if previous_center is None:
            return "Still"

        delta_x = current_center[0] - previous_center[0]
        delta_y = current_center[1] - previous_center[1]
        dist = math.hypot(delta_x, delta_y)

        # Chuẩn hóa khoảng cách di chuyển theo palm_size nếu có
        norm_dist = dist / max(palm_size, 1e-6)
        if norm_dist < self.thresholds.movement_distance:
            return "Still"

        angle = math.degrees(math.atan2(delta_x, -delta_y))
        if -45 <= angle <= 45:
            return "Move Up"
        if 45 < angle <= 135:
            return "Move Right"
        if angle > 135 or angle < -135:
            return "Move Down"
        return "Move Left"

    def is_finger_up(
        self,
        landmarks: List[Any],
        tip_id: int,
        joint_id: int,
        mcp_id: Optional[int] = None,
        wrist_id: int = 0,
    ) -> bool:
        """Kiểm tra ngón tay duỗi ra dựa trên góc khớp chuẩn TIP -> PIP -> MCP.

        Args:
            landmarks: Danh sách các điểm mốc bàn tay.
            tip_id: ID điểm mốc đầu ngón.
            joint_id: ID điểm mốc khớp PIP (khớp giữa).
            mcp_id: ID điểm mốc khớp MCP (khớp gốc). Nếu None, tự động tra cứu từ FINGER_LANDMARKS.
            wrist_id: ID điểm mốc cổ tay (mặc định 0).
        """
        pip_id = joint_id
        if mcp_id is None:
            # Tra cứu chuẩn xác từ FINGER_LANDMARKS thay vì phỏng đoán joint_id + 1
            mapping_by_tip = {
                FINGER_LANDMARKS["index"]["tip"]: FINGER_LANDMARKS["index"]["mcp"],      # 8 -> 5
                FINGER_LANDMARKS["middle"]["tip"]: FINGER_LANDMARKS["middle"]["mcp"],    # 12 -> 9
                FINGER_LANDMARKS["ring"]["tip"]: FINGER_LANDMARKS["ring"]["mcp"],        # 16 -> 13
                FINGER_LANDMARKS["pinky"]["tip"]: FINGER_LANDMARKS["pinky"]["mcp"],      # 20 -> 17
            }
            mapping_by_pip = {
                FINGER_LANDMARKS["index"]["pip"]: FINGER_LANDMARKS["index"]["mcp"],      # 6 -> 5
                FINGER_LANDMARKS["middle"]["pip"]: FINGER_LANDMARKS["middle"]["mcp"],    # 10 -> 9
                FINGER_LANDMARKS["ring"]["pip"]: FINGER_LANDMARKS["ring"]["mcp"],        # 14 -> 13
                FINGER_LANDMARKS["pinky"]["pip"]: FINGER_LANDMARKS["pinky"]["mcp"],      # 18 -> 17
            }
            mcp_id = mapping_by_tip.get(tip_id, mapping_by_pip.get(joint_id, max(0, joint_id - 1)))

        return is_finger_extended(
            landmarks,
            tip_id,
            pip_id,
            mcp_id,
            min_angle_deg=self.thresholds.min_finger_extension_angle_deg,
        )

    def is_thumb_up(
        self, landmarks: List[Any], tip_id: int = 4, joint_id: int = 3, wrist_id: int = 17
    ) -> bool:
        """Kiểm tra ngón cái duỗi ra khỏi lòng bàn tay."""
        return calculate_distance_2d(
            landmarks[tip_id], landmarks[wrist_id]
        ) > calculate_distance_2d(landmarks[joint_id], landmarks[wrist_id])

    def _finger_states(self, landmarks: List[Any]) -> Tuple[bool, bool, bool, bool, bool]:
        """Trả về trạng thái duỗi/gập của 5 ngón tay: [Cái, Trỏ, Giữa, Áp Út, Út] sử dụng MCP chuẩn."""
        return (
            self.is_thumb_up(
                landmarks,
                tip_id=FINGER_LANDMARKS["thumb"]["tip"],
                joint_id=FINGER_LANDMARKS["thumb"]["ip"],
            ),
            self.is_finger_up(
                landmarks,
                tip_id=FINGER_LANDMARKS["index"]["tip"],
                joint_id=FINGER_LANDMARKS["index"]["pip"],
                mcp_id=FINGER_LANDMARKS["index"]["mcp"],
            ),
            self.is_finger_up(
                landmarks,
                tip_id=FINGER_LANDMARKS["middle"]["tip"],
                joint_id=FINGER_LANDMARKS["middle"]["pip"],
                mcp_id=FINGER_LANDMARKS["middle"]["mcp"],
            ),
            self.is_finger_up(
                landmarks,
                tip_id=FINGER_LANDMARKS["ring"]["tip"],
                joint_id=FINGER_LANDMARKS["ring"]["pip"],
                mcp_id=FINGER_LANDMARKS["ring"]["mcp"],
            ),
            self.is_finger_up(
                landmarks,
                tip_id=FINGER_LANDMARKS["pinky"]["tip"],
                joint_id=FINGER_LANDMARKS["pinky"]["pip"],
                mcp_id=FINGER_LANDMARKS["pinky"]["mcp"],
            ),
        )

    def _is_on_off_start(
        self,
        landmarks: List[Any],
        finger_states: Tuple[bool, bool, bool, bool, bool],
        palm_size: float,
    ) -> bool:
        """Trạng thái bắt đầu cử chỉ On/Off."""
        _, index_up, middle_up, ring_up, pinky_up = finger_states
        d8_4 = normalized_distance(landmarks[8], landmarks[4], palm_size)
        d12_4 = normalized_distance(landmarks[12], landmarks[4], palm_size)
        return (
            index_up
            and middle_up
            and not ring_up
            and not pinky_up
            and d8_4 >= 0.5
            and d12_4 < 0.35
        )

    def _is_on_off_end(
        self,
        landmarks: List[Any],
        finger_states: Tuple[bool, bool, bool, bool, bool],
        palm_size: float,
    ) -> bool:
        """Trạng thái kết thúc cử chỉ On/Off."""
        _, index_up, middle_up, ring_up, pinky_up = finger_states
        d8_4 = normalized_distance(landmarks[8], landmarks[4], palm_size)
        d12_4 = normalized_distance(landmarks[12], landmarks[4], palm_size)
        return (
            index_up
            and not middle_up
            and not ring_up
            and not pinky_up
            and d8_4 >= 0.5
            and d12_4 >= 0.35
        )

    def detect_static_gesture_result(self, landmarks: Any) -> GestureResult:
        """Nhận diện cử chỉ tĩnh và trả về GestureResult có rule_score / rule confidence.

        Thứ tự ưu tiên (Precedence Table):
        1. Fist: Không có ngón nào duỗi (fingers_up == 0)
        2. Stop: Tất cả 5 ngón duỗi (fingers_up == 5)
        3. Peace: Ngón trỏ & giữa duỗi, ngón cái gập (fingers_up == 2)
        4. Pinch branch (d_norm(thumb, index) < select_distance):
           a. Options: d_norm(middle, index) < options_distance (chụm 3 ngón)
           b. Select: d_norm(middle, index) > 0.4 và fingers_up == 2 (chụm 2 ngón)
           c. OK: fingers_up >= 3
        5. Thumbs Up / Thumbs Down: Chỉ ngón cái duỗi (fingers_up == 1)

        Args:
            landmarks: Đối tượng chứa 21 điểm mốc MediaPipe.

        Returns:
            GestureResult: Kết quả phân loại kèm điểm heuristic rule_score [0.0, 1.0].
        """
        if not landmarks or not hasattr(landmarks, "landmark"):
            return GestureResult(label="Unknown", confidence=0.0, rule_score=0.0, source="rule_based")

        points = landmarks.landmark
        if len(points) < 21:
            return GestureResult(label="Unknown", confidence=0.0, rule_score=0.0, source="rule_based")

        palm_sz = calculate_palm_size(landmarks)
        wrist = points[0]
        thumb_tip = points[4]
        index_tip = points[8]
        middle_tip = points[12]

        finger_states = self._finger_states(points)
        fingers_up = sum(finger_states)
        thumb_up, index_up, middle_up, _, _ = finger_states

        gesture = "Unknown"
        rule_score = 0.50

        norm_d_thumb_index = normalized_distance(thumb_tip, index_tip, palm_sz)
        norm_d_middle_index = normalized_distance(middle_tip, index_tip, palm_sz)

        if fingers_up == 0:
            gesture = "Fist"
            rule_score = 0.95
        elif fingers_up == 5:
            gesture = "Stop"
            rule_score = 0.95
        elif fingers_up == 2 and index_up and middle_up and not thumb_up and norm_d_thumb_index >= self.thresholds.select_distance:
            gesture = "Peace"
            rule_score = 0.90
        elif norm_d_thumb_index < self.thresholds.select_distance:
            pinch_margin = max(0.0, min(1.0, 1.0 - norm_d_thumb_index / max(self.thresholds.select_distance, 1e-6)))
            if norm_d_middle_index < self.thresholds.options_distance:
                gesture = "Options"
                opt_margin = max(0.0, min(1.0, 1.0 - norm_d_middle_index / max(self.thresholds.options_distance, 1e-6)))
                rule_score = round(0.70 + 0.25 * ((pinch_margin + opt_margin) / 2.0), 2)
            elif norm_d_middle_index > 0.4 and fingers_up == 2:
                gesture = "Select"
                rule_score = round(0.70 + 0.25 * pinch_margin, 2)
            elif fingers_up >= 3:
                gesture = "OK"
                rule_score = round(0.70 + 0.20 * pinch_margin, 2)
        elif fingers_up == 1 and self.is_thumb_up(points, 4, 2):
            gesture = "Thumbs Up" if thumb_tip.y < wrist.y else "Thumbs Down"
            y_diff = abs(thumb_tip.y - wrist.y)
            thumb_margin = max(0.0, min(1.0, y_diff / max(palm_sz, 1e-6)))
            rule_score = round(0.70 + 0.25 * thumb_margin, 2)

        return GestureResult(
            label=gesture,
            confidence=rule_score,
            rule_score=rule_score,
            source="rule_based",
        )

    def detect_static_gesture(
        self, landmarks: Any
    ) -> Tuple[str, Tuple[int, int, int]]:
        """API tương thích cũ trả về (tên_cử_chỉ, màu_RGB)."""
        result = self.detect_static_gesture_result(landmarks)
        return result.label, self.COLORS.get(result.label, self.COLORS["Unknown"])

    def _reset_wave(self) -> None:
        self.gestureTracking = ""
        self.wave_direction = ""
        self.wave_count = 0
        self.prev_center = None

    def _track_wave(self, landmarks: Any) -> str:
        current_center = self.get_hand_center(landmarks)
        palm_sz = calculate_palm_size(landmarks)
        move_direction = self.get_move_direction(current_center, self.prev_center, palm_sz)
        now = time.perf_counter()

        if self.gestureTracking != "Start Wave":
            self.gestureTracking = "Start Wave"
            self.wave_direction = ""
            self.wave_count = 0
            self.sTime = now
        else:
            direction = {"Move Left": "left", "Move Right": "right"}.get(move_direction)
            if direction:
                if not self.wave_direction:
                    self.wave_direction = direction
                elif direction != self.wave_direction:
                    self.wave_direction = direction
                    self.wave_count += 1

        if now - self.sTime >= self.thresholds.wave_timeout_seconds:
            self._reset_wave()
            return "Still"

        self.prev_center = current_center
        if self.wave_count >= self.thresholds.wave_direction_changes:
            self._reset_wave()
            return "Wave"
        return move_direction


    def detect_motion_gesture(
        self, landmarks: Any
    ) -> Tuple[Tuple[str, Tuple[int, int, int]], str]:
        """Nhận diện cử chỉ chuyển động sử dụng FSM."""
        if not landmarks or not hasattr(landmarks, "landmark"):
            return self.last_detect_gesture, self.gestureTracking

        points = landmarks.landmark
        palm_sz = calculate_palm_size(landmarks)
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
            if elapsed < self.thresholds.sos_timeout_seconds:
                gesture = "SOS"
            self.gestureTracking = ""
        elif self._is_on_off_start(points, finger_states, palm_sz):
            self.gestureTracking = "Start onoff"
        elif (
            self.gestureTracking == "Start onoff"
            and self._is_on_off_end(points, finger_states, palm_sz)
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
        """Phương thức điều phối nhận diện cử chỉ theo mode."""
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
