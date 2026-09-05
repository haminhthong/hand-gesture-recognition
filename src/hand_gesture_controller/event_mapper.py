import time
from enum import Enum
from typing import Dict, Optional, Set


class GestureEvent(Enum):
    """Tập các sự kiện hành động không phụ thuộc trực tiếp vào tên cử chỉ cụ thể."""

    NONE = "none"
    START_DRAG = "start_drag"
    DRAG = "drag"
    STOP_DRAG = "stop_drag"
    CHANGE_COLOR = "change_color"
    DELETE_OBJECT = "delete_object"
    TOGGLE_CANVAS = "toggle_canvas"
    OPEN_MENU = "open_menu"
    EMERGENCY_SOS = "emergency_sos"


CONTINUOUS_EVENTS: Set[GestureEvent] = {
    GestureEvent.START_DRAG,
    GestureEvent.DRAG,
    GestureEvent.STOP_DRAG,
}

EDGE_TRIGGERED_EVENTS: Set[GestureEvent] = {
    GestureEvent.CHANGE_COLOR,
    GestureEvent.DELETE_OBJECT,
    GestureEvent.TOGGLE_CANVAS,
    GestureEvent.OPEN_MENU,
    GestureEvent.EMERGENCY_SOS,
}

DEFAULT_COOLDOWNS: Dict[GestureEvent, float] = {
    GestureEvent.CHANGE_COLOR: 0.5,
    GestureEvent.DELETE_OBJECT: 0.5,
    GestureEvent.TOGGLE_CANVAS: 0.8,
    GestureEvent.OPEN_MENU: 0.5,
    GestureEvent.EMERGENCY_SOS: 1.0,
}


class GestureEventMapper:
    """Bộ chuyển đổi nhãn cử chỉ và trạng thái FSM thành GestureEvent cho Canvas Manager.

    Hỗ trợ phân tách Continuous Events (kéo thả con trỏ) và Edge-Triggered Events
    (xóa, đổi màu, menu, toggle) kích hoạt duy nhất tại sườn lên (Rising Edge: Inactive -> Active)
    kèm cơ chế Cooldown ngăn ngừa hiện tượng kích hoạt lặp liên tục mỗi khung hình.
    """

    def __init__(self, cooldowns: Optional[Dict[GestureEvent, float]] = None) -> None:
        """Khởi tạo GestureEventMapper.

        Args:
            cooldowns: Từ điển thời gian chờ (giây) cho từng loại sự kiện edge-triggered.
        """
        self.is_dragging: bool = False
        self.prev_gesture: Optional[str] = None
        self.prev_motion: Optional[str] = None
        self.cooldowns: Dict[GestureEvent, float] = (
            cooldowns if cooldowns is not None else DEFAULT_COOLDOWNS.copy()
        )
        self.last_event_timestamps: Dict[GestureEvent, float] = {}

    def _can_trigger_edge_event(self, event: GestureEvent, now: float) -> bool:
        """Kiểm tra sự kiện edge-triggered có thỏa mãn thời gian cooldown hay không."""
        last_time = self.last_event_timestamps.get(event, -1e9)
        cooldown = self.cooldowns.get(event, 0.5)
        if now - last_time >= cooldown:
            self.last_event_timestamps[event] = now
            return True
        return False

    def map_gesture_to_event(
        self,
        gesture_label: str,
        motion_label: str = "Still",
        timestamp: Optional[float] = None,
    ) -> GestureEvent:
        """Ánh xạ nhãn cử chỉ tĩnh và chuyển động thành GestureEvent tương ứng.

        Args:
            gesture_label: Nhãn cử chỉ tĩnh đã làm mượt (Select, Options, Stop, Fist, Peace, ...).
            motion_label: Nhãn cử chỉ động (On/Off, SOS, Wave, Still).
            timestamp: Thời điểm khung hình (giây). Nếu None, sử dụng time.perf_counter().

        Returns:
            GestureEvent: Sự kiện điều khiển hệ thống.
        """
        now = timestamp if timestamp is not None else time.perf_counter()

        # 1. Ưu tiên các cử chỉ động (Motion FSM) - Edge Triggered
        if motion_label == "On/Off":
            if self.is_dragging:
                self.is_dragging = False
            if self.prev_motion != "On/Off":
                self.prev_motion = motion_label
                self.prev_gesture = gesture_label
                if self._can_trigger_edge_event(GestureEvent.TOGGLE_CANVAS, now):
                    return GestureEvent.TOGGLE_CANVAS
            return GestureEvent.NONE

        if motion_label == "SOS":
            if self.is_dragging:
                self.is_dragging = False
            if self.prev_motion != "SOS":
                self.prev_motion = motion_label
                self.prev_gesture = gesture_label
                if self._can_trigger_edge_event(GestureEvent.EMERGENCY_SOS, now):
                    return GestureEvent.EMERGENCY_SOS
            return GestureEvent.NONE

        self.prev_motion = motion_label

        # 2. Ánh xạ cử chỉ kéo thả (Continuous Event)
        if gesture_label == "Select":
            self.prev_gesture = gesture_label
            if not self.is_dragging:
                self.is_dragging = True
                return GestureEvent.START_DRAG
            return GestureEvent.DRAG

        # Thoát trạng thái kéo thả khi không còn Select
        if self.is_dragging:
            self.is_dragging = False
            self.prev_gesture = gesture_label
            return GestureEvent.STOP_DRAG

        # 3. Ánh xạ các cử chỉ tĩnh tác vụ (Edge Triggered: chỉ kích hoạt khi cử chỉ vừa chuyển sang)
        is_rising_edge = (gesture_label != self.prev_gesture)
        self.prev_gesture = gesture_label

        if is_rising_edge:
            if gesture_label == "Options":
                if self._can_trigger_edge_event(GestureEvent.CHANGE_COLOR, now):
                    return GestureEvent.CHANGE_COLOR

            elif gesture_label == "Stop":
                if self._can_trigger_edge_event(GestureEvent.DELETE_OBJECT, now):
                    return GestureEvent.DELETE_OBJECT

            elif gesture_label == "Peace":
                if self._can_trigger_edge_event(GestureEvent.OPEN_MENU, now):
                    return GestureEvent.OPEN_MENU

        return GestureEvent.NONE

    def reset(self) -> None:
        """Đặt lại toàn bộ trạng thái event mapper khi mất dấu bàn tay."""
        self.is_dragging = False
        self.prev_gesture = None
        self.prev_motion = None
        self.last_event_timestamps.clear()

