"""Mô-đun ánh xạ cử chỉ thành sự kiện ứng dụng (GestureEventMapper)."""

from enum import Enum


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


class GestureEventMapper:
    """Bộ chuyển đổi nhãn cử chỉ và trạng thái FSM thành GestureEvent cho Canvas Manager."""

    def __init__(self) -> None:
        """Khởi tạo GestureEventMapper."""
        self.is_dragging: bool = False

    def map_gesture_to_event(
        self,
        gesture_label: str,
        motion_label: str = "Still",
    ) -> GestureEvent:
        """Ánh xạ nhãn cử chỉ tĩnh và chuyển động thành GestureEvent tương ứng.

        Args:
            gesture_label: Nhãn cử chỉ tĩnh đã làm mượt (Select, Options, Stop, Fist, Peace, ...).
            motion_label: Nhãn cử chỉ động (On/Off, SOS, Wave, Still).

        Returns:
            GestureEvent: Sự kiện điều khiển hệ thống.
        """
        # Ưu tiên các cử chỉ động
        if motion_label == "On/Off":
            self.is_dragging = False
            return GestureEvent.TOGGLE_CANVAS
        if motion_label == "SOS":
            self.is_dragging = False
            return GestureEvent.EMERGENCY_SOS

        # Ánh xạ cử chỉ tĩnh
        if gesture_label == "Select":
            if not self.is_dragging:
                self.is_dragging = True
                return GestureEvent.START_DRAG
            return GestureEvent.DRAG

        if self.is_dragging:
            self.is_dragging = False
            return GestureEvent.STOP_DRAG

        if gesture_label == "Options":
            return GestureEvent.CHANGE_COLOR

        if gesture_label == "Stop":
            return GestureEvent.DELETE_OBJECT

        if gesture_label == "Peace":
            return GestureEvent.OPEN_MENU

        return GestureEvent.NONE

    def reset(self) -> None:
        """Đặt lại trạng thái event mapper khi mất dấu bàn tay."""
        self.is_dragging = False
