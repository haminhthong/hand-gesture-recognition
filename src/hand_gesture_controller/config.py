"""Mô-đun cấu hình chứa các tham số và ngưỡng hình học (GestureThresholds)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class GestureThresholds:
    """Dataclass tập trung toàn bộ các ngưỡng khoảng cách đã chuẩn hóa và thời gian chờ FSM.

    Attributes:
        pinch_distance (float): Ngưỡng khoảng cách giữa ngón cái và ngón trỏ cho cử chỉ chụm ngón.
        select_distance (float): Ngưỡng khoảng cách chuẩn hóa cho cử chỉ Select.
        options_distance (float): Ngưỡng khoảng cách chuẩn hóa cho cử chỉ Options.
        movement_distance (float): Ngưỡng dịch chuyển tối thiểu lòng bàn tay để nhận diện di chuyển.
        sos_timeout_seconds (float): Thời gian tối đa (giây) hoàn tất chuỗi cử chỉ khẩn cấp SOS.
        wave_timeout_seconds (float): Thời gian tối đa (giây) duy trì hành vi vẫy tay.
        wave_direction_changes (int): Số lần đổi hướng tối thiểu để xác nhận vẫy tay.
        min_finger_extension_angle_deg (float): Góc khớp tối thiểu (độ) để xác định ngón duỗi thẳng.
    """

    pinch_distance: float = 0.35
    select_distance: float = 0.60
    options_distance: float = 0.35
    movement_distance: float = 0.12
    sos_timeout_seconds: float = 1.5
    wave_timeout_seconds: float = 2.0
    wave_direction_changes: int = 3
    min_finger_extension_angle_deg: float = 140.0


DEFAULT_THRESHOLDS = GestureThresholds()
