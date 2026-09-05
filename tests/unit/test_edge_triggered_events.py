"""Unit tests xác thực tính năng Edge-Triggered Events và Cooldown trong GestureEventMapper.

Đảm bảo các hành động như CHANGE_COLOR, DELETE_OBJECT, TOGGLE_CANVAS không bao giờ
bị kích hoạt lặp liên tục ở mỗi khung hình khi giữ nguyên cử chỉ.
"""

from hand_gesture_controller.event_mapper import GestureEvent, GestureEventMapper


def test_event_change_color_is_edge_triggered():
    """CHANGE_COLOR chỉ kích hoạt một lần duy nhất tại sườn lên (ENTER)."""
    mapper = GestureEventMapper()

    # Frame 1: Cử chỉ Options kích hoạt -> CHANGE_COLOR
    e1 = mapper.map_gesture_to_event("Options", "Still", timestamp=10.0)
    assert e1 == GestureEvent.CHANGE_COLOR

    # Frame 2: Cử chỉ Options tiếp tục giữ nguyên (HOLD) -> NONE
    e2 = mapper.map_gesture_to_event("Options", "Still", timestamp=10.033)
    assert e2 == GestureEvent.NONE

    # Frame 3: Vẫn giữ Options -> NONE
    e3 = mapper.map_gesture_to_event("Options", "Still", timestamp=10.066)
    assert e3 == GestureEvent.NONE

    # Frame 4: Thả tay sang Unknown (EXIT) -> NONE
    e4 = mapper.map_gesture_to_event("Unknown", "Still", timestamp=10.100)
    assert e4 == GestureEvent.NONE


def test_delete_is_edge_triggered():
    """DELETE_OBJECT chỉ kích hoạt một lần khi chuyển sang Stop."""
    mapper = GestureEventMapper()

    e1 = mapper.map_gesture_to_event("Stop", "Still", timestamp=5.0)
    assert e1 == GestureEvent.DELETE_OBJECT

    # Các frame kế tiếp khi vẫn giữ Stop không được xóa tiếp
    for t in (5.033, 5.066, 5.100, 5.200):
        assert mapper.map_gesture_to_event("Stop", "Still", timestamp=t) == GestureEvent.NONE


def test_toggle_canvas_does_not_repeat():
    """TOGGLE_CANVAS chỉ kích hoạt khi On/Off mới bắt đầu."""
    mapper = GestureEventMapper()

    e1 = mapper.map_gesture_to_event("Unknown", "On/Off", timestamp=1.0)
    assert e1 == GestureEvent.TOGGLE_CANVAS

    # Giữ nguyên On/Off trong các frame tiếp theo
    e2 = mapper.map_gesture_to_event("Unknown", "On/Off", timestamp=1.05)
    assert e2 == GestureEvent.NONE


def test_cooldown_rejection_and_expiration():
    """Kiểm tra thời gian cooldown ngăn kích hoạt lại quá nhanh ngay cả khi re-enter."""
    mapper = GestureEventMapper()  # CHANGE_COLOR cooldown = 0.5s

    # Lần 1: Kích hoạt lúc t=2.0
    assert mapper.map_gesture_to_event("Options", "Still", timestamp=2.0) == GestureEvent.CHANGE_COLOR

    # Thoát cử chỉ lúc t=2.1
    assert mapper.map_gesture_to_event("Unknown", "Still", timestamp=2.1) == GestureEvent.NONE

    # Vào lại Options lúc t=2.3 (chưa hết 0.5s cooldown) -> Phải bị từ chối (NONE)
    assert mapper.map_gesture_to_event("Options", "Still", timestamp=2.3) == GestureEvent.NONE

    # Vào lại Options lúc t=2.6 (> 2.0 + 0.5s cooldown) -> Thành công
    assert mapper.map_gesture_to_event("Unknown", "Still", timestamp=2.5) == GestureEvent.NONE
    assert mapper.map_gesture_to_event("Options", "Still", timestamp=2.6) == GestureEvent.CHANGE_COLOR


def test_continuous_drag_maintains_drag_event():
    """Kiểm tra sự kiện continuous (Select) vẫn duy trì DRAG ở mọi khung hình khi giữ."""
    mapper = GestureEventMapper()

    # Frame 1: Select -> START_DRAG
    assert mapper.map_gesture_to_event("Select", "Still", timestamp=0.1) == GestureEvent.START_DRAG
    assert mapper.is_dragging is True

    # Frame 2..5: Tiếp tục Select -> DRAG mỗi khung hình
    for t in (0.133, 0.166, 0.200, 0.233):
        assert mapper.map_gesture_to_event("Select", "Still", timestamp=t) == GestureEvent.DRAG

    # Thả tay khỏi Select -> STOP_DRAG
    assert mapper.map_gesture_to_event("Fist", "Still", timestamp=0.266) == GestureEvent.STOP_DRAG
    assert mapper.is_dragging is False
