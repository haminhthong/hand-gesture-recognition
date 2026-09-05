"""Unit tests xác thực cơ chế Hysteresis & Release trong GestureSmoother.

Đảm bảo loại bỏ hiện tượng 'sticky state' khi chuyển từ tư thế cũ sang trạng thái thả lỏng.
"""

from hand_gesture_controller.gesture_smoother import GestureSmoother


def test_smoother_releases_sticky_state_after_missing_frames():
    """Kiểm tra GestureSmoother tự động giải phóng (release) sau 2 frame vắng mặt."""
    smoother = GestureSmoother(window_size=5, minimum_votes=3, release_frames=2)
    green = (0, 255, 0)
    white = (255, 255, 255)

    # 3 frame liên tiếp "Select" -> kích hoạt trạng thái ổn định "Select"
    smoother.update("Select", green)
    smoother.update("Select", green)
    res3 = smoother.update("Select", green)
    assert res3[0] == "Select"

    # Frame 4: người dùng bắt đầu thả tay -> "Unknown" (mới 1 frame vắng mặt, vẫn giữ Select)
    res4 = smoother.update("Unknown", white)
    assert res4[0] == "Select"

    # Frame 5: tiếp tục "Unknown" (đã 2 frame liên tiếp vắng mặt Select -> trigger Release)
    res5 = smoother.update("Unknown", white)
    assert res5[0] == "Unknown"

    # Frame 6: không bị dính (sticky) Select nữa
    res6 = smoother.update("Unknown", white)
    assert res6[0] == "Unknown"


def test_smoother_requires_minimum_votes_to_activate():
    """Kiểm tra cần đủ minimum_votes (3/5) mới kích hoạt được trạng thái ổn định."""
    smoother = GestureSmoother(window_size=5, minimum_votes=3, release_frames=2)
    red = (0, 0, 255)
    blue = (255, 0, 0)

    # 2 frame "Fist" (chưa đủ 3) -> trả về nhãn hiện tại, chưa set stable
    res1 = smoother.update("Fist", red)
    assert res1[0] == "Fist"

    res2 = smoother.update("Fist", red)
    assert res2[0] == "Fist"

    # 1 frame "Stop" chen vào
    res3 = smoother.update("Stop", blue)
    assert res3[0] == "Stop"

    # Thêm 1 frame "Fist" nữa -> tổng Fist trong history = 3 -> trở thành stable
    res4 = smoother.update("Fist", red)
    assert res4[0] == "Fist"
    assert smoother._stable_result is not None
    assert smoother._stable_result[0] == "Fist"
