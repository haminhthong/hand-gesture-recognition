"""Unit tests cho mô-đun gesture_features."""

import pytest
from types import SimpleNamespace

from hand_gesture_controller.gesture_features import (
    calculate_distance_2d,
    calculate_joint_angle,
    calculate_palm_size,
    is_finger_extended,
    normalized_distance,
)


def create_mock_point(x: float, y: float, z: float = 0.0) -> SimpleNamespace:
    return SimpleNamespace(x=x, y=y, z=z)


def test_calculate_distance_2d():
    p1 = create_mock_point(0.0, 0.0)
    p2 = create_mock_point(3.0, 4.0)
    assert calculate_distance_2d(p1, p2) == pytest.approx(5.0)


def test_calculate_palm_size():
    landmarks = SimpleNamespace(
        landmark=[create_mock_point(0.0, 0.0)] * 10
    )
    landmarks.landmark[0] = create_mock_point(0.0, 0.0)
    landmarks.landmark[9] = create_mock_point(0.3, 0.4)

    assert calculate_palm_size(landmarks) == pytest.approx(0.5)


def test_calculate_palm_size_invalid():
    assert calculate_palm_size(None) == 0.0
    assert calculate_palm_size(SimpleNamespace(landmark=[])) == 0.0


def test_normalized_distance():
    p1 = create_mock_point(0.0, 0.0)
    p2 = create_mock_point(0.1, 0.0)
    ref_size = 0.2
    assert normalized_distance(p1, p2, ref_size) == pytest.approx(0.5)
    assert normalized_distance(p1, p2, 0.0) == 0.0


def test_calculate_joint_angle():
    # Góc vuông 90 độ: (1, 0) -> (0, 0) -> (0, 1)
    p_a = create_mock_point(1.0, 0.0)
    joint = create_mock_point(0.0, 0.0)
    p_b = create_mock_point(0.0, 1.0)
    assert calculate_joint_angle(p_a, joint, p_b) == pytest.approx(90.0)

    # Thẳng hàng 180 độ: (-1, 0) -> (0, 0) -> (1, 0)
    p_a2 = create_mock_point(-1.0, 0.0)
    p_b2 = create_mock_point(1.0, 0.0)
    assert calculate_joint_angle(p_a2, joint, p_b2) == pytest.approx(180.0)


def test_is_finger_extended():
    wrist = create_mock_point(0.0, 1.0)
    mcp = create_mock_point(0.0, 0.7)
    pip = create_mock_point(0.0, 0.4)
    tip = create_mock_point(0.0, 0.1)

    landmarks = [wrist, None, None, None, None, None, mcp, None, tip]
    landmarks[6] = mcp  # MCP
    landmarks[7] = pip  # PIP
    landmarks[8] = tip  # TIP

    # 180 deg extension
    assert is_finger_extended(landmarks, tip_id=8, pip_id=7, mcp_id=6, min_angle_deg=140.0) is True
