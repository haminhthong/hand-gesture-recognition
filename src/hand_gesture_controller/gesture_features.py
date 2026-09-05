"""Mô-đun trích xuất đặc trưng hình học chuẩn hóa theo kích thước lòng bàn tay (Palm Normalization)."""

import math
from typing import Any, Dict, List

# Bảng tra cứu chuẩn hóa 21 điểm mốc MediaPipe Hands cho từng ngón tay
FINGER_LANDMARKS: Dict[str, Dict[str, int]] = {
    "thumb": {"tip": 4, "ip": 3, "mcp": 2, "cmc": 1},
    "index": {"tip": 8, "pip": 6, "mcp": 5},
    "middle": {"tip": 12, "pip": 10, "mcp": 9},
    "ring": {"tip": 16, "pip": 14, "mcp": 13},
    "pinky": {"tip": 20, "pip": 18, "mcp": 17},
}



def calculate_distance_2d(point_a: Any, point_b: Any) -> float:
    """Tính khoảng cách Euclidean 2D giữa hai điểm mốc (landmarks).

    Args:
        point_a: Điểm mốc thứ nhất có thuộc tính x, y.
        point_b: Điểm mốc thứ hai có thuộc tính x, y.

    Returns:
        float: Khoảng cách Euclidean 2D.
    """
    return math.hypot(point_a.x - point_b.x, point_a.y - point_b.y)


def calculate_palm_size(landmarks: Any) -> float:
    """Tính kích thước lòng bàn tay tham chiếu (palm size) dựa trên khoảng cách từ Cổ tay (ID 0) đến Khớp gốc ngón giữa Middle MCP (ID 9).

    Args:
        landmarks: Đối tượng chứa 21 điểm mốc MediaPipe (thuộc tính .landmark).

    Returns:
        float: Kích thước tham chiếu lòng bàn tay (chuẩn hóa 2D).
    """
    if not landmarks or not hasattr(landmarks, "landmark"):
        return 0.0
    points = landmarks.landmark
    if len(points) <= 9:
        return 0.0
    return calculate_distance_2d(points[0], points[9])


def normalized_distance(
    point_a: Any,
    point_b: Any,
    reference_size: float,
    epsilon: float = 1e-6,
) -> float:
    """Tính khoảng cách 2D đã chuẩn hóa theo kích thước lòng bàn tay.

    Args:
        point_a: Điểm mốc thứ nhất.
        point_b: Điểm mốc thứ hai.
        reference_size: Kích thước lòng bàn tay tham chiếu (palm_size).
        epsilon: Hằng số an toàn tránh chia cho 0.

    Returns:
        float: Khoảng cách đã chia cho reference_size. Trả về 0.0 nếu reference_size <= epsilon.
    """
    if reference_size <= epsilon:
        return 0.0
    return calculate_distance_2d(point_a, point_b) / reference_size


def calculate_joint_angle(
    point_a: Any,
    joint: Any,
    point_b: Any,
) -> float:
    """Tính góc (bằng độ - degrees) giữa hai vector hình thành từ 3 điểm (PointA -> Joint -> PointB).

    Sử dụng tích vô hướng (dot product) và arccos để tính góc chính xác bất kể hướng xoay bàn tay.

    Args:
        point_a: Điểm mốc đầu ngón.
        joint: Điểm mốc khớp giữa (đỉnh góc).
        point_b: Điểm mốc gốc ngón.

    Returns:
        float: Góc giữa hai đoạn thẳng tính bằng độ [0.0, 180.0].
    """
    v1_x = point_a.x - joint.x
    v1_y = point_a.y - joint.y
    v2_x = point_b.x - joint.x
    v2_y = point_b.y - joint.y

    norm1 = math.hypot(v1_x, v1_y)
    norm2 = math.hypot(v2_x, v2_y)

    if norm1 <= 1e-6 or norm2 <= 1e-6:
        return 0.0

    dot = v1_x * v2_x + v1_y * v2_y
    cos_angle = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return math.degrees(math.acos(cos_angle))


def is_finger_extended(
    landmarks_list: List[Any],
    tip_id: int,
    pip_id: int,
    mcp_id: int,
    min_angle_deg: float = 140.0,
) -> bool:
    """Kiểm tra ngón tay có đang duỗi thẳng hay không dựa trên góc tại khớp PIP và khoảng cách tới cổ tay.

    Args:
        landmarks_list: Danh sách 21 điểm mốc bàn tay.
        tip_id: ID điểm mốc đầu ngón.
        pip_id: ID điểm mốc khớp PIP.
        mcp_id: ID điểm mốc khớp MCP.
        min_angle_deg: Góc tối thiểu (độ) để ngón tay được tính là duỗi thẳng.

    Returns:
        bool: True nếu ngón tay duỗi thẳng.
    """
    if max(tip_id, pip_id, mcp_id) >= len(landmarks_list):
        return False

    angle = calculate_joint_angle(
        landmarks_list[tip_id],
        landmarks_list[pip_id],
        landmarks_list[mcp_id],
    )
    dist_tip_wrist = calculate_distance_2d(landmarks_list[tip_id], landmarks_list[0])
    dist_pip_wrist = calculate_distance_2d(landmarks_list[pip_id], landmarks_list[0])

    return angle >= min_angle_deg and dist_tip_wrist > dist_pip_wrist


def is_finger_extended_named(
    landmarks_list: List[Any],
    finger_name: str,
    min_angle_deg: float = 140.0,
) -> bool:
    """Kiểm tra ngón tay duỗi thẳng theo tên ngón tay sử dụng bảng tra cứu FINGER_LANDMARKS.

    Args:
        landmarks_list: Danh sách 21 điểm mốc bàn tay.
        finger_name: Tên ngón ("thumb", "index", "middle", "ring", "pinky").
        min_angle_deg: Góc tối thiểu (độ).

    Returns:
        bool: True nếu ngón tay duỗi thẳng.
    """
    if finger_name not in FINGER_LANDMARKS:
        raise ValueError(
            f"Tên ngón tay không hợp lệ: '{finger_name}'. Hỗ trợ: {list(FINGER_LANDMARKS.keys())}"
        )

    mapping = FINGER_LANDMARKS[finger_name]
    if finger_name == "thumb":
        return is_finger_extended(
            landmarks_list,
            tip_id=mapping["tip"],
            pip_id=mapping["ip"],
            mcp_id=mapping["mcp"],
            min_angle_deg=min_angle_deg,
        )

    return is_finger_extended(
        landmarks_list,
        tip_id=mapping["tip"],
        pip_id=mapping["pip"],
        mcp_id=mapping["mcp"],
        min_angle_deg=min_angle_deg,
    )

