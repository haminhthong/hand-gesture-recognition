import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


class FingerNumber:
    """Mã hóa trạng thái 5 ngón tay (Cái, Trỏ, Giữa, Áp Út, Út) thành chuỗi mã nhị phân 5 ký tự ("00000" đến "11111").

    Tải và hiển thị ảnh overlay icon bàn tay tương ứng với mã số ngón tay đang đếm được.

    Attributes:
        hand_left_path (str): Đường dẫn thư mục ảnh icon cho bàn tay trái.
        hand_right_path (str): Đường dẫn thư mục ảnh icon cho bàn tay phải.
        left_hand_images (Dict[str, Optional[np.ndarray]]): Bộ nhớ đệm (cache) ảnh bàn tay trái.
        right_hand_images (Dict[str, Optional[np.ndarray]]): Bộ nhớ đệm (cache) ảnh bàn tay phải.
    """

    def __init__(self) -> None:
        """Khởi tạo đường dẫn thư mục tài nguyên ảnh và tạo bộ nhớ đệm."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.hand_left_path = os.path.join(base_dir, "Image", "hand_left")
        self.hand_right_path = os.path.join(base_dir, "Image", "hand_right")

        self.left_hand_images: Dict[str, Optional[np.ndarray]] = {}
        self.right_hand_images: Dict[str, Optional[np.ndarray]] = {}

    @staticmethod
    def distance(p1: Any, p2: Any) -> float:
        """Tính khoảng cách Euclidean 2D giữa 2 điểm mốc."""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def is_finger_up(
        self, landmarks: List[Any], tip_id: int, mcp_id: int, wrist_id: int = 0
    ) -> bool:
        """Kiểm tra ngón tay có duỗi ra xa hơn khớp gốc hay không."""
        return self.distance(landmarks[tip_id], landmarks[wrist_id]) > self.distance(
            landmarks[mcp_id], landmarks[wrist_id]
        )

    def is_thumb_up(
        self, landmarks: List[Any], tip_id: int, mcp_id: int, wrist_id: int = 17
    ) -> bool:
        """Kiểm tra ngón cái có xòe ra khỏi lòng bàn tay hay không."""
        return self.distance(landmarks[tip_id], landmarks[wrist_id]) > self.distance(
            landmarks[mcp_id], landmarks[wrist_id]
        )

    def detect_gesture(
        self, landmarks: Any, hand_label: str = "Right"
    ) -> Tuple[str, Optional[np.ndarray]]:
        """Xác định mã nhị phân ngón tay và lấy ảnh icon tương ứng trong cache.

        Args:
            landmarks: 21 điểm mốc bàn tay MediaPipe.
            hand_label: "Left" (bàn tay trái) hoặc "Right" (bàn tay phải).

        Returns:
            Tuple[str, Optional[np.ndarray]]: (Chuỗi mã 5 chữ số nhị phân, Mảng ảnh BGR/BGRA icon).
        """
        if not landmarks or not hasattr(landmarks, "landmark"):
            return "00000", None

        lm = landmarks.landmark
        finger_code_value = 0
        if self.is_thumb_up(lm, 4, 3):
            finger_code_value += 1
        if self.is_finger_up(lm, 8, 6):
            finger_code_value += 10
        if self.is_finger_up(lm, 12, 10):
            finger_code_value += 100
        if self.is_finger_up(lm, 16, 14):
            finger_code_value += 1000
        if self.is_finger_up(lm, 20, 18):
            finger_code_value += 10000

        finger_code = f"{finger_code_value:05d}"

        if hand_label == "Left":
            img_path = os.path.join(self.hand_left_path, f"{finger_code}.png")
            cache = self.left_hand_images
        else:
            img_path = os.path.join(self.hand_right_path, f"{finger_code}.png")
            cache = self.right_hand_images

        # Chỉ đọc ảnh từ đĩa ở lần đầu tiên gặp mã cử chỉ này
        if finger_code not in cache:
            if os.path.exists(img_path):
                cache[finger_code] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            else:
                cache[finger_code] = None

        return finger_code, cache[finger_code]
