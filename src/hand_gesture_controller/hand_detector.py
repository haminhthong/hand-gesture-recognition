"""Mô-đun HandDetector bao bọc MediaPipe Hands giúp phát hiện 21 điểm mốc bàn tay."""

import logging
from typing import Any, Optional
import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger(__name__)


class HandDetector:
    """Bộ bọc (Wrapper) cho MediaPipe Hands giúp phát hiện và vẽ 21 điểm mốc bàn tay.

    Attributes:
        mode (bool): Chế độ xử lý ảnh tĩnh (True) hoặc video stream (False).
        max_hands (int): Số lượng bàn tay tối đa có thể phát hiện cùng lúc.
        detection_con (float): Ngưỡng tin cậy tối thiểu để phát hiện bàn tay (0.0 - 1.0).
        track_con (float): Ngưỡng tin cậy tối thiểu để theo dõi bàn tay (0.0 - 1.0).
        results: Kết quả đầu ra gần nhất chứa các điểm mốc (landmarks).
    """

    def __init__(
        self,
        mode: bool = False,
        maxHands: int = 1,
        detectionCon: float = 0.5,
        trackCon: float = 0.5,
    ) -> None:
        """Khởi tạo mô hình nhận diện bàn tay MediaPipe."""
        if mp is None:
            logger.warning("MediaPipe chưa được cài đặt hoặc không hỗ trợ phiên bản Python này.")
            self.mp_hands = None
            self.hands = None
            self.mp_draw = None
            self.results = None
            return

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.results: Optional[Any] = None

    def close(self) -> None:
        """Giải phóng tài nguyên và mô hình MediaPipe Hands."""
        if hasattr(self, "hands") and self.hands:
            try:
                self.hands.close()
            except Exception as e:
                logger.warning("Lỗi giải phóng MediaPipe Hands: %s", e)

    def __enter__(self) -> "HandDetector":
        """Hỗ trợ cú pháp Context Manager (with HandDetector() as detector:)."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Tự động giải phóng tài nguyên khi thoát khỏi khối Context Manager."""
        self.close()

    def findHands(self, img: Optional[np.ndarray], draw: bool = True) -> Optional[np.ndarray]:
        """Phát hiện bàn tay trong khung hình BGR và vẽ bộ khung xương bàn tay lên ảnh."""
        if img is None or self.hands is None:
            return img

        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        if self.results and self.results.multi_hand_landmarks and draw and self.mp_draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )
        return img
