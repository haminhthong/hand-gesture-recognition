import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Any


class HandDetector:
    """Bộ bọc (Wrapper) cho MediaPipe Hands giúp phát hiện và vẽ 21 điểm mốc bàn tay.

    Attributes:
        mode (bool): Chế độ xử lý ảnh tĩnh (True) hoặc video stream (False).
        max_hands (int): Số lượng bàn tay tối đa có thể phát hiện cùng lúc.
        detection_con (float): Ngưỡng tin cậy tối thiểu để phát hiện bàn tay (0.0 - 1.0).
        track_con (float): Ngưỡng tin cậy tối thiểu để theo dõi bàn tay (0.0 - 1.0).
        mp_hands: Phân tích cú pháp đối tượng MediaPipe Hands.
        hands: Thể hiện mô hình xử lý MediaPipe Hands.
        mp_draw: Công cụ hỗ trợ vẽ điểm mốc và liên kết xương bàn tay từ MediaPipe.
        results: Kết quả đầu ra gần nhất chứa các điểm mốc (landmarks) và thông tin bàn tay.
    """

    def __init__(
        self,
        mode: bool = False,
        maxHands: int = 1,
        detectionCon: float = 0.5,
        trackCon: float = 0.5,
    ) -> None:
        """Khởi tạo mô hình nhận diện bàn tay MediaPipe.

        Args:
            mode: True nếu xử lý từng ảnh rời rạc, False cho luồng video liên tục.
            maxHands: Số bàn tay tối đa được nhận diện.
            detectionCon: Độ tin cậy tối thiểu cho bước nhận diện đầu tiên.
            trackCon: Độ tin cậy tối thiểu cho bước theo dõi liên tục.
        """
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
            self.hands.close()

    def __enter__(self) -> "HandDetector":
        """Hỗ trợ cú pháp Context Manager (with HandDetector() as detector:)."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Tự động giải phóng tài nguyên khi thoát khỏi khối context Manager."""
        self.close()

    def findHands(self, img: Optional[np.ndarray], draw: bool = True) -> Optional[np.ndarray]:
        """Phát hiện bàn tay trong khung hình BGR và vẽ bộ khung xương bàn tay lên ảnh.

        Args:
            img (Optional[np.ndarray]): Khung hình ảnh dạng mảng BGR NumPy.
            draw (bool): Cho phép vẽ các điểm mốc và đoạn nối xương hay không.

        Returns:
            Optional[np.ndarray]: Khung hình BGR đã được vẽ kết quả (hoặc ảnh gốc nếu img là None).
        """
        if img is None:
            return img

        # MediaPipe yêu cầu ảnh định dạng RGB thay vì BGR mặc định của OpenCV
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        if self.results and self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                )
        return img
