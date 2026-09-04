"""Mô-đun tiền xử lý điểm mốc 3D (LandmarkPreprocessor) dùng chung cho Inference và Training ML."""

from typing import Any, Optional
import numpy as np


class LandmarkPreprocessor:
    """Bộ tiền xử lý chuẩn hóa 21 điểm mốc 3D MediaPipe thành vector đặc trưng 63 chiều bất biến với vị trí và tỉ lệ.

    Attributes:
        mirror_left_hand (bool): Nếu True, tự động lật tọa độ X của tay trái để đưa về cùng hệ quy chiếu với tay phải.
    """

    def __init__(self, mirror_left_hand: bool = True) -> None:
        """Khởi tạo LandmarkPreprocessor.

        Args:
            mirror_left_hand: Tự động lật trục X bàn tay trái về hệ quy chiếu bàn tay phải.
        """
        self.mirror_left_hand = mirror_left_hand

    def transform_landmarks_object(
        self,
        landmarks: Any,
        handedness: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Chuẩn hóa đối tượng MediaPipe landmarks thành mảng NumPy 1D 63 chiều.

        Args:
            landmarks: Đối tượng chứa thuộc tính .landmark (danh sách 21 điểm mốc).
            handedness: "Left" hoặc "Right" (nếu có).

        Returns:
            Optional[np.ndarray]: Mảng 1D 63 phần tử (float32) hoặc None nếu landmarks không hợp lệ.
        """
        if not landmarks or not hasattr(landmarks, "landmark"):
            return None
        points = landmarks.landmark
        if len(points) < 21:
            return None

        coords = np.array([[lm.x, lm.y, lm.z] for lm in points[:21]], dtype=np.float32)
        return self.transform(coords, handedness=handedness)

    def transform(
        self,
        coords: np.ndarray,
        handedness: Optional[str] = None,
    ) -> np.ndarray:
        """Thực hiện tiền xử lý chuẩn hóa mảng tọa độ 2D/3D.

        Các bước xử lý:
        1. Dịch cổ tay (index 0) về gốc tọa độ (0, 0, 0).
        2. Tính palm_size = distance(wrist, middle_mcp) và chia các điểm mốc cho palm_size.
        3. Phản chiếu tay trái nếu mirror_left_hand = True.
        4. Trải phẳng thành vector 63 chiều.

        Args:
            coords: Mảng NumPy kích thước (21, 3) đại diện 21 điểm mốc (x, y, z).
            handedness: Nhãn "Left" hoặc "Right".

        Returns:
            np.ndarray: Vector 1D độ dài 63 (float32).
        """
        if coords.shape != (21, 3):
            raise ValueError(f"Kích thước coords phải là (21, 3), nhận được {coords.shape}")

        coords = coords.copy()

        # Step 1: Translate wrist (index 0) to origin
        wrist = coords[0].copy()
        coords = coords - wrist

        # Step 2: Scale by palm size (distance between wrist [0] and middle_mcp [9])
        palm_size = float(np.linalg.norm(coords[9]))
        if palm_size > 1e-6:
            coords = coords / palm_size

        # Step 3: Mirror left hand along X axis to match right hand
        if self.mirror_left_hand and handedness == "Left":
            coords[:, 0] = -coords[:, 0]

        # Step 4: Flatten vector
        return coords.reshape(-1).astype(np.float32)
