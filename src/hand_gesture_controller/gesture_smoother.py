"""Mô-đun GestureSmoother làm mượt nhãn cử chỉ tĩnh bằng thuật toán Biểu quyết Đa số Cửa sổ Trượt."""

from collections import Counter, deque
from typing import Deque, Optional, Tuple


class GestureSmoother:
    """Bộ làm mượt nhãn cử chỉ sử dụng thuật toán Biểu quyết Đa số trong Cửa sổ Trượt kết hợp Hysteresis & Release.

    Ngăn chặn triệt để hiện tượng 'sticky state' (cử chỉ cũ bị giữ mãi khi không có cử chỉ mới đạt đa số)
    bằng chính sách giải phóng (release policy) khi cử chỉ ổn định vắng mặt trong các khung hình gần nhất.

    Attributes:
        window_size (int): Số khung hình tối đa lưu giữ trong bộ nhớ đệm.
        minimum_votes (int): Số lượng phiếu tối thiểu cần thiết để xác nhận nhãn mượt (Activation).
        release_frames (int): Số khung hình liên tiếp vắng mặt để giải phóng trạng thái ổn định (Release).
    """

    def __init__(
        self,
        window_size: int = 5,
        minimum_votes: int = 3,
        release_frames: int = 2,
    ) -> None:
        """Khởi tạo GestureSmoother.

        Args:
            window_size: Kích thước cửa sổ trượt.
            minimum_votes: Số phiếu đồng thuận tối thiểu để kích hoạt.
            release_frames: Số khung hình liên tiếp không thấy cử chỉ cũ để giải phóng.

        Raises:
            ValueError: Nếu tham số cấu hình không hợp lệ.
        """
        if window_size < 1:
            raise ValueError("Kích thước cửa sổ phải lớn hơn 0.")
        if not 1 <= minimum_votes <= window_size:
            raise ValueError("Số phiếu tối thiểu phải nằm trong kích thước cửa sổ.")
        if release_frames < 1:
            raise ValueError("Số frame giải phóng (release_frames) phải lớn hơn hoặc bằng 1.")

        self.window_size = window_size
        self.minimum_votes = minimum_votes
        self.release_frames = release_frames
        self._history: Deque[Tuple[str, Tuple[int, int, int]]] = deque(maxlen=window_size)
        self._stable_result: Optional[Tuple[str, Tuple[int, int, int]]] = None

    def update(
        self, label: str, color: Tuple[int, int, int]
    ) -> Tuple[str, Tuple[int, int, int]]:
        """Thêm nhãn cử chỉ từ khung hình mới và tính toán nhãn cử chỉ mượt nhất.

        Args:
            label: Nhãn cử chỉ từ khung hình hiện tại.
            color: Màu RGB đại diện cho cử chỉ đó.

        Returns:
            Tuple[str, Tuple[int, int, int]]: (Nhãn cử chỉ đã mượt, Màu tương ứng).
        """
        self._history.append((label, color))

        # 1. Hysteresis Release: Giải phóng stable_result nếu cử chỉ ổn định đã biến mất
        if self._stable_result is not None:
            stable_label = self._stable_result[0]
            votes_for_stable = sum(1 for item in self._history if item[0] == stable_label)
            if votes_for_stable == 0:
                self._stable_result = None
            elif len(self._history) >= self.release_frames:
                recent_labels = [item[0] for item in list(self._history)[-self.release_frames:]]
                if stable_label not in recent_labels:
                    self._stable_result = None

        # 2. Activation: Kích hoạt khi có nhãn đạt đủ minimum_votes VÀ có mặt trong recent_frames
        majority_label, votes = Counter(
            item[0] for item in self._history
        ).most_common(1)[0]

        if votes >= self.minimum_votes:
            recent_labels = [item[0] for item in list(self._history)[-self.release_frames:]]
            if majority_label in recent_labels:
                majority_color = next(
                    item_color
                    for item_label, item_color in reversed(self._history)
                    if item_label == majority_label
                )
                self._stable_result = majority_label, majority_color


        return self._stable_result or (label, color)

    def reset(self) -> None:
        """Xóa toàn bộ lịch sử biểu quyết khi bàn tay biến mất khỏi khung hình."""
        self._history.clear()
        self._stable_result = None

