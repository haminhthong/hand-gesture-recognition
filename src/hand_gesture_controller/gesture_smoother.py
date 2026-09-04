"""Mô-đun GestureSmoother làm mượt nhãn cử chỉ tĩnh bằng thuật toán Biểu quyết Đa số Cửa sổ Trượt."""

from collections import Counter, deque
from typing import Deque, Optional, Tuple


class GestureSmoother:
    """Bộ làm mượt nhãn cử chỉ sử dụng thuật toán Biểu quyết Đa số trong Cửa sổ Trượt (Sliding Window Majority Voting).

    Attributes:
        window_size (int): Số khung hình tối đa lưu giữ trong bộ nhớ đệm.
        minimum_votes (int): Số lượng phiếu tối thiểu cần thiết để xác nhận nhãn mượt.
    """

    def __init__(self, window_size: int = 5, minimum_votes: int = 3) -> None:
        """Khởi tạo GestureSmoother.

        Args:
            window_size: Kích thước cửa sổ trượt.
            minimum_votes: Số phiếu đồng thuận tối thiểu.

        Raises:
            ValueError: Nếu window_size < 1 hoặc minimum_votes không thuộc [1, window_size].
        """
        if window_size < 1:
            raise ValueError("Kích thước cửa sổ phải lớn hơn 0.")
        if not 1 <= minimum_votes <= window_size:
            raise ValueError("Số phiếu tối thiểu phải nằm trong kích thước cửa sổ.")

        self.window_size = window_size
        self.minimum_votes = minimum_votes
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
        majority_label, votes = Counter(
            item[0] for item in self._history
        ).most_common(1)[0]

        if votes >= self.minimum_votes:
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
