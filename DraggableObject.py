import math
import random
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np


class DraggableObject:
    """Lớp cơ sở biểu diễn vật thể hình chữ nhật có thể kéo thả trên màn hình.

    Attributes:
        x (int): Tọa độ X góc trên bên trái.
        y (int): Tọa độ Y góc trên bên trái.
        width (int): Chiều rộng vật thể.
        height (int): Chiều cao vật thể.
        color (Tuple[int, int, int]): Màu sắc BGR của vật thể.
        name (str): Tên gọi của vật thể.
        is_dragging (bool): Trạng thái vật thể có đang được kéo thả hay không.
        offset_x (int): Khoảng cách lệch X giữa con trỏ và vị trí vật thể khi bắt đầu kéo.
        offset_y (int): Khoảng cách lệch Y giữa con trỏ và vị trí vật thể khi bắt đầu kéo.
    """

    def __init__(
        self,
        x: int,
        y: int,
        width: int,
        height: int,
        color: Tuple[int, int, int] = (0, 255, 0),
        name: str = "Object",
    ) -> None:
        """Khởi tạo một vật thể hình chữ nhật kéo thả.

        Args:
            x: Tọa độ X ban đầu.
            y: Tọa độ Y ban đầu.
            width: Chiều rộng vật thể.
            height: Chiều cao vật thể.
            color: Màu BGR ban đầu.
            name: Tên nhận diện vật thể.
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.name = name
        self.is_dragging = False
        self.offset_x = 0
        self.offset_y = 0

    def is_point_inside(self, px: int, py: int) -> bool:
        """Kiểm tra xem tọa độ (px, py) của con trỏ có nằm trong vùng vật thể hay không.

        Args:
            px: Tọa độ X con trỏ.
            py: Tọa độ Y con trỏ.

        Returns:
            bool: True nếu con trỏ nằm bên trong vật thể.
        """
        return (
            self.x <= px <= self.x + self.width
            and self.y <= py <= self.y + self.height
        )

    def start_drag(self, px: int, py: int) -> bool:
        """Bắt đầu trạng thái kéo thả nếu con trỏ chạm vào vật thể.

        Args:
            px: Tọa độ X con trỏ.
            py: Tọa độ Y con trỏ.

        Returns:
            bool: True nếu kích hoạt kéo thả thành công.
        """
        if self.is_point_inside(px, py):
            self.is_dragging = True
            self.offset_x = px - self.x
            self.offset_y = py - self.y
            return True
        return False

    def update_position(
        self,
        px: int,
        py: int,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> None:
        """Cập nhật vị trí vật thể theo tọa độ con trỏ và giới hạn trong khung hình.

        Args:
            px: Tọa độ X mới của con trỏ.
            py: Tọa độ Y mới của con trỏ.
            frame_width: Chiều rộng khung hình để giới hạn.
            frame_height: Chiều cao khung hình để giới hạn.
        """
        if self.is_dragging:
            self.x = px - self.offset_x
            self.y = py - self.offset_y
            if frame_width is not None:
                self.x = max(0, min(self.x, max(0, frame_width - self.width)))
            if frame_height is not None:
                self.y = max(0, min(self.y, max(0, frame_height - self.height)))

    def stop_drag(self) -> None:
        """Kết thúc trạng thái kéo thả."""
        self.is_dragging = False

    def change_color(self) -> None:
        """Đổi màu ngẫu nhiên cho vật thể khi nhận được cử chỉ tương tác."""
        self.color = (
            random.randint(50, 255),
            random.randint(50, 255),
            random.randint(50, 255),
        )

    def clone(self, x: int, y: int) -> "DraggableObject":
        """Tạo bản sao mới của vật thể tại vị trí (x, y)."""
        return DraggableObject(x, y, self.width, self.height, self.color, self.name)

    def draw(self, frame: np.ndarray) -> None:
        """Vẽ vật thể hình chữ nhật lên khung hình OpenCV.

        Args:
            frame: Khung hình BGR cần vẽ.
        """
        color = self.color if not self.is_dragging else (0, 255, 255)
        top_left = (int(self.x), int(self.y))
        bottom_right = (int(self.x + self.width), int(self.y + self.height))
        cv2.rectangle(frame, top_left, bottom_right, color, -1)
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)


class CircleObject(DraggableObject):
    """Vật thể hình tròn có thể kéo thả trên màn hình."""

    def __init__(
        self,
        x: int,
        y: int,
        radius: int,
        color: Tuple[int, int, int] = (0, 255, 0),
        name: str = "Circle",
    ) -> None:
        super().__init__(x, y, radius * 2, radius * 2, color, name)
        self.radius = radius

    def clone(self, x: int, y: int) -> "CircleObject":
        return CircleObject(x, y, self.radius, self.color, self.name)

    def is_point_inside(self, px: int, py: int) -> bool:
        """Kiểm tra khoảng cách từ con trỏ tới tâm tròn có nhỏ hơn hoặc bằng bán kính."""
        center_x = self.x + self.radius
        center_y = self.y + self.radius
        distance = math.hypot(px - center_x, py - center_y)
        return distance <= self.radius

    def draw(self, frame: np.ndarray) -> None:
        """Vẽ hình tròn lên khung hình OpenCV."""
        center_x = int(self.x + self.radius)
        center_y = int(self.y + self.radius)
        color = self.color if not self.is_dragging else (0, 255, 255)
        cv2.circle(frame, (center_x, center_y), int(self.radius), color, -1)
        cv2.circle(frame, (center_x, center_y), int(self.radius), (255, 255, 255), 2)


class TriangleObject(DraggableObject):
    """Vật thể hình tam giác cân có thể kéo thả trên màn hình."""

    def __init__(
        self,
        x: int,
        y: int,
        size: int,
        color: Tuple[int, int, int] = (0, 255, 0),
        name: str = "Triangle",
    ) -> None:
        super().__init__(x, y, size, size, color, name)
        self.size = size

    def clone(self, x: int, y: int) -> "TriangleObject":
        return TriangleObject(x, y, self.size, self.color, self.name)

    def get_triangle_points(self) -> np.ndarray:
        """Lấy mảng chứa tọa độ 3 đỉnh của tam giác."""
        center_x = self.x + self.size // 2
        top_y = self.y
        bottom_y = self.y + self.size

        pt1 = (center_x, top_y)
        pt2 = (self.x, bottom_y)
        pt3 = (self.x + self.size, bottom_y)

        return np.array([pt1, pt2, pt3], np.int32)

    def is_point_inside(self, px: int, py: int) -> bool:
        """Kiểm tra con trỏ nằm trong đa giác tam giác bằng pointPolygonTest."""
        pts = self.get_triangle_points()
        return cv2.pointPolygonTest(pts, (float(px), float(py)), False) >= 0

    def draw(self, frame: np.ndarray) -> None:
        """Vẽ tam giác lên khung hình OpenCV."""
        pts = self.get_triangle_points()
        color = self.color if not self.is_dragging else (0, 255, 255)
        cv2.fillPoly(frame, [pts], color)
        cv2.polylines(frame, [pts], True, (255, 255, 255), 2)


class StarObject(DraggableObject):
    """Vật thể hình ngôi sao 5 cánh có thể kéo thả trên màn hình."""

    def __init__(
        self,
        x: int,
        y: int,
        size: int,
        color: Tuple[int, int, int] = (0, 255, 0),
        name: str = "Star",
    ) -> None:
        super().__init__(x, y, size, size, color, name)
        self.size = size

    def clone(self, x: int, y: int) -> "StarObject":
        return StarObject(x, y, self.size, self.color, self.name)

    def get_star_points(self) -> np.ndarray:
        """Lấy mảng 10 đỉnh (ngoại tiếp và nội tiếp) của ngôi sao 5 cánh."""
        center_x = self.x + self.size // 2
        center_y = self.y + self.size // 2
        outer_radius = self.size // 2
        inner_radius = self.size // 4

        points = []
        for i in range(10):
            angle = math.pi / 2 + i * math.pi / 5
            radius = outer_radius if i % 2 == 0 else inner_radius
            px = int(center_x + radius * math.cos(angle))
            py = int(center_y - radius * math.sin(angle))
            points.append([px, py])

        return np.array(points, np.int32)

    def is_point_inside(self, px: int, py: int) -> bool:
        """Kiểm tra con trỏ nằm trong đa giác ngôi sao bằng pointPolygonTest."""
        pts = self.get_star_points()
        return cv2.pointPolygonTest(pts, (float(px), float(py)), False) >= 0

    def draw(self, frame: np.ndarray) -> None:
        """Vẽ ngôi sao lên khung hình OpenCV."""
        pts = self.get_star_points()
        color = self.color if not self.is_dragging else (0, 255, 255)
        cv2.fillPoly(frame, [pts], color)
        cv2.polylines(frame, [pts], True, (255, 255, 255), 2)


class DraggableObjectManager:
    """Trình quản lý tập hợp các vật thể kéo thả và con trỏ điều khiển bằng cử chỉ tay.

    Tích hợp bộ lọc mượt tọa độ EMA (Exponential Moving Average) giúp chuyển động con trỏ
    siêu mượt và chống rung tay khi tương tác.

    Attributes:
        objects (List[DraggableObject]): Danh sách các vật thể đang có trên màn hình.
        active_object (Optional[DraggableObject]): Vật thể đang được kéo thả hiện tại.
        visible (bool): Trạng thái ẩn/hiện của giao diện tương tác vật thể.
        smooth_alpha (float): Hệ số làm mượt EMA con trỏ (0.0 < alpha <= 1.0).
    """

    def __init__(self, smooth_alpha: float = 0.4) -> None:
        """Khởi tạo quản lý vật thể và bộ lọc mượt con trỏ EMA.

        Args:
            smooth_alpha: Hệ số làm mượt EMA (giá trị mặc định 0.4 cho phản hồi mượt & nhạy).
        """
        self.objects: List[DraggableObject] = []
        self.active_object: Optional[DraggableObject] = None
        self.prev_gesture: Optional[str] = None
        self.visible: bool = False
        self.prev_motion_gesture: Optional[str] = None

        # Cấu hình lọc mượt tọa độ con trỏ (EMA filter)
        self.smooth_alpha: float = smooth_alpha
        self.smooth_x: Optional[float] = None
        self.smooth_y: Optional[float] = None

    def add_object(self, obj: DraggableObject) -> None:
        """Thêm một vật thể mới vào danh sách quản lý."""
        self.objects.append(obj)

    def toggle_visibility(self) -> None:
        """Bật hoặc tắt trạng thái hiển thị của các vật thể."""
        self.visible = not self.visible
        if not self.visible and self.active_object is not None:
            self.active_object.stop_drag()
            self.active_object = None

    def get_thumb_index_midpoint(
        self,
        hand_landmarks: Any,
        frame_width: int,
        frame_height: int,
        smooth: bool = True,
    ) -> Tuple[int, int]:
        """Tính tọa độ điểm giữa ngón cái (ID 4) và ngón trỏ (ID 8) có lọc mượt EMA.

        Args:
            hand_landmarks: 21 điểm mốc bàn tay từ MediaPipe.
            frame_width: Chiều rộng khung hình.
            frame_height: Chiều cao khung hình.
            smooth: Có áp dụng bộ lọc mượt EMA hay không.

        Returns:
            Tuple[int, int]: Tọa độ (X, Y) của con trỏ điều khiển.
        """
        lm = hand_landmarks.landmark
        thumb_tip = lm[4]
        index_tip = lm[8]

        raw_x = (thumb_tip.x + index_tip.x) / 2.0 * frame_width
        raw_y = (thumb_tip.y + index_tip.y) / 2.0 * frame_height

        if smooth:
            if self.smooth_x is None or self.smooth_y is None:
                self.smooth_x, self.smooth_y = raw_x, raw_y
            else:
                self.smooth_x = (
                    self.smooth_alpha * raw_x + (1 - self.smooth_alpha) * self.smooth_x
                )
                self.smooth_y = (
                    self.smooth_alpha * raw_y + (1 - self.smooth_alpha) * self.smooth_y
                )
            return int(self.smooth_x), int(self.smooth_y)

        return int(raw_x), int(raw_y)

    def update(
        self,
        hand_landmarks: Any,
        static_gesture: str,
        motion_gesture: str,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Cập nhật trạng thái tương tác kéo thả, đổi màu và xóa vật thể theo cử chỉ.

        Args:
            hand_landmarks: 21 điểm mốc bàn tay.
            static_gesture: Nhãn cử chỉ tĩnh đã làm mượt.
            motion_gesture: Nhãn cử chỉ chuyển động.
            frame_width: Chiều rộng khung hình.
            frame_height: Chiều cao khung hình.
        """
        if hand_landmarks is None or not hasattr(hand_landmarks, "landmark"):
            self.prev_motion_gesture = None
            self.prev_gesture = None
            self.smooth_x = None
            self.smooth_y = None
            if self.active_object is not None:
                self.active_object.stop_drag()
                self.active_object = None
            return

        # Xử lý cử chỉ On/Off để ẩn/hiện giao diện
        if motion_gesture == "On/Off" and self.prev_motion_gesture != "On/Off":
            self.toggle_visibility()

        self.prev_motion_gesture = motion_gesture

        if not self.visible:
            return

        # Lấy tọa độ con trỏ đã được làm mượt qua EMA
        mid_x, mid_y = self.get_thumb_index_midpoint(
            hand_landmarks, frame_width, frame_height
        )

        # Cử chỉ "Stop" (xòe bàn tay): Xóa vật thể tại con trỏ
        if static_gesture == "Stop" and self.prev_gesture != "Stop":
            for obj in reversed(self.objects):
                if obj.is_point_inside(mid_x, mid_y):
                    self.objects.remove(obj)
                    if self.active_object == obj:
                        self.active_object = None
                    break

        # Cử chỉ "Options": Đổi màu ngẫu nhiên vật thể tại con trỏ
        if static_gesture == "Options" and self.prev_gesture != "Options":
            for obj in reversed(self.objects):
                if obj.is_point_inside(mid_x, mid_y):
                    obj.change_color()
                    break

        # Cử chỉ "Select": Bắt đầu kéo hoặc tiếp tục di chuyển vật thể
        if static_gesture == "Select":
            if self.active_object is None:
                for obj in reversed(self.objects):
                    if obj.start_drag(mid_x, mid_y):
                        self.active_object = obj
                        # Đưa vật thể đang chọn lên đầu danh sách (ưu tiên hiển thị trên cùng)
                        self.objects.remove(obj)
                        self.objects.append(obj)
                        break
            else:
                self.active_object.update_position(
                    mid_x, mid_y, frame_width, frame_height
                )
        else:
            if self.active_object is not None:
                self.active_object.stop_drag()
                self.active_object = None

        self.prev_gesture = static_gesture

    def draw_all(self, frame: np.ndarray) -> None:
        """Vẽ tất cả các vật thể lên khung hình nếu giao diện đang bật."""
        if self.visible:
            for obj in self.objects:
                obj.draw(frame)

    def draw_cursor(
        self,
        frame: np.ndarray,
        hand_landmarks: Any,
        frame_width: int,
        frame_height: int,
    ) -> None:
        """Vẽ con trỏ điều khiển tròn với viền nổi bật tại vị trí giữa ngón cái và trỏ."""
        if hand_landmarks is not None and hasattr(hand_landmarks, "landmark"):
            mid_x, mid_y = self.get_thumb_index_midpoint(
                hand_landmarks, frame_width, frame_height
            )
            cv2.circle(frame, (mid_x, mid_y), 10, (0, 0, 255), -1)
            cv2.circle(frame, (mid_x, mid_y), 12, (255, 255, 255), 2)


class ShapeMenu:
    """Trình đơn chọn hình để thêm vật thể mới (Menu giao diện).

    Attributes:
        is_open (bool): Trạng thái mở/đóng của menu.
        shapes (List[DraggableObject]): Các hình mẫu thu nhỏ hiển thị trong menu.
    """

    def __init__(self) -> None:
        """Khởi tạo danh sách các hình mẫu thu nhỏ."""
        self.is_open: bool = False
        self.shapes: List[DraggableObject] = [
            DraggableObject(0, 0, 60, 50, (100, 200, 100), "Rectangle"),
            CircleObject(0, 0, 30, (100, 100, 200), "Circle"),
            TriangleObject(0, 0, 60, (200, 100, 100), "Triangle"),
            StarObject(0, 0, 60, (200, 200, 100), "Star"),
        ]
        self.prev: bool = False

    def handle_click(self, px: int, py: int) -> Optional[int]:
        """Xử lý nhấp chọn một mục hình dạng trong menu.

        Args:
            px: Tọa độ X nhấp.
            py: Tọa độ Y nhấp.

        Returns:
            Optional[int]: Chỉ số của hình được chọn hoặc None.
        """
        if not self.is_open:
            return None
        for i, shape in enumerate(self.shapes):
            if shape.is_point_inside(px, py):
                self.is_open = False
                return i
        return None

    def update(
        self,
        hand_landmarks: Any,
        static_gesture: str,
        w: int,
        h: int,
        mgr: DraggableObjectManager,
    ) -> None:
        """Cập nhật trạng thái mở menu và khởi tạo vật thể mới khi người dùng chạm vào menu."""
        if not hand_landmarks or not hasattr(hand_landmarks, "landmark") or not mgr.visible:
            self.prev = False
            self.is_open = False
            return

        mx, my = mgr.get_thumb_index_midpoint(hand_landmarks, w, h)
        is_select = static_gesture == "Select"

        if is_select and not self.prev:
            # Vùng nút MENU góc dưới bên trái (10 <= x <= 110, h-70 <= y <= h-10)
            if 10 <= mx <= 110 and h - 70 <= my <= h - 10:
                self.is_open = not self.is_open
            else:
                idx = self.handle_click(mx, my)
                if idx is not None:
                    target = self.shapes[idx]
                    cx, cy = w // 2 - 50, h // 2 - 50
                    # Tạo vật thể kích thước đầy đủ ở giữa màn hình
                    if isinstance(target, CircleObject):
                        new_obj: DraggableObject = CircleObject(cx, cy, 50, target.color, target.name)
                    elif isinstance(target, TriangleObject):
                        new_obj = TriangleObject(cx, cy, 100, target.color, target.name)
                    elif isinstance(target, StarObject):
                        new_obj = StarObject(cx, cy, 100, target.color, target.name)
                    else:
                        new_obj = DraggableObject(cx, cy, 100, 80, target.color, target.name)
                    mgr.add_object(new_obj)
        self.prev = is_select

    def draw(self, frame: np.ndarray, visible: bool = True) -> None:
        """Vẽ nút MENU và danh sách các hình mẫu thu nhỏ khi mở menu."""
        if not visible:
            return
        h = frame.shape[0]
        # Vẽ nút MENU góc dưới trái
        cv2.rectangle(frame, (10, h - 70), (110, h - 10), (60, 60, 60), -1)
        cv2.rectangle(frame, (10, h - 70), (110, h - 10), (255, 255, 255), 2)
        cv2.putText(
            frame,
            "MENU",
            (25, h - 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Hiển thị các hình mẫu theo cột đứng đi lên từ nút MENU
        if self.is_open:
            spacing = 70
            for i, shape in enumerate(self.shapes):
                y_pos = h - 70 - (i + 1) * spacing
                shape.x = 25
                shape.y = y_pos
                shape.draw(frame)


