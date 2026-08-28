import math
import random

import cv2
import numpy as np


class DraggableObject:
    def __init__(self, x, y, width, height, color=(0, 255, 0), name="Object"):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.name = name
        self.is_dragging = False
        self.offset_x = 0
        self.offset_y = 0
    
    def is_point_inside(self, px, py):
        return (
            self.x <= px <= self.x + self.width
            and self.y <= py <= self.y + self.height
        )
    
    def start_drag(self, px, py):
        if self.is_point_inside(px, py):
            self.is_dragging = True
            self.offset_x = px - self.x
            self.offset_y = py - self.y
            return True
        return False
    
    def update_position(self, px, py, frame_width=None, frame_height=None):
        if self.is_dragging:
            self.x = px - self.offset_x
            self.y = py - self.offset_y
            if frame_width is not None:
                self.x = max(0, min(self.x, max(0, frame_width - self.width)))
            if frame_height is not None:
                self.y = max(0, min(self.y, max(0, frame_height - self.height)))
    
    def stop_drag(self):
        self.is_dragging = False
    
    def change_color(self):
        self.color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    
    def clone(self, x, y):
        return DraggableObject(x, y, self.width, self.height, self.color, self.name)
    
    def draw(self, frame):
        color = self.color if not self.is_dragging else (0, 255, 255)
        top_left = (self.x, self.y)
        bottom_right = (self.x + self.width, self.y + self.height)
        cv2.rectangle(frame, top_left, bottom_right, color, -1)
        cv2.rectangle(frame, top_left, bottom_right, (255, 255, 255), 2)


class CircleObject(DraggableObject):
    def __init__(self, x, y, radius, color=(0, 255, 0), name="Circle"):
        super().__init__(x, y, radius * 2, radius * 2, color, name)
        self.radius = radius
    
    def clone(self, x, y):
        return CircleObject(x, y, self.radius, self.color, self.name)
    
    def is_point_inside(self, px, py):
        center_x = self.x + self.radius
        center_y = self.y + self.radius
        distance = math.sqrt((px - center_x)**2 + (py - center_y)**2)
        return distance <= self.radius
    
    def draw(self, frame):
        """Vẽ hình tròn."""
        center_x = self.x + self.radius
        center_y = self.y + self.radius
        color = self.color if not self.is_dragging else (0, 255, 255)
        cv2.circle(frame, (center_x, center_y), self.radius, color, -1)
        cv2.circle(frame, (center_x, center_y), self.radius, (255, 255, 255), 2)


class TriangleObject(DraggableObject):
    """Vật thể hình tam giác."""

    def __init__(self, x, y, size, color=(0, 255, 0), name="Triangle"):
        super().__init__(x, y, size, size, color, name)
        self.size = size
    
    def clone(self, x, y):
        return TriangleObject(x, y, self.size, self.color, self.name)
    
    def get_triangle_points(self):
        """Lấy ba đỉnh của tam giác."""
        center_x = self.x + self.size // 2
        top_y = self.y
        bottom_y = self.y + self.size
        
        pt1 = (center_x, top_y)
        pt2 = (self.x, bottom_y)
        pt3 = (self.x + self.size, bottom_y)
        
        return np.array([pt1, pt2, pt3], np.int32)
    
    def is_point_inside(self, px, py):
        """Kiểm tra một điểm có nằm trong tam giác hay không."""
        pts = self.get_triangle_points()
        return cv2.pointPolygonTest(pts, (float(px), float(py)), False) >= 0
    
    def draw(self, frame):
        """Vẽ hình tam giác."""
        pts = self.get_triangle_points()
        color = self.color if not self.is_dragging else (0, 255, 255)
        cv2.fillPoly(frame, [pts], color)
        cv2.polylines(frame, [pts], True, (255, 255, 255), 2)


class StarObject(DraggableObject):
    """Vật thể hình ngôi sao."""

    def __init__(self, x, y, size, color=(0, 255, 0), name="Star"):
        super().__init__(x, y, size, size, color, name)
        self.size = size
    
    def clone(self, x, y):
        return StarObject(x, y, self.size, self.color, self.name)
    
    def get_star_points(self):
        """Lấy các đỉnh của ngôi sao năm cánh."""
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
    
    def is_point_inside(self, px, py):
        """Kiểm tra một điểm có nằm trong ngôi sao hay không."""
        pts = self.get_star_points()
        return cv2.pointPolygonTest(pts, (float(px), float(py)), False) >= 0
    
    def draw(self, frame):
        """Vẽ hình ngôi sao."""
        pts = self.get_star_points()
        color = self.color if not self.is_dragging else (0, 255, 255)
        cv2.fillPoly(frame, [pts], color)
        cv2.polylines(frame, [pts], True, (255, 255, 255), 2)


class DraggableObjectManager:
    def __init__(self):
        """Quản lý các vật thể có thể kéo thả."""
        self.objects = []
        self.active_object = None
        self.prev_gesture = None
        self.visible = False
        self.prev_motion_gesture = None
    
    def add_object(self, obj):
        self.objects.append(obj)
    
    def toggle_visibility(self):
        self.visible = not self.visible
        # Nếu đang ẩn, dừng kéo thả
        if not self.visible and self.active_object is not None:
            self.active_object.stop_drag()
            self.active_object = None
    
    def get_thumb_index_midpoint(self, hand_landmarks, frame_width, frame_height):
        lm = hand_landmarks.landmark
        thumb_tip = lm[4]
        index_tip = lm[8]
        
        # Tính điểm giữa
        mid_x = int((thumb_tip.x + index_tip.x) / 2 * frame_width)
        mid_y = int((thumb_tip.y + index_tip.y) / 2 * frame_height)
        
        return mid_x, mid_y
    
    def update(self, hand_landmarks, static_gesture, motion_gesture, frame_width, frame_height):
        """Cập nhật trạng thái kéo thả dựa trên cử chỉ."""
        if hand_landmarks is None:
            self.prev_motion_gesture = None
            self.prev_gesture = None
            if self.active_object is not None:
                self.active_object.stop_drag()
                self.active_object = None
            return
        
        # Xử lý cử chỉ On/Off để ẩn/hiện vật thể
        if motion_gesture == "On/Off" and self.prev_motion_gesture != "On/Off":
            self.toggle_visibility()
        
        self.prev_motion_gesture = motion_gesture
        
        # Chỉ cho phép kéo thả khi vật thể đang hiển thị
        if not self.visible:
            return
        
        # Lấy tọa độ điểm giữa ngón cái và ngón trỏ
        mid_x, mid_y = self.get_thumb_index_midpoint(hand_landmarks, frame_width, frame_height)
        
        # Xử lý cử chỉ Stop để xóa vật thể
        if static_gesture == "Stop" and self.prev_gesture != "Stop":
            for obj in reversed(self.objects):
                if obj.is_point_inside(mid_x, mid_y):
                    self.objects.remove(obj)
                    if self.active_object == obj:
                        self.active_object = None
                    break
        
        # Xử lý cử chỉ Options để đổi màu vật thể
        if static_gesture == "Options" and self.prev_gesture != "Options":
            for obj in reversed(self.objects):
                if obj.is_point_inside(mid_x, mid_y):
                    obj.change_color()
                    break
        
        # Nếu đang nhận cử chỉ "Select"
        if static_gesture == "Select":
            if self.active_object is None:
                for obj in reversed(self.objects):
                    if obj.start_drag(mid_x, mid_y):
                        self.active_object = obj
                        # Đưa vật thể đang chọn lên trên cùng.
                        self.objects.remove(obj)
                        self.objects.append(obj)
                        break
            else:
                self.active_object.update_position(mid_x, mid_y, frame_width, frame_height)
        else:
            if self.active_object is not None:
                self.active_object.stop_drag()
                self.active_object = None
        
        self.prev_gesture = static_gesture
    
    def draw_all(self, frame):
        """Vẽ tất cả vật thể lên khung hình."""
        if self.visible:
            for obj in self.objects:
                obj.draw(frame)
    
    def draw_cursor(self, frame, hand_landmarks, frame_width, frame_height):
        """Vẽ con trỏ tại điểm giữa ngón cái và ngón trỏ."""
        if hand_landmarks is not None:
            mid_x, mid_y = self.get_thumb_index_midpoint(hand_landmarks, frame_width, frame_height)
            cv2.circle(frame, (mid_x, mid_y), 10, (0, 0, 255), -1)
            cv2.circle(frame, (mid_x, mid_y), 12, (255, 255, 255), 2)


class ShapeMenu:
    def __init__(self):
        self.is_open = False
        # Các mẫu thu nhỏ dùng trong trình đơn chọn hình.
        self.shapes = [
            DraggableObject(0, 0, 60, 50, (100, 200, 100), "Rectangle"),
            CircleObject(0, 0, 30, (100, 100, 200), "Circle"),
            TriangleObject(0, 0, 60, (200, 100, 100), "Triangle"),
            StarObject(0, 0, 60, (200, 200, 100), "Star")
        ]
        self.prev = False
    
    def handle_click(self, px, py):
        if not self.is_open:
            return None
        for i, s in enumerate(self.shapes):
            if s.is_point_inside(px, py):
                self.is_open = False
                return i
        return None
    
    def update(self, hand_landmarks, static_gesture, w, h, mgr):
        if not hand_landmarks or not mgr.visible:
            self.prev = False
            self.is_open = False
            return
        
        mx, my = mgr.get_thumb_index_midpoint(hand_landmarks, w, h)
        sel = static_gesture == "Select"
        
        if sel and not self.prev:
            if 10 <= mx <= 110 and h - 70 <= my <= h - 10:
                self.is_open = not self.is_open
            else:
                idx = self.handle_click(mx, my)
                if idx is not None:
                    t = self.shapes[idx]
                    cx, cy = w // 2 - 50, h // 2 - 50
                    # Tạo vật thể mới ở kích thước đầy đủ.
                    if isinstance(t, CircleObject):
                        obj = CircleObject(cx, cy, 50, t.color, t.name)
                    elif isinstance(t, TriangleObject):
                        obj = TriangleObject(cx, cy, 100, t.color, t.name)
                    elif isinstance(t, StarObject):
                        obj = StarObject(cx, cy, 100, t.color, t.name)
                    else:
                        obj = DraggableObject(cx, cy, 100, 80, t.color, t.name)
                    mgr.add_object(obj)
        self.prev = sel
    
    def draw(self, frame, visible=True):
        if not visible:
            return
        h = frame.shape[0]
        # Nút menu ở góc dưới trái
        cv2.rectangle(frame, (10, h - 70), (110, h - 10), (60, 60, 60), -1)
        cv2.rectangle(frame, (10, h - 70), (110, h - 10), (255, 255, 255), 2)
        cv2.putText(frame, "MENU", (25, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Hiển thị các mẫu theo chiều dọc, từ nút trình đơn đi lên.
        if self.is_open:
            spacing = 70
            for i, s in enumerate(self.shapes):
                y_pos = h - 70 - (i + 1) * spacing
                s.x = 25
                s.y = y_pos
                s.draw(frame)

