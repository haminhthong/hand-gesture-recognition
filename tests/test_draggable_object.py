import unittest
from types import SimpleNamespace
import numpy as np

from DraggableObject import (
    CircleObject,
    DraggableObject,
    DraggableObjectManager,
    ShapeMenu,
    StarObject,
    TriangleObject,
)


class DraggableObjectTests(unittest.TestCase):
    def test_draggable_object_bounds_and_drag(self):
        obj = DraggableObject(x=100, y=100, width=50, height=50)
        self.assertTrue(obj.is_point_inside(120, 120))
        self.assertFalse(obj.is_point_inside(80, 80))

        started = obj.start_drag(120, 120)
        self.assertTrue(started)
        self.assertTrue(obj.is_dragging)

        obj.update_position(150, 150, frame_width=640, frame_height=480)
        self.assertEqual(obj.x, 130)
        self.assertEqual(obj.y, 130)

        obj.stop_drag()
        self.assertFalse(obj.is_dragging)

    def test_circle_object_collision(self):
        circle = CircleObject(x=100, y=100, radius=30)
        self.assertTrue(circle.is_point_inside(130, 130))  # Tâm tròn
        self.assertFalse(circle.is_point_inside(100, 100))  # Góc ngoài hình vuông bao quanh

    def test_triangle_object_collision(self):
        triangle = TriangleObject(x=100, y=100, size=60)
        self.assertTrue(triangle.is_point_inside(130, 110))
        self.assertFalse(triangle.is_point_inside(100, 100))

    def test_star_object_collision(self):
        star = StarObject(x=100, y=100, size=60)
        self.assertTrue(star.is_point_inside(130, 130))

    def test_object_manager_ema_smoothing(self):
        mgr = DraggableObjectManager(smooth_alpha=0.5)

        def make_hand(x, y):
            landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(21)]
            landmarks[4] = SimpleNamespace(x=x, y=y)
            landmarks[8] = SimpleNamespace(x=x, y=y)
            return SimpleNamespace(landmark=landmarks)

        h1 = make_hand(0.2, 0.4)
        x1, y1 = mgr.get_thumb_index_midpoint(h1, 100, 100, smooth=True)
        self.assertEqual((x1, y1), (20, 40))

        h2 = make_hand(0.4, 0.6)
        x2, y2 = mgr.get_thumb_index_midpoint(h2, 100, 100, smooth=True)
        # EMA: 0.5 * 40 + 0.5 * 20 = 30; 0.5 * 60 + 0.5 * 40 = 50
        self.assertEqual((x2, y2), (30, 50))

    def test_manager_toggle_and_actions(self):
        mgr = DraggableObjectManager()
        obj = DraggableObject(x=50, y=50, width=50, height=50)
        mgr.add_object(obj)

        mgr.toggle_visibility()
        self.assertTrue(mgr.visible)

        def make_hand(x, y):
            landmarks = [SimpleNamespace(x=0.0, y=0.0) for _ in range(21)]
            landmarks[4] = SimpleNamespace(x=x, y=y)
            landmarks[8] = SimpleNamespace(x=x, y=y)
            return SimpleNamespace(landmark=landmarks)

        hand = make_hand(0.75 / 10.0, 0.75 / 10.0)  # Tọa độ ~ (75, 75) trên khung 1000x1000

        # Cử chỉ Options -> Đổi màu
        original_color = obj.color
        mgr.update(hand, "Options", "Still", 1000, 1000)

        # Cử chỉ Stop -> Xóa vật thể
        mgr.update(hand, "Stop", "Still", 1000, 1000)
        self.assertEqual(len(mgr.objects), 0)


if __name__ == "__main__":
    unittest.main()
