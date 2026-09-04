import unittest
from types import SimpleNamespace
from unittest.mock import patch

from hand_gesture_controller.gesture_detector import GestureDetector


def create_open_hand(center_x=0.5):
    points = [SimpleNamespace(x=center_x, y=0.8) for _ in range(21)]
    for tip, joint, x in (
        (4, 3, center_x - 0.2),
        (8, 6, center_x - 0.1),
        (12, 10, center_x),
        (16, 14, center_x + 0.1),
        (20, 18, center_x + 0.2),
    ):
        points[joint] = SimpleNamespace(x=x, y=0.55)
        points[tip] = SimpleNamespace(x=x, y=0.25)
    return SimpleNamespace(landmark=points)


class GestureDetectorTests(unittest.TestCase):
    def test_small_motion_is_still(self):
        detector = GestureDetector()
        direction = detector.get_move_direction((0.501, 0.5), (0.5, 0.5))
        self.assertEqual(direction, "Still")

    def test_wave_timeout_resets_tracking(self):
        detector = GestureDetector()
        with patch("hand_gesture_controller.gesture_detector.time.perf_counter", return_value=100.0):
            detector.detect_motion_gesture(create_open_hand())
        with patch("hand_gesture_controller.gesture_detector.time.perf_counter", return_value=103.0):
            (gesture, _), _ = detector.detect_motion_gesture(
                create_open_hand(0.6)
            )
        self.assertEqual(gesture, "Still")
        self.assertIsNone(detector.prev_center)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            GestureDetector().detect_gesture(create_open_hand(), mode="invalid")


if __name__ == "__main__":
    unittest.main()
