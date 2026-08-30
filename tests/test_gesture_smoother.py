import unittest

from GestureSmoother import GestureSmoother


class GestureSmootherTests(unittest.TestCase):
    def test_majority_becomes_stable_result(self):
        smoother = GestureSmoother(window_size=5, minimum_votes=3)
        red = (0, 0, 255)
        smoother.update("Fist", red)
        smoother.update("Unknown", (255, 255, 255))
        smoother.update("Fist", red)
        self.assertEqual(smoother.update("Fist", red), ("Fist", red))

    def test_reset_clears_previous_result(self):
        smoother = GestureSmoother(window_size=3, minimum_votes=2)
        smoother.update("Fist", (0, 0, 255))
        smoother.update("Fist", (0, 0, 255))
        smoother.reset()
        result = smoother.update("Unknown", (255, 255, 255))
        self.assertEqual(result[0], "Unknown")

    def test_invalid_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            GestureSmoother(window_size=2, minimum_votes=3)


if __name__ == "__main__":
    unittest.main()
