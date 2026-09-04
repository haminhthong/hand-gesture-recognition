import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hand_gesture_controller.performance_monitor import PerformanceMonitor


class PerformanceMonitorTests(unittest.TestCase):
    def test_summary_contains_reproducible_metrics(self):
        with patch(
            "hand_gesture_controller.performance_monitor.time.perf_counter",
            side_effect=[0.0, 1.0, 2.0, 3.0],
        ):
            monitor = PerformanceMonitor(window_size=5)
            monitor.record_frame(0.5)
            monitor.record_frame(1.5)
            summary = monitor.summary()
        self.assertEqual(summary["total_frames"], 2)
        self.assertEqual(summary["average_fps"], 1.0)
        self.assertEqual(summary["average_latency_ms"], 500.0)

    def test_save_writes_valid_json(self):
        monitor = PerformanceMonitor()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "benchmark.json"
            monitor.save(output)
            data = json.loads(output.read_text(encoding="utf-8"))
        self.assertIn("average_fps", data)
        self.assertIn("average_latency_ms", data)


if __name__ == "__main__":
    unittest.main()
