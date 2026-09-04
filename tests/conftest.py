"""Configuration for pytest."""

import os
import sys

# Thêm thư mục src vào sys.path để pytest tự động nạp gói hand_gesture_controller
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
