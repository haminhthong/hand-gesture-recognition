"""File tương thích công cụ thu thập dữ liệu MediaPipe Landmarks.

Chuyển hướng lời gọi đến tools/collect_landmarks.py.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tools.collect_landmarks import main

if __name__ == "__main__":
    main()
