"""Mô-đun PerformanceMonitor theo dõi FPS và Latency (ms) theo cửa sổ trượt."""

import json
import logging
import statistics
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Bộ giám sát hiệu năng theo dõi Tốc độ khung hình (FPS) và Độ trễ xử lý (Latency ms).

    Attributes:
        window_size (int): Số khung hình lưu giữ trong bộ nhớ đệm.
        frame_times (Deque[float]): Đệm lưu khoảng thời gian giữa các khung hình (giây).
        latencies_ms (Deque[float]): Đệm lưu thời gian xử lý từng khung hình (mili-giây).
        total_frames (int): Tổng số khung hình đã xử lý.
        started_at (float): Thời điểm bắt đầu đo.
    """

    def __init__(self, window_size: int = 120) -> None:
        """Khởi tạo PerformanceMonitor.

        Args:
            window_size: Kích thước cửa sổ trượt (mặc định 120 khung hình).

        Raises:
            ValueError: Nếu window_size < 1.
        """
        if window_size < 1:
            raise ValueError("Kích thước cửa sổ phải lớn hơn 0.")
        self.window_size = window_size
        self.frame_times: Deque[float] = deque(maxlen=window_size)
        self.latencies_ms: Deque[float] = deque(maxlen=window_size)
        self.total_frames: int = 0
        self.started_at: float = time.perf_counter()
        self.previous_frame_at: Optional[float] = None

    def record_frame(self, processing_started_at: float) -> None:
        """Ghi nhận mốc thời gian hoàn tất khung hình và đo độ trễ xử lý.

        Args:
            processing_started_at: Thời điểm bắt đầu xử lý khung hình (time.perf_counter()).
        """
        now = time.perf_counter()
        if self.previous_frame_at is not None:
            self.frame_times.append(now - self.previous_frame_at)
        self.previous_frame_at = now
        self.latencies_ms.append((now - processing_started_at) * 1000.0)
        self.total_frames += 1

    @property
    def average_fps(self) -> float:
        """Tính số khung hình trung bình trên giây (FPS) theo cửa sổ trượt."""
        if not self.frame_times:
            return 0.0
        average_frame_time = statistics.fmean(self.frame_times)
        return 1.0 / average_frame_time if average_frame_time > 0 else 0.0

    @property
    def average_latency_ms(self) -> float:
        """Tính độ trễ xử lý trung bình (Latency ms) theo cửa sổ trượt."""
        if not self.latencies_ms:
            return 0.0
        return statistics.fmean(self.latencies_ms)

    def get_percentile_latencies(self) -> Dict[str, float]:
        """Tính phân vị độ trễ P50, P95, P99 trong cửa sổ trượt."""
        if not self.latencies_ms:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        arr = np.array(list(self.latencies_ms))
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def summary(self) -> Dict[str, Union[int, float]]:
        """Xuất báo cáo tổng hợp thông số hiệu năng dưới dạng Dictionary."""
        elapsed = time.perf_counter() - self.started_at
        percentiles = self.get_percentile_latencies()
        return {
            "total_frames": self.total_frames,
            "elapsed_seconds": round(elapsed, 3),
            "average_fps": round(self.average_fps, 2),
            "average_latency_ms": round(self.average_latency_ms, 2),
            "p50_latency_ms": round(percentiles["p50"], 2),
            "p95_latency_ms": round(percentiles["p95"], 2),
            "p99_latency_ms": round(percentiles["p99"], 2),
        }

    def save(self, output_path: Union[str, Path]) -> None:
        """Xuất báo cáo hiệu năng ra tệp định dạng JSON.

        Args:
            output_path: Đường dẫn tệp JSON đầu ra.
        """
        try:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(self.summary(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.info("Đã lưu báo cáo benchmark thành công: %s", path)
        except OSError as error:
            logger.error("Không thể lưu benchmark: %s", error)
