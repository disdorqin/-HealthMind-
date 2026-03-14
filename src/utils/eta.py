from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ETAEstimator:
    """Estimate remaining time from the first completed unit (epoch/model)."""

    total_units: int
    first_unit_duration: Optional[float] = None
    started_at: float = 0.0

    def start(self) -> None:
        self.started_at = time.time()

    def observe_first_unit(self) -> None:
        if self.started_at <= 0:
            self.start()
        if self.first_unit_duration is None:
            self.first_unit_duration = max(0.01, time.time() - self.started_at)

    def estimate_remaining_seconds(self, completed_units: int) -> Optional[float]:
        if self.first_unit_duration is None:
            return None
        remaining_units = max(self.total_units - completed_units, 0)
        return remaining_units * self.first_unit_duration

    @staticmethod
    def format_seconds(seconds: Optional[float]) -> str:
        if seconds is None:
            return "estimating..."
        secs = int(max(seconds, 0))
        hours, rem = divmod(secs, 3600)
        minutes, sec = divmod(rem, 60)
        if hours > 0:
            return f"{hours}h {minutes}m {sec}s"
        if minutes > 0:
            return f"{minutes}m {sec}s"
        return f"{sec}s"
