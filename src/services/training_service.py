from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional


@dataclass
class TrainingProgress:
    model: str
    model_index: int
    model_total: int
    epoch: int
    epochs: int
    train_loss: float | None
    val_loss: float | None
    progress_ratio: float
    eta_seconds: float | None


class TrainingService:
    """Progress adapter with first-epoch ETA extrapolation."""

    def __init__(self):
        self._first_epoch_seconds: float | None = None
        self._first_epoch_started: float | None = None

    @staticmethod
    def _overall_progress(model_index: int, model_total: int, epoch: int, epochs: int) -> float:
        completed_models = max(model_index - 1, 0)
        return min(1.0, (completed_models + epoch / max(epochs, 1)) / max(model_total, 1))

    def _estimate_eta(self, epochs: int, model_total: int, progress_ratio: float) -> float | None:
        if self._first_epoch_seconds is None:
            return None
        total_seconds = self._first_epoch_seconds * max(epochs, 1) * max(model_total, 1)
        return max(0.0, total_seconds * (1.0 - progress_ratio))

    @staticmethod
    def format_eta(seconds: float | None) -> str:
        if seconds is None:
            return "estimating..."
        secs = int(max(0, seconds))
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        if m > 0:
            return f"{m}m {s}s"
        return f"{s}s"

    def build_callback(
        self,
        epochs: int,
        selected_models: List[str],
        on_progress: Callable[[TrainingProgress], None],
    ) -> Callable[[Dict[str, Any]], None]:
        model_total = max(1, len(selected_models))

        def _callback(payload: Dict[str, Any]) -> None:
            model = str(payload.get("model", "unknown"))
            model_index = int(payload.get("model_index", 1))
            epoch = int(payload.get("epoch", 1))

            if epoch == 1:
                if self._first_epoch_started is None:
                    self._first_epoch_started = time.time()
                elif self._first_epoch_seconds is None:
                    self._first_epoch_seconds = max(0.01, time.time() - self._first_epoch_started)

            progress_ratio = self._overall_progress(model_index, model_total, epoch, epochs)
            eta_seconds = self._estimate_eta(epochs, model_total, progress_ratio)

            on_progress(
                TrainingProgress(
                    model=model,
                    model_index=model_index,
                    model_total=model_total,
                    epoch=epoch,
                    epochs=epochs,
                    train_loss=payload.get("train_loss"),
                    val_loss=payload.get("val_loss"),
                    progress_ratio=progress_ratio,
                    eta_seconds=eta_seconds,
                )
            )

        return _callback
