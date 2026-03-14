from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


class BaseForecastModel(ABC):
    """Unified interface for all base models."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        progress_callback: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        pass
