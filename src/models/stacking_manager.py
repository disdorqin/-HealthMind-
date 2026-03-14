from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.linear_model import Ridge


class StackingManager:
    """Linear meta learner over base model outputs."""

    def __init__(self, alpha: float = 1.0):
        self.meta_learner = Ridge(alpha=alpha, random_state=42)
        self.is_fitted = False

    def train(self, base_predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> Dict[str, float]:
        keys = sorted(base_predictions.keys())
        X_meta = np.column_stack([base_predictions[k].reshape(-1) for k in keys])
        self.meta_learner.fit(X_meta, y_true)
        self.is_fitted = True

        pred = self.meta_learner.predict(X_meta)
        mae = float(np.mean(np.abs(pred - y_true)))
        rmse = float(np.sqrt(np.mean((pred - y_true) ** 2)))
        return {"mae": mae, "rmse": rmse}

    def predict(self, base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("StackingManager is not fitted")
        keys = sorted(base_predictions.keys())
        X_meta = np.column_stack([base_predictions[k].reshape(-1) for k in keys])
        return self.meta_learner.predict(X_meta).astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"meta_learner": self.meta_learner, "is_fitted": self.is_fitted}, path)

    def load(self, path: Path) -> None:
        payload = joblib.load(path)
        self.meta_learner = payload["meta_learner"]
        self.is_fitted = bool(payload.get("is_fitted", True))
