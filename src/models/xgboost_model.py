from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from xgboost import XGBRegressor

from src.core.utils.logger import logger
from .base_model import BaseForecastModel


class XGBoostForecastModel(BaseForecastModel):
    def __init__(self):
        super().__init__(name="xgboost")
        self.model = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        )

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        if X.ndim == 3:
            n, t, f = X.shape
            return X.reshape(n, t * f)
        return X

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        progress_callback: Optional[Any] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        X_train_2d = self._flatten(X_train)

        eval_set = None
        if X_val is not None and y_val is not None and len(X_val) > 0:
            eval_set = [(self._flatten(X_val), y_val)]

        self.model.fit(
            X_train_2d,
            y_train,
            eval_set=eval_set,
            verbose=False,
        )

        if progress_callback is not None:
            progress_callback({"model": self.name, "epoch": 1, "epochs": 1, "train_loss": None, "val_loss": None})

        logger.info("XGBoost training completed")
        return {"history": [{"epoch": 1}]}

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        X_2d = self._flatten(X)
        pred = self.model.predict(X_2d)
        return pred.astype(np.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Path) -> None:
        self.model = joblib.load(path)
