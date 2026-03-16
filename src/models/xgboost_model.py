from __future__ import annotations

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, List

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.core.utils.logger import logger
from .base_model import BaseForecastModel


class XGBoostForecastModel(BaseForecastModel):
    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        name: str = "xgboost",
    ):
        super().__init__(name=name)
        self.params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "objective": "reg:squarederror",
            "n_jobs": -1,
        }
        self.model = XGBRegressor(**self.params)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        save_path: str = "models/xgboost_best.joblib",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Train XGBoost with Numerical Trend Audit (Validation monitoring) and Saving Condition.
        """
        logger.info(f"Training XGBoost with {X_train.shape} samples.")
        
        eval_set = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_train, y_train), (X_val, y_val)]
        else:
            eval_set = [(X_train, y_train)]
            
        self.model.fit(
            X_train, 
            y_train, 
            eval_set=eval_set, 
            verbose=False
        )
        
        results = self.model.evals_result()
        train_rmse = results['validation_0']['rmse']
        val_rmse = results.get('validation_1', {}).get('rmse', [])
        
        # --- Numerical Trend Audit ---
        if len(train_rmse) > 10:
            std_dev = np.std(train_rmse[-10:])
            if std_dev < 1e-4:
                logger.warning(f"XGBoost Loss flatline (std={std_dev:.6f}). Training has stabilized.")
        
        # Validation Metrics
        metrics = {}
        if X_val is not None:
            val_preds = self.model.predict(X_val)
            val_r2 = r2_score(y_val, val_preds)
            val_mae = mean_absolute_error(y_val, val_preds)
            logger.info(f"XGBoost Validation - R2: {val_r2:.4f}, MAE: {val_mae:.4f}")
            
            # --- Saving Condition ---
            if val_r2 > 0:
                self.save(Path(save_path))
                logger.info(f"Saved XGBoost model to {save_path} (R2 > 0 check passed)")
            else:
                logger.warning(f"XGBoost model R2 ({val_r2:.4f}) <= 0. Not saving model.")
        else:
             self.save(Path(save_path))

        self._plot_metrics(train_rmse, val_rmse, Path(save_path).parent)
        return metrics

    def predict(self, X: np.ndarray, **kwargs: Any) -> np.ndarray:
        return self.model.predict(X)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path: Union[str, Path]) -> None:
        self.model = joblib.load(path)
        
    def _plot_metrics(self, train_loss: List[float], val_loss: List[float], save_dir: Path):
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label='Train RMSE')
        if val_loss:
            plt.plot(val_loss, label='Val RMSE')
        plt.title('XGBoost Training RMSE')
        plt.xlabel('Round')
        plt.ylabel('RMSE')
        plt.legend()
        plot_path = save_dir / 'metrics' / 'xgboost_loss.png'
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path)
        plt.close()
